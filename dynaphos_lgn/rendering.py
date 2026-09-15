"""Turning recruited tissue into a rendered percept.

Two renderers, chosen per electrode rather than globally:

`EllipseGaussianRenderer`
    Evaluate the local Jacobian once at the electrode centre and stretch
    the whole blob by that single linear map. One closed-form elliptical
    Gaussian per electrode -- cheap, smooth, differentiable, and it
    subsumes the isotropic case rather than being a separate path.

`ScatterRenderer`
    Let every recruited voxel's own atlas position, weighted by its
    recruitment, be the phosphene. Assumes no ellipse, so nothing has to
    hold across the patch. The splat kernel that makes the cloud
    continuous is a rendering technicality, sized from resolution and
    point spacing and never from a magnification factor.

Both return activation maps normalised to a peak of 1, the convention
the Dynaphos state machinery expects: amplitude lives in `brightness`,
shape lives here.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dynaphos_lgn.params import require


def visual_field_grid(params: Mapping, device=None, dtype=None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pixel-centre coordinates of the rendered image, in degrees."""
    res_x, res_y = require(params, 'run.resolution')
    x_org, y_org = require(params, 'run.origin')
    half = require(params, 'run.view_angle') / 2
    x = torch.linspace(x_org - half, x_org + half, res_x,
                       device=device, dtype=dtype)
    y = torch.linspace(y_org - half, y_org + half, res_y,
                       device=device, dtype=dtype)
    return torch.meshgrid(x, y, indexing='xy')


class EllipseGaussianRenderer:
    """Anisotropic Gaussian phosphenes, one per electrode.

    :param centers_xy: (n, 2) phosphene centres in visual-field degrees.
    :param orientation_rad: (n,) major-axis direction, radians.
    :param params: Dynaphos-style parameter dict (needs `run`).
    :param data_kwargs: ``dict(device=..., dtype=...)``.
    """

    def __init__(self, centers_xy: np.ndarray, orientation_rad: np.ndarray,
                 params: Mapping, data_kwargs: Optional[dict] = None):
        data_kwargs = data_kwargs or {}
        grid_x, grid_y = visual_field_grid(params, **data_kwargs)
        centers = torch.as_tensor(np.asarray(centers_xy), **data_kwargs)
        theta = torch.as_tensor(np.asarray(orientation_rad), **data_kwargs)
        theta = torch.nan_to_num(theta, nan=0.0)

        dx = grid_x[None] - centers[:, 0, None, None]
        dy = grid_y[None] - centers[:, 1, None, None]
        cos = torch.cos(theta)[:, None, None]
        sin = torch.sin(theta)[:, None, None]

        # Fixed at setup: only the sigmas move at runtime.
        self.u = dx * cos + dy * sin          # along the major axis
        self.v = -dx * sin + dy * cos         # along the minor axis
        self.n_phosphenes = len(centers)
        self.resolution = (self.u.shape[-2], self.u.shape[-1])
        self._eps = torch.finfo(self.u.dtype).tiny

    def render(self, sigma_major: torch.Tensor,
               sigma_minor: torch.Tensor) -> torch.Tensor:
        """Activation maps, peak 1.

        :param sigma_major: (..., n, 1, 1) major-axis sigma, degrees.
        :param sigma_minor: (..., n, 1, 1) minor-axis sigma, degrees.
        """
        sa = torch.clamp(sigma_major, min=self._eps)
        sb = torch.clamp(sigma_minor, min=self._eps)
        return torch.exp(-0.5 * ((self.u / sa) ** 2 + (self.v / sb) ** 2))


def _gaussian_kernel1d(sigma_px: torch.Tensor, radius: int) -> torch.Tensor:
    """(n, 2*radius+1) normalised 1-D Gaussian kernels, differentiable
    in `sigma_px`."""
    offsets = torch.arange(-radius, radius + 1, device=sigma_px.device,
                           dtype=sigma_px.dtype)
    s = torch.clamp(sigma_px, min=1e-3)[:, None]
    k = torch.exp(-0.5 * (offsets[None, :] / s) ** 2)
    return k / k.sum(dim=1, keepdim=True)


class ScatterRenderer:
    """Per-voxel scatter rendering with a small, fixed splat kernel.

    Point positions are fixed at setup and only the weights move, so
    which pixels each point touches is precomputed as bilinear splat
    weights and the forward pass is a scatter-add plus a separable
    Gaussian blur -- O(points + pixels * kernel) rather than the
    O(points * pixels) a dense distance matrix would cost. Each
    electrode renders into its own patch, not the whole frame.

    :param points_xy: (n_electrodes, n_points, 2) point positions, deg.
    :param centers_xy: (n_electrodes, 2) patch centres, deg.
    :param params: Dynaphos-style parameter dict.
    :param patch_radius_deg: (n_electrodes,) half-width of each patch.
    :param bandwidth_floor_deg: (n_electrodes,) lower clamp on the splat
        kernel, from the point cloud's own spacing.
    :param bandwidth_ceiling_deg: (n_electrodes,) upper clamp, so the
        kernel can never approach the cloud's overall extent even in a
        degenerate case such as a single dominant voxel.
    """

    def __init__(self, points_xy: np.ndarray, centers_xy: np.ndarray,
                 params: Mapping, patch_radius_deg: np.ndarray,
                 bandwidth_floor_deg: np.ndarray,
                 bandwidth_ceiling_deg: np.ndarray,
                 data_kwargs: Optional[dict] = None):
        data_kwargs = data_kwargs or {}
        self.params = params
        device = data_kwargs.get('device', None)
        dtype = data_kwargs.get('dtype', torch.float32)

        res_x, res_y = require(params, 'run.resolution')
        x_org, y_org = require(params, 'run.origin')
        half = require(params, 'run.view_angle') / 2
        self.resolution = (res_y, res_x)
        self.deg_per_px_x = (2 * half) / max(res_x - 1, 1)
        self.deg_per_px_y = (2 * half) / max(res_y - 1, 1)
        self.x_min, self.y_min = x_org - half, y_org - half

        points = np.asarray(points_xy, dtype=float)
        centers = np.asarray(centers_xy, dtype=float)
        self.n_electrodes, self.n_points, _ = points.shape

        # -- patch geometry (fixed) ---------------------------------
        radius_px_x = np.ceil(np.asarray(patch_radius_deg)
                              / self.deg_per_px_x).astype(int)
        radius_px_y = np.ceil(np.asarray(patch_radius_deg)
                              / self.deg_per_px_y).astype(int)
        self.patch_radius_px = int(max(1, min(int(np.nanmax(
            np.maximum(radius_px_x, radius_px_y))), max(res_x, res_y))))
        self.patch_size = 2 * self.patch_radius_px + 1

        cx = np.round((centers[:, 0] - self.x_min) / self.deg_per_px_x)
        cy = np.round((centers[:, 1] - self.y_min) / self.deg_per_px_y)
        self.origin_px = np.stack([cx - self.patch_radius_px,
                                   cy - self.patch_radius_px], axis=1)

        # -- bilinear splat weights (fixed) --------------------------
        px = (points[..., 0] - self.x_min) / self.deg_per_px_x
        py = (points[..., 1] - self.y_min) / self.deg_per_px_y
        lx = px - self.origin_px[:, 0:1]
        ly = py - self.origin_px[:, 1:2]

        x0 = np.floor(lx).astype(int)
        y0 = np.floor(ly).astype(int)
        fx = lx - x0
        fy = ly - y0

        idx_list, w_list = [], []
        for dy in (0, 1):
            for dx in (0, 1):
                xi, yi = x0 + dx, y0 + dy
                inside = ((xi >= 0) & (xi < self.patch_size)
                          & (yi >= 0) & (yi < self.patch_size))
                w = ((fx if dx else 1 - fx) * (fy if dy else 1 - fy))
                idx_list.append(np.where(inside,
                                         yi * self.patch_size + xi, 0))
                w_list.append(np.where(inside, w, 0.0))

        self.splat_index = torch.as_tensor(
            np.stack(idx_list, axis=-1).reshape(self.n_electrodes, -1),
            device=device, dtype=torch.long)
        self.splat_weight = torch.as_tensor(
            np.stack(w_list, axis=-1).reshape(self.n_electrodes, -1),
            device=device, dtype=dtype)

        # -- patch -> image scatter indices (fixed) ------------------
        ay = np.arange(self.patch_size)
        gx, gy = np.meshgrid(ay, ay, indexing='xy')
        abs_x = self.origin_px[:, 0:1, None] + gx[None]
        abs_y = self.origin_px[:, 1:2, None] + gy[None]
        inside = ((abs_x >= 0) & (abs_x < res_x)
                  & (abs_y >= 0) & (abs_y < res_y))
        flat = np.where(inside, abs_y * res_x + abs_x, 0).astype(np.int64)
        self.image_index = torch.as_tensor(
            flat.reshape(self.n_electrodes, -1), device=device,
            dtype=torch.long)
        self.image_mask = torch.as_tensor(
            inside.reshape(self.n_electrodes, -1), device=device,
            dtype=dtype)

        # -- bandwidth clamps ---------------------------------------
        px_deg = min(self.deg_per_px_x, self.deg_per_px_y)
        floor_px = require(params, 'rendering.scatter_bandwidth_floor_pixels')
        floor = np.maximum(np.nan_to_num(bandwidth_floor_deg),
                           floor_px * px_deg)
        ceiling = np.maximum(np.nan_to_num(bandwidth_ceiling_deg),
                             floor * 1.001)
        self.bandwidth_floor = torch.as_tensor(floor, device=device,
                                               dtype=dtype)
        self.bandwidth_ceiling = torch.as_tensor(ceiling, device=device,
                                                 dtype=dtype)
        self.max_kernel_radius_px = require(params,
                                            'rendering.max_kernel_radius_px')
        self._smooth_clamp_softness = require(
            params, 'rendering.smooth_clamp_softness_fraction')
        self._dtype, self._device = dtype, device

    # ------------------------------------------------------------------
    def bandwidth_deg(self, weights: torch.Tensor,
                      points_xy: torch.Tensor,
                      adaptive: bool = True) -> torch.Tensor:
        """Splat bandwidth per electrode, degrees.

        With ``adaptive=False`` this is just the precomputed floor (the
        point cloud's own median nearest-neighbour spacing), which costs
        nothing at runtime.

        With ``adaptive=True`` it is recomputed from the live
        recruitment weights by a Scott/Silverman-style rule,
        ``h = s * n_eff ** (-1/6)``, with ``s`` the weighted spread about
        the weighted centroid and ``n_eff`` Kish's effective sample size.
        That matters at low amplitude, where only a tight cluster of
        voxels is active and a neighbourhood-sized bandwidth would
        over-smooth it. The clamps are a smooth blend, not a hard
        min/max, so no gradient is zeroed.
        """
        floor = self.bandwidth_floor
        if not adaptive:
            return floor

        w = torch.clamp(weights, min=0)
        total = w.sum(dim=-1, keepdim=True) + 1e-12
        centroid = (w.unsqueeze(-1) * points_xy).sum(dim=-2,
                                                     keepdim=True) / \
            total.unsqueeze(-1)
        delta = points_xy - centroid
        var = (w.unsqueeze(-1) * delta ** 2).sum(dim=-2) / total
        spread = torch.sqrt(var.sum(dim=-1) + 1e-12)
        n_eff = total.squeeze(-1) ** 2 / ((w ** 2).sum(dim=-1) + 1e-12)
        h = spread * torch.clamp(n_eff, min=1.0) ** (-1.0 / 6.0)
        return _smooth_clamp(h, floor, self.bandwidth_ceiling,
                             self._smooth_clamp_softness)

    def render(self, weights: torch.Tensor,
               bandwidth_deg: torch.Tensor) -> torch.Tensor:
        """Rendered activation maps, peak 1.

        :param weights: (n_electrodes, n_points) recruitment weights.
        :param bandwidth_deg: (n_electrodes,) splat bandwidth.
        :return: (n_electrodes, res_y, res_x)
        """
        n = self.n_electrodes
        w = torch.clamp(weights, min=0)
        # Each point contributes to the four pixels it falls between,
        # and `splat_weight` is point-major (p occupies slots 4p..4p+3),
        # so repeating each weight four times lines the two up.
        contrib = w.repeat_interleave(4, dim=-1) * self.splat_weight

        patch = torch.zeros(n, self.patch_size * self.patch_size,
                            device=w.device, dtype=w.dtype)
        patch = patch.scatter_add(1, self.splat_index, contrib)
        patch = patch.reshape(1, n, self.patch_size, self.patch_size)

        sigma_px = bandwidth_deg / min(self.deg_per_px_x, self.deg_per_px_y)
        radius = int(min(self.max_kernel_radius_px,
                         max(1, math.ceil(3.0 * float(
                             sigma_px.detach().max())))))
        kernel = _gaussian_kernel1d(sigma_px, radius)
        kx = kernel.reshape(n, 1, 1, -1)
        ky = kernel.reshape(n, 1, -1, 1)
        patch = F.conv2d(F.pad(patch, (radius, radius, 0, 0),
                               mode='constant'), kx, groups=n)
        patch = F.conv2d(F.pad(patch, (0, 0, radius, radius),
                               mode='constant'), ky, groups=n)

        patch = patch.reshape(n, -1)
        peak = patch.amax(dim=1, keepdim=True)
        patch = patch / (peak + 1e-12)
        patch = patch * self.image_mask

        res_y, res_x = self.resolution
        image = torch.zeros(n, res_y * res_x, device=w.device, dtype=w.dtype)
        image = image.scatter_add(1, self.image_index, patch)
        return image.reshape(n, res_y, res_x)


def _smooth_max(a: torch.Tensor, b: torch.Tensor,
                softness: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + b + torch.sqrt((a - b) ** 2 + softness ** 2))


def _smooth_min(a: torch.Tensor, b: torch.Tensor,
                softness: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + b - torch.sqrt((a - b) ** 2 + softness ** 2))


def _smooth_clamp(x: torch.Tensor, low: torch.Tensor,
                  high: torch.Tensor, softness_fraction: float = 0.05
                  ) -> torch.Tensor:
    """Differentiable stand-in for ``clamp(x, low, high)``.

    A hard clamp zeroes the gradient outside the interval. The
    smooth-min/max pair ``0.5 * (a + b +- sqrt((a - b)^2 + k^2))`` is
    indistinguishable from the identity well inside it and still passes
    a small gradient through a saturated value.
    """
    span = torch.clamp(high - low, min=1e-9)
    softness = softness_fraction * span
    return _smooth_min(_smooth_max(x, low, softness), high, softness)
