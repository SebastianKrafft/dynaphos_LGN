"""Renderers: the ellipse shortcut and the per-voxel scatter path."""
import math

import numpy as np
import pytest
import torch

from dynaphos_lgn.rendering import (EllipseGaussianRenderer, ScatterRenderer,
                                    _smooth_clamp, visual_field_grid)

KW = {'dtype': torch.float32}


@pytest.fixture(scope='module')
def PARAMS(request):
    """The shipped config, shrunk. Renderers read their clamps and
    kernel limits from `rendering`, so a hand-rolled dict would not
    exercise the real one."""
    return request.getfixturevalue('lgn_params')


def _peak_xy(image, params):
    gx, gy = visual_field_grid(params, **KW)
    flat = int(torch.argmax(image))
    return float(gx.flatten()[flat]), float(gy.flatten()[flat])


class TestEllipseRenderer:
    def test_peak_sits_at_the_requested_center(self, PARAMS):
        centers = np.array([[0.0, 0.0], [4.0, -3.0]])
        r = EllipseGaussianRenderer(centers, np.zeros(2), PARAMS, KW)
        img = r.render(torch.full((2, 1, 1), 1.0),
                       torch.full((2, 1, 1), 1.0))
        for i, c in enumerate(centers):
            x, y = _peak_xy(img[i], PARAMS)
            assert abs(x - c[0]) < 0.3 and abs(y - c[1]) < 0.3

    def test_peak_value_is_one(self, PARAMS):
        """Peak 1, up to pixel quantization."""
        img = EllipseGaussianRenderer(
            np.zeros((1, 2)), np.zeros(1), PARAMS, KW).render(
            torch.full((1, 1, 1), 1.2), torch.full((1, 1, 1), 0.6))
        assert float(img.max()) == pytest.approx(1.0, abs=0.02)

    def test_equal_sigmas_give_a_circle(self, PARAMS):
        r = EllipseGaussianRenderer(np.zeros((1, 2)), np.zeros(1), PARAMS, KW)
        img = r.render(torch.full((1, 1, 1), 1.5),
                       torch.full((1, 1, 1), 1.5))[0]
        horizontal = img[img.shape[0] // 2, :]
        vertical = img[:, img.shape[1] // 2]
        assert torch.allclose(horizontal, vertical, atol=1e-5)

    def test_unequal_sigmas_stretch_along_the_orientation(self, PARAMS):
        r = EllipseGaussianRenderer(np.zeros((1, 2)), np.zeros(1), PARAMS, KW)
        img = r.render(torch.full((1, 1, 1), 2.0),
                       torch.full((1, 1, 1), 0.5))[0]
        mid = img.shape[0] // 2
        assert img[mid].sum() > img[:, mid].sum()

    def test_rotating_by_ninety_degrees_swaps_the_axes(self, PARAMS):
        flat = EllipseGaussianRenderer(np.zeros((1, 2)), np.zeros(1),
                                       PARAMS, KW)
        upright = EllipseGaussianRenderer(np.zeros((1, 2)),
                                          np.array([np.pi / 2]), PARAMS, KW)
        a = flat.render(torch.full((1, 1, 1), 2.0),
                        torch.full((1, 1, 1), 0.5))[0]
        b = upright.render(torch.full((1, 1, 1), 2.0),
                           torch.full((1, 1, 1), 0.5))[0]
        assert torch.allclose(a, b.T, atol=1e-4)

    def test_gradients_reach_both_sigmas(self, PARAMS):
        r = EllipseGaussianRenderer(np.zeros((1, 2)), np.zeros(1), PARAMS, KW)
        sa = torch.full((1, 1, 1), 1.0, requires_grad=True)
        sb = torch.full((1, 1, 1), 0.7, requires_grad=True)
        r.render(sa, sb).sum().backward()
        assert float(sa.grad.abs()) > 0 and float(sb.grad.abs()) > 0

    def test_nan_orientation_does_not_poison_the_image(self, PARAMS):
        """A foveal voxel can have an undefined principal direction; the
        renderer must degrade to an axis-aligned blob, not to NaN."""
        r = EllipseGaussianRenderer(np.zeros((1, 2)), np.array([np.nan]),
                                    PARAMS, KW)
        img = r.render(torch.full((1, 1, 1), 1.0), torch.full((1, 1, 1), 1.0))
        assert torch.isfinite(img).all()


class TestScatterRenderer:
    def _make(self, PARAMS, spread=0.8, n=400, seed=0):
        rng = np.random.default_rng(seed)
        pts = rng.normal(0.0, spread, (1, n, 2))
        return ScatterRenderer(pts, np.zeros((1, 2)), PARAMS,
                               np.array([4.0]), np.array([0.08]),
                               np.array([0.8]), KW), pts

    def test_peak_sits_near_the_cloud_centroid(self, PARAMS):
        r, pts = self._make(PARAMS)
        w = torch.ones(1, pts.shape[1])
        img = r.render(w, r.bandwidth_deg(w, torch.as_tensor(
            pts, dtype=torch.float32), adaptive=False))
        x, y = _peak_xy(img[0], PARAMS)
        assert abs(x) < 0.8 and abs(y) < 0.8

    def test_peak_value_is_one(self, PARAMS):
        r, pts = self._make(PARAMS)
        w = torch.rand(1, pts.shape[1])
        img = r.render(w, r.bandwidth_deg(w, torch.as_tensor(
            pts, dtype=torch.float32)))
        assert float(img.max()) == pytest.approx(1.0, abs=1e-5)

    def test_an_isotropic_cloud_renders_isotropically(self, PARAMS):
        """Scatter mode should agree with the ellipse shortcut where the
        shortcut's assumption holds -- that is the sanity check that the
        two paths are rendering the same thing."""
        r, pts = self._make(PARAMS, spread=1.0, n=3000, seed=3)
        w = torch.ones(1, pts.shape[1])
        img = r.render(w, r.bandwidth_deg(w, torch.as_tensor(
            pts, dtype=torch.float32), adaptive=False))[0]
        ys, xs = torch.meshgrid(torch.arange(img.shape[0]),
                                torch.arange(img.shape[1]), indexing='ij')
        total = img.sum()
        cx = (img * xs).sum() / total
        cy = (img * ys).sum() / total
        var_x = (img * (xs - cx) ** 2).sum() / total
        var_y = (img * (ys - cy) ** 2).sum() / total
        assert float(var_x / var_y) == pytest.approx(1.0, rel=0.25)

    def test_a_stretched_cloud_renders_stretched(self, PARAMS):
        rng = np.random.default_rng(5)
        pts = rng.normal(0, 1, (1, 3000, 2)) * np.array([2.0, 0.4])
        r = ScatterRenderer(pts, np.zeros((1, 2)), PARAMS, np.array([6.0]),
                            np.array([0.08]), np.array([0.6]), KW)
        w = torch.ones(1, pts.shape[1])
        img = r.render(w, r.bandwidth_deg(w, torch.as_tensor(
            pts, dtype=torch.float32), adaptive=False))[0]
        mid = int(torch.argmax(img.sum(dim=1)))
        assert img[mid].sum() > img[:, int(torch.argmax(img.sum(dim=0)))].sum()

    def test_gradient_reaches_the_weights(self, PARAMS):
        r, pts = self._make(PARAMS)
        w = torch.rand(1, pts.shape[1], requires_grad=True)
        h = r.bandwidth_deg(w, torch.as_tensor(pts, dtype=torch.float32))
        r.render(w, h).sum().backward()
        assert float(w.grad.abs().sum()) > 0

    def test_adaptive_bandwidth_shrinks_for_a_concentrated_cloud(self, PARAMS):
        """At low amplitude only a tight cluster is meaningfully active,
        and a bandwidth sized for the whole candidate neighborhood
        would over-smooth it."""
        r, pts = self._make(PARAMS, n=600, seed=7)
        t_pts = torch.as_tensor(pts, dtype=torch.float32)
        diffuse = torch.ones(1, pts.shape[1])
        radius = torch.linalg.norm(t_pts, dim=-1)
        concentrated = torch.exp(-8.0 * radius ** 2)
        h_diffuse = r.bandwidth_deg(diffuse, t_pts)
        h_concentrated = r.bandwidth_deg(concentrated, t_pts)
        assert float(h_concentrated) < float(h_diffuse)

    def test_bandwidth_respects_its_clamps(self, PARAMS):
        r, pts = self._make(PARAMS)
        t_pts = torch.as_tensor(pts, dtype=torch.float32)
        for w in (torch.ones(1, pts.shape[1]),
                  torch.zeros(1, pts.shape[1]).scatter(
                      1, torch.tensor([[0]]), 1.0)):
            h = r.bandwidth_deg(w, t_pts)
            assert float(h) >= float(r.bandwidth_floor) * 0.9
            assert float(h) <= float(r.bandwidth_ceiling) * 1.1

    def _single_point(self, PARAMS, point_xy, sigma_px, center_xy=(0.0, 0.0)):
        """Render one point at a wide bandwidth; the ceiling is set just
        above it, as the simulator's would be for a broad cloud."""
        gx, _ = visual_field_grid(PARAMS, **KW)
        sigma_deg = sigma_px * float(gx[0, 1] - gx[0, 0])
        r = ScatterRenderer(np.array([[point_xy]]), np.array([center_xy]),
                            PARAMS, np.array([2.0]), np.array([0.05]),
                            np.array([1.2 * sigma_deg]), KW)
        return r.render(torch.ones(1, 1), torch.tensor([sigma_deg]))[0], r

    def test_a_wide_splat_falls_off_smoothly_not_in_a_square(self, PARAMS):
        """One point blurred at a sigma of several pixels must look like
        a Gaussian all the way out. A kernel truncated near 2 sigma left
        a 13% step at its edge, so every point became a hard-edged
        square -- the artifact scatter phosphenes showed at 3-6 deg."""
        gx, gy = visual_field_grid(PARAMS, **KW)
        # A pixel center, so the bilinear splat does not smear it.
        x0, y0 = float(gx[48, 48]), float(gy[48, 48])
        sigma_px = 6.0
        img, r = self._single_point(PARAMS, (x0, y0), sigma_px)
        px = min(r.deg_per_px_x, r.deg_per_px_y)
        r2 = ((gx - x0) ** 2 + (gy - y0) ** 2) / px ** 2
        expected = torch.exp(-0.5 * r2 / sigma_px ** 2)
        assert float((img - expected).abs().max()) < 0.01
        # No step anywhere: neighboring pixels differ by at most the
        # steepest slope of the Gaussian itself (~0.1 per px here).
        steepest = math.exp(-0.5) / sigma_px
        assert float(img.diff(dim=0).abs().max()) < 1.1 * steepest
        assert float(img.diff(dim=1).abs().max()) < 1.1 * steepest

    def test_a_point_at_the_patch_rim_is_not_cropped(self, PARAMS):
        """The patch is padded by the widest kernel the ceiling allows,
        so a point at the edge of the kept region renders the same splat
        as one at the patch center."""
        gx, gy = visual_field_grid(PARAMS, **KW)
        x0, y0 = float(gx[48, 48]), float(gy[48, 48])
        rim_x = float(gx[48, 48 + 11])     # 11 px, inside the 2 deg core
        centered, _ = self._single_point(PARAMS, (rim_x, y0), 5.0,
                                        center_xy=(rim_x, y0))
        at_rim, r = self._single_point(PARAMS, (rim_x, y0), 5.0,
                                       center_xy=(x0, y0))
        assert torch.allclose(at_rim, centered, atol=1e-5)

    def test_gradient_reaches_the_bandwidth(self, PARAMS):
        """The taper that removes the kernel's edge step is a function of
        sigma too, so the blur stays differentiable in the bandwidth."""
        r, pts = self._make(PARAMS)
        w = torch.rand(1, pts.shape[1])
        h = torch.tensor([0.5], requires_grad=True)
        r.render(w, h).sum().backward()
        assert torch.isfinite(h.grad).all() and float(h.grad.abs()) > 0

    def test_points_outside_the_patch_are_dropped_not_wrapped(self, PARAMS):
        """A point beyond the patch must contribute nothing, rather than
        aliasing back in at the opposite edge."""
        pts = np.array([[[0.0, 0.0], [50.0, 50.0]]])
        r = ScatterRenderer(pts, np.zeros((1, 2)), PARAMS, np.array([2.0]),
                            np.array([0.1]), np.array([0.5]), KW)
        w = torch.tensor([[0.0, 1.0]])
        img = r.render(w, torch.tensor([0.2]))
        assert float(img.sum()) == pytest.approx(0.0, abs=1e-6)


def test_smooth_clamp_is_smooth_and_bounded(PARAMS):
    """A soft clamp trades exactness at the bound for a non-zero
    gradient outside it, which is the whole point -- a hard clamp would
    stop a saturated bandwidth from ever recovering. The overshoot is
    bounded by the softness fraction (5% of the span by default), so it
    is checked against that rather than against the bound itself."""
    low, high = torch.tensor(0.1), torch.tensor(1.0)
    span = float(high - low)
    x = torch.linspace(-2.0, 4.0, 200, requires_grad=True)
    y = _smooth_clamp(x, low, high)
    assert float(y.min().detach()) >= float(low) - 0.05 * span
    assert float(y.max().detach()) <= float(high) + 0.05 * span
    # Well inside the interval it is indistinguishable from the identity.
    mid = _smooth_clamp(torch.tensor(0.55), low, high)
    assert float(mid) == pytest.approx(0.55, abs=0.01)
    y.sum().backward()
    # A hard clamp would give exactly zero gradient outside the range.
    assert float(x.grad[0].abs()) > 0 and float(x.grad[-1].abs()) > 0
