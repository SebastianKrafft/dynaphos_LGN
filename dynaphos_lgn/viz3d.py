"""Three-dimensional visualisation of the LGN atlas, its maps and an
electrode array inside it.

Draws the nucleus as a voxel surface in Horsley-Clarke millimetres and
paints per-voxel atlas quantities onto it: eccentricity, inclination,
laminar identity, cell class, eye of origin, cell density, and the
quantities derived from the per-voxel Jacobian of the retinotopy --
the two principal magnifications, their ratio and orientation, and the
fit's own residual and confidence flags. Electrodes are drawn in the
same millimetres, optionally with the tissue each one recruits.

Two backends, same geometry and the same colours:

- **plotly** (default) writes a self-contained HTML file: rotate, zoom,
  switch overlay from a dropdown, and click the legend for a hoverable
  point cloud carrying every overlay's number at that voxel.
- **matplotlib** needs no extra dependency and writes a PNG, one panel
  per overlay.

Usage::

    python -m dynaphos_lgn.viz3d                       # eccentricity, plotly
    python -m dynaphos_lgn.viz3d --overlay all
    python -m dynaphos_lgn.viz3d --hemispheres left right --electrodes layout
    python -m dynaphos_lgn.viz3d --clip ap:6:9 --overlay layer
    python -m dynaphos_lgn.viz3d --backend matplotlib --overlay eccentricity layer
    python -m dynaphos_lgn.viz3d --list-overlays

What it draws is the atlas, not a model of it. Nothing here smooths,
interpolates or fills in: a rendered voxel takes its value from one
real atlas voxel, tissue with no value is grey rather than coloured at
one end of the scale, and the two derived maps the package itself
flags as shaky (`jacobian_flags`, `jacobian_residual`) are overlays you
can look at rather than something hidden behind the pretty ones.

Every default -- stride, colormaps, colour limits, marker sizes,
lighting, output paths -- lives in the `visualization` block of
`config/params_lgn.yaml`, and every one has a command-line override.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from dynaphos_lgn.params import require

# The topology of a cube: for each (axis, sign) face, the four corners
# of that face in units of one voxel, wound so the normal points
# outwards. Not in the config, unlike every number in this package,
# because it is not a number about the LGN -- a cube has six faces and
# four corners each whatever atlas it is describing.
_FACE_CORNERS: Dict[Tuple[int, int], np.ndarray] = {
    (0, +1): np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]]),
    (0, -1): np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]]),
    (1, +1): np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]),
    (1, -1): np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]]),
    (2, +1): np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]),
    (2, -1): np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]]),
}

_AXIS_NAMES = ('ml', 'dv', 'ap')


# ======================================================================
# Geometry
# ======================================================================
def axis_mm(atlas, axis: int) -> np.ndarray:
    """Horsley-Clarke millimetres along one atlas axis.

    Goes through the atlas's own `index_to_horsley_clarke`, so a
    `MirroredAtlas` -- whose ML axis runs the other way -- reports the
    millimetres it actually occupies rather than the left nucleus's.

    :param atlas: A built atlas.
    :param axis: 0 (ML), 1 (DV) or 2 (AP).
    :return: (n,) array of millimetres, one per index along that axis.
    """
    n = atlas.atlas_shape[axis]
    args = [np.zeros(n, dtype=int) for _ in range(3)]
    args[axis] = np.arange(n)
    return np.asarray(atlas.index_to_horsley_clarke(*args)[axis], dtype=float)


def blocks_of(mask: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce a voxel mask to a coarser grid of drawable blocks.

    A block is drawn when ANY atlas voxel inside it is tissue -- so a
    thin lamina survives a stride that would otherwise step over it --
    and is represented by the tissue voxel nearest its centre. That
    voxel's own values are what gets painted: no averaging, no
    interpolation, so every colour on the surface is a number the atlas
    actually holds.

    :param mask: (ML, DV, AP) boolean volume of voxels to draw.
    :param stride: Atlas voxels per block along each axis.
    :return: (block_mask, representative) -- the coarse boolean volume,
        and for every block the (ml, dv, ap) index of the atlas voxel
        standing for it. `representative` is meaningless where
        `block_mask` is False.
    """
    if stride < 1:
        raise ValueError(f"stride must be at least 1, got {stride}.")
    shape = np.asarray(mask.shape)
    n_blocks = -(-shape // stride)                  # ceiling division
    padded_shape = n_blocks * stride

    if np.any(padded_shape != shape):
        padded = np.zeros(tuple(padded_shape), dtype=bool)
        padded[:shape[0], :shape[1], :shape[2]] = mask
    else:
        padded = mask

    a, b, c = (int(v) for v in n_blocks)
    cube = padded.reshape(a, stride, b, stride, c, stride)
    cube = cube.transpose(0, 2, 4, 1, 3, 5).reshape(a, b, c, stride ** 3)
    block_mask = cube.any(axis=-1)

    # Squared distance from the block centre, in the block's own
    # flattened order, so argmin picks the most central tissue voxel.
    offsets = np.arange(stride) - (stride - 1) / 2.0
    distance = (offsets[:, None, None] ** 2 + offsets[None, :, None] ** 2
                + offsets[None, None, :] ** 2).ravel().astype(np.float32)
    choice = np.argmin(np.where(cube, distance, np.float32(np.inf)), axis=-1)
    within = np.stack(np.unravel_index(choice, (stride,) * 3), axis=-1)

    grid = np.meshgrid(np.arange(a), np.arange(b), np.arange(c),
                       indexing='ij')
    origin = np.stack(grid, axis=-1) * stride
    representative = np.minimum(origin + within, shape - 1)
    return block_mask, representative.astype(np.int32)


def exposed_faces(block_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Every face of a voxel volume with empty space on the far side.

    That includes the flat faces a clip plane creates, which is what
    makes `--clip` a cross-section rather than a way of hiding tissue.

    :param block_mask: Coarse boolean volume from `blocks_of`.
    :return: (corners, owners) -- (n_faces, 4, 3) integer lattice
        corners in block units, and the (n_faces, 3) block index each
        face belongs to.
    """
    corners: List[np.ndarray] = []
    owners: List[np.ndarray] = []
    for (axis, sign), face in _FACE_CORNERS.items():
        neighbour = np.zeros_like(block_mask)
        src: List[slice] = [slice(None)] * 3
        dst: List[slice] = [slice(None)] * 3
        size = block_mask.shape[axis]
        if sign > 0:
            dst[axis], src[axis] = slice(0, size - 1), slice(1, size)
        else:
            dst[axis], src[axis] = slice(1, size), slice(0, size - 1)
        neighbour[tuple(dst)] = block_mask[tuple(src)]

        index = np.argwhere(block_mask & ~neighbour)
        if not len(index):
            continue
        corners.append(index[:, None, :] + face[None, :, :])
        owners.append(index)

    if not corners:
        return (np.zeros((0, 4, 3), dtype=np.int32),
                np.zeros((0, 3), dtype=np.int32))
    return (np.concatenate(corners).astype(np.int32),
            np.concatenate(owners).astype(np.int32))


@dataclass
class VoxelMesh:
    """The drawable surface of one nucleus.

    :ivar vertices_mm: (n_vertices, 3) scene coordinates, millimetres.
        x is signed ML (see `visualization.scene.left_nucleus_x_sign`);
        y and z are Horsley-Clarke DV and AP.
    :ivar quads: (n_faces, 4) indices into `vertices_mm`.
    :ivar block_of_quad: (n_faces,) row of `voxel_indices` each face
        belongs to, i.e. which voxel's value colours it.
    :ivar voxel_indices: (n_blocks, 3) the atlas voxel representing
        each drawn block. Overlays are evaluated here, once.
    :ivar centres_mm: (n_blocks, 3) scene coordinates of those voxels.
    :ivar hemisphere: 'left' or 'right'.
    :ivar stride: Atlas voxels per block along each axis.
    :ivar n_atlas_voxels: How many atlas voxels the selected volume
        holds, so the surface can say what it is the surface of.
    :ivar n_blocks_total: How many blocks that volume fills, of which
        only the ones with an exposed face are drawn.
    """

    vertices_mm: np.ndarray
    quads: np.ndarray
    block_of_quad: np.ndarray
    voxel_indices: np.ndarray
    centres_mm: np.ndarray
    hemisphere: str
    stride: int
    n_atlas_voxels: int
    n_blocks_total: int

    @property
    def n_faces(self) -> int:
        return len(self.quads)

    @property
    def n_blocks(self) -> int:
        return len(self.voxel_indices)

    @property
    def quad_vertices_mm(self) -> np.ndarray:
        """(n_faces, 4, 3) corner positions, for matplotlib."""
        return self.vertices_mm[self.quads]

    @property
    def triangles(self) -> np.ndarray:
        """(2 * n_faces, 3) triangulation of the quads, for plotly."""
        a, b, c, d = (self.quads[:, i] for i in range(4))
        return np.concatenate([np.stack([a, b, c], axis=1),
                               np.stack([a, c, d], axis=1)])

    @property
    def block_of_triangle(self) -> np.ndarray:
        """(2 * n_faces,) block index per triangle, for cell colouring."""
        return np.concatenate([self.block_of_quad, self.block_of_quad])


def build_mesh(atlas, mask: np.ndarray, stride: int, x_sign: float
               ) -> VoxelMesh:
    """Turn a voxel mask into the surface of the tissue it selects.

    :param atlas: The built atlas the mask indexes.
    :param mask: (ML, DV, AP) boolean volume of voxels to draw.
    :param stride: Atlas voxels per drawn block along each axis.
    :param x_sign: +1 or -1, which side of the midline to draw on.
    :return: A `VoxelMesh`, with zero faces if the mask selects
        nothing -- which the caller reports rather than crashes on.
    """
    block_mask, representative = blocks_of(mask, stride)
    corners, owners = exposed_faces(block_mask)
    n_atlas_voxels = int(mask.sum())
    n_blocks_total = int(block_mask.sum())

    if not len(corners):
        empty = np.zeros((0, 3), dtype=np.float32)
        return VoxelMesh(empty, np.zeros((0, 4), dtype=np.int32),
                         np.zeros(0, dtype=np.int32),
                         np.zeros((0, 3), dtype=np.int32), empty,
                         getattr(atlas, 'hemisphere', 'left'), stride,
                         n_atlas_voxels, n_blocks_total)

    # --- unique blocks, so an overlay is evaluated once per voxel -----
    shape_blocks = np.asarray(block_mask.shape)
    flat_owner = np.ravel_multi_index(tuple(owners.T), tuple(shape_blocks))
    unique_blocks, block_of_quad = np.unique(flat_owner, return_inverse=True)
    voxel_indices = representative.reshape(-1, 3)[unique_blocks]

    # --- unique vertices, so a corner shared by four faces is stored
    #     once. The lattice is integral, so the packing is exact.
    lattice = corners.reshape(-1, 3)
    span = tuple(shape_blocks + 1)
    flat_vertex = np.ravel_multi_index(tuple(lattice.T), span)
    unique_vertices, vertex_of_corner = np.unique(flat_vertex,
                                                  return_inverse=True)
    lattice_unique = np.stack(np.unravel_index(unique_vertices, span),
                              axis=-1).astype(np.float64)

    # A block spans atlas indices [i * stride, i * stride + stride - 1],
    # whose voxel cubes together span [i * stride - 0.5, (i + 1) * stride
    # - 0.5]. Lattice corner L therefore sits at atlas index
    # L * stride - 0.5 -- fractional, and converted through the atlas's
    # own inverse so that a mirrored ML axis comes out on the right side.
    fractional = lattice_unique * stride - 0.5
    vertices_mm = np.stack(
        atlas.index_to_horsley_clarke(fractional[:, 0], fractional[:, 1],
                                      fractional[:, 2]), axis=-1)
    vertices_mm[:, 0] *= x_sign

    centres_mm = np.stack(
        atlas.index_to_horsley_clarke(voxel_indices[:, 0],
                                      voxel_indices[:, 1],
                                      voxel_indices[:, 2]), axis=-1)
    centres_mm[:, 0] *= x_sign

    return VoxelMesh(
        vertices_mm=vertices_mm.astype(np.float32),
        quads=vertex_of_corner.reshape(-1, 4).astype(np.int32),
        block_of_quad=block_of_quad.astype(np.int32),
        voxel_indices=voxel_indices.astype(np.int32),
        centres_mm=centres_mm.astype(np.float32),
        hemisphere=getattr(atlas, 'hemisphere', 'left'),
        stride=stride,
        n_atlas_voxels=n_atlas_voxels,
        n_blocks_total=n_blocks_total)


# ======================================================================
# Which tissue to draw
# ======================================================================
def tissue_mask(atlas, params: Mapping,
                cell_class: Optional[str] = None,
                layers: Optional[Sequence[int]] = None,
                min_eccentricity_deg: Optional[float] = None,
                max_eccentricity_deg: Optional[float] = None,
                clips: Optional[Sequence[Tuple[str, float, float]]] = None,
                include_ipsi_placeholder: Optional[bool] = None
                ) -> np.ndarray:
    """The boolean volume a figure should draw.

    Starts from the atlas's own `valid` -- reconstructed tissue with an
    eccentricity, an inclination and a lamina -- and narrows it.

    :param cell_class: 'magno' or 'parvo' to keep only that class.
    :param layers: Explicit LAYERS.DAT codes to keep.
    :param min_eccentricity_deg: Drop tissue below this eccentricity.
    :param max_eccentricity_deg: Drop tissue above it.
    :param clips: ``(axis, low_mm, high_mm)`` half-space cuts, where
        axis is 'ml', 'dv' or 'ap'. Cuts the volume open: the exposed
        face is drawn and painted like any other surface, which is how
        interior structure such as the laminae becomes visible.
    :param include_ipsi_placeholder: Keep the coarse +-135 deg
        ipsilateral-hemifield placeholder voxels. Defaults to the
        opposite of `atlas.exclude_ipsi_flat_inclination`, so a figure
        shows the same tissue the simulator uses.
    :return: (ML, DV, AP) boolean volume.
    """
    mask = atlas.valid.copy()

    if include_ipsi_placeholder is None:
        include_ipsi_placeholder = not require(
            params, 'atlas.exclude_ipsi_flat_inclination')
    if not include_ipsi_placeholder:
        mask &= ~atlas.is_ipsi_sentinel()

    if cell_class is not None:
        codes = require(params, 'atlas.layers.classes')
        if cell_class not in codes:
            raise ValueError(f"Unknown cell class {cell_class!r}; "
                             f"atlas.layers.classes defines "
                             f"{', '.join(sorted(codes))}.")
        mask &= np.isin(atlas.layer, list(codes[cell_class]))

    if layers:
        mask &= np.isin(atlas.layer, [int(code) for code in layers])

    if min_eccentricity_deg is not None:
        mask &= atlas.eccentricity_deg >= min_eccentricity_deg
    if max_eccentricity_deg is not None:
        mask &= atlas.eccentricity_deg <= max_eccentricity_deg

    for axis_name, low, high in (clips or ()):
        axis = _AXIS_NAMES.index(axis_name)
        coordinate = axis_mm(atlas, axis)
        keep = (coordinate >= low) & (coordinate <= high)
        shape = [1, 1, 1]
        shape[axis] = -1
        mask &= keep.reshape(shape)

    return mask


# ======================================================================
# Overlays
# ======================================================================
@dataclass
class Overlay:
    """One quantity painted onto the surface, already evaluated.

    :ivar name: Its key, as the CLI and the config spell it.
    :ivar label: Human-readable name, with units.
    :ivar values: (n_blocks,) float values, NaN where the atlas has
        none. Categorical overlays hold integer codes as floats.
    :ivar kind: 'continuous' or 'categorical'.
    :ivar categories: Category names, in code order, for 'categorical'.
    :ivar note: What the reader should know before believing the
        picture. Shown under the title.
    :ivar unit: Physical unit, or '' for none.
    """

    name: str
    label: str
    values: np.ndarray
    kind: str
    categories: Optional[List[str]] = None
    note: str = ''
    unit: str = ''

    @property
    def finite(self) -> np.ndarray:
        return self.values[np.isfinite(self.values)]


@dataclass
class AtlasContext:
    """What overlays are computed from, with the Jacobian loaded lazily.

    The cached Jacobian is a ~110 MB file that unpacks to half a
    gigabyte, so it is loaded only if an overlay derived from it is
    actually asked for.

    :ivar atlas: The built (possibly mirrored) atlas.
    :ivar params: The parameter dictionary.
    :ivar jacobian_loader: Callable returning the matching
        `JacobianAtlas`, or None when none is available.
    """

    atlas: object
    params: Mapping
    jacobian_loader: Optional[Callable[[], object]] = None
    _jacobian: object = None
    _jacobian_tried: bool = False

    @property
    def jacobian(self):
        """The Jacobian atlas, loaded on first use. None if absent."""
        if not self._jacobian_tried:
            self._jacobian_tried = True
            if self.jacobian_loader is not None:
                self._jacobian = self.jacobian_loader()
        return self._jacobian


def _categorical(values: np.ndarray, code_to_name: Mapping) -> Tuple[
        np.ndarray, List[str]]:
    """Map raw codes to consecutive category indices.

    :param values: Raw per-block codes.
    :param code_to_name: {code: display name}, iterated in sorted code
        order so the legend is stable between figures.
    :return: (codes as float with NaN for anything unmapped, names).
    """
    order = sorted(code_to_name)
    lookup = {code: i for i, code in enumerate(order)}
    out = np.full(len(values), np.nan)
    for code, index in lookup.items():
        out[values == code] = index
    return out, [str(code_to_name[code]) for code in order]


def _principal_magnification(context: AtlasContext, voxels: np.ndarray):
    """(major, minor, orientation_rad, valid) at the given voxels.

    Evaluated only at the voxels actually drawn, so the SVD costs tens
    of thousands of 2x3 decompositions rather than the atlas's 3.6
    million.
    """
    from dynaphos_lgn.magnification import JacobianMagnification

    jacobian = context.jacobian
    if jacobian is None:
        nan = np.full(len(voxels), np.nan)
        return nan, nan.copy(), nan.copy(), np.zeros(len(voxels), dtype=bool)

    index = tuple(voxels.T)
    major, minor, orientation = JacobianMagnification.decompose(
        jacobian.jacobian_at(voxels))
    valid = np.asarray(jacobian.jacobian_valid[index], dtype=bool)
    major = np.where(valid, major, np.nan)
    minor = np.where(valid, minor, np.nan)
    orientation = np.where(valid, orientation, np.nan)
    return major, minor, orientation, valid


_JACOBIAN_NOTE = ('Derived from a per-voxel local linear fit of the '
                  'atlas retinotopy, not published data.')


def overlay_eccentricity(context: AtlasContext, voxels: np.ndarray
                         ) -> Overlay:
    values = np.asarray(context.atlas.eccentricity_deg[tuple(voxels.T)],
                        dtype=float)
    return Overlay('eccentricity', 'Eccentricity (deg)', values,
                   'continuous', unit='deg',
                   note='Erwin et al. (1999), as published. Inside '
                        f'{require(context.params, "atlas.isotropic_construction_radius_deg")} deg '
                        'it is Malpeli et al.\'s formula written in, not '
                        'interpolated data.')


def overlay_inclination(context: AtlasContext, voxels: np.ndarray) -> Overlay:
    atlas = context.atlas
    values = np.asarray(atlas.inclination_deg[tuple(voxels.T)], dtype=float)
    placeholder = atlas.is_ipsi_sentinel(
        np.asarray(atlas.inclination[tuple(voxels.T)]))
    values = np.where(placeholder, np.nan, values)
    return Overlay('inclination', 'Inclination (deg)', values, 'continuous',
                   unit='deg',
                   note='0 deg is the horizontal meridian, +-90 deg the '
                        'vertical one. The coarse ipsilateral placeholder '
                        'is drawn as missing, not as an angle.')


def overlay_cell_class(context: AtlasContext, voxels: np.ndarray) -> Overlay:
    classes = require(context.params, 'atlas.layers.classes')
    code_to_name = {int(code): name
                    for name, codes in classes.items() for code in codes}
    raw = np.asarray(context.atlas.layer[tuple(voxels.T)], dtype=int)
    values, names = _categorical(raw, code_to_name)
    return Overlay('cell_class', 'Cell class', values, 'categorical',
                   categories=names,
                   note='Magnocellular vs parvocellular, from the '
                        'LAYERS.DAT codes. Koniocellular tissue is not '
                        'in the atlas at all.')


def overlay_layer(context: AtlasContext, voxels: np.ndarray) -> Overlay:
    names = require(context.params, 'atlas.layers.names')
    code_to_name = {int(code): name for code, name in names.items()}
    raw = np.asarray(context.atlas.layer[tuple(voxels.T)], dtype=int)
    values, labels = _categorical(raw, code_to_name)
    return Overlay('layer', 'Lamina', values, 'categorical',
                   categories=labels,
                   note='"contra"/"ipsi" here is EYE OF ORIGIN, a '
                        'different axis from the hemifield an inclination '
                        'describes.')


def overlay_eye(context: AtlasContext, voxels: np.ndarray) -> Overlay:
    eyes = require(context.params, 'atlas.layers.eyes')
    code_to_name = {int(code): f'{eye}lateral eye'
                    for eye, codes in eyes.items() for code in codes}
    raw = np.asarray(context.atlas.layer[tuple(voxels.T)], dtype=int)
    values, labels = _categorical(raw, code_to_name)
    return Overlay('eye', 'Eye of origin', values, 'categorical',
                   categories=labels,
                   note='Which eye drives each lamina. Contralateral '
                        'here means the eye, not the hemifield.')


def overlay_cells(context: AtlasContext, voxels: np.ndarray) -> Overlay:
    values = np.asarray(context.atlas.cells_per_voxel[tuple(voxels.T)],
                        dtype=float)
    voxel_mm3 = require(context.params, 'atlas.voxel_size_mm') ** 3
    return Overlay('cells_per_voxel', 'Cells per voxel', values,
                   'continuous', unit='cells',
                   note=f'One voxel is {voxel_mm3 * 1e3:.4f} mm^3 x 1e-3. '
                        'Erwin et al. reconstructed counts, not a density '
                        'model.')


def overlay_magnification_major(context: AtlasContext, voxels: np.ndarray
                                ) -> Overlay:
    major, _, _, _ = _principal_magnification(context, voxels)
    return Overlay('magnification_major',
                   'Magnification, major axis (deg/mm)', major, 'continuous',
                   unit='deg/mm',
                   note='Larger principal singular value of the local '
                        f'Jacobian. {_JACOBIAN_NOTE}')


def overlay_magnification_minor(context: AtlasContext, voxels: np.ndarray
                                ) -> Overlay:
    _, minor, _, _ = _principal_magnification(context, voxels)
    return Overlay('magnification_minor',
                   'Magnification, minor axis (deg/mm)', minor, 'continuous',
                   unit='deg/mm',
                   note='Smaller principal singular value of the local '
                        f'Jacobian. {_JACOBIAN_NOTE}')


def overlay_anisotropy(context: AtlasContext, voxels: np.ndarray) -> Overlay:
    major, minor, _, _ = _principal_magnification(context, voxels)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(minor > 0, major / minor, np.nan)
    return Overlay('anisotropy', 'Anisotropy (major / minor)', ratio,
                   'continuous',
                   note='1 is a circular phosphene. A validation target '
                        'only -- never a simulator input -- and inside the '
                        'central isotropic-by-construction patch it is an '
                        'artefact of how the atlas was built.')


def overlay_magnification_areal(context: AtlasContext, voxels: np.ndarray
                                ) -> Overlay:
    major, minor, _, _ = _principal_magnification(context, voxels)
    return Overlay('magnification_areal',
                   'Areal magnification (deg^2/mm^2)', major * minor,
                   'continuous', unit='deg^2/mm^2',
                   note=f'Product of the two principal magnifications. '
                        f'{_JACOBIAN_NOTE}')


def overlay_magnification_orientation(context: AtlasContext,
                                      voxels: np.ndarray) -> Overlay:
    _, _, orientation, _ = _principal_magnification(context, voxels)
    return Overlay('magnification_orientation',
                   'Major-axis orientation (deg)', np.rad2deg(orientation),
                   'continuous', unit='deg',
                   note='Direction of the major axis in VISUAL-FIELD '
                        'coordinates, not in the tissue. Cyclic, so the '
                        'colour scale wraps.')


def overlay_jacobian_residual(context: AtlasContext, voxels: np.ndarray
                              ) -> Overlay:
    jacobian = context.jacobian
    if jacobian is None:
        values = np.full(len(voxels), np.nan)
    else:
        values = np.asarray(jacobian.residual_rms[tuple(voxels.T)],
                            dtype=float)
    return Overlay('jacobian_residual', 'Jacobian fit residual (deg)',
                   values, 'continuous', unit='deg',
                   note='In-neighbourhood RMS residual of the local linear '
                        'fit. High values mark tissue where a single local '
                        'magnification does not describe the retinotopy.')


def overlay_jacobian_flags(context: AtlasContext, voxels: np.ndarray
                           ) -> Overlay:
    jacobian = context.jacobian
    n = len(voxels)
    if jacobian is None:
        return Overlay('jacobian_flags', 'Jacobian confidence',
                       np.full(n, np.nan), 'categorical',
                       categories=['no fit available'],
                       note='No Jacobian atlas was loaded.')

    index = tuple(voxels.T)
    fitted = np.asarray(jacobian.jacobian_valid[index], dtype=bool)
    isotropic = np.asarray(jacobian.isotropic_flag[index], dtype=bool)
    unreliable = np.asarray(
        jacobian.unreliable_beyond_neighborhood_flag[index], dtype=bool)

    # Later conditions win: an unreliable fit is the thing worth seeing
    # even where the voxel is also inside the central patch.
    values = np.full(n, np.nan)
    values[fitted] = 0.0
    values[fitted & isotropic] = 1.0
    values[fitted & unreliable] = 2.0
    values[~fitted] = 3.0
    return Overlay(
        'jacobian_flags', 'Jacobian confidence', values, 'categorical',
        categories=['fit accepted',
                    'central patch: isotropic by construction',
                    'do not extrapolate beyond neighbourhood',
                    'no fit'],
        note='The package\'s own confidence flags, drawn rather than '
             'buried. Electrodes in "do not extrapolate" tissue are '
             'routed to per-voxel scatter rendering.')


#: Every overlay this module knows how to draw, by the name the CLI and
#: the config use. `visualization.overlays` selects from these.
OVERLAY_BUILDERS: Dict[str, Callable[[AtlasContext, np.ndarray], Overlay]] = {
    'eccentricity': overlay_eccentricity,
    'inclination': overlay_inclination,
    'cell_class': overlay_cell_class,
    'layer': overlay_layer,
    'eye': overlay_eye,
    'cells_per_voxel': overlay_cells,
    'magnification_major': overlay_magnification_major,
    'magnification_minor': overlay_magnification_minor,
    'anisotropy': overlay_anisotropy,
    'magnification_areal': overlay_magnification_areal,
    'magnification_orientation': overlay_magnification_orientation,
    'jacobian_residual': overlay_jacobian_residual,
    'jacobian_flags': overlay_jacobian_flags,
}

#: Which overlays need the cached Jacobian, so the CLI can say what a
#: missing cache costs before spending a minute finding out.
JACOBIAN_OVERLAYS = frozenset({
    'magnification_major', 'magnification_minor', 'anisotropy',
    'magnification_areal', 'magnification_orientation',
    'jacobian_residual', 'jacobian_flags'})


def compute_overlay(name: str, context: AtlasContext, voxels: np.ndarray
                    ) -> Overlay:
    """Evaluate one overlay at the drawn voxels.

    :param name: A key of `OVERLAY_BUILDERS`.
    :param context: Atlas, params and the lazy Jacobian.
    :param voxels: (n_blocks, 3) atlas indices.
    :raises KeyError: naming the overlay and listing the valid ones.
    """
    if name not in OVERLAY_BUILDERS:
        raise KeyError(f"Unknown overlay {name!r}. Available: "
                       f"{', '.join(sorted(OVERLAY_BUILDERS))}.")
    return OVERLAY_BUILDERS[name](context, voxels)


# ======================================================================
# Colour
# ======================================================================
@dataclass
class ColourSpec:
    """How one overlay's numbers become colours, in both backends.

    Missing values get their own reserved band at the bottom of the
    scale rather than the scale's first colour, so "the atlas has no
    value here" never reads as "the value here is small".

    :ivar values: (n,) values with missing entries moved into the
        reserved band, safe to hand to plotly as `intensity`.
    :ivar cmin: Low end of the colour axis, inside the reserved band.
    :ivar cmax: High end.
    :ivar colorscale: Plotly colorscale, sampled from the matplotlib
        colormap so both backends agree.
    :ivar tickvals: Colourbar tick positions, or None for automatic.
    :ivar ticktext: Colourbar tick labels, or None.
    :ivar rgba: (n, 4) colours for matplotlib, missing entries already
        in the missing colour.
    """

    values: np.ndarray
    cmin: float
    cmax: float
    colorscale: List[List]
    tickvals: Optional[List[float]]
    ticktext: Optional[List[str]]
    rgba: np.ndarray


def colormap(name: str):
    from matplotlib import colormaps
    return colormaps[name]


def category_colour(cmap, index: int, n_categories: int):
    """One distinct colour per category.

    A qualitative colormap (Set1, Set2, ...) holds a short list of
    colours whose last entry is a neutral grey, so sampling it by
    fraction hands the last category the same grey this module uses for
    "no value". Taking its entries by position instead keeps the two
    apart; a continuous colormap has no such list and is still sampled
    by fraction.
    """
    from matplotlib.colors import ListedColormap
    if isinstance(cmap, ListedColormap) and cmap.N <= 20:
        return np.array(cmap(index % cmap.N))
    return np.array(cmap(index / max(n_categories - 1, 1)))


def _to_rgb_string(rgba) -> str:
    r, g, b = (int(round(255 * float(c))) for c in rgba[:3])
    return f'rgb({r},{g},{b})'


def continuous_limits(overlay: Overlay, params: Mapping) -> Tuple[float,
                                                                  float]:
    """Low and high ends of a continuous overlay's colour axis.

    Percentile-clipped for the overlays `visualization.colour_limits.
    clipped_overlays` names -- the magnification maps have a tail long
    enough that one voxel otherwise flattens everything else -- and the
    full range for the rest.
    """
    finite = overlay.finite
    if not len(finite):
        return 0.0, 1.0
    clipped = require(params, 'visualization.colour_limits.clipped_overlays')
    if overlay.name in clipped:
        low, high = require(params, 'visualization.colour_limits.percentile')
        lo, hi = np.percentile(finite, [low, high])
    else:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def colour_spec(overlay: Overlay, params: Mapping,
                limits: Optional[Tuple[float, float]] = None) -> ColourSpec:
    """Turn one overlay's values into everything both backends need.

    :param limits: Colour-axis ends to use instead of ones derived from
        these values. The slice viewer passes the whole volume's limits
        here, so that one colour means one number on every slice
        rather than being renormalised plane by plane.
    """
    import matplotlib.colors as mcolors

    cmap = colormap(require(params,
                             f'visualization.colormaps.{overlay.name}'))
    missing = require(params, 'visualization.missing_colour')
    missing_rgba = np.array(mcolors.to_rgba(missing))
    band = float(require(params, 'visualization.missing_band_fraction'))
    finite = np.isfinite(overlay.values)

    if overlay.kind == 'categorical':
        n = max(len(overlay.categories or []), 1)
        # One slot per category plus one for "missing", drawn as a step
        # function so a category boundary is a hard edge.
        cmin, cmax = -1.5, n - 0.5
        span = cmax - cmin
        colours = [missing_rgba] + [category_colour(cmap, i, n)
                                    for i in range(n)]
        scale: List[List] = []
        for i, rgba in enumerate(colours):
            start = ((cmin + i) - cmin) / span
            stop = ((cmin + i + 1) - cmin) / span
            scale.append([start, _to_rgb_string(rgba)])
            scale.append([min(stop, 1.0), _to_rgb_string(rgba)])
        values = np.where(finite, overlay.values, -1.0)
        tickvals = [-1.0] + [float(i) for i in range(n)]
        ticktext = ['no value'] + list(overlay.categories or [])
        rgba = np.repeat(missing_rgba[None, :], len(overlay.values), axis=0)
        for i in range(n):
            rgba[overlay.values == i] = colours[i + 1]
        return ColourSpec(values, cmin, cmax, scale, tickvals, ticktext, rgba)

    lo, hi = (continuous_limits(overlay, params) if limits is None
              else (float(limits[0]), float(limits[1])))
    reserved = (hi - lo) * band
    cmin, cmax = lo - reserved, hi
    span = cmax - cmin
    edge = reserved / span
    samples = np.linspace(0.0, 1.0, 64)
    scale = [[0.0, _to_rgb_string(missing_rgba)],
             [max(edge - 1e-6, 0.0), _to_rgb_string(missing_rgba)]]
    scale += [[edge + (1.0 - edge) * t, _to_rgb_string(cmap(t))]
              for t in samples]
    scale[-1][0] = 1.0

    values = np.where(finite, np.clip(overlay.values, lo, hi),
                      lo - reserved / 2.0)
    norm = mcolors.Normalize(vmin=lo, vmax=hi)
    rgba = np.asarray(cmap(norm(np.where(finite, overlay.values, lo))))
    rgba[~finite] = missing_rgba
    return ColourSpec(values, cmin, cmax, scale, None, None, rgba)


# ======================================================================
# Electrodes
# ======================================================================
@dataclass
class ElectrodeSet:
    """Contacts to draw, in scene millimetres.

    :ivar positions_mm: (n, 3) scene coordinates.
    :ivar labels: Short per-contact labels.
    :ivar hover: Per-contact multi-line description.
    :ivar group: (n,) integer group index, for colouring.
    :ivar group_names: Names of those groups.
    :ivar radius_mm: Recruitment radius per contact, or None.
    :ivar voxel_indices: (n, 3) atlas indices, or None where a contact
        falls outside the atlas volume entirely.
    :ivar source: How this set was built, for the figure subtitle.
    :ivar outside: (n,) True where the contact is off the atlas grid.
    """

    positions_mm: np.ndarray
    labels: List[str]
    hover: List[str]
    group: np.ndarray
    group_names: List[str]
    radius_mm: Optional[np.ndarray]
    voxel_indices: Optional[np.ndarray]
    source: str
    outside: np.ndarray

    def __len__(self) -> int:
        return len(self.positions_mm)


def hemisphere_sign(params: Mapping, hemisphere: str) -> float:
    """Which side of the scene's midline a nucleus is drawn on."""
    sign = float(require(params, 'visualization.scene.left_nucleus_x_sign'))
    return sign if hemisphere == 'left' else -sign


def _voxels_for_positions(atlas, ml_mm, dv_mm, ap_mm):
    """Atlas indices for stereotaxic positions, without raising.

    :return: ((n, 3) clipped indices, (n,) True where the position was
        off the grid and the index is therefore only the nearest one).
    """
    index = atlas.horsley_clarke_to_index(np.asarray(ml_mm, dtype=float),
                                          np.asarray(dv_mm, dtype=float),
                                          np.asarray(ap_mm, dtype=float),
                                          validate=False)
    stacked = np.stack(index, axis=-1)
    shape = np.asarray(atlas.atlas_shape)
    outside = np.any((stacked < 0) | (stacked >= shape), axis=-1)
    return np.clip(stacked, 0, shape - 1), outside


def _contact_hover(atlas, params: Mapping, voxels: np.ndarray,
                   outside: np.ndarray, labels: Sequence[str],
                   hemisphere: Sequence[str]) -> List[str]:
    """What the atlas says about the tissue each contact sits in."""
    names = {int(code): name for code, name
             in require(params, 'atlas.layers.names').items()}
    index = tuple(voxels.T)
    ecc = np.asarray(atlas.eccentricity_deg[index], dtype=float)
    incl = np.asarray(atlas.inclination_deg[index], dtype=float)
    layer = np.asarray(atlas.layer[index], dtype=int)
    cells = np.asarray(atlas.cells_per_voxel[index], dtype=float)
    on_tissue = np.asarray(atlas.valid[index], dtype=bool)

    lines = []
    for i in range(len(voxels)):
        parts = [f'<b>{labels[i]}</b>',
                 f'nucleus: {hemisphere[i]} LGN']
        if outside[i]:
            parts.append('OUTSIDE the atlas volume')
        elif not on_tissue[i]:
            parts.append('on a placeholder voxel (no reconstructed tissue)')
        else:
            parts.append(f'eccentricity: {ecc[i]:.2f} deg')
            parts.append(f'inclination: {incl[i]:.1f} deg')
            parts.append(f'lamina: {names.get(int(layer[i]), layer[i])}')
            parts.append(f'cells/voxel: {cells[i]:.3f}')
        parts.append(f'voxel (ml,dv,ap): {tuple(int(v) for v in voxels[i])}')
        lines.append('<br>'.join(parts))
    return lines


def _group_by(params: Mapping, atlas, voxels: np.ndarray,
              hemisphere: Sequence[str], array=None
              ) -> Tuple[np.ndarray, List[str]]:
    """Per-contact colour groups, per `visualization.electrodes.colour_by`."""
    mode = require(params, 'visualization.electrodes.colour_by')
    n = len(voxels)
    if mode == 'none':
        return np.zeros(n, dtype=int), ['electrode']

    if mode == 'hemisphere':
        names = sorted(set(hemisphere))
        lookup = {name: i for i, name in enumerate(names)}
        return (np.array([lookup[h] for h in hemisphere]),
                [f'{name} LGN' for name in names])

    if mode in ('layer', 'cell_class'):
        if mode == 'layer':
            table = {int(code): str(name) for code, name
                     in require(params, 'atlas.layers.names').items()}
        else:
            table = {int(code): name for name, codes
                     in require(params, 'atlas.layers.classes').items()
                     for code in codes}
        raw = np.asarray(atlas.layer[tuple(voxels.T)], dtype=int)
        order = sorted(table)
        lookup = {code: i for i, code in enumerate(order)}
        group = np.array([lookup.get(int(code), len(order)) for code in raw])
        return group, [table[code] for code in order] + ['not in a lamina']

    if array is None:
        raise ValueError(
            f"visualization.electrodes.colour_by = {mode!r} needs a built "
            f"electrode array (--electrodes array), which carries the "
            f"flags and renderer choice it colours by.")
    if mode == 'render_mode':
        modes = np.asarray(array.render_mode)
        names = ['ellipse', 'scatter']
        return (np.array([names.index(m) for m in modes]), names)
    if mode == 'flagged':
        flagged = array.flags.any_flag()
        return flagged.astype(int), ['no flag', 'at least one flag']

    raise ValueError(
        f"Unknown visualization.electrodes.colour_by {mode!r}; expected "
        f"none, layer, cell_class, hemisphere, flagged or render_mode.")


def electrodes_from_layout(atlas, params: Mapping, n_electrodes: int,
                           rng: np.random.Generator) -> ElectrodeSet:
    """Place an array the way the simulator would, and draw only that.

    Uses the same `get_electrode_layout` the simulator's own
    `spread_in_visual_field` constructor uses, so the picture is of the
    array a run with this config would build -- but skips building the
    array itself, and so needs no Jacobian.
    """
    from dynaphos_lgn.lgn_utils import get_electrode_layout

    _, voxels = get_electrode_layout(
        atlas, n_electrodes,
        cell_class=require(params, 'electrodes.cell_class'),
        min_eccentricity=require(params, 'electrodes.min_eccentricity_deg'),
        max_eccentricity=require(params, 'electrodes.max_eccentricity_deg'),
        bin_size_deg=require(params, 'electrodes.bin_size_deg'),
        exclude_ipsi_flat_inclination=require(
            params, 'atlas.exclude_ipsi_flat_inclination'),
        depth_selection=require(params, 'electrodes.depth_selection'),
        rng=rng)

    hemisphere = getattr(atlas, 'hemisphere', 'left')
    sign = hemisphere_sign(params, hemisphere)
    positions = np.stack(atlas.index_to_horsley_clarke(
        voxels[:, 0], voxels[:, 1], voxels[:, 2]), axis=-1)
    positions[:, 0] *= sign

    labels = [f'{hemisphere[0].upper()}{i}' for i in range(len(voxels))]
    hemispheres = [hemisphere] * len(voxels)
    outside = np.zeros(len(voxels), dtype=bool)
    group, group_names = _group_by(params, atlas, voxels, hemispheres)
    return ElectrodeSet(
        positions_mm=positions,
        labels=labels,
        hover=_contact_hover(atlas, params, voxels, outside, labels,
                             hemispheres),
        group=group, group_names=group_names,
        radius_mm=None, voxel_indices=voxels,
        source=f'{len(voxels)} electrodes, farthest-point sampled in the '
               f'visual field ({hemisphere} LGN)',
        outside=outside)


def electrodes_from_array(array, params: Mapping) -> ElectrodeSet:
    """Draw a built `LGNElectrodeArray`, flags and all.

    Costs a Jacobian load, and in exchange the hover carries what the
    simulator itself concluded about each electrode: its realised
    retinotopic position, the two principal magnifications, which
    renderer it was routed to, and every confidence flag it raised.
    """
    atlas = array.atlas
    hemisphere = array.hemisphere
    sign = hemisphere_sign(params, hemisphere)
    voxels = array.voxel_indices
    positions = np.stack(atlas.index_to_horsley_clarke(
        voxels[:, 0], voxels[:, 1], voxels[:, 2]), axis=-1)
    positions[:, 0] *= sign

    labels = [f'{hemisphere[0].upper()}{i}' for i in range(len(voxels))]
    hemispheres = [hemisphere] * len(voxels)
    outside = np.zeros(len(voxels), dtype=bool)
    hover = _contact_hover(atlas, params, voxels, outside, labels,
                           hemispheres)

    magnification = array.local_magnification
    mode = array.render_mode
    flags = array.flags
    flag_names = [('central-1 deg isotropic', flags.central_isotropic),
                  ('do not extrapolate', flags.unreliable_beyond_neighborhood),
                  ('touches placeholder', flags.touches_placeholder),
                  ('magno+parvo footprint', flags.class_heterogeneous),
                  ('scalar magnification', flags.scalar_magnification)]
    for i in range(len(voxels)):
        raised = [name for name, mask in flag_names if bool(mask[i])]
        hover[i] += (
            f'<br>renderer: {mode[i]}'
            f'<br>magnification: {magnification.major_deg_per_mm[i]:.2f} / '
            f'{magnification.minor_deg_per_mm[i]:.2f} deg/mm'
            f'<br>flags: {", ".join(raised) if raised else "none"}')

    group, group_names = _group_by(params, atlas, voxels, hemispheres, array)
    return ElectrodeSet(
        positions_mm=positions, labels=labels, hover=hover,
        group=group, group_names=group_names,
        radius_mm=None, voxel_indices=voxels,
        source=f'{len(voxels)}-electrode LGNElectrodeArray ({hemisphere} '
               f'LGN), as built by the simulator',
        outside=outside)


def _parse_spec(spec: str) -> Dict[str, str]:
    """``key=value;key=value`` into a dict, with a useful error."""
    out: Dict[str, str] = {}
    for item in spec.split(';'):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(
                f"Malformed layout spec {spec!r}: {item!r} is not key=value.")
        key, value = item.split('=', 1)
        out[key.strip()] = value.strip()
    return out


def _floats(text: str, n: Optional[int] = None) -> np.ndarray:
    values = np.array([float(v) for v in text.split(',')])
    if n is not None and len(values) != n:
        raise ValueError(f"Expected {n} comma-separated numbers, got "
                         f"{text!r}.")
    return values


def _contacts_to_set(atlas_by_hemisphere: Mapping, params: Mapping,
                     positions: np.ndarray, hemispheres: List[str],
                     labels: List[str], source: str) -> ElectrodeSet:
    """Common tail of every geometric layout: look each contact up in
    the atlas of its own nucleus and describe it."""
    positions = np.asarray(positions, dtype=float)
    scene = positions.copy()
    voxels = np.zeros((len(positions), 3), dtype=int)
    outside = np.zeros(len(positions), dtype=bool)
    hover: List[str] = [''] * len(positions)
    group = np.zeros(len(positions), dtype=int)
    group_names: List[str] = []
    index_of: Dict[str, int] = {}

    # Per nucleus, not once for all of them: a voxel index means a
    # different voxel in the two atlases, so describing a right-nucleus
    # contact from the left atlas's arrays would report its mirror
    # image's tissue as its own.
    for name in sorted(set(hemispheres)):
        if name not in atlas_by_hemisphere:
            raise ValueError(
                f"A contact was placed in the {name!r} LGN, which this "
                f"figure does not draw. Add it with --hemispheres.")
        rows = np.flatnonzero([h == name for h in hemispheres])
        atlas = atlas_by_hemisphere[name]
        index, off = _voxels_for_positions(atlas, positions[rows, 0],
                                           positions[rows, 1],
                                           positions[rows, 2])
        voxels[rows] = index
        outside[rows] = off
        scene[rows, 0] *= hemisphere_sign(params, name)

        here = [name] * len(rows)
        for row, text in zip(rows, _contact_hover(
                atlas, params, index, off, [labels[r] for r in rows], here)):
            hover[row] = text

        codes, names = _group_by(params, atlas, index, here)
        for group_name in names:
            if group_name not in index_of:
                index_of[group_name] = len(group_names)
                group_names.append(group_name)
        group[rows] = np.array([index_of[n] for n in names])[codes]

    return ElectrodeSet(positions_mm=scene, labels=labels, hover=hover,
                        group=group, group_names=group_names,
                        radius_mm=None, voxel_indices=voxels, source=source,
                        outside=outside)


def probe_contacts(spec: str, atlas_by_hemisphere: Mapping, params: Mapping,
                   default_hemisphere: str) -> ElectrodeSet:
    """Contacts along a straight shank, the way a depth electrode sits.

    ``entry=ml,dv,ap;target=ml,dv,ap;n=8`` spreads `n` contacts evenly
    between the two points, inclusive. Adding ``spacing=0.5`` instead
    steps `n` contacts from `entry` towards `target` at that pitch in
    millimetres, which is how a probe is actually specified. All
    coordinates are Horsley-Clarke millimetres, with `ml` the lateral
    distance from the midline -- positive in both nuclei, so
    ``hemisphere=right`` is what puts a shank in the right one.
    """
    fields = _parse_spec(spec)
    missing = {'entry', 'target', 'n'} - set(fields)
    if missing:
        raise ValueError(
            f"Probe spec {spec!r} is missing {sorted(missing)}. Example: "
            f"'entry=10.0,4.5,7.0;target=10.0,1.0,7.0;n=8'.")

    entry = _floats(fields['entry'], 3)
    target = _floats(fields['target'], 3)
    n = int(fields['n'])
    if n < 1:
        raise ValueError(f"Probe spec {spec!r}: n must be at least 1.")
    direction = target - entry
    length = float(np.linalg.norm(direction))
    if length == 0:
        raise ValueError(f"Probe spec {spec!r}: entry and target are the "
                         f"same point, so the shank has no direction.")
    unit = direction / length

    if 'spacing' in fields:
        step = float(fields['spacing'])
        positions = entry[None, :] + np.arange(n)[:, None] * step * unit
    else:
        fraction = np.linspace(0.0, 1.0, n) if n > 1 else np.zeros(1)
        positions = entry[None, :] + fraction[:, None] * direction

    hemisphere = fields.get('hemisphere', default_hemisphere)
    name = fields.get('label', 'probe')
    labels = [f'{name}-{i}' for i in range(n)]
    return _contacts_to_set(
        atlas_by_hemisphere, params, positions, [hemisphere] * n, labels,
        f'{n}-contact probe {name!r} in the {hemisphere} LGN')


def grid_contacts(spec: str, atlas_by_hemisphere: Mapping, params: Mapping,
                  default_hemisphere: str) -> ElectrodeSet:
    """A planar array, the way a Utah-style grid sits.

    ``centre=ml,dv,ap;plane=dv,ap;n=4,4;pitch=0.4,0.4`` lays `n` by `n`
    contacts out in the plane spanned by two of the three stereotaxic
    axes, centred on `centre`. The remaining axis is the insertion
    direction and is left alone -- so a grid inserted along ML is
    ``plane=dv,ap``.
    """
    fields = _parse_spec(spec)
    centre_key = 'centre' if 'centre' in fields else 'center'
    missing = {centre_key, 'plane', 'n', 'pitch'} - set(fields)
    if missing:
        raise ValueError(
            f"Grid spec {spec!r} is missing {sorted(missing)}. Example: "
            f"'centre=10.5,2.0,7.5;plane=dv,ap;n=4,4;pitch=0.5,0.5'.")

    centre = _floats(fields[centre_key], 3)
    axes = [a.strip().lower() for a in fields['plane'].split(',')]
    if len(axes) != 2 or any(a not in _AXIS_NAMES for a in axes) \
            or axes[0] == axes[1]:
        raise ValueError(
            f"Grid spec {spec!r}: plane must name two different axes out "
            f"of {', '.join(_AXIS_NAMES)}, e.g. 'plane=dv,ap'.")
    counts = _floats(fields['n'], 2).astype(int)
    pitch = _floats(fields['pitch'], 2)

    steps = [(np.arange(c) - (c - 1) / 2.0) * p for c, p in zip(counts,
                                                                pitch)]
    first, second = np.meshgrid(steps[0], steps[1], indexing='ij')
    positions = np.repeat(centre[None, :], first.size, axis=0)
    positions[:, _AXIS_NAMES.index(axes[0])] += first.ravel()
    positions[:, _AXIS_NAMES.index(axes[1])] += second.ravel()

    hemisphere = fields.get('hemisphere', default_hemisphere)
    name = fields.get('label', 'grid')
    labels = [f'{name}-{i}' for i in range(len(positions))]
    return _contacts_to_set(
        atlas_by_hemisphere, params, positions, [hemisphere] * len(positions),
        labels,
        f'{counts[0]}x{counts[1]} grid {name!r} in the {axes[0]}-{axes[1]} '
        f'plane of the {hemisphere} LGN')


def electrodes_from_file(path: Path, atlas_by_hemisphere: Mapping,
                         params: Mapping, units: str,
                         default_hemisphere: str) -> ElectrodeSet:
    """Read contacts from a file.

    Accepts a ``.npy`` of shape (n, 3), or a CSV with a header naming
    three coordinate columns (``ml``, ``dv``, ``ap``, with or without a
    ``_mm`` suffix) and optionally ``hemisphere`` and ``label``.

    :param units: 'mm' for Horsley-Clarke millimetres, 'vox' for atlas
        indices.
    """
    path = Path(path)
    hemispheres: List[str]
    labels: Optional[List[str]] = None

    if path.suffix.lower() == '.npy':
        raw = np.load(path)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError(f"{path} holds an array of shape {raw.shape}; "
                             f"expected (n_contacts, 3).")
        coordinates = raw.astype(float)
        hemispheres = [default_hemisphere] * len(coordinates)
    else:
        import csv
        with open(path, newline='') as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"{path} has a header but no contacts.")
        available = {key.strip().lower(): key for key in rows[0]}

        def column(axis):
            for candidate in (axis, f'{axis}_mm', f'{axis}_vox', f'{axis}_idx'):
                if candidate in available:
                    return available[candidate]
            raise ValueError(
                f"{path} has no {axis!r} column. Found: "
                f"{', '.join(sorted(available))}. Expected columns named "
                f"ml, dv, ap (or ml_mm, dv_mm, ap_mm).")

        keys = [column(axis) for axis in _AXIS_NAMES]
        coordinates = np.array([[float(row[key]) for key in keys]
                                for row in rows])
        if 'hemisphere' in available:
            hemispheres = [row[available['hemisphere']].strip().lower()
                           for row in rows]
        else:
            hemispheres = [default_hemisphere] * len(rows)
        if 'label' in available:
            labels = [row[available['label']] for row in rows]

    if units == 'vox':
        converted = np.empty_like(coordinates, dtype=float)
        for name in sorted(set(hemispheres)):
            if name not in atlas_by_hemisphere:
                raise ValueError(
                    f"{path} places a contact in the {name!r} LGN, which "
                    f"this figure does not draw. Add it with --hemispheres.")
            selection = np.array([h == name for h in hemispheres])
            atlas = atlas_by_hemisphere[name]
            index = coordinates[selection].astype(int)
            converted[selection] = np.stack(atlas.index_to_horsley_clarke(
                index[:, 0], index[:, 1], index[:, 2]), axis=-1)
        coordinates = converted
    elif units != 'mm':
        raise ValueError(f"Unknown units {units!r}; expected 'mm' or 'vox'.")

    if labels is None:
        labels = [f'{path.stem}-{i}' for i in range(len(coordinates))]
    return _contacts_to_set(atlas_by_hemisphere, params, coordinates,
                            hemispheres, labels,
                            f'{len(coordinates)} contacts from {path.name}')


def merge_electrodes(sets: Sequence[ElectrodeSet]) -> Optional[ElectrodeSet]:
    """Combine several layouts into one drawable set.

    Groups are merged by NAME, not by position, so one lamina keeps one
    colour and one legend entry across a shank, a grid and a configured
    array drawn together -- which is the whole point of colouring by
    something other than which layout a contact came from.
    """
    sets = [s for s in sets if s is not None and len(s)]
    if not sets:
        return None
    if len(sets) == 1:
        return sets[0]

    names: List[str] = []
    index_of: Dict[str, int] = {}
    groups: List[np.ndarray] = []
    for one in sets:
        for name in one.group_names:
            if name not in index_of:
                index_of[name] = len(names)
                names.append(name)
        renumber = np.array([index_of[name] for name in one.group_names])
        groups.append(renumber[one.group])

    radii = [s.radius_mm for s in sets]
    return ElectrodeSet(
        positions_mm=np.concatenate([s.positions_mm for s in sets]),
        labels=[label for s in sets for label in s.labels],
        hover=[text for s in sets for text in s.hover],
        group=np.concatenate(groups), group_names=names,
        radius_mm=(None if any(r is None for r in radii)
                   else np.concatenate(radii)),
        voxel_indices=np.concatenate([s.voxel_indices for s in sets]),
        source='; '.join(s.source for s in sets),
        outside=np.concatenate([s.outside for s in sets]))


def recruitment_radius_mm(params: Mapping, n: int) -> Tuple[np.ndarray, str]:
    """The radius of the sphere drawn around each contact.

    Straight from the simulator's own `RecruitmentKernel`, at
    `visualization.electrodes.spread_current_ua`, so the picture and the
    forward pass cannot disagree about how far a current reaches.

    :return: (radius per contact, a label saying which radius it is).
    """
    from dynaphos_lgn.current_spread import RecruitmentKernel

    kernel = RecruitmentKernel.from_params(params)
    current = float(require(params,
                            'visualization.electrodes.spread_current_ua'))
    which = require(params, 'visualization.electrodes.spread_radius')
    if which == 'nominal':
        radius = float(kernel.radius_mm(current))
        label = f'Stoney radius at {current:.0f} uA'
    elif which == 'effective':
        radius = float(kernel.effective_radius_mm(
            current, require(params, 'current_spread.tail_weight')))
        label = (f'gathered radius at {current:.0f} uA (kernel tail at '
                 f'{require(params, "current_spread.tail_weight")})')
    else:
        raise ValueError(
            f"Unknown visualization.electrodes.spread_radius {which!r}; "
            f"expected 'nominal' or 'effective'.")
    return np.full(n, radius), label


def unit_sphere(resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vertices and triangles of a unit UV sphere.

    :param resolution: Points per parametric direction.
    :return: ((n_vertices, 3), (n_triangles, 3)).
    """
    theta = np.linspace(0.0, np.pi, resolution)
    phi = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    grid_theta, grid_phi = np.meshgrid(theta, phi, indexing='ij')
    vertices = np.stack([
        (np.sin(grid_theta) * np.cos(grid_phi)).ravel(),
        (np.sin(grid_theta) * np.sin(grid_phi)).ravel(),
        np.cos(grid_theta).ravel()], axis=-1)

    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution):
            j_next = (j + 1) % resolution
            a = i * resolution + j
            b = i * resolution + j_next
            c = (i + 1) * resolution + j
            d = (i + 1) * resolution + j_next
            triangles.append([a, b, d])
            triangles.append([a, d, c])
    return vertices, np.array(triangles, dtype=int)


# ======================================================================
# The scene
# ======================================================================
@dataclass
class Scene:
    """Everything a backend needs to draw one figure.

    :ivar meshes: One surface per nucleus.
    :ivar overlays: {overlay name: one `Overlay` per mesh}, already
        evaluated at the drawn voxels.
    :ivar order: The overlay names, in the order they were asked for.
    :ivar electrodes: Contacts to draw, or None.
    :ivar params: The parameter dictionary, which every backend reads
        its own defaults from.
    :ivar banner: Which atlas this is, stated on the figure -- a
        synthetic run says so.
    :ivar selection: What tissue was selected, in words.
    """

    meshes: List[VoxelMesh]
    overlays: Dict[str, List[Overlay]]
    order: List[str]
    electrodes: Optional[ElectrodeSet]
    params: Mapping
    banner: str
    selection: str

    def describe(self) -> str:
        """A text summary, in the style of the package's other
        `describe` methods: what was drawn, how coarsely, and out of
        how much."""
        lines = [f"LGN 3-D scene: {self.banner}",
                 f"  tissue drawn      : {self.selection}"]
        for mesh in self.meshes:
            lines.append(
                f"  {mesh.hemisphere:>5} LGN        : {mesh.n_faces:,} faces "
                f"over {mesh.n_blocks:,} surface voxels, the surface of "
                f"{mesh.n_blocks_total:,} drawn voxels "
                f"({mesh.n_atlas_voxels:,} atlas voxels at stride "
                f"{mesh.stride})")
        lines.append(f"  overlays          : {', '.join(self.order)}")
        for name in self.order:
            covered = sum(int(np.isfinite(o.values).sum())
                          for o in self.overlays[name])
            total = sum(len(o.values) for o in self.overlays[name])
            share = 100.0 * covered / max(total, 1)
            lines.append(f"    {name:<26} {share:5.1f}% of drawn voxels "
                         f"have a value")
        if self.electrodes is not None:
            n_outside = int(self.electrodes.outside.sum())
            lines.append(f"  electrodes        : {len(self.electrodes)} "
                         f"({self.electrodes.source})")
            if n_outside:
                lines.append(f"    {n_outside} contact(s) fall outside the "
                             f"atlas volume and are drawn hollow")
        return '\n'.join(lines)


def build_scene(params: Mapping, atlases: Mapping,
                jacobian_for: Optional[Callable[[str], object]],
                overlay_names: Sequence[str],
                mask_options: Optional[Mapping] = None,
                electrodes: Optional[ElectrodeSet] = None,
                banner: str = '') -> Scene:
    """Assemble the surfaces and every overlay on them.

    :param atlases: {hemisphere: built atlas}.
    :param jacobian_for: Callable returning the `JacobianAtlas` for one
        hemisphere, or None where no Jacobian is available. Called at
        most once per hemisphere, and only if a Jacobian-derived
        overlay was actually asked for.
    :param overlay_names: Which overlays to evaluate.
    :param mask_options: Keyword arguments for `tissue_mask`.
    :param electrodes: Contacts to draw.
    :param banner: Which atlas this is.
    """
    stride = int(require(params, 'visualization.stride_vox'))
    mask_options = dict(mask_options or {})

    meshes: List[VoxelMesh] = []
    contexts: List[AtlasContext] = []
    for hemisphere, atlas in atlases.items():
        mask = tissue_mask(atlas, params, **mask_options)
        mesh = build_mesh(atlas, mask, stride,
                          hemisphere_sign(params, hemisphere))
        if mesh.n_faces == 0:
            logging.warning(
                "The %s LGN has no tissue left after the selection, so "
                "nothing is drawn for it.", hemisphere)
        meshes.append(mesh)
        loader = (None if jacobian_for is None
                  else (lambda h=hemisphere: jacobian_for(h)))
        contexts.append(AtlasContext(atlas, params, loader))

    overlays: Dict[str, List[Overlay]] = {}
    for name in overlay_names:
        overlays[name] = [compute_overlay(name, context, mesh.voxel_indices)
                          for context, mesh in zip(contexts, meshes)]

    selection = _describe_selection(mask_options, params)
    return Scene(meshes=meshes, overlays=overlays, order=list(overlay_names),
                 electrodes=electrodes, params=params, banner=banner,
                 selection=selection)


def _describe_selection(mask_options: Mapping, params: Mapping) -> str:
    """The tissue filter, in words, for the figure and the summary."""
    parts: List[str] = []
    if mask_options.get('cell_class'):
        parts.append(f"{mask_options['cell_class']} laminae only")
    if mask_options.get('layers'):
        parts.append('laminae ' + ', '.join(str(c) for c
                                            in mask_options['layers']))
    low = mask_options.get('min_eccentricity_deg')
    high = mask_options.get('max_eccentricity_deg')
    if low is not None or high is not None:
        parts.append(f"eccentricity {low if low is not None else 0}"
                     f"-{high if high is not None else 'max'} deg")
    for axis, lo, hi in (mask_options.get('clips') or ()):
        parts.append(f"{axis.upper()} clipped to {lo}-{hi} mm")
    include_ipsi = mask_options.get('include_ipsi_placeholder')
    if include_ipsi is None:
        include_ipsi = not require(params,
                                   'atlas.exclude_ipsi_flat_inclination')
    if include_ipsi:
        parts.append('including the coarse ipsilateral placeholder')
    return '; '.join(parts) if parts else 'all reconstructed tissue'


def hover_text(scene: Scene, mesh_index: int, blocks: np.ndarray
               ) -> List[str]:
    """Every overlay's value at a set of drawn voxels, as one tooltip.

    Independent of which overlay is on screen, so the point cloud
    answers "what else is true here?" rather than repeating the colour.
    """
    mesh = scene.meshes[mesh_index]
    position = mesh.centres_mm[blocks]
    lines = [[f'<b>{mesh.hemisphere} LGN</b>',
              f'ML {p[0]:+.2f}, DV {p[1]:+.2f}, AP {p[2]:+.2f} mm']
             for p in position]

    for name in scene.order:
        overlay = scene.overlays[name][mesh_index]
        values = overlay.values[blocks]
        for row, value in zip(lines, values):
            if not np.isfinite(value):
                row.append(f'{overlay.label}: -')
            elif overlay.kind == 'categorical':
                row.append(f'{overlay.label}: '
                           f'{overlay.categories[int(value)]}')
            else:
                row.append(f'{overlay.label}: {value:.3g}')
    return ['<br>'.join(row) for row in lines]


def _hover_blocks(scene: Scene, mesh_index: int, rng: np.random.Generator
                  ) -> np.ndarray:
    """Which drawn voxels get a hover point, subsampled to the budget."""
    mesh = scene.meshes[mesh_index]
    budget = int(require(scene.params, 'visualization.hover_points.max_points'))
    if mesh.n_blocks <= budget:
        return np.arange(mesh.n_blocks)
    return np.sort(rng.choice(mesh.n_blocks, budget, replace=False))


def group_colours(params: Mapping, n_groups: int) -> List[str]:
    """One colour per electrode group.

    A single group -- the default, `colour_by: none` -- keeps the
    configured electrode colour rather than picking the first entry of
    a qualitative colormap, so the common case looks the way the config
    says it should.
    """
    import matplotlib.colors as mcolors
    if n_groups <= 1:
        return [require(params, 'visualization.electrodes.colour')]
    cmap = colormap(require(params, 'visualization.colormaps.cell_class'))
    return [mcolors.to_hex(category_colour(cmap, i, n_groups))
            for i in range(n_groups)]


def _open_symbol(symbol: str) -> str:
    """The hollow version of a marker symbol, for a contact that falls
    outside the atlas volume -- so it is visibly a position the atlas
    cannot describe rather than one it can."""
    if symbol in ('circle', 'square', 'diamond'):
        return f'{symbol}-open'
    return 'x'


def _figure_title(scene: Scene, overlay_name: str) -> Tuple[str, str]:
    """(title, subtitle) for one overlay."""
    overlay = scene.overlays[overlay_name][0]
    title = f'LGN atlas — {overlay.label}'
    subtitle = f'{scene.banner} | {scene.selection}'
    if overlay.note:
        subtitle += f'<br>{overlay.note}'
    return title, subtitle


# ======================================================================
# Plotly backend
# ======================================================================
def render_plotly(scene: Scene, out_path: Path,
                  open_browser: Optional[bool] = None) -> Path:
    """Write an interactive HTML figure.

    One `Mesh3d` per nucleus, recoloured in place by a dropdown, so the
    geometry is stored once however many overlays are offered. The
    hoverable point cloud and the electrodes are separate traces, and
    both start hidden behind a legend entry when they would otherwise
    get in the way.

    :param scene: What to draw.
    :param out_path: Where to write the .html file.
    :param open_browser: Open it when done. Defaults to
        `visualization.output.open_browser`.
    :return: `out_path`.
    """
    import plotly.graph_objects as go

    params = scene.params
    rng = np.random.default_rng(require(params, 'run.seed'))
    lighting = require(params, 'visualization.surface.lighting')
    figure = go.Figure()

    # --- the nuclei ---------------------------------------------------
    specs: Dict[str, List[ColourSpec]] = {
        name: [colour_spec(overlay, params) for overlay in scene.overlays[name]]
        for name in scene.order}
    first = scene.order[0]
    mesh_traces: List[int] = []

    for i, mesh in enumerate(scene.meshes):
        if mesh.n_faces == 0:
            continue
        triangles = mesh.triangles
        spec = specs[first][i]
        figure.add_trace(go.Mesh3d(
            x=mesh.vertices_mm[:, 0], y=mesh.vertices_mm[:, 1],
            z=mesh.vertices_mm[:, 2],
            i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
            intensity=spec.values[mesh.block_of_triangle],
            intensitymode='cell',
            colorscale=spec.colorscale, cmin=spec.cmin, cmax=spec.cmax,
            showscale=not mesh_traces,
            colorbar=dict(title=dict(text=scene.overlays[first][i].label,
                                     side='right'),
                          tickvals=spec.tickvals, ticktext=spec.ticktext,
                          len=0.7, x=1.02),
            flatshading=bool(require(params,
                                     'visualization.surface.flat_shading')),
            opacity=float(require(params, 'visualization.surface.opacity')),
            lighting=dict(ambient=lighting['ambient'],
                          diffuse=lighting['diffuse'],
                          specular=lighting['specular'],
                          roughness=lighting['roughness'],
                          fresnel=lighting['fresnel']),
            name=f'{mesh.hemisphere} LGN', showlegend=True,
            hoverinfo='skip'))
        mesh_traces.append(len(figure.data) - 1)

    # --- hoverable voxel centres --------------------------------------
    if require(params, 'visualization.hover_points.enabled'):
        for i, mesh in enumerate(scene.meshes):
            if mesh.n_blocks == 0:
                continue
            blocks = _hover_blocks(scene, i, rng)
            figure.add_trace(go.Scatter3d(
                x=mesh.centres_mm[blocks, 0], y=mesh.centres_mm[blocks, 1],
                z=mesh.centres_mm[blocks, 2], mode='markers',
                marker=dict(
                    size=require(params,
                                 'visualization.hover_points.marker_size'),
                    color=require(params, 'visualization.hover_points.colour')),
                text=hover_text(scene, i, blocks), hoverinfo='text',
                name=f'{mesh.hemisphere} voxel values ({len(blocks):,} of '
                     f'{mesh.n_blocks:,})',
                visible='legendonly'))

    # --- electrodes ---------------------------------------------------
    electrodes = scene.electrodes
    if electrodes is not None and len(electrodes):
        colours = group_colours(params, len(electrodes.group_names))
        size = require(params, 'visualization.electrodes.marker_size')
        symbol = require(params, 'visualization.electrodes.marker_symbol')
        label_every = int(require(params,
                                  'visualization.electrodes.label_every'))
        for group in np.unique(electrodes.group):
            for outside in (False, True):
                selection = ((electrodes.group == group)
                             & (electrodes.outside == outside))
                index = np.flatnonzero(selection)
                if not len(index):
                    continue
                mode = 'markers'
                text = None
                if label_every > 0:
                    mode = 'markers+text'
                    text = [electrodes.labels[j] if (j % label_every == 0)
                            else '' for j in index]
                name = electrodes.group_names[int(group)]
                figure.add_trace(go.Scatter3d(
                    x=electrodes.positions_mm[selection, 0],
                    y=electrodes.positions_mm[selection, 1],
                    z=electrodes.positions_mm[selection, 2],
                    mode=mode, text=text, textposition='top center',
                    marker=dict(size=size,
                                symbol=_open_symbol(symbol) if outside
                                else symbol,
                                color=colours[int(group)],
                                line=dict(width=1, color='#000000'),
                                opacity=1.0),
                    customdata=[electrodes.hover[j] for j in index],
                    hovertemplate='%{customdata}<extra></extra>',
                    name=f'{name} (outside the atlas)' if outside else name))

        if require(params, 'visualization.electrodes.show_spread'):
            _add_spread_spheres(figure, electrodes, params)

    # --- the overlay dropdown ----------------------------------------
    if len(scene.order) > 1 and mesh_traces:
        figure.update_layout(updatemenus=[dict(
            buttons=_overlay_buttons(scene, specs, mesh_traces),
            direction='down', showactive=True,
            x=0.0, xanchor='left', y=1.0, yanchor='top')],
            annotations=[dict(text='Overlay', x=0.0, xanchor='left', y=1.05,
                              yanchor='bottom', xref='paper', yref='paper',
                              showarrow=False)])

    title, subtitle = _figure_title(scene, first)
    camera = require(params, 'visualization.scene.camera_eye')
    labels = require(params, 'visualization.scene.axis_labels')
    figure.update_layout(
        title=dict(text=f'{title}<br><sup>{subtitle}</sup>', x=0.5,
                   xanchor='center'),
        width=require(params, 'visualization.figure.width'),
        height=require(params, 'visualization.figure.height'),
        paper_bgcolor=require(params, 'visualization.scene.background'),
        scene=dict(aspectmode='data',
                   xaxis_title=labels['x'], yaxis_title=labels['y'],
                   zaxis_title=labels['z'],
                   camera=dict(eye=dict(x=camera[0], y=camera[1],
                                        z=camera[2]))),
        legend=dict(x=0.0, y=0.90, xanchor='left', yanchor='top',
                    bgcolor='rgba(255,255,255,0.75)'),
        margin=dict(l=0, r=0, t=90, b=0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 'inline' is what the config calls plotly's own `True`: the whole
    # library written into the file, so it opens with no network.
    plotlyjs = require(params, 'visualization.output.plotlyjs')
    figure.write_html(str(out_path),
                      include_plotlyjs=True if plotlyjs == 'inline'
                      else plotlyjs)

    if open_browser is None:
        open_browser = require(params, 'visualization.output.open_browser')
    if open_browser:
        import webbrowser
        webbrowser.open(out_path.resolve().as_uri())
    return out_path


def _quantised(values: np.ndarray, cmin: float, cmax: float) -> List[float]:
    """Intensities rounded to what the colour axis can actually show.

    A dropdown entry carries one number per triangle, and unlike trace
    data those numbers are written into the HTML as plain JSON rather
    than base64. Full float precision there costs megabytes to express
    differences finer than one ten-thousandth of the colour range,
    which no screen resolves.
    """
    span = max(float(cmax) - float(cmin), 1e-12)
    decimals = int(max(0, 4 - np.floor(np.log10(span))))
    return np.round(np.asarray(values, dtype=float), decimals).tolist()


def _overlay_buttons(scene: Scene, specs: Mapping, mesh_traces: List[int]
                     ) -> List[dict]:
    """Dropdown entries that recolour the meshes in place.

    A `restyle` of the intensity and the colour scale, not a second
    copy of the geometry per overlay -- which is the difference between
    a figure that opens and one that does not.
    """
    buttons = []
    drawn = [i for i, mesh in enumerate(scene.meshes) if mesh.n_faces > 0]
    for name in scene.order:
        overlay = scene.overlays[name][drawn[0]]
        title, subtitle = _figure_title(scene, name)
        buttons.append(dict(
            label=overlay.label,
            method='update',
            args=[{'intensity': [
                       _quantised(specs[name][i].values[
                           scene.meshes[i].block_of_triangle],
                           specs[name][i].cmin, specs[name][i].cmax)
                       for i in drawn],
                   'colorscale': [specs[name][i].colorscale for i in drawn],
                   'cmin': [specs[name][i].cmin for i in drawn],
                   'cmax': [specs[name][i].cmax for i in drawn],
                   'colorbar.title.text': [scene.overlays[name][i].label
                                           for i in drawn],
                   'colorbar.tickvals': [specs[name][i].tickvals
                                         for i in drawn],
                   'colorbar.ticktext': [specs[name][i].ticktext
                                         for i in drawn]},
                  {'title.text': f'{title}<br><sup>{subtitle}</sup>'},
                  mesh_traces]))
    return buttons


def _add_spread_spheres(figure, electrodes: ElectrodeSet, params: Mapping):
    """The tissue each contact recruits, as one translucent mesh."""
    import plotly.graph_objects as go

    radius, label = recruitment_radius_mm(params, len(electrodes))
    resolution = int(require(params,
                             'visualization.electrodes.spread_resolution'))
    sphere_vertices, sphere_triangles = unit_sphere(resolution)

    vertices = []
    triangles = []
    for i, centre in enumerate(electrodes.positions_mm):
        vertices.append(sphere_vertices * radius[i] + centre)
        triangles.append(sphere_triangles + i * len(sphere_vertices))
    vertices = np.concatenate(vertices)
    triangles = np.concatenate(triangles)

    figure.add_trace(go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        color=require(params, 'visualization.electrodes.colour'),
        opacity=float(require(params,
                              'visualization.electrodes.spread_opacity')),
        name=f'recruited tissue ({label})', showlegend=True,
        hoverinfo='skip'))


# ======================================================================
# Matplotlib backend
# ======================================================================
def _shade(quad_vertices: np.ndarray, rgba: np.ndarray, params: Mapping
           ) -> np.ndarray:
    """Lambert-shade face colours so the surface reads as a solid.

    Matplotlib's 3-D axes do not light anything, so without this a
    voxel volume renders as a flat silhouette. The same ambient and
    diffuse terms the plotly backend passes to its own lighting model
    are used here, so the two pictures match.
    """
    lighting = require(params, 'visualization.surface.lighting')
    direction = np.asarray(lighting['light_direction'], dtype=float)
    direction = direction / np.linalg.norm(direction)

    edge_a = quad_vertices[:, 1] - quad_vertices[:, 0]
    edge_b = quad_vertices[:, 3] - quad_vertices[:, 0]
    normal = np.cross(edge_a, edge_b)
    norm = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = np.divide(normal, norm, out=np.zeros_like(normal),
                       where=norm > 0)

    factor = (lighting['ambient']
              + lighting['diffuse'] * np.abs(normal @ direction))
    shaded = rgba.copy()
    shaded[:, :3] = np.clip(rgba[:, :3] * factor[:, None], 0.0, 1.0)
    return shaded


def render_matplotlib(scene: Scene, out_path: Path, show: bool = False
                      ) -> Path:
    """Write a PNG, one 3-D panel per overlay.

    No extra dependency, and no interactivity: the panels are fixed
    views. Use it for a figure that goes in a document, and the plotly
    backend for looking around.
    """
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    params = scene.params
    n_panels = len(scene.order)
    columns = min(int(require(params,
                              'visualization.figure.matplotlib_panel_columns')),
                  n_panels)
    rows = int(np.ceil(n_panels / columns))
    # The configured size is ONE panel's, so a grid of overlays grows
    # with the grid instead of cramming every panel into one page.
    width, height = require(params, 'visualization.figure.matplotlib_figsize')
    figure = plt.figure(figsize=(width * columns, height * rows))

    show_edges = bool(require(params, 'visualization.surface.show_edges'))
    edge_width = float(require(params, 'visualization.surface.edge_width'))
    camera = require(params, 'visualization.scene.camera_eye')
    labels = require(params, 'visualization.scene.axis_labels')

    for panel, name in enumerate(scene.order):
        axes = figure.add_subplot(rows, columns, panel + 1, projection='3d')
        limits = []
        for i, mesh in enumerate(scene.meshes):
            if mesh.n_faces == 0:
                continue
            spec = colour_spec(scene.overlays[name][i], params)
            quads = mesh.quad_vertices_mm
            colours = _shade(quads, spec.rgba[mesh.block_of_quad], params)
            collection = Poly3DCollection(
                quads, facecolors=colours,
                edgecolors=('#00000022' if show_edges else 'none'),
                linewidths=edge_width if show_edges else 0.0)
            axes.add_collection3d(collection)
            limits.append(mesh.vertices_mm)

        _add_matplotlib_electrodes(axes, scene, params)
        _set_equal_box(axes, limits, scene, labels, camera)

        overlay = scene.overlays[name][0]
        axes.set_title(_wrap(overlay.label + '\n' + overlay.note, 64),
                       fontsize=9)
        _add_matplotlib_legend(figure, axes, scene, name, params)

    figure.suptitle(f'LGN atlas in Horsley-Clarke coordinates\n'
                    f'{scene.banner} | {scene.selection}', fontsize=11)
    # h_pad, because a 3-D axis's tick labels stick out further than
    # tight_layout accounts for, and a second row of panels lands on
    # the first row's axis titles without it.
    figure.tight_layout(rect=(0.015, 0, 0.995, 0.95), h_pad=3.0)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path,
                   dpi=int(require(params,
                                   'visualization.figure.matplotlib_dpi')))
    if show:
        plt.show()
    else:
        plt.close(figure)
    return out_path


def _wrap(text: str, width: int) -> str:
    import textwrap
    return '\n'.join(textwrap.wrap(text, width) or [''])


def _set_equal_box(axes, limits: List[np.ndarray], scene: Scene,
                   labels: Mapping, camera: Sequence[float]):
    """Millimetres equal in all three directions, and a matching view."""
    if not limits:
        return
    stacked = np.concatenate(limits)
    low, high = stacked.min(axis=0), stacked.max(axis=0)
    span = np.maximum(high - low, 1e-6)
    for setter, lo, hi in zip((axes.set_xlim, axes.set_ylim, axes.set_zlim),
                              low, high):
        setter(lo, hi)
    axes.set_box_aspect(tuple(span))
    axes.set_xlabel(labels['x'], fontsize=7)
    axes.set_ylabel(labels['y'], fontsize=7)
    axes.set_zlabel(labels['z'], fontsize=7)
    axes.tick_params(labelsize=6)
    # Same viewpoint as the plotly camera, expressed the way matplotlib
    # wants it: an elevation and an azimuth rather than an eye position.
    eye = np.asarray(camera, dtype=float)
    radius = float(np.linalg.norm(eye))
    axes.view_init(elev=float(np.degrees(np.arcsin(eye[2] / radius))),
                   azim=float(np.degrees(np.arctan2(eye[1], eye[0]))))


def _add_matplotlib_electrodes(axes, scene: Scene, params: Mapping):
    """Contacts, and their labels if the config asks for them."""
    electrodes = scene.electrodes
    if electrodes is None or not len(electrodes):
        return
    colours = group_colours(params, len(electrodes.group_names))
    label_every = int(require(params, 'visualization.electrodes.label_every'))
    size = float(require(params, 'visualization.electrodes.marker_size')) ** 2
    for group in np.unique(electrodes.group):
        for outside in (False, True):
            selection = ((electrodes.group == group)
                         & (electrodes.outside == outside))
            if not selection.any():
                continue
            colour = colours[int(group)]
            axes.scatter(electrodes.positions_mm[selection, 0],
                         electrodes.positions_mm[selection, 1],
                         electrodes.positions_mm[selection, 2],
                         s=size, marker='D',
                         facecolors='none' if outside else colour,
                         edgecolors=colour if outside else 'black',
                         linewidths=1.0 if outside else 0.4,
                         depthshade=False, zorder=5,
                         label=(f'{electrodes.group_names[int(group)]} '
                                f'(outside the atlas)' if outside
                                else electrodes.group_names[int(group)]))
    if label_every > 0:
        for i, (position, label) in enumerate(zip(electrodes.positions_mm,
                                                  electrodes.labels)):
            if i % label_every == 0:
                axes.text(position[0], position[1], position[2], label,
                          fontsize=6)


def _add_matplotlib_legend(figure, axes, scene: Scene, name: str,
                           params: Mapping):
    """A colourbar for a continuous overlay, a patch legend for a
    categorical one -- plus the electrode groups, if any."""
    import matplotlib.colors as mcolors
    from matplotlib import cm
    from matplotlib.patches import Patch

    overlay = scene.overlays[name][0]
    handles = []

    if overlay.kind == 'categorical':
        cmap = colormap(require(params, f'visualization.colormaps.{name}'))
        n = len(overlay.categories or [])
        handles = [Patch(facecolor=category_colour(cmap, i, n),
                         label=category)
                   for i, category in enumerate(overlay.categories or [])]
        handles.append(Patch(
            facecolor=require(params, 'visualization.missing_colour'),
            label='no value'))
    else:
        low, high = continuous_limits(overlay, params)
        mappable = cm.ScalarMappable(
            norm=mcolors.Normalize(vmin=low, vmax=high),
            cmap=colormap(require(params,
                                   f'visualization.colormaps.{name}')))
        bar = figure.colorbar(mappable, ax=axes, shrink=0.6, pad=0.08)
        bar.set_label(overlay.unit or overlay.label, fontsize=7)
        bar.ax.tick_params(labelsize=6)

    if scene.electrodes is not None and len(scene.electrodes):
        # Prefixed, because the contacts and the surface can be keyed
        # by the same thing -- lamina, say -- in two different colour
        # schemes, and one unlabelled legend box would read as one key.
        colours = group_colours(params, len(scene.electrodes.group_names))
        for group in np.unique(scene.electrodes.group):
            handles.append(Patch(
                facecolor=colours[int(group)],
                label=f'electrode: '
                      f'{scene.electrodes.group_names[int(group)]}'))
    if handles:
        axes.legend(handles=handles, fontsize=6, loc='upper left',
                    framealpha=0.85)


# ======================================================================
# Putting it together
# ======================================================================
def atlas_banner(atlas) -> str:
    """Which atlas a figure is of, said on the figure itself."""
    if getattr(atlas, 'is_synthetic', False):
        return ('SYNTHETIC ATLAS -- a code demonstration, not the LGN')
    return 'Erwin, Baker, Busen & Malpeli (1999) macaque LGN atlas'


def jacobian_provider(params: Mapping, atlas, atlas_dir: Optional[Path],
                      recompute: bool = False
                      ) -> Callable[[str], Optional[object]]:
    """A per-hemisphere Jacobian getter that loads at most one cache.

    The right nucleus's field is derived from the left one's by
    reflection rather than refitted, exactly as the simulator does it,
    so asking for both costs one load and no second fit.
    """
    from dynaphos_lgn.atlas import MirroredJacobianAtlas
    from dynaphos_lgn.build import build_jacobian

    cache: Dict[str, object] = {}

    def provide(hemisphere: str):
        if 'left' not in cache:
            directory = (None if getattr(atlas, 'is_synthetic', False)
                         else Path(atlas_dir
                                   or require(params, 'atlas.directory')))
            if directory is not None and not (
                    directory
                    / require(params, 'atlas.jacobian_cache')).exists():
                logging.warning(
                    "No cached Jacobian in %s, so it has to be fitted now "
                    "-- roughly 90 s and 4.2 GB of peak memory. "
                    "`python -m dynaphos_lgn.build --cache-jacobian` does "
                    "this once and writes the cache.", directory)
            cache['left'] = build_jacobian(params, atlas, directory,
                                           recompute)
        if hemisphere == 'left':
            return cache['left']
        if 'right' not in cache:
            cache['right'] = MirroredJacobianAtlas(cache['left'], params)
        return cache['right']

    return provide


def collect_electrodes(params: Mapping, atlases: Mapping, options,
                       rng: np.random.Generator,
                       provide_jacobian: Optional[Callable] = None
                       ) -> Optional[ElectrodeSet]:
    """Every layout the command line asked for, merged into one set.

    The sources can be combined: a configured array in one nucleus and
    a hand-placed shank in the other is one `--electrodes layout
    --probe ...` away.
    """
    from dynaphos_lgn.build import (build_electrode_array, build_kernel,
                                    build_magnification_model,
                                    n_electrodes_for)

    sets: List[ElectrodeSet] = []
    default_hemisphere = next(iter(atlases))

    if options.electrodes == 'layout':
        for hemisphere, atlas in atlases.items():
            sets.append(electrodes_from_layout(
                atlas, params, n_electrodes_for(params, hemisphere), rng))
    elif options.electrodes == 'array':
        if provide_jacobian is None:
            raise ValueError("--electrodes array needs the Jacobian atlas, "
                             "which is not available here.")
        kernel = build_kernel(params)
        for hemisphere, atlas in atlases.items():
            jacobian = provide_jacobian(hemisphere)
            magnification = build_magnification_model(params, atlas, jacobian)
            array = build_electrode_array(
                params, atlas, jacobian, kernel, rng,
                n_electrodes=n_electrodes_for(params, hemisphere),
                magnification_model=magnification)
            array.magnification_model = magnification
            sets.append(electrodes_from_array(array, params))
    elif options.electrodes != 'none':
        raise ValueError(f"Unknown --electrodes {options.electrodes!r}; "
                         f"expected none, layout or array.")

    if options.electrodes_file:
        sets.append(electrodes_from_file(
            Path(options.electrodes_file), atlases, params,
            options.electrodes_units, default_hemisphere))
    for spec in (options.probe or []):
        sets.append(probe_contacts(spec, atlases, params, default_hemisphere))
    for spec in (options.grid or []):
        sets.append(grid_contacts(spec, atlases, params, default_hemisphere))

    return merge_electrodes(sets)


def _parse_clip(text: str) -> Tuple[str, float, float]:
    """``axis:low:high`` in millimetres, e.g. ``ap:6:9``."""
    parts = text.split(':')
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--clip wants AXIS:LOW:HIGH in millimetres, e.g. 'ap:6:9'; "
            f"got {text!r}.")
    axis = parts[0].strip().lower()
    if axis not in _AXIS_NAMES:
        raise argparse.ArgumentTypeError(
            f"--clip axis must be one of {', '.join(_AXIS_NAMES)}; got "
            f"{axis!r}.")
    try:
        return axis, float(parts[1]), float(parts[2])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--clip bounds must be numbers in millimetres; got {text!r}.")


def list_overlays() -> str:
    """One line per overlay: its name and what it costs to draw."""
    lines = ['Overlays (--overlay NAME [NAME ...], or --overlay all):']
    for name in OVERLAY_BUILDERS:
        needs = ('needs the cached Jacobian' if name in JACOBIAN_OVERLAYS
                 else 'straight from the .DAT files')
        lines.append(f'  {name:<28} {needs}')
    return '\n'.join(lines)


def _apply_overrides(params: dict, options) -> dict:
    """Fold command-line choices into the parameter dictionary.

    Everything downstream then reads the config and only the config,
    so a flag and a config key cannot disagree about what was drawn.

    Reads through `getattr` because the slice viewer shares most of
    these flags but not the ones that only mean something for a
    surface.
    """
    visualization = params['visualization']
    if getattr(options, 'stride', None) is not None:
        visualization['stride_vox'] = int(options.stride)
    if getattr(options, 'surface_opacity', None) is not None:
        visualization['surface']['opacity'] = float(options.surface_opacity)
    if options.backend is not None:
        visualization['backend'] = options.backend
    if options.overlay:
        visualization['overlays'] = (list(OVERLAY_BUILDERS)
                                     if options.overlay == ['all']
                                     else list(options.overlay))
    if options.colour_by is not None:
        visualization['electrodes']['colour_by'] = options.colour_by
    if options.label_every is not None:
        visualization['electrodes']['label_every'] = int(options.label_every)
    if getattr(options, 'show_spread', False):
        visualization['electrodes']['show_spread'] = True
    if options.spread_current is not None:
        visualization['electrodes']['spread_current_ua'] = float(
            options.spread_current)
    if options.no_browser:
        visualization['output']['open_browser'] = False
    if options.n_electrodes is not None:
        params['electrodes']['n_electrodes'] = int(options.n_electrodes)
    if options.hemispheres:
        params['atlas']['hemispheres'] = list(options.hemispheres)
    return params


def add_atlas_arguments(parser: argparse.ArgumentParser):
    """Which atlas, and how much of it. Shared with the slice viewer,
    so the two programs cannot drift apart on what `--synthetic` or
    `--hemispheres` mean."""
    source = parser.add_argument_group('atlas')
    source.add_argument('--params', default=None,
                        help='Parameter file (default: '
                             'config/params_lgn.yaml beside the package).')
    source.add_argument('--atlas-dir', default=None,
                        help='Directory holding the four .DAT files.')
    source.add_argument('--synthetic', action='store_true',
                        help='Use the synthetic stand-in atlas. Exercises '
                             'the code; means nothing biologically.')
    source.add_argument('--hemispheres', nargs='+', choices=['left', 'right'],
                        default=None,
                        help='Which nuclei to draw (default: '
                             'atlas.hemispheres).')
    source.add_argument('--recompute-jacobian', action='store_true',
                        help='Refit the Jacobian instead of using the cache.')
    return parser


def add_tissue_arguments(parser: argparse.ArgumentParser,
                         surface: bool = True):
    """Which tissue to show and what to paint on it.

    :param surface: Include the options that only mean something for a
        rendered surface -- the block stride and its opacity. A slice
        is one voxel thick and fully opaque by construction, so the
        slice viewer leaves them out rather than accepting them and
        doing nothing.
    """
    what = parser.add_argument_group('what to draw')
    what.add_argument('--overlay', nargs='+', default=None,
                      help="Overlays to offer, or 'all'. See "
                           "--list-overlays.")
    if surface:
        what.add_argument('--stride', type=int, default=None,
                          help='Atlas voxels per drawn voxel, per axis '
                               '(default: visualization.stride_vox).')
    what.add_argument('--cell-class', choices=['magno', 'parvo'],
                      default=None, help='Draw only one cell class.')
    what.add_argument('--layers', nargs='+', type=int, default=None,
                      help='Draw only these LAYERS.DAT codes.')
    what.add_argument('--ecc-min', type=float, default=None,
                      help='Drop tissue below this eccentricity (deg).')
    what.add_argument('--ecc-max', type=float, default=None,
                      help='Drop tissue above this eccentricity (deg).')
    what.add_argument('--clip', action='append', type=_parse_clip,
                      default=None, metavar='AXIS:LOW:HIGH',
                      help='Cut the volume open, in millimetres, e.g. '
                           "'ap:6:9'. Repeatable. The cut face is drawn "
                           'and painted like any other surface.')
    if surface:
        what.add_argument('--surface-opacity', type=float, default=None,
                          help='0 to 1. Electrodes sit INSIDE the nucleus, '
                               'so an opaque surface hides them; about 0.35 '
                               'shows them through it, and --clip cuts the '
                               'volume open instead.')
    what.add_argument('--include-ipsi', action='store_true',
                      help='Keep the coarse +-135 deg ipsilateral '
                           'placeholder voxels, which the simulator drops.')
    return parser


def add_electrode_arguments(parser: argparse.ArgumentParser):
    """Where the contacts come from and what their colours mean."""
    contacts = parser.add_argument_group('electrodes')
    contacts.add_argument('--electrodes', default='none',
                          choices=['none', 'layout', 'array'],
                          help="'layout' places the array this config "
                               "describes; 'array' builds the real "
                               "LGNElectrodeArray, which costs a Jacobian "
                               "load and in exchange carries its flags.")
    contacts.add_argument('--n-electrodes', type=int, default=None,
                          help='Per nucleus (default: '
                               'electrodes.n_electrodes).')
    contacts.add_argument('--electrodes-file', default=None,
                          help='CSV with ml/dv/ap columns, or an .npy of '
                               'shape (n, 3).')
    contacts.add_argument('--electrodes-units', default='mm',
                          choices=['mm', 'vox'],
                          help='Units in --electrodes-file (default: mm).')
    contacts.add_argument('--probe', action='append', default=None,
                          metavar='SPEC',
                          help="A shank: 'entry=ml,dv,ap;target=ml,dv,ap;"
                               "n=8[;spacing=0.5][;hemisphere=left]"
                               "[;label=A]'. Repeatable.")
    contacts.add_argument('--grid', action='append', default=None,
                          metavar='SPEC',
                          help="A planar array: 'centre=ml,dv,ap;plane=dv,"
                               "ap;n=4,4;pitch=0.5,0.5[;hemisphere=left]'. "
                               "Repeatable.")
    contacts.add_argument('--colour-by', default=None,
                          choices=['none', 'layer', 'cell_class',
                                   'hemisphere', 'flagged', 'render_mode'],
                          help='What the contact colours mean.')
    contacts.add_argument('--label-every', type=int, default=None,
                          help='Label every Nth contact; 0 for none.')
    contacts.add_argument('--spread-current', type=float, default=None,
                          help='Current the recruited tissue is drawn at '
                               '(uA).')
    return parser


def add_output_arguments(parser: argparse.ArgumentParser):
    """Backend, destination and verbosity."""
    output = parser.add_argument_group('output')
    output.add_argument('--backend', choices=['plotly', 'matplotlib'],
                        default=None)
    output.add_argument('--out', default=None,
                        help='Output file (default: '
                             'visualization.output.directory plus the '
                             "backend's configured name).")
    output.add_argument('--no-browser', action='store_true',
                        help='Do not open the HTML when it is written.')
    output.add_argument('--show', action='store_true',
                        help='matplotlib only: open an interactive window '
                             'as well as writing the PNG.')
    output.add_argument('--list-overlays', action='store_true',
                        help='List the overlays and exit.')
    output.add_argument('-v', '--verbose', action='store_true')
    return parser


def build_parser() -> argparse.ArgumentParser:
    """The 3-D viewer's command line: the shared groups plus the two
    options that only a rendered surface has."""
    parser = argparse.ArgumentParser(
        prog='python -m dynaphos_lgn.viz3d',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_atlas_arguments(parser)
    add_tissue_arguments(parser, surface=True)
    add_electrode_arguments(parser)
    parser.add_argument('--show-spread', action='store_true',
                        help='Draw the tissue each contact recruits, as a '
                             'translucent sphere.')
    add_output_arguments(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    from dynaphos.utils import load_params
    from dynaphos_lgn.build import (build_atlas, build_hemisphere_atlas,
                                    configured_hemispheres)

    options = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if options.verbose
                        else logging.WARNING,
                        format='%(levelname)s: %(message)s')

    if options.list_overlays:
        print(list_overlays())
        return 0

    repository = Path(__file__).resolve().parents[1]
    params_path = Path(options.params or
                       repository / 'config' / 'params_lgn.yaml')
    params = _apply_overrides(load_params(str(params_path)), options)

    unknown = [name for name in require(params, 'visualization.overlays')
               if name not in OVERLAY_BUILDERS]
    if unknown:
        print(f"Unknown overlay(s) {', '.join(unknown)}.\n"
              f"{list_overlays()}", file=sys.stderr)
        return 2

    # --- the atlas, and its mirror image if both nuclei are wanted ----
    # Through the builder's own helpers, so the picture is of exactly
    # the nuclei a run with this config would simulate.
    atlas = build_atlas(params, options.atlas_dir, options.synthetic)
    atlases = {name: build_hemisphere_atlas(params, atlas, name)
               for name in configured_hemispheres(params)}

    overlays = list(require(params, 'visualization.overlays'))
    needs_jacobian = bool(set(overlays) & JACOBIAN_OVERLAYS) \
        or options.electrodes == 'array'
    provide = (jacobian_provider(params, atlas, options.atlas_dir,
                                 options.recompute_jacobian)
               if needs_jacobian else None)

    rng = np.random.default_rng(require(params, 'run.seed'))
    electrodes = collect_electrodes(params, atlases, options, rng, provide)

    if (electrodes is not None and len(electrodes)
            and float(require(params, 'visualization.surface.opacity')) >= 1.0
            and not options.clip):
        print("Note: electrodes sit inside the nucleus, and an opaque "
              "surface hides them."
              "\n      Add --surface-opacity 0.35 to see through it, or "
              "--clip AXIS:LOW:HIGH to cut it open.")

    scene = build_scene(
        params, atlases, provide, overlays,
        mask_options=dict(cell_class=options.cell_class,
                          layers=options.layers,
                          min_eccentricity_deg=options.ecc_min,
                          max_eccentricity_deg=options.ecc_max,
                          clips=options.clip,
                          include_ipsi_placeholder=(True if
                                                    options.include_ipsi
                                                    else None)),
        electrodes=electrodes, banner=atlas_banner(atlas))
    print(scene.describe())

    backend = require(params, 'visualization.backend')
    directory = Path(require(params, 'visualization.output.directory'))
    if not directory.is_absolute():
        directory = repository / directory

    if backend == 'plotly':
        from importlib.util import find_spec
        if find_spec('plotly') is None:
            print("The plotly backend needs plotly: `pip install plotly`, "
                  "or use --backend matplotlib, which needs nothing "
                  "further.", file=sys.stderr)
            return 3
        out = Path(options.out or
                   directory / require(params,
                                       'visualization.output.html_name'))
        render_plotly(scene, out)
        print(f"\nWrote {out} ({out.stat().st_size / 1e6:.1f} MB).")
    elif backend == 'matplotlib':
        out = Path(options.out or
                   directory / require(params,
                                       'visualization.output.png_name'))
        render_matplotlib(scene, out, show=options.show)
        print(f"\nWrote {out}.")
    else:
        print(f"Unknown backend {backend!r}; expected plotly or "
              f"matplotlib.", file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':       # pragma: no cover
    raise SystemExit(main())
