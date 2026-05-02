"""Slicing, colormap, and rendering utilities. Mirror MATLAB sliceS / makeCL."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlotParams:
    """Axis metadata for a 2D slice. Mirror of MATLAB plotParams struct.

    `horz_axis_name`, `vert_axis_name`, `slice_axis_name` are 'x', 'y', or 'z' —
    used by `render_slice` to project optode coordinates onto the plane and
    decide which optodes intersect the slice. `slice_dr` is the voxel size,
    used as the in-plane tolerance for optode visibility.
    """

    horz_axis: np.ndarray
    horz_label: str
    horz_axis_name: str
    vert_axis: np.ndarray
    vert_label: str
    vert_axis_name: str
    slice_axis_name: str
    slice_value: float
    slice_dr: float


def slice_s(S: np.ndarray, params, axis: str, value: float):
    """Slice a 3D S array for 2D plotting. Port of sliceS.m.

    Parameters
    ----------
    S     : ndarray, shape (Nx, Ny, Nz)
    params: GridParams with axes .x, .y, .z
    axis  : "x", "y", or "z" — the axis to hold fixed
    value : float — value along `axis` to slice at (snaps to nearest voxel)

    Returns
    -------
    S_plane   : 2D ndarray — oriented for imshow (vert axis × horz axis)
    plot_params : PlotParams
    """
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be one of 'x', 'y', 'z'; got {axis!r}")

    axis_vec = getattr(params, axis)
    idx = int(np.argmin(np.abs(axis_vec - value)))
    snapped_value = float(axis_vec[idx])

    # Infer voxel size from any non-degenerate axis. The slice axis itself may
    # be a single point (e.g. yl=[0,0]); fall back to the first axis with >=2.
    dr = next(
        (float(v[1] - v[0]) for v in (params.x, params.y, params.z) if v.size >= 2),
        1.0,
    )

    if axis == "x":
        plane = S[idx, :, :].T
        pp = PlotParams(
            horz_axis=params.y, horz_label="$y$ (mm)", horz_axis_name="y",
            vert_axis=params.z, vert_label="$z$ (mm)", vert_axis_name="z",
            slice_axis_name="x", slice_value=snapped_value, slice_dr=dr,
        )
    elif axis == "y":
        plane = S[:, idx, :].T
        pp = PlotParams(
            horz_axis=params.x, horz_label="$x$ (mm)", horz_axis_name="x",
            vert_axis=params.z, vert_label="$z$ (mm)", vert_axis_name="z",
            slice_axis_name="y", slice_value=snapped_value, slice_dr=dr,
        )
    else:  # z
        plane = S[:, :, idx].T
        pp = PlotParams(
            horz_axis=params.x, horz_label="$x$ (mm)", horz_axis_name="x",
            vert_axis=params.y, vert_label="$y$ (mm)", vert_axis_name="y",
            slice_axis_name="z", slice_value=snapped_value, slice_dr=dr,
        )
    return plane, pp


import matplotlib.pyplot as _plt


def make_color_limits(x, quantiles=(0.05, 0.95)):
    """Quantile-based color limits and saturated-ends jet colormap. Port of makeCL.m.

    Parameters
    ----------
    x         : array-like — values whose quantiles set the color limits
    quantiles : (lo, hi) floats in [0, 1]

    Returns
    -------
    clim : (vmin, vmax) tuple of floats
    cmap : ndarray of shape (100, 4) — jet with row 0 black and row 99 white
    """
    arr = np.asarray(x).ravel()
    clim = (float(np.quantile(arr, quantiles[0])), float(np.quantile(arr, quantiles[1])))
    jet = _plt.get_cmap("jet")
    cmap = jet(np.linspace(0, 1, 100))
    cmap[0, :3] = 0.0     # first row black
    cmap[-1, :3] = 1.0    # last row white
    return clim, cmap


from matplotlib.colors import ListedColormap


def render_slice(ax, S_plane, plot_params, clim, cmap, rs=None, rd=None,
                 pert=(1.0, 1.0, 1.0), contour_alpha=0.5, *, colorbar=True,
                 set_title=True):
    """Draw a 2D slice with dashed contour overlay on a matplotlib Axes.

    Port of the imagesc + contour pattern from MATLAB example1_DT.m.

    Parameters
    ----------
    ax           : matplotlib.axes.Axes
    S_plane      : 2D ndarray (vert × horz) from slice_s
    plot_params  : PlotParams from slice_s
    clim         : (vmin, vmax)
    cmap         : (N, 4) RGBA array OR a matplotlib Colormap
    rs           : (N_s, 3) source coordinates [mm] (optional)
    rd           : (N_d, 3) detector coordinates [mm] (optional)
    pert         : (px, py, pz) perturbation box size [mm]
    contour_alpha: float in [0, 1] for the overlay gray level

    Returns
    -------
    dict with keys "image", "contour", "colorbar", "sources", "detectors" — the artists.
    """
    ax.clear()

    cmap_obj = cmap if hasattr(cmap, "__call__") else ListedColormap(cmap)

    # When an axis has only one voxel, fall back to a half-step pad so imshow
    # has finite extent on that axis.
    def _padded_extent(vec, fallback=0.5):
        if vec.size == 1:
            return float(vec[0]) - fallback, float(vec[0]) + fallback
        return float(vec[0]), float(vec[-1])

    h0, h1 = _padded_extent(plot_params.horz_axis)
    v0, v1 = _padded_extent(plot_params.vert_axis)
    extent = (h0, h1, v0, v1)
    image = ax.imshow(
        S_plane,
        extent=extent, origin="lower", aspect="auto",
        vmin=clim[0], vmax=clim[1], cmap=cmap_obj,
        interpolation="nearest",
    )
    fig = ax.figure
    cbar = None
    if colorbar:
        cbar = fig.colorbar(image, ax=ax)
        p_str = f"{pert[0]:g} x {pert[1]:g} x {pert[2]:g}"
        cbar.set_label(
            rf"$\mathcal{{S}}$ to a ( {p_str} ) mm$^3$ absorption perturbation"
        )

    if set_title:
        ax.set_title(r"$\mathcal{S} = \partial \mu_{a,meas} / \partial \mu_{a,pert}$" "\n"
                     r"fractional measurement sensitivity to absorption perturbations",
                     fontsize=10)

    # Contour overlay. Use the colorbar's tick values when available;
    # otherwise pick 8 levels evenly across `clim`.
    contour = None
    if S_plane.shape[0] >= 2 and S_plane.shape[1] >= 2:
        if cbar is not None:
            levels = [v for v in cbar.get_ticks() if clim[0] <= v <= clim[1]]
        else:
            levels = list(np.linspace(clim[0], clim[1], 8))
        if levels:
            contour = ax.contour(
                plot_params.horz_axis, plot_params.vert_axis, S_plane,
                levels=levels, linestyles="--",
                colors=[(contour_alpha, contour_alpha, contour_alpha)],
                linewidths=0.8,
            )

    _AXIS_IDX = {"x": 0, "y": 1, "z": 2}
    h_idx = _AXIS_IDX[plot_params.horz_axis_name]
    v_idx = _AXIS_IDX[plot_params.vert_axis_name]
    s_idx = _AXIS_IDX[plot_params.slice_axis_name]
    half_dr = plot_params.slice_dr / 2.0

    def _in_plane(coords: np.ndarray) -> np.ndarray:
        return np.abs(coords[:, s_idx] - plot_params.slice_value) <= half_dr

    sources_artist = None
    if rs is not None:
        rs = np.atleast_2d(rs)
        mask = _in_plane(rs)
        if mask.any():
            sources_artist = ax.scatter(
                rs[mask, h_idx], rs[mask, v_idx],
                marker="v", color="red", s=50, edgecolors="white", zorder=10,
            )

    detectors_artist = None
    if rd is not None:
        rd = np.atleast_2d(rd)
        mask = _in_plane(rd)
        if mask.any():
            detectors_artist = ax.scatter(
                rd[mask, h_idx], rd[mask, v_idx],
                marker="^", color="blue", s=50, edgecolors="white", zorder=10,
            )

    ax.set_xlabel(plot_params.horz_label)
    ax.set_ylabel(plot_params.vert_label)
    # When the vertical axis is depth (z), put z=0 at the top — NIRS convention.
    if plot_params.vert_axis_name == "z":
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    return {
        "image": image, "contour": contour, "colorbar": cbar,
        "sources": sources_artist, "detectors": detectors_artist
    }


def render_three_view(fig, S, params, slice_xyz, *,
                       rs=None, rd=None, pert=(1.0, 1.0, 1.0),
                       quantiles=(0.05, 0.95)):
    """Build a 2x2 third-angle projection on the given Figure.

    Layout (matches paper Figs. 28/45):
        [ x-y plane (z = sz) ] [ 3D context           ]
        [ x-z plane (y = sy) ] [ y-z plane (x = sx)   ]

    All four panels share one colormap and color limits computed once
    from `S`. The 3D context shows the three slice planes as colored
    quads embedded in 3D space.

    Parameters
    ----------
    fig         : matplotlib.figure.Figure (cleared and rebuilt).
    S           : (Nx, Ny, Nz) sensitivity volume.
    params      : GridParams.
    slice_xyz   : (sx, sy, sz) slice values [mm].
    rs, rd, pert: forwarded to per-panel render_slice.

    Returns
    -------
    dict[str, Axes] with keys "xy", "3d", "xz", "yz".
    """
    fig.clear()
    sx, sy, sz = (float(v) for v in slice_xyz)

    clim, cmap = make_color_limits(S, quantiles=quantiles)
    cmap_obj = ListedColormap(cmap)

    # Build 2x2 grid; cell (0,1) is 3D.
    ax_xy = fig.add_subplot(2, 2, 1)
    ax_3d = fig.add_subplot(2, 2, 2, projection="3d")
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    # 2D panels — share the precomputed clim/cmap; suppress per-panel colorbars.
    plane_xy, pp_xy = slice_s(S, params, "z", sz)
    render_slice(ax_xy, plane_xy, pp_xy, clim, cmap, rs=rs, rd=rd,
                 pert=pert, colorbar=False, set_title=False)
    ax_xy.set_title(f"$z = {pp_xy.slice_value:g}$ mm")

    plane_xz, pp_xz = slice_s(S, params, "y", sy)
    render_slice(ax_xz, plane_xz, pp_xz, clim, cmap, rs=rs, rd=rd,
                 pert=pert, colorbar=False, set_title=False)
    ax_xz.set_title(f"$y = {pp_xz.slice_value:g}$ mm")

    plane_yz, pp_yz = slice_s(S, params, "x", sx)
    render_slice(ax_yz, plane_yz, pp_yz, clim, cmap, rs=rs, rd=rd,
                 pert=pert, colorbar=False, set_title=False)
    ax_yz.set_title(f"$x = {pp_yz.slice_value:g}$ mm")

    # 3D context — render each slice plane as a colored quad in 3D.
    _render_3d_slice_planes(ax_3d, S, params, (sx, sy, sz), clim, cmap_obj,
                             rs=rs, rd=rd)

    # Shared colorbar — anchored to the 3D axes for spatial balance.
    image_xy = ax_xy.images[0]
    cbar = fig.colorbar(image_xy, ax=[ax_xy, ax_3d, ax_xz, ax_yz], shrink=0.85)
    p_str = f"{pert[0]:g} x {pert[1]:g} x {pert[2]:g}"
    cbar.set_label(
        rf"$\mathcal{{S}}$ to a ( {p_str} ) mm$^3$ absorption perturbation"
    )

    fig.suptitle(
        r"$\mathcal{S} = \partial \mu_{a,meas} / \partial \mu_{a,pert}$ — "
        r"fractional measurement sensitivity (third-angle projection)",
        fontsize=10,
    )

    return {"xy": ax_xy, "3d": ax_3d, "xz": ax_xz, "yz": ax_yz, "colorbar": cbar}


def _render_3d_slice_planes(ax3d, S, params, slice_xyz, clim, cmap_obj,
                             rs=None, rd=None):
    """Render three slice planes embedded in 3D space on `ax3d`."""
    sx, sy, sz = slice_xyz
    x = np.asarray(params.x); y = np.asarray(params.y); z = np.asarray(params.z)

    norm = _plt.Normalize(vmin=clim[0], vmax=clim[1])

    def _plane_facecolors(plane_2d):
        # plane_2d: (M, N) sampled values at the panel grid.
        return cmap_obj(norm(plane_2d))

    # x-y plane at z = sz
    iz = int(np.argmin(np.abs(z - sz)))
    XX, YY = np.meshgrid(x, y, indexing="ij")
    ZZ = np.full_like(XX, float(z[iz]))
    fc = _plane_facecolors(S[:, :, iz])
    if XX.shape[0] >= 2 and XX.shape[1] >= 2:
        ax3d.plot_surface(XX, YY, ZZ, facecolors=fc, shade=False,
                           rstride=1, cstride=1, edgecolor="none",
                           antialiased=False)

    # x-z plane at y = sy
    iy = int(np.argmin(np.abs(y - sy)))
    XX, ZZ = np.meshgrid(x, z, indexing="ij")
    YY = np.full_like(XX, float(y[iy]))
    fc = _plane_facecolors(S[:, iy, :])
    if XX.shape[0] >= 2 and XX.shape[1] >= 2:
        ax3d.plot_surface(XX, YY, ZZ, facecolors=fc, shade=False,
                           rstride=1, cstride=1, edgecolor="none",
                           antialiased=False)

    # y-z plane at x = sx
    ix = int(np.argmin(np.abs(x - sx)))
    YY, ZZ = np.meshgrid(y, z, indexing="ij")
    XX = np.full_like(YY, float(x[ix]))
    fc = _plane_facecolors(S[ix, :, :])
    if YY.shape[0] >= 2 and YY.shape[1] >= 2:
        ax3d.plot_surface(XX, YY, ZZ, facecolors=fc, shade=False,
                           rstride=1, cstride=1, edgecolor="none",
                           antialiased=False)

    # Optodes
    if rs is not None:
        rs = np.atleast_2d(rs)
        ax3d.scatter(rs[:, 0], rs[:, 1], rs[:, 2],
                     marker="v", color="red", s=50, edgecolors="white",
                     depthshade=False)
    if rd is not None:
        rd = np.atleast_2d(rd)
        ax3d.scatter(rd[:, 0], rd[:, 1], rd[:, 2],
                     marker="^", color="blue", s=50, edgecolors="white",
                     depthshade=False)

    ax3d.set_xlabel("$x$ (mm)"); ax3d.set_ylabel("$y$ (mm)"); ax3d.set_zlabel("$z$ (mm)")
    ax3d.invert_zaxis()  # NIRS convention: depth grows downward
