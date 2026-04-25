"""Slicing, colormap, and rendering utilities. Mirror MATLAB sliceS / makeCL."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlotParams:
    """Axis metadata for a 2D slice. Mirror of MATLAB plotParams struct."""

    horz_axis: np.ndarray
    horz_label: str
    vert_axis: np.ndarray
    vert_label: str


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

    if axis == "x":
        plane = S[idx, :, :].T
        pp = PlotParams(
            horz_axis=params.y, horz_label="$y$ (mm)",
            vert_axis=params.z, vert_label="$z$ (mm)",
        )
    elif axis == "y":
        plane = S[:, idx, :].T
        pp = PlotParams(
            horz_axis=params.x, horz_label="$x$ (mm)",
            vert_axis=params.z, vert_label="$z$ (mm)",
        )
    else:  # z
        plane = S[:, :, idx].T
        pp = PlotParams(
            horz_axis=params.x, horz_label="$x$ (mm)",
            vert_axis=params.y, vert_label="$y$ (mm)",
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


def render_slice(ax, S_plane, plot_params, clim, cmap, rs=None, rd=None, pert=(1.0, 1.0, 1.0), contour_alpha=0.5):
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
    colorbar = fig.colorbar(image, ax=ax)
    
    # Dynamic colorbar label
    p_str = f"{pert[0]:g} x {pert[1]:g} x {pert[2]:g}"
    colorbar.set_label(f"S to a ( {p_str} ) $mm^3$ absorption perturbation")

    # Title with LaTeX and new line
    ax.set_title(r"$\mathcal{S} = \partial \mu_{a,meas} / \partial \mu_{a,pert}$" "\n"
                 r"fractional measurement sensitivity to absorption perturbations",
                 fontsize=10)

    # Contour overlay at colorbar tick values. Skip when either axis is
    # degenerate — matplotlib.contour requires a (>=2, >=2) array.
    contour = None
    if S_plane.shape[0] >= 2 and S_plane.shape[1] >= 2:
        levels = colorbar.get_ticks()
        # Filter levels to be within clim to avoid warnings
        levels = [v for v in levels if clim[0] <= v <= clim[1]]
        if levels:
            contour = ax.contour(
                plot_params.horz_axis, plot_params.vert_axis, S_plane,
                levels=levels, linestyles="--",
                colors=[(contour_alpha, contour_alpha, contour_alpha)],
                linewidths=0.8,
            )

    # Plot optodes if provided
    # Identify labels to find coordinates
    h_lab = plot_params.horz_label.lower()
    v_lab = plot_params.vert_label.lower()
    
    def get_indices(lab):
        if 'x' in lab: return 0
        if 'y' in lab: return 1
        if 'z' in lab: return 2
        return None

    h_idx = get_indices(h_lab)
    v_idx = get_indices(v_lab)
    
    sources_artist = None
    if rs is not None:
        rs = np.atleast_2d(rs)
        sources_artist = ax.scatter(
            rs[:, h_idx], rs[:, v_idx],
            marker='v', color='red', s=50, label='Sources', edgecolors='white',
            zorder=10
        )

    detectors_artist = None
    if rd is not None:
        rd = np.atleast_2d(rd)
        detectors_artist = ax.scatter(
            rd[:, h_idx], rd[:, v_idx],
            marker='^', color='blue', s=50, label='Detectors', edgecolors='white',
            zorder=10
        )

    ax.set_xlabel(plot_params.horz_label)
    ax.set_ylabel(plot_params.vert_label)
    # When the vertical axis is depth (z), put z=0 at the top — NIRS convention.
    if "$z$" in plot_params.vert_label:
        ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    return {
        "image": image, "contour": contour, "colorbar": colorbar,
        "sources": sources_artist, "detectors": detectors_artist
    }
