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


def render_slice(ax, S_plane, plot_params, clim, cmap, contour_alpha=0.5):
    """Draw a 2D slice with dashed contour overlay on a matplotlib Axes.

    Port of the imagesc + contour pattern from MATLAB example1_DT.m.

    Parameters
    ----------
    ax           : matplotlib.axes.Axes
    S_plane      : 2D ndarray (vert × horz) from slice_s
    plot_params  : PlotParams from slice_s
    clim         : (vmin, vmax)
    cmap         : (N, 4) RGBA array OR a matplotlib Colormap
    contour_alpha: float in [0, 1] for the overlay gray level

    Returns
    -------
    dict with keys "image", "contour", "colorbar" — the matplotlib artists.
    """
    ax.clear()

    cmap_obj = cmap if hasattr(cmap, "__call__") else ListedColormap(cmap)

    extent = (
        plot_params.horz_axis[0], plot_params.horz_axis[-1],
        plot_params.vert_axis[0], plot_params.vert_axis[-1],
    )
    image = ax.imshow(
        S_plane,
        extent=extent, origin="lower", aspect="equal",
        vmin=clim[0], vmax=clim[1], cmap=cmap_obj,
        interpolation="nearest",
    )
    fig = ax.figure
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$\mathcal{S}$")

    # Contour overlay at colorbar tick values
    levels = colorbar.get_ticks()
    contour = ax.contour(
        plot_params.horz_axis, plot_params.vert_axis, S_plane,
        levels=levels, linestyles="--",
        colors=[(contour_alpha, contour_alpha, contour_alpha)],
        linewidths=0.8,
    )

    ax.set_xlabel(plot_params.horz_label)
    ax.set_ylabel(plot_params.vert_label)
    ax.set_aspect("equal", adjustable="box")

    return {"image": image, "contour": contour, "colorbar": colorbar}
