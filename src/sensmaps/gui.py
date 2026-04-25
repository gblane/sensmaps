"""Tkinter GUI — MainWindow, ParameterPanel, PlotCanvas.

This module is built up incrementally across Tasks 14–16. PlotCanvas lands
first (it has no Tk-widget dependencies of its own beyond a Frame). The
ParameterPanel and MainWindow arrive in the following tasks.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sensmaps.views import make_color_limits, render_slice, slice_s


class PlotCanvas:
    """Matplotlib canvas embedded in a Tk Frame, showing a single 2D slice."""

    def __init__(self, master: tk.Misc):
        self._frame = ttk.Frame(master)
        self._figure = Figure(figsize=(7, 5), dpi=100)
        self._ax = self._figure.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._figure, master=self._frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    @property
    def widget(self) -> ttk.Frame:
        return self._frame

    def show(self, *, S, params, axis: str, value: float,
             quantiles: tuple[float, float] = (0.05, 0.95)) -> None:
        """Render a slice of S at (axis, value) with given color quantiles."""
        plane, pp = slice_s(S, params, axis=axis, value=value)
        clim, cmap = make_color_limits(S, quantiles=quantiles)
        # Remove any old colorbar to avoid stacking
        self._figure.clf()
        self._ax = self._figure.add_subplot(111)
        render_slice(self._ax, plane, pp, clim, cmap)
        self._canvas.draw()

    def clear(self) -> None:
        self._figure.clf()
        self._ax = self._figure.add_subplot(111)
        self._canvas.draw()
