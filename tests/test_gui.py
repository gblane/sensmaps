"""GUI smoke tests.

These tests construct Tkinter widgets with a hidden root and exercise a
minimum of behavior; they intentionally do not assert anything interactive.
"""
import os
import sys

import numpy as np
import pytest


def _require_display():
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        pytest.skip("Tkinter tests require a DISPLAY on Linux")


@pytest.fixture
def tk_root():
    _require_display()
    import tkinter as tk
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"Tk failed to initialize: {e}")
    root.withdraw()
    yield root
    root.destroy()


def test_plot_canvas_constructs_and_shows_slice(tk_root, cw_sd_i_ref):
    from sensmaps.compute import GridParams
    from sensmaps.gui import PlotCanvas
    x = np.asarray(cw_sd_i_ref["x"], dtype=float)
    y = np.asarray(cw_sd_i_ref["y"], dtype=float)
    z = np.asarray(cw_sd_i_ref["z"], dtype=float)
    S = np.asarray(cw_sd_i_ref["S"], dtype=float)
    params = GridParams(
        x=x, y=y, z=z,
        xl_valid=(float(x[0]), float(x[-1])),
        yl_valid=(float(y[0]), float(y[-1])),
        zl_valid=(float(z[0]), float(z[-1])),
    )
    canvas = PlotCanvas(master=tk_root)
    canvas.show(S=S, params=params, axis="y", value=0.0, quantiles=(0.05, 0.95))
    # Widget is packable
    assert canvas.widget is not None
