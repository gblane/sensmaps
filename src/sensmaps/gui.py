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


from typing import Any, Callable


# Map parameter name → "cheap" or "expensive"
PARAM_CLASS: dict[str, str] = {
    "type_str": "expensive",
    "rs": "expensive",
    "rd": "expensive",
    "opt_prop": "expensive",
    "xl": "expensive",
    "yl": "expensive",
    "zl": "expensive",
    "dr": "expensive",
    "pert": "cheap",
    "slice_axis": "cheap",
    "slice_value": "cheap",
    "quantiles": "cheap",
}


_DEFAULTS: dict[str, Any] = {
    "type_str": "CW_SD_I",
    "rs": [0.0, 0.0, 0.0],
    "rd": [35.0, 0.0, 0.0],
    "opt_prop": {
        "n_in": 1.333, "n_out": 1.0, "musp": 1.1, "mua": 0.011,
    },
    "xl": [-10.0, 70.0],
    "yl": [0.0, 0.0],
    "zl": [0.0, 25.0],
    "dr": 1.0,
    "pert": [1.0, 1.0, 1.0],
    "slice_axis": "y",
    "slice_value": 0.0,
    "quantiles": [0.05, 0.95],
}


def _parse_float_list(text: str, n: int) -> list[float]:
    """Parse a comma- or whitespace-separated string of n floats."""
    parts = [p for p in text.replace(",", " ").split() if p.strip()]
    if len(parts) != n:
        raise ValueError(f"expected {n} values, got {len(parts)}: {text!r}")
    return [float(p) for p in parts]


class ParameterPanel:
    """Form panel with entries for every input parameter.

    Exposes get_values() → dict, set_values(dict), and subscribe(callback).
    Callback is invoked with (param_name, new_value) whenever a field changes.
    """

    def __init__(self, master: tk.Misc):
        self._frame = ttk.Frame(master)
        self._subscribers: list[Callable[[str, Any], None]] = []

        # Backing variables
        self._vars: dict[str, tk.Variable] = {}
        self._build_widgets()
        self.set_values(_DEFAULTS)

    @property
    def widget(self) -> ttk.Frame:
        return self._frame

    def subscribe(self, callback: Callable[[str, Any], None]) -> None:
        self._subscribers.append(callback)

    def _notify(self, name: str, value: Any) -> None:
        for cb in self._subscribers:
            cb(name, value)

    def _build_widgets(self) -> None:
        f = self._frame
        row = 0

        # Type
        ttk.Label(f, text="Type").grid(row=row, column=0, sticky="w")
        self._vars["type_str"] = tk.StringVar(value="CW_SD_I")
        ttk.Combobox(f, textvariable=self._vars["type_str"],
                     values=["CW_SD_I"], state="readonly", width=12
                     ).grid(row=row, column=1, sticky="ew"); row += 1

        # Optodes
        ttk.Label(f, text="rs [x y z]").grid(row=row, column=0, sticky="w")
        self._vars["rs"] = tk.StringVar()
        ttk.Entry(f, textvariable=self._vars["rs"], width=20
                  ).grid(row=row, column=1, sticky="ew"); row += 1

        ttk.Label(f, text="rd [x y z]").grid(row=row, column=0, sticky="w")
        self._vars["rd"] = tk.StringVar()
        ttk.Entry(f, textvariable=self._vars["rd"], width=20
                  ).grid(row=row, column=1, sticky="ew"); row += 1

        # Optical properties
        ttk.Label(f, text="Optical: n_in n_out musp mua").grid(
            row=row, column=0, columnspan=2, sticky="w"); row += 1
        for key in ("n_in", "n_out", "musp", "mua"):
            ttk.Label(f, text=f"  {key}").grid(row=row, column=0, sticky="w")
            self._vars[f"opt_prop.{key}"] = tk.StringVar()
            ttk.Entry(f, textvariable=self._vars[f"opt_prop.{key}"], width=10
                      ).grid(row=row, column=1, sticky="w"); row += 1

        # Grid limits
        for key in ("xl", "yl", "zl"):
            ttk.Label(f, text=f"{key} [min max]").grid(row=row, column=0, sticky="w")
            self._vars[key] = tk.StringVar()
            ttk.Entry(f, textvariable=self._vars[key], width=20
                      ).grid(row=row, column=1, sticky="ew"); row += 1

        ttk.Label(f, text="dr (mm)").grid(row=row, column=0, sticky="w")
        self._vars["dr"] = tk.StringVar()
        ttk.Entry(f, textvariable=self._vars["dr"], width=8
                  ).grid(row=row, column=1, sticky="w"); row += 1

        # Perturbation
        ttk.Label(f, text="pert [x y z]").grid(row=row, column=0, sticky="w")
        self._vars["pert"] = tk.StringVar()
        ttk.Entry(f, textvariable=self._vars["pert"], width=20
                  ).grid(row=row, column=1, sticky="ew"); row += 1

        # Slice
        ttk.Label(f, text="slice axis").grid(row=row, column=0, sticky="w")
        self._vars["slice_axis"] = tk.StringVar(value="y")
        ttk.Combobox(f, textvariable=self._vars["slice_axis"],
                     values=["x", "y", "z"], state="readonly", width=4
                     ).grid(row=row, column=1, sticky="w"); row += 1

        ttk.Label(f, text="slice value").grid(row=row, column=0, sticky="w")
        self._vars["slice_value"] = tk.StringVar()
        ttk.Entry(f, textvariable=self._vars["slice_value"], width=10
                  ).grid(row=row, column=1, sticky="w"); row += 1

        # Color quantiles
        ttk.Label(f, text="quantiles [lo hi]").grid(row=row, column=0, sticky="w")
        self._vars["quantiles"] = tk.StringVar()
        ttk.Entry(f, textvariable=self._vars["quantiles"], width=14
                  ).grid(row=row, column=1, sticky="ew"); row += 1

        # Wire change callbacks
        for name, var in self._vars.items():
            var.trace_add("write", lambda *_a, n=name: self._on_var_changed(n))

    def _on_var_changed(self, name: str) -> None:
        try:
            values = self.get_values()
        except Exception:
            return  # swallow transient parse errors during typing
        if name.startswith("opt_prop."):
            self._notify("opt_prop", values["opt_prop"])
        else:
            self._notify(name, values[name])

    def get_values(self) -> dict[str, Any]:
        v = self._vars
        return {
            "type_str": v["type_str"].get(),
            "rs": _parse_float_list(v["rs"].get(), 3),
            "rd": _parse_float_list(v["rd"].get(), 3),
            "opt_prop": {
                "n_in":  float(v["opt_prop.n_in"].get()),
                "n_out": float(v["opt_prop.n_out"].get()),
                "musp":  float(v["opt_prop.musp"].get()),
                "mua":   float(v["opt_prop.mua"].get()),
            },
            "xl": _parse_float_list(v["xl"].get(), 2),
            "yl": _parse_float_list(v["yl"].get(), 2),
            "zl": _parse_float_list(v["zl"].get(), 2),
            "dr": float(v["dr"].get()),
            "pert": _parse_float_list(v["pert"].get(), 3),
            "slice_axis": v["slice_axis"].get(),
            "slice_value": float(v["slice_value"].get()),
            "quantiles": _parse_float_list(v["quantiles"].get(), 2),
        }

    def set_values(self, values: dict[str, Any]) -> None:
        def _fmt_list(xs): return " ".join(f"{x:g}" for x in xs)

        self._vars["type_str"].set(values["type_str"])
        self._vars["rs"].set(_fmt_list(values["rs"]))
        self._vars["rd"].set(_fmt_list(values["rd"]))
        op = values["opt_prop"]
        for k in ("n_in", "n_out", "musp", "mua"):
            self._vars[f"opt_prop.{k}"].set(f"{op[k]:g}")
        self._vars["xl"].set(_fmt_list(values["xl"]))
        self._vars["yl"].set(_fmt_list(values["yl"]))
        self._vars["zl"].set(_fmt_list(values["zl"]))
        self._vars["dr"].set(f"{values['dr']:g}")
        self._vars["pert"].set(_fmt_list(values["pert"]))
        self._vars["slice_axis"].set(values["slice_axis"])
        self._vars["slice_value"].set(f"{values['slice_value']:g}")
        self._vars["quantiles"].set(_fmt_list(values["quantiles"]))
