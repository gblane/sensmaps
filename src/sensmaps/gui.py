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
             quantiles: tuple[float, float] = (0.05, 0.95),
             rs=None, rd=None, pert=(1.0, 1.0, 1.0)) -> None:
        """Render a slice of S at (axis, value) with given color quantiles."""
        plane, pp = slice_s(S, params, axis=axis, value=value)
        clim, cmap = make_color_limits(S, quantiles=quantiles)
        # Remove any old colorbar to avoid stacking
        self._figure.clf()
        self._ax = self._figure.add_subplot(111)
        render_slice(self._ax, plane, pp, clim, cmap, rs=rs, rd=rd, pert=pert)
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
    "fmod": "expensive",                  # NEW
    "tg": "expensive",                    # NEW (v1.2)
    "tend": "expensive",                  # NEW (v1.2)
    "ndt": "expensive",                   # NEW (v1.2)
    "td_override": "expensive",           # NEW (v1.2)
    "pert": "cheap",
    "pert_override": "cheap",
    "slice_axis": "cheap",
    "slice_value": "cheap",
    "quantiles": "cheap",
}


_DEFAULTS: dict[str, Any] = {
    "type_str": "CW_SD_I",
    "rs": [[0.0, 0.0, 0.0]],
    "rd": [[35.0, 0.0, 0.0]],
    "opt_prop": {
        "n_in": 1.333, "n_out": 1.0, "musp": 1.1, "mua": 0.011,
    },
    "xl": [-10.0, 70.0],
    "yl": [0.0, 0.0],
    "zl": [0.0, 25.0],
    "dr": 1.0,
    "fmod": 100.0,                  # NEW — MHz
    "tg": [[1.0, 2.0]],             # NEW (v1.2) — ns; single-row matrix for parser reuse
    "tend": 10.0,                   # NEW (v1.2) — ns
    "ndt": 10000,                   # NEW (v1.2)
    "td_override": False,           # NEW (v1.2)
    "pert": [1.0, 1.0, 1.0],
    "pert_override": False,
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


def _parse_float_matrix(text: str, ncols: int) -> list[list[float]]:
    """Parse a `;`-separated list of `ncols`-wide float rows.

    Each row is then whitespace/comma-separated. Trailing `;` is ignored.
    Empty input (or only `;`) raises ValueError.
    """
    rows = [r for r in text.split(";") if r.strip()]
    if not rows:
        raise ValueError(f"expected at least one row of {ncols} values; got {text!r}")
    return [_parse_float_list(r, ncols) for r in rows]


class ParameterPanel:
    """Form panel with entries for every input parameter.

    Exposes get_values() → dict, set_values(dict), and subscribe(callback).
    Callback is invoked with (param_name, new_value) whenever a field changes.
    """

    def __init__(self, master: tk.Misc):
        self._frame = ttk.Frame(master)
        self._subscribers: list[Callable[[str, Any], None]] = []
        self._inputs: list[tk.Widget] = []

        # Backing variables
        self._vars: dict[str, tk.Variable] = {}
        self._build_widgets()
        self.set_values(_DEFAULTS)

    @property
    def widget(self) -> ttk.Frame:
        return self._frame

    def subscribe(self, callback: Callable[[str, Any], None]) -> None:
        self._subscribers.append(callback)

    def bind_return(self, callback: Callable[[tk.Event], Any]) -> None:
        """Bind the Return key to all input widgets."""
        for widget in self._inputs:
            widget.bind("<Return>", callback)

    def _notify(self, name: str, value: Any) -> None:
        for cb in self._subscribers:
            cb(name, value)

    def _build_widgets(self) -> None:
        f = self._frame
        row = 0

        # Type
        _VALID_TYPES = [
            "CW_SD_I", "CW_SS_I", "CW_DS_I",
            "FD_SD_I", "FD_SS_I", "FD_DS_I",
            "FD_SD_P", "FD_SS_P", "FD_DS_P",
            "TD_SD_GI", "TD_SS_GI", "TD_DS_GI",
        ]
        ttk.Label(f, text="Type").grid(row=row, column=0, sticky="w")
        self._vars["type_str"] = tk.StringVar(value="CW_SD_I")
        cb_type = ttk.Combobox(f, textvariable=self._vars["type_str"],
                               values=_VALID_TYPES, state="readonly", width=12)
        cb_type.grid(row=row, column=1, sticky="ew")
        self._inputs.append(cb_type)
        row += 1

        # Optodes
        ttk.Label(f, text="rs [x y z; ...] (mm)").grid(row=row, column=0, sticky="w")
        self._vars["rs"] = tk.StringVar()
        en_rs = ttk.Entry(f, textvariable=self._vars["rs"], width=20)
        en_rs.grid(row=row, column=1, sticky="ew")
        self._inputs.append(en_rs)
        row += 1

        ttk.Label(f, text="rd [x y z; ...] (mm)").grid(row=row, column=0, sticky="w")
        self._vars["rd"] = tk.StringVar()
        en_rd = ttk.Entry(f, textvariable=self._vars["rd"], width=20)
        en_rd.grid(row=row, column=1, sticky="ew")
        self._inputs.append(en_rd)
        row += 1

        # Optical properties
        ttk.Label(f, text="Optical properties").grid(
            row=row, column=0, columnspan=2, sticky="w"); row += 1
        for key, unit in [("n_in", ""), ("n_out", ""), ("musp", " (1/mm)"), ("mua", " (1/mm)")]:
            ttk.Label(f, text=f"  {key}{unit}").grid(row=row, column=0, sticky="w")
            self._vars[f"opt_prop.{key}"] = tk.StringVar()
            en_op = ttk.Entry(f, textvariable=self._vars[f"opt_prop.{key}"], width=10)
            en_op.grid(row=row, column=1, sticky="w")
            self._inputs.append(en_op)
            row += 1

        # Grid limits
        for key in ("xl", "yl", "zl"):
            ttk.Label(f, text=f"{key} [min max] (mm)").grid(row=row, column=0, sticky="w")
            self._vars[key] = tk.StringVar()
            en_lim = ttk.Entry(f, textvariable=self._vars[key], width=20)
            en_lim.grid(row=row, column=1, sticky="ew")
            self._inputs.append(en_lim)
            row += 1

        ttk.Label(f, text="dr (mm)").grid(row=row, column=0, sticky="w")
        self._vars["dr"] = tk.StringVar()
        en_dr = ttk.Entry(f, textvariable=self._vars["dr"], width=8)
        en_dr.grid(row=row, column=1, sticky="w")
        self._inputs.append(en_dr)
        row += 1

        # Modulation frequency (FD only — disabled when type is CW_*)
        ttk.Label(f, text="fmod (MHz)").grid(row=row, column=0, sticky="w")
        self._vars["fmod"] = tk.StringVar()
        self._fmod_entry = ttk.Entry(f, textvariable=self._vars["fmod"], width=10)
        self._fmod_entry.grid(row=row, column=1, sticky="w")
        self._inputs.append(self._fmod_entry)
        row += 1

        # Gate window for TD_*_GI (entered as `start; end` in ns; ps internally)
        ttk.Label(f, text="tg [start; end] (ns)").grid(row=row, column=0, sticky="w")
        self._vars["tg"] = tk.StringVar()
        self._tg_entry = ttk.Entry(f, textvariable=self._vars["tg"], width=14)
        self._tg_entry.grid(row=row, column=1, sticky="ew")
        self._inputs.append(self._tg_entry)
        row += 1

        # TD numerics (tend / ndt) — disabled by default behind override checkbox
        ttk.Label(f, text="tend (ns)").grid(row=row, column=0, sticky="w")
        td_frame = ttk.Frame(f)
        td_frame.grid(row=row, column=1, sticky="ew")

        self._vars["tend"] = tk.StringVar()
        self._tend_entry = ttk.Entry(td_frame, textvariable=self._vars["tend"], width=8)
        self._tend_entry.pack(side=tk.LEFT)
        self._inputs.append(self._tend_entry)

        self._vars["td_override"] = tk.BooleanVar(value=False)
        self._td_override_check = ttk.Checkbutton(
            td_frame, text="Override", variable=self._vars["td_override"]
        )
        self._td_override_check.pack(side=tk.LEFT, padx=(4, 0))
        row += 1

        ttk.Label(f, text="ndt").grid(row=row, column=0, sticky="w")
        self._vars["ndt"] = tk.StringVar()
        self._ndt_entry = ttk.Entry(f, textvariable=self._vars["ndt"], width=10)
        self._ndt_entry.grid(row=row, column=1, sticky="w")
        self._inputs.append(self._ndt_entry)
        row += 1

        # Perturbation
        ttk.Label(f, text="pert [x y z] (mm)").grid(row=row, column=0, sticky="w")
        pert_frame = ttk.Frame(f)
        pert_frame.grid(row=row, column=1, sticky="ew")

        self._vars["pert"] = tk.StringVar()
        self._pert_entry = ttk.Entry(pert_frame, textvariable=self._vars["pert"], width=12)
        self._pert_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._inputs.append(self._pert_entry)

        self._vars["pert_override"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(pert_frame, text="Override", variable=self._vars["pert_override"]
                        ).pack(side=tk.LEFT, padx=(4, 0))
        row += 1

        # Slice
        ttk.Label(f, text="slice axis").grid(row=row, column=0, sticky="w")
        self._vars["slice_axis"] = tk.StringVar(value="y")
        cb_slice = ttk.Combobox(f, textvariable=self._vars["slice_axis"],
                                values=["x", "y", "z"], state="readonly", width=4)
        cb_slice.grid(row=row, column=1, sticky="w")
        self._inputs.append(cb_slice)
        row += 1

        ttk.Label(f, text="slice value (mm)").grid(row=row, column=0, sticky="w")
        self._vars["slice_value"] = tk.StringVar()
        en_val = ttk.Entry(f, textvariable=self._vars["slice_value"], width=10)
        en_val.grid(row=row, column=1, sticky="w")
        self._inputs.append(en_val)
        row += 1

        # Color quantiles
        ttk.Label(f, text="quantiles [lo hi]").grid(row=row, column=0, sticky="w")
        self._vars["quantiles"] = tk.StringVar()
        en_q = ttk.Entry(f, textvariable=self._vars["quantiles"], width=14)
        en_q.grid(row=row, column=1, sticky="ew")
        self._inputs.append(en_q)
        row += 1

        # Wire change callbacks
        for name, var in self._vars.items():
            var.trace_add("write", lambda *_a, n=name: self._on_var_changed(n))

    def _on_var_changed(self, name: str) -> None:
        try:
            values = self.get_values()
        except Exception:
            return  # swallow transient parse errors during typing

        # Handle pert syncing if not overridden
        override = self._vars["pert_override"].get()
        if not override:
            self._pert_entry.config(state="disabled")
            if name == "dr" or name == "pert_override":
                dr = values["dr"]
                self._vars["pert"].set(f"{dr:g} {dr:g} {dr:g}")
                # Refresh values to pick up the newly set pert
                values = self.get_values()
        else:
            self._pert_entry.config(state="normal")

        # Disable fmod entry when the selected type is CW_* (it's unused there).
        if values["type_str"].startswith("FD_"):
            self._fmod_entry.config(state="normal")
        else:
            self._fmod_entry.config(state="disabled")

        # TD-only fields: tg always editable for _GI; tend/ndt gated by checkbox.
        is_td = values["type_str"].startswith("TD_")
        self._tg_entry.config(state="normal" if values["type_str"].endswith("_GI") else "disabled")
        self._td_override_check.config(state="normal" if is_td else "disabled")
        td_edit = is_td and values["td_override"]
        self._tend_entry.config(state="normal" if td_edit else "disabled")
        self._ndt_entry.config(state="normal" if td_edit else "disabled")

        if name.startswith("opt_prop."):
            self._notify("opt_prop", values["opt_prop"])
        else:
            self._notify(name, values[name])

    def get_values(self) -> dict[str, Any]:
        v = self._vars
        return {
            "type_str": v["type_str"].get(),
            "rs": _parse_float_matrix(v["rs"].get(), 3),
            "rd": _parse_float_matrix(v["rd"].get(), 3),
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
            "fmod": float(v["fmod"].get()),
            "tg": _parse_float_matrix(v["tg"].get(), 2),
            "tend": float(v["tend"].get()),
            "ndt": int(float(v["ndt"].get())),
            "td_override": bool(v["td_override"].get()),
            "pert": _parse_float_list(v["pert"].get(), 3),
            "pert_override": bool(v["pert_override"].get()),
            "slice_axis": v["slice_axis"].get(),
            "slice_value": float(v["slice_value"].get()),
            "quantiles": _parse_float_list(v["quantiles"].get(), 2),
        }

    _VALID_KEYS = frozenset(_DEFAULTS.keys())
    _VALID_OPT_PROP_KEYS = frozenset(("n_in", "n_out", "musp", "mua"))

    def set_values(self, values: dict[str, Any]) -> None:
        def _fmt_list(xs):    return " ".join(f"{x:g}" for x in xs)
        def _fmt_matrix(xss): return "; ".join(_fmt_list(xs) for xs in xss)

        unknown = set(values) - self._VALID_KEYS
        if unknown:
            raise ValueError(
                f"Unknown keys in set_values: {sorted(unknown)}; "
                f"expected subset of {sorted(self._VALID_KEYS)}"
            )
        if "opt_prop" in values:
            unknown_op = set(values["opt_prop"]) - self._VALID_OPT_PROP_KEYS
            if unknown_op:
                raise ValueError(
                    f"Unknown opt_prop keys: {sorted(unknown_op)}; "
                    f"expected subset of {sorted(self._VALID_OPT_PROP_KEYS)}"
                )

        if "type_str" in values:
            self._vars["type_str"].set(values["type_str"])
        if "rs" in values:
            self._vars["rs"].set(_fmt_matrix(values["rs"]))
        if "rd" in values:
            self._vars["rd"].set(_fmt_matrix(values["rd"]))
        if "opt_prop" in values:
            op = values["opt_prop"]
            for k in ("n_in", "n_out", "musp", "mua"):
                if k in op:
                    self._vars[f"opt_prop.{k}"].set(f"{op[k]:g}")
        if "xl" in values:
            self._vars["xl"].set(_fmt_list(values["xl"]))
        if "yl" in values:
            self._vars["yl"].set(_fmt_list(values["yl"]))
        if "zl" in values:
            self._vars["zl"].set(_fmt_list(values["zl"]))
        if "dr" in values:
            self._vars["dr"].set(f"{values['dr']:g}")
        if "fmod" in values:
            self._vars["fmod"].set(f"{values['fmod']:g}")
        if "tg" in values:
            self._vars["tg"].set(_fmt_matrix(values["tg"]))
        if "tend" in values:
            self._vars["tend"].set(f"{values['tend']:g}")
        if "ndt" in values:
            self._vars["ndt"].set(f"{int(values['ndt']):d}")
        if "td_override" in values:
            self._vars["td_override"].set(values["td_override"])
        if "pert_override" in values:
            self._vars["pert_override"].set(values["pert_override"])
        if "pert" in values:
            self._vars["pert"].set(_fmt_list(values["pert"]))
        if "slice_axis" in values:
            self._vars["slice_axis"].set(values["slice_axis"])
        if "slice_value" in values:
            self._vars["slice_value"].set(f"{values['slice_value']:g}")
        if "quantiles" in values:
            self._vars["quantiles"].set(_fmt_list(values["quantiles"]))


import json
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np

from dataclasses import replace as _replace

from sensmaps.compute import apply_pert_kernel, make_s_full
from sensmaps.physics import OpticalProperties


SESSION_FILE = Path("last_session.json")


def _opt_prop_from_dict(d: dict) -> OpticalProperties:
    return OpticalProperties(
        n_in=d["n_in"], n_out=d["n_out"], musp=d["musp"], mua=d["mua"],
    )


class MainWindow:
    """Top-level Tk window wiring ParameterPanel, PlotCanvas, and actions."""

    def __init__(self, master: tk.Misc):
        self._root = master
        # Container frame (so test can pass in a withdrawn root)
        self._container = ttk.Frame(master)
        self._container.pack(fill=tk.BOTH, expand=True)

        # Two-pane split: params on left, plot on right
        left = ttk.Frame(self._container)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self._container)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.params_panel = ParameterPanel(master=left)
        self.params_panel.widget.pack(padx=8, pady=8, fill=tk.Y)
        self.params_panel.subscribe(self._on_param_changed)
        self.params_panel.bind_return(lambda _e: self.recalculate())

        self.plot_canvas = PlotCanvas(master=right)
        self.plot_canvas.widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Action bar
        bar = ttk.Frame(master)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._dirty_label = ttk.Label(bar, text="● image updated", foreground="green")
        self._dirty_label.pack(side=tk.LEFT, padx=8)

        self._recalc_btn = ttk.Button(bar, text="Recalculate", command=self.recalculate)
        self._recalc_btn.pack(side=tk.LEFT, padx=4, pady=4)

        self._revert_btn = ttk.Button(bar, text="Revert", command=self.revert)
        self._revert_btn.pack(side=tk.LEFT, padx=4, pady=4)

        ttk.Button(bar, text="Save Figure…", command=self.save_figure
                   ).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(bar, text="Save Data…", command=self.save_data
                   ).pack(side=tk.LEFT, padx=4, pady=4)

        # Style for highlighting
        self.style = ttk.Style()
        # Some themes don't support foreground on TButton easily, so we'll try to use a distinct style
        self.style.configure("Highlighted.TButton", font=("TkDefaultFont", 10, "bold"))

        # State
        self._cache = None   # SensitivityResult or None
        self._last_inputs: dict | None = None
        self._dirty = False

        # Load session and compute on open
        self.load_session()
        self.recalculate()

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def _set_dirty(self, flag: bool) -> None:
        self._dirty = flag
        if flag:
            self._dirty_label.config(text="● click recalculate",
                                     foreground="orange")
            self._recalc_btn.config(style="Highlighted.TButton")
            self._revert_btn.config(style="Highlighted.TButton")
        else:
            self._dirty_label.config(text="● image updated", foreground="green")
            self._recalc_btn.config(style="TButton")
            self._revert_btn.config(style="TButton")

    def _on_param_changed(self, name: str, value) -> None:
        klass = PARAM_CLASS.get(name, "expensive")
        if klass == "expensive":
            if self._cache is not None:
                self._set_dirty(True)
        else:
            if self._cache is not None:
                self._apply_cheap_change(name)

    def _apply_cheap_change(self, name: str) -> None:
        """Re-slice and redraw without recomputing the physics."""
        try:
            values = self.params_panel.get_values()
        except Exception:
            return
        try:
            # Pert is cheap: re-conv Svox with new kernel before slicing.
            if name == "pert" and list(values["pert"]) != list(self._cache.pert):
                new_pert = tuple(values["pert"])
                S_new = apply_pert_kernel(
                    self._cache.Svox, new_pert, self._cache.dr,
                )
                self._cache = _replace(self._cache, S=S_new, pert=new_pert)
            self.plot_canvas.show(
                S=self._cache.S,
                params=self._cache.params,
                axis=values["slice_axis"],
                value=values["slice_value"],
                quantiles=tuple(values["quantiles"]),
                rs=self._cache.rs,
                rd=self._cache.rd,
                pert=self._cache.pert,
            )
        except Exception:
            # A bad render shouldn't kill the live-update path; log to stderr.
            import traceback
            traceback.print_exc()

    def recalculate(self) -> None:
        try:
            values = self.params_panel.get_values()
        except Exception as e:
            messagebox.showerror("Invalid input", f"Could not parse form: {e}")
            return
        try:
            op = _opt_prop_from_dict(values["opt_prop"])
            type_str = values["type_str"]
            fmod_hz = values["fmod"] * 1e6 if type_str.startswith("FD_") else None

            tg_ps = tend_ps = ndt_val = None
            if type_str.startswith("TD_"):
                tg_ps = np.asarray(values["tg"][0], dtype=float) * 1000.0
                if values["td_override"]:
                    tend_ps = float(values["tend"]) * 1000.0
                    ndt_val = int(values["ndt"])

            result = make_s_full(
                type_str=type_str,
                rs=np.asarray(values["rs"], dtype=float),
                rd=np.asarray(values["rd"], dtype=float),
                opt_prop=op,
                xl=tuple(values["xl"]),
                yl=tuple(values["yl"]),
                zl=tuple(values["zl"]),
                dr=values["dr"],
                pert=tuple(values["pert"]),
                fmod=fmod_hz,
                tg=tg_ps, tend=tend_ps, ndt=ndt_val,
            )
        except (ValueError, NotImplementedError) as e:
            messagebox.showerror("Recalculate failed", str(e))
            return
        self._cache = result
        self._last_inputs = values
        self._set_dirty(False)
        self.plot_canvas.show(
            S=result.S, params=result.params,
            axis=values["slice_axis"], value=values["slice_value"],
            quantiles=tuple(values["quantiles"]),
            rs=result.rs, rd=result.rd,
            pert=result.pert,
        )

    def revert(self) -> None:
        if self._last_inputs is None:
            return
        self.params_panel.set_values(self._last_inputs)
        self._set_dirty(False)

    def save_figure(self) -> None:
        # Don't pass defaultextension — Tk on Linux always appends it regardless
        # of the filter the user picked. Honor the extension the user typed; if
        # there isn't one, fall back to .png.
        path = filedialog.asksaveasfilename(
            filetypes=[
                ("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        from pathlib import Path as _Path
        if not _Path(path).suffix:
            path = path + ".png"
        try:
            self.plot_canvas._figure.savefig(path)
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def save_data(self) -> None:
        if self._cache is None:
            messagebox.showinfo("No data", "Run Recalculate first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy archive", "*.npz")],
        )
        if not path:
            return
        c = self._cache
        from sensmaps import __version__ as version
        try:
            np.savez(
                path,
                S=c.S, Svox=c.Svox,
                x=c.params.x, y=c.params.y, z=c.params.z,
                rs=c.rs, rd=c.rd, pert=np.asarray(c.pert), dr=c.dr,
                type_str=c.type_str,
                n_in=c.opt_prop.n_in, n_out=c.opt_prop.n_out,
                musp=c.opt_prop.musp, mua=c.opt_prop.mua,
                fmod=(np.nan if c.fmod is None else c.fmod),    # NEW (v1.1)
                tg=(np.full(2, np.nan) if c.tg is None else c.tg),   # NEW (v1.2) ps
                tend=(np.nan if c.tend is None else c.tend),         # NEW (v1.2) ps
                ndt=(-1 if c.ndt is None else c.ndt),                # NEW (v1.2)
                sensmaps_version=version,
            )
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def save_session(self) -> None:
        try:
            values = self.params_panel.get_values()
        except Exception:
            return
        SESSION_FILE.write_text(json.dumps(values, indent=2))

    def load_session(self) -> None:
        if not SESSION_FILE.exists():
            return
        try:
            values = json.loads(SESSION_FILE.read_text())
            self.params_panel.set_values(values)
        except Exception:
            # Corrupt or schema-incompatible session — leave panel at defaults.
            return
