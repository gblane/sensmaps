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


def test_parameter_panel_reads_and_writes_values(tk_root):
    from sensmaps.gui import ParameterPanel
    panel = ParameterPanel(master=tk_root)

    # Default values should round-trip through get_values
    defaults = panel.get_values()
    assert defaults["type_str"] == "CW_SD_I"
    assert defaults["slice_axis"] in ("x", "y", "z")
    assert isinstance(defaults["rs"], list)
    assert len(defaults["rs"]) == 1  # single row
    assert isinstance(defaults["opt_prop"], dict)
    assert "musp" in defaults["opt_prop"]

    # Round-trip set_values → get_values
    new_vals = dict(defaults)
    new_vals["opt_prop"] = dict(defaults["opt_prop"], musp=1.2, mua=0.02)
    new_vals["rs"] = [[1.0, 2.0, 3.0]]
    panel.set_values(new_vals)
    got = panel.get_values()
    assert got["opt_prop"]["musp"] == 1.2
    assert got["opt_prop"]["mua"] == 0.02
    assert got["rs"] == [[1.0, 2.0, 3.0]]


def test_parameter_panel_pert_syncing(tk_root):
    from sensmaps.gui import ParameterPanel
    panel = ParameterPanel(master=tk_root)

    # By default pert_override is False, so pert should sync with dr
    panel.set_values({"dr": 0.5, "pert_override": False})
    vals = panel.get_values()
    assert vals["pert"] == [0.5, 0.5, 0.5]

    # Change dr, pert should follow
    panel.set_values({"dr": 0.2})
    assert panel.get_values()["pert"] == [0.2, 0.2, 0.2]

    # Enable override
    panel.set_values({"pert_override": True, "pert": [1.0, 1.0, 1.0]})
    assert panel.get_values()["pert"] == [1.0, 1.0, 1.0]

    # Change dr, pert should NOT follow
    panel.set_values({"dr": 0.3})
    assert panel.get_values()["pert"] == [1.0, 1.0, 1.0]

    # Disable override, pert should snap back to dr
    panel.set_values({"pert_override": False})
    assert panel.get_values()["pert"] == [0.3, 0.3, 0.3]


def test_parameter_panel_classifies_cheap_vs_expensive(tk_root):
    from sensmaps.gui import ParameterPanel, PARAM_CLASS
    panel = ParameterPanel(master=tk_root)
    # Sample classifications
    assert PARAM_CLASS["slice_axis"] == "cheap"
    assert PARAM_CLASS["slice_value"] == "cheap"
    assert PARAM_CLASS["quantiles"] == "cheap"
    assert PARAM_CLASS["pert"] == "cheap"
    assert PARAM_CLASS["rs"] == "expensive"
    assert PARAM_CLASS["opt_prop"] == "expensive"
    assert PARAM_CLASS["xl"] == "expensive"
    assert PARAM_CLASS["dr"] == "expensive"
    assert PARAM_CLASS["type_str"] == "expensive"


def test_main_window_constructs_and_recalculates(tk_root, tmp_path, monkeypatch, cw_sd_i_ref):
    """Constructs MainWindow, triggers a recalc, verifies S was computed and cached."""
    monkeypatch.chdir(tmp_path)  # so last_session.json writes into tmp_path
    from sensmaps.gui import MainWindow
    ref = cw_sd_i_ref
    mw = MainWindow(master=tk_root)
    # Set the form to values that match the fixture
    values = {
        "type_str": "CW_SD_I",
        "rs": [[0.0, 0.0, 0.0]],
        "rd": [[float(ref["rho"]), 0.0, 0.0]],
        "opt_prop": {
            "n_in":  float(ref["nin"]),
            "n_out": float(ref["nout"]),
            "musp":  float(ref["musp"]),
            "mua":   float(ref["mua"]),
        },
        "xl": [float(ref["xl"][0]), float(ref["xl"][1])],
        "yl": [float(ref["yl"][0]), float(ref["yl"][1])],
        "zl": [float(ref["zl"][0]), float(ref["zl"][1])],
        "dr": float(ref["dr"]),
        "pert": [float(p) for p in ref["pert"]],
        "slice_axis": "y",
        "slice_value": 0.0,
        "quantiles": [0.05, 0.95],
    }
    mw.params_panel.set_values(values)
    mw.recalculate()
    assert mw._cache is not None
    S_ref = np.asarray(ref["S"], dtype=float)
    np.testing.assert_allclose(mw._cache.S, S_ref, rtol=1e-8, atol=1e-12)


def test_session_persistence_round_trip(tk_root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from sensmaps.gui import MainWindow
    mw1 = MainWindow(master=tk_root)
    vals = mw1.params_panel.get_values()
    vals["opt_prop"]["musp"] = 1.7
    vals["rs"] = [[1.0, 2.0, 3.0]]
    mw1.params_panel.set_values(vals)
    mw1.save_session()

    mw2 = MainWindow(master=tk_root)
    mw2.load_session()
    got = mw2.params_panel.get_values()
    assert got["opt_prop"]["musp"] == 1.7
    assert got["rs"] == [[1.0, 2.0, 3.0]]


def test_revert_restores_to_last_computed(tk_root, tmp_path, monkeypatch, cw_sd_i_ref):
    monkeypatch.chdir(tmp_path)
    from sensmaps.gui import MainWindow
    ref = cw_sd_i_ref
    mw = MainWindow(master=tk_root)
    values = {
        "type_str": "CW_SD_I",
        "rs": [[0.0, 0.0, 0.0]],
        "rd": [[float(ref["rho"]), 0.0, 0.0]],
        "opt_prop": {"n_in": 1.333, "n_out": 1.0, "musp": 1.1, "mua": 0.011},
        "xl": [-5.0, 40.0], "yl": [0.0, 0.0], "zl": [0.0, 20.0],
        "dr": 1.0, "pert": [1.0, 1.0, 1.0],
        "slice_axis": "y", "slice_value": 0.0, "quantiles": [0.05, 0.95],
    }
    mw.params_panel.set_values(values)
    mw.recalculate()
    # Mutate an expensive field
    dirty = dict(values); dirty["rs"] = [[5.0, 5.0, 5.0]]
    mw.params_panel.set_values(dirty)
    assert mw.is_dirty
    mw.revert()
    got = mw.params_panel.get_values()
    assert got["rs"] == [[0.0, 0.0, 0.0]]
    assert not mw.is_dirty


def test_main_smoke_constructs_and_exits(tmp_path, monkeypatch):
    """Ensure main() can construct a MainWindow and shut down cleanly."""
    _require_display()
    monkeypatch.chdir(tmp_path)
    from sensmaps.__main__ import main
    exit_code = main(argv=["--smoke-test"])
    assert exit_code == 0


def test_parse_float_matrix_single_row():
    from sensmaps.gui import _parse_float_matrix
    assert _parse_float_matrix("0 0 0", 3) == [[0.0, 0.0, 0.0]]


def test_parse_float_matrix_two_rows_semicolon():
    from sensmaps.gui import _parse_float_matrix
    assert _parse_float_matrix("0 0 0; 30 0 0", 3) == [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]]


def test_parse_float_matrix_trailing_semicolon():
    from sensmaps.gui import _parse_float_matrix
    assert _parse_float_matrix("0 0 0;", 3) == [[0.0, 0.0, 0.0]]


def test_parse_float_matrix_mixed_separators():
    from sensmaps.gui import _parse_float_matrix
    assert _parse_float_matrix("0,0,0; 30, 0, 0", 3) == [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]]


def test_parse_float_matrix_rejects_wrong_column_count():
    from sensmaps.gui import _parse_float_matrix
    with pytest.raises(ValueError, match="expected 3"):
        _parse_float_matrix("0 0", 3)


def test_parse_float_matrix_rejects_empty_input():
    from sensmaps.gui import _parse_float_matrix
    with pytest.raises(ValueError, match="at least one row"):
        _parse_float_matrix("", 3)
    with pytest.raises(ValueError, match="at least one row"):
        _parse_float_matrix(";", 3)


def test_parameter_panel_round_trips_multi_row_rs(tk_root):
    from sensmaps.gui import ParameterPanel
    panel = ParameterPanel(master=tk_root)
    panel.set_values({
        "rs": [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        "rd": [[35.0, 0.0, 0.0]],
    })
    got = panel.get_values()
    assert got["rs"] == [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    assert got["rd"] == [[35.0, 0.0, 0.0]]


def test_parameter_panel_default_rs_is_single_row(tk_root):
    """Existing v1 default rs/rd must still parse as a single row."""
    from sensmaps.gui import ParameterPanel
    panel = ParameterPanel(master=tk_root)
    got = panel.get_values()
    assert got["rs"] == [[0.0, 0.0, 0.0]]
    assert got["rd"] == [[35.0, 0.0, 0.0]]


def test_parameter_panel_default_fmod(tk_root):
    from sensmaps.gui import ParameterPanel
    panel = ParameterPanel(master=tk_root)
    got = panel.get_values()
    assert got["fmod"] == 100.0   # MHz, defaults from _DEFAULTS


def test_fmod_field_disabled_for_cw(tk_root):
    from sensmaps.gui import ParameterPanel
    panel = ParameterPanel(master=tk_root)
    panel.set_values({"type_str": "CW_SD_I"})
    assert str(panel._fmod_entry.cget("state")) == "disabled"


def test_fmod_field_enabled_for_fd(tk_root):
    from sensmaps.gui import ParameterPanel
    panel = ParameterPanel(master=tk_root)
    panel.set_values({"type_str": "FD_SD_I"})
    assert str(panel._fmod_entry.cget("state")) == "normal"


def test_param_class_classifies_fmod_as_expensive(tk_root):
    from sensmaps.gui import PARAM_CLASS
    assert PARAM_CLASS["fmod"] == "expensive"


def test_type_combobox_lists_v1_1_combos(tk_root):
    from sensmaps.gui import ParameterPanel
    panel = ParameterPanel(master=tk_root)
    # The combobox is the first widget in panel._inputs (built first in _build_widgets).
    cb_type = panel._inputs[0]
    expected = [
        "CW_SD_I", "CW_SS_I", "CW_DS_I",
        "FD_SD_I", "FD_SS_I", "FD_DS_I",
        "FD_SD_P", "FD_SS_P", "FD_DS_P",
    ]
    assert list(cb_type.cget("values")) == expected


def test_recalculate_passes_fmod_in_hz_for_fd(tk_root, tmp_path, monkeypatch, cw_sd_i_ref):
    """When type is FD_*, MainWindow.recalculate must convert MHz → Hz at the boundary."""
    monkeypatch.chdir(tmp_path)
    from sensmaps.gui import MainWindow
    import sensmaps.compute as compute_mod
    captured = {}
    real_make_s_full = compute_mod.make_s_full

    def spy(*args, **kwargs):
        captured.update(kwargs)
        # Defer to a CW path so we don't need an FD fixture here:
        kwargs["type_str"] = "CW_SD_I"
        kwargs["fmod"] = None
        return real_make_s_full(*args, **kwargs)

    monkeypatch.setattr(compute_mod, "make_s_full", spy)
    monkeypatch.setattr("sensmaps.gui.make_s_full", spy)

    mw = MainWindow(master=tk_root)
    mw.params_panel.set_values({"type_str": "FD_SD_I", "fmod": 100.0})
    mw.recalculate()
    assert captured["fmod"] == 100.0 * 1e6
