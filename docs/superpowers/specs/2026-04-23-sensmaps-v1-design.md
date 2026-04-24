# sensmaps v1 — Design

**Date:** 2026-04-23
**Author:** Giles Blaney (with Claude)
**Status:** Draft, pending review before implementation plan is written.

## 1. Purpose

`sensmaps` is an interactive Python GUI for exploring 2D colormap slices of the Jacobian

$$\mathcal{S}(\vec{r}) = \frac{\partial Y / \partial \mu_a(\vec{r})}{\partial Y / \partial \mu_{a,\text{homogeneous}}}$$

of a detected optical signal `Y` with respect to local absorption changes — "sensitivity to absorption change" maps. It is the Python companion to the MATLAB `SensitivityCompendium` (Blaney, Sassaroli, and Fantini, *JIOHS* **17**(04), 2430001 (2024)) and is intended to be a "graphing-calculator" for testing how measurement parameters affect the spatial sensitivity.

## 2. Versioning

| Version | Scope |
|---|---|
| **v1** *(this spec)* | Single measurement type: **CW_SD_I** (continuous-wave, single-distance, intensity). Diffusion-theory compute only. Tkinter GUI. Hybrid interactivity. Single 2D slice view. Save PNG/PDF + `.npz`. Lab-friendly install via `pip install -e .`. |
| **v2** *(backend)* | Broaden to all ~30 type combos (CW/FD/TD × SD/SS/DS × I/GI/DGI/P/T/V). Vectorize / cache to push all parameter changes toward realtime updates. Retire the explicit `Recalculate` button once compute is fast enough. |
| **v3** *(frontend + MC)* | Third-angle multi-pane view. Parameter-sweep mode (multi-panel paper figures). LaTeX figure export. Optional Monte Carlo backend via `pmcx`, installed as `pip install sensmaps[mc]`. |

## 3. v1 scope in detail

### 3.1 Physics

Port the CW continuous-wave path-length integrals for single-distance intensity (`CW_SD_I`) from `DOIT-Public/SensitivityCompendium/deps/`:

- `continuousTotPathLen.m` → `continuous_tot_path_len`
- `continuousPartPathLen.m` → `continuous_part_path_len`

Both are real-valued CW closed-form analytical solutions. They are used rather than the `complex*` forms at ω=0 because real arithmetic is strictly faster than complex arithmetic in NumPy with no precision loss at ω=0. If empirical benchmarking in implementation shows the `complex*`-at-ω=0 path is faster for some reason (unlikely), we may switch; default is `continuous*`.

The v1 sensitivity is then

$$\mathcal{S} = \frac{\ell_{\text{pert}}}{L},$$

where `L` is the total path length (scalar) and `ℓ_pert` is the per-voxel partial path length convolved with a box-kernel representing the perturbation size. This matches MATLAB `makeS.m` for `CW_SD_I`.

### 3.2 Compute layer

`compute.make_s(type_str, rs, rd, opt_prop, **kw)` mirrors MATLAB `makeS`:

- Parses `type_str` of the form `AA_BB_C` (v1 only accepts `CW_SD_I`; other values raise `NotImplementedError` with a clear message).
- Builds the voxel grid from `xl`, `yl`, `zl`, `dr`.
- Applies the source z-offset (`1/musp` for DT).
- Dispatches to `physics.continuous_tot_path_len` and `physics.continuous_part_path_len`.
- Applies the perturbation convolution via `scipy.signal.fftconvolve` with a box kernel of shape `pert / dr`.
- Returns `(S, params)` where `params` is a dataclass holding `x`, `y`, `z` axis vectors and `xl_valid`, `yl_valid`, `zl_valid`.
- Also caches `Svox` (pre-perturbation) on the returned struct so the GUI can re-convolve quickly when only `pert` changes.

The `(temporal, data_type) → (L, Y, ll)` + `(arrangement) → combine` dispatch pattern from MATLAB `makeS` is present in v1 even though only one entry (`CW, I, SD`) is filled in. Adding a new type in v2 is an entry in the dispatch table plus one new `physics` function.

### 3.3 Views layer

- `slice_s(S, params, axis, value)` → `(S_plane, plot_params)` where `plot_params` holds `vert_axis`, `vert_label`, `horz_axis`, `horz_label`. Mirrors MATLAB `sliceS`.
- `make_color_limits(S, quantiles=(0.05, 0.95))` → `(clim, cmap)` where `cmap` is the jet colormap with saturated ends (lowest row black, highest row white) per MATLAB `makeCL`.
- `render_slice(ax, S_plane, plot_params, clim, cmap)` draws `imshow` (equivalent to MATLAB `imagesc`) + dashed-grey contour lines at colorbar ticks, onto a provided matplotlib Axes. No Tkinter import.

No default colormap registry or other global state.

### 3.4 GUI layer

**Framework:** Tkinter (`tkinter` from stdlib) with matplotlib's `FigureCanvasTkAgg` for the plot embedding. No Qt, no web stack.

**Window layout** (resizable; opens ~1100 × 750 px):

```
┌────────────────────────────────────────────────────────────────────┐
│ sensmaps                                                            │
├─────────────────────────────┬──────────────────────────────────────┤
│  ParameterPanel             │  PlotCanvas                          │
│                             │                                      │
│  ▾ Type         CW_SD_I ▾   │       ┌──────────────────┐           │
│  ▾ Optodes (rs, rd)         │       │   2D slice       │           │
│  ▾ Optical properties       │       │   imshow +       │           │
│  ▾ Grid (xl, yl, zl, dr)    │       │   contour        │           │
│  ▾ Perturbation (pert)      │       └──────────────────┘           │
│  ▾ Slice (axis, value)      │       (colorbar)                     │
│  ▾ Color (quantiles)        │                                      │
│                             │                                      │
├─────────────────────────────┴──────────────────────────────────────┤
│ [● dirty]  [Recalculate]  [Revert]  [Save Figure…]  [Save Data…]    │
└────────────────────────────────────────────────────────────────────┘
```

**Controls:**

- Entry fields for scalars and short vectors. Focus-out validator parses to float or list of floats.
- Dropdowns for `Type` (one option in v1) and slice `axis` (`x`, `y`, `z`).
- Slider for slice `value`, range auto-set from grid limits, step `dr`.
- Paired min/max entries for `xl`, `yl`, `zl`.

**Cheap vs expensive parameter classification:**

| Group | Params | Class |
|---|---|---|
| Slice | axis, value | cheap |
| Color | quantiles | cheap |
| Perturbation | pert | cheap (just re-convolves cached `Svox`) |
| Optodes | rs, rd | expensive |
| Optical props | n_in, n_out, musp, mua | expensive |
| Grid | xl, yl, zl, dr | expensive |
| Type | type_str | expensive |

- Cheap change → `views.slice_s` (or re-convolve for `pert`) + canvas redraw.
- Expensive change → state marked dirty, dirty indicator turns amber, plot is not updated until user clicks `Recalculate`.
- `Revert` → restores all form fields to the values that produced the currently-displayed `S`. Cached `S` untouched.

**Dirty indicator:** small colored circle plus a short label ("in sync" / "form changed — click Recalculate").

**Save Figure:** file dialog; extension (`.png`, `.pdf`, `.svg`) selects format via matplotlib.

**Save Data:** file dialog → writes `.npz` with `S`, `Svox`, `x`, `y`, `z`, all input parameters, `type_str`, `sensmaps` version.

**Session persistence:** on close, write current form values to `./last_session.json` in the current working directory. On launch, load if present. Delete the file to reset to defaults. (This keeps the session file adjacent to the clone for discoverability.)

### 3.5 Entry points

- `python -m sensmaps` (via `src/sensmaps/__main__.py`)
- `sensmaps` console-script (installed by `pyproject.toml`)
- `python src/sensmaps/__main__.py` (runs from a clone without installing)

## 4. Architecture

### 4.1 Repo layout

```
sensmaps/
├── .gitignore
├── README.md
├── pyproject.toml
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-04-23-sensmaps-v1-design.md   # this document
├── src/
│   └── sensmaps/
│       ├── __init__.py         # public API: make_s, slice_s, make_color_limits
│       ├── __main__.py         # launches GUI
│       ├── compute.py
│       ├── gui.py
│       ├── physics.py
│       └── views.py
└── tests/
    ├── fixtures/
    │   ├── generate_fixtures.m     # committed; regenerates reference data
    │   └── cw_sd_i_example1.mat    # committed; small canonical reference
    ├── test_compute.py
    ├── test_gui.py
    ├── test_physics.py
    └── test_views.py
```

`LICENSE` and CI configuration (`.github/workflows/`) are deferred — added when the license is chosen and when we approach public release (see §9).

### 4.2 Module responsibilities

**`physics.py`** — Pure NumPy. Analytical DT formulas. v1 exports `continuous_tot_path_len`, `continuous_part_path_len`. No matplotlib, no Tkinter, no global state. Each function takes arrays of positions + an `OpticalProperties` dataclass, returns arrays.

**`compute.py`** — `make_s(type_str, rs, rd, opt_prop, **kw)` top-level dispatcher. Handles grid construction, source z-offset, type-string parsing, dispatch to `physics.*`, perturbation convolution, assembly of return struct. Dispatch table structure is v2-ready.

**`views.py`** — `slice_s`, `make_color_limits`, `render_slice`. No Tkinter imports.

**`gui.py`** — `MainWindow`, `ParameterPanel`, `PlotCanvas`. Holds cached `(S, Svox, params)` on the window instance. Routes param changes through cheap/expensive classification. Embeds matplotlib canvas. All Tkinter code lives here.

**`__main__.py`** — `main()` function parses optional CLI flags, constructs `MainWindow`, starts Tk mainloop.

**`__init__.py`** — re-exports `make_s`, `slice_s`, `make_color_limits` for library users.

### 4.3 Data flow

```
cheap param change    → on_cheap_change  → views.slice_s / re-conv → canvas redraw
expensive change      → mark state dirty → dirty indicator amber
Recalculate clicked   → compute.make_s   → physics.*                → cache S, Svox, params
                                         → views.slice_s            → canvas redraw
                                         → mark state in-sync
Revert clicked        → restore form fields from last-computed inputs
Save Figure clicked   → matplotlib savefig (PNG / PDF / SVG by extension)
Save Data clicked     → numpy.savez      (S, Svox, axes, inputs, type, version)
Window close          → write ./last_session.json
Window open           → read  ./last_session.json if present
```

## 5. MATLAB-fidelity principle

The Python code stays close to the MATLAB structure. Variable names are snake_case but otherwise recognizable (`typStr` → `type_str`, `optProp` → `opt_prop`, `makeS` → `make_s`, `sliceS` → `slice_s`, `makeCL` → `make_color_limits`). Formula structure is preserved line-by-line where practical. Coordinate conventions, quantile choices, contour positions, and colormap saturation match MATLAB. Structs become dataclasses; otherwise no refactoring for Pythonic style unless behavior-preserving and clearly beneficial.

## 6. Testing strategy

**Framework:** `pytest`.

**Layout:**

```
tests/
├── fixtures/
│   ├── generate_fixtures.m         # committed; MATLAB-side reference generator
│   └── cw_sd_i_example1.mat        # committed; small (~1 MB) reference
├── test_physics.py
├── test_compute.py
├── test_views.py
└── test_gui.py
```

**Fixture generation:** A small `generate_fixtures.m` exercises `continuousPartPathLen` and `continuousTotPathLen` on a canonical set of inputs and saves via MATLAB `save` as `.mat`. The Python tests load with `scipy.io.loadmat`. Committed fixture is deliberately small (single rho, compact grid) so it stays under ~1 MB. The pre-existing `CW_l_L.mat` (~980 MB, 10 distances) at `/home/giles/TuftsBox_local/DOIT/SenVol/230407_CWsen_forJIOHSreview/` is a local-only "wide" cross-check, not committed.

**Tolerances:** `numpy.testing.assert_allclose(rtol=1e-10, atol=0)` for path-length formulas; relax to `rtol=1e-8` for FFT-convolved `S` if needed.

**Tests:**

- `test_physics.py` — regression test each ported physics function against its fixture.
- `test_compute.py` — end-to-end `make_s("CW_SD_I", …)` on `example1_DT.m` inputs vs. fixture.
- `test_views.py` — `slice_s` shape/orientation/labels; `make_color_limits` quantile values (no fixture needed).
- `test_gui.py` — smoke test: construct `MainWindow` inside a hidden Tk root, verify it initializes, destroy. No interactive assertions.

**Out of scope for v1:** CI (GitHub Actions), coverage reporting, interactive GUI test harness.

## 7. Install & run

### 7.1 `pyproject.toml` skeleton

```toml
[project]
name = "sensmaps"
version = "0.1.0"
description = "Interactive GUI for 2D sensitivity maps in diffuse optical imaging"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.24",
  "scipy>=1.11",
  "matplotlib>=3.7",
]

[project.optional-dependencies]
dev = ["pytest>=8"]
mc  = ["pmcx"]                  # placeholder for v3

[project.scripts]
sensmaps = "sensmaps.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 7.2 README.md outline

1. Title + one-line description.
2. Screenshot (added after the first successful run).
3. Install (primary: venv):
   ```bash
   git clone https://github.com/gblane/sensmaps.git
   cd sensmaps
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   sensmaps                     # or: python -m sensmaps
   ```
4. Alternative install notes for conda and uv (one block each).
5. Usage — controls summary, screenshot annotations.
6. Relationship to the compendium paper (link + BibTeX).
7. Development — `pip install -e .[dev]`, `pytest`.
8. License + authorship.

### 7.3 `.gitignore`

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
last_session.json
*.npz
tests/fixtures/CW_l_L.mat
tests/fixtures/DSsetSDSSDSsen.mat
.vscode/
.idea/
```

## 8. Extension points for v2 and v3

**v2 — backend broadening:**

- New physics functions land in `physics.py` one per MATLAB `deps/` file. When `physics.py` exceeds ~500 lines it splits into a `physics/` subpackage (one file per formula family: `cw_path_lengths.py`, `fd_path_lengths.py`, `td_path_lengths.py`, `td_gated_path_lengths.py`, etc.). No other layer changes.
- `compute.make_s` dispatch table gains entries for each new `(temporal, data_type)` pair. The `(arrangement) → combine` step handles `SS`, `DS`, `SD_DIFF` without further physics changes (it's pure linear combination of `L` and `ll`).
- Once physics is fast enough, the cheap/expensive split retires: all params become "cheap-ish", `on_change` schedules a debounced recompute, and the `Recalculate` / `Revert` / dirty banner controls become hidden or removed. The MainWindow state cache then just serves as the most-recent-compute memo.

**v3 — frontend + MC:**

- `views.py` gains `render_third_angle(fig, S, params, xsl, ysl, zsl, …)` alongside `render_slice`. `MainWindow` adds a `View: single / third-angle` dropdown and dispatches to the chosen renderer.
- `compute.py` gains `make_s_sweep(type_str, rs, rd, opt_prop, sweep_param, sweep_values, **kw)` returning `(S_all, params, sweep_values)` with a leading sweep axis. GUI adds a "Sweep…" modal; sweep results open in a secondary window with a multi-panel layout.
- `views.py` gains `save_figure_latex(fig, path, size_cm, …)` mirroring MATLAB `saveFig_senCompen` + `makeLatexFigure`. Emits paired `.pdf` + `.tex`.
- `physics/mc.py` lazily imports `pmcx`. `compute.make_s` honors `sim_typ="MC"` and dispatches. Missing `pmcx` with `sim_typ="MC"` raises `ModuleNotFoundError` with a hint: `pip install sensmaps[mc]`.

**What v1 locks in to enable all of this without rework:**

1. `compute.make_s` signature includes `type_str`, `sim_typ` kwargs even though only `"CW_SD_I"` / `"DT"` work.
2. Dispatch table structure in `compute.py` is present even with one entry.
3. `views.py` has `slice_s`, `make_color_limits`, `render_slice` as separate small functions.
4. `gui.py` routes rendering through a single view-mode handle, even if only "single slice" is wired.
5. `compute.make_s` caches `Svox` (pre-convolution) alongside `S` so a v2 live-perturbation slider works without a re-dispatch.

## 9. Deferred / open decisions

1. **License** — Deferred until public release. Candidates: MIT, BSD-3-Clause, or whatever `DOIT-Public` adopts.
2. **CI** — Deferred. Add GitHub Actions when moving toward public release.
3. **Coverage reporting** — Deferred. Optional even at public release time.
4. **BibTeX entry in README** — Pull the standard JIOHS citation when writing the README.
5. **Screenshot in README** — Generate after first working v1 run.
6. **Window default size and color theme** — Start with Tkinter defaults; adjust after first visual check.
