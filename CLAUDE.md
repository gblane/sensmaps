# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`sensmaps` is an interactive Python/Tkinter GUI for exploring 2D slice maps of the Jacobian
`S = ∂Y/∂μₐ(r)` of a detected optical signal `Y` with respect to local absorption changes —
"sensitivity to absorption change" maps for diffuse optical spectroscopy and imaging. It is
the Python companion to the MATLAB
[`SensitivityCompendium`](https://github.com/DOIT-Lab/DOIT-Public/tree/main/SensitivityCompendium)
that accompanies Blaney, Sassaroli & Fantini, *JIOHS* **17**(04), 2430001 (2024).

v1.2 implements 12 measurement types under diffusion theory:
- CW × {SD, SS, DS} × {I}
- FD × {SD, SS, DS} × {I, P}
- TD × {SD, SS, DS} × {GI}

Remaining TD data types (`DGI` / `T` / `V`) land in v1.3, completing the full ~30-combo
table from the MATLAB compendium. v3 adds a Monte Carlo backend (`pmcx`) and parameter
sweeps without restructuring.

## Common commands

All commands assume the venv is active: `source .venv/bin/activate`.

```bash
# Run the GUI
sensmaps                         # console-script entry point
python -m sensmaps               # equivalent
sensmaps --smoke-test            # build window, tick once, exit 0 (used in tests)

# Tests
pytest                           # full suite (~26 tests, ~4 s)
pytest tests/test_physics.py -v  # one file
pytest tests/test_compute.py::test_make_s_cw_sd_i_matches_matlab -v   # one test
```

GUI tests skip on Linux when `$DISPLAY` is unset. On Fedora, tkinter requires
`sudo dnf install python3-tkinter`; on Debian/Ubuntu, `python3-tk`.

## Architecture (the part that takes reading multiple files)

The codebase is **four layers**, strictly bottom-up; nothing imports a layer above it.

```
physics.py   →   compute.py   →   views.py   →   gui.py   →   __main__.py
(numpy)          (dispatcher)     (matplotlib)    (tkinter)
```

- **`physics.py`** — pure NumPy ports of `DOIT-Public/SensitivityCompendium/deps/*.m`.
  Variable names mirror the MATLAB source (snake_case), formulas are line-by-line.
  All `continuous_*` functions are thin `omega=0` wrappers around `complex_*`. `_split_source`
  generalizes to N-row source arrays so `complex_part_path_len` can call `complex_reflectance`
  with a voxel grid acting as N independent sources.

- **`compute.py`** — `make_s_full(type_str, rs, rd, opt_prop, xl, yl, zl, dr, pert, sim_typ)`
  is the dispatcher. Mirrors MATLAB `makeS.m`: parses the `AA_BB_C` type string (`ParsedType`),
  builds the voxel grid (`GridParams.from_limits`), z-offsets the source by `1/musp`, calls
  the physics functions, applies the perturbation box-kernel via `scipy.signal.fftconvolve`,
  and returns a `SensitivityResult` with both `S` (post-conv) and `Svox` (pre-conv) cached.
  `make_s` is a thin wrapper returning just `(S, params)`. v1 only accepts `type_str="CW_SD_I"`
  and `sim_typ="DT"`; anything else raises `NotImplementedError` with a clear message.
  The `(temporal, data_type) → (L, Y, ll)` dispatch structure is intentionally laid out for v2
  to add entries without restructuring. `Svox` is cached so the GUI can re-convolve with a
  new perturbation kernel without redoing the diffusion-theory pass.
- v1.1 introduces `_PHYSICS_DISPATCH[(temporal, data_type)] -> (L_fn, ll_fn, Y_fn)`
  and `_ARRANGEMENT_COMBINE[arrangement]`. Adding a new combo is one row in
  each table plus, if needed, a new physics function. `_expand_optodes`
  validates optode counts against the arrangement and applies the z-offset.
- v1.2 adds the `("TD", "GI")` dispatch entry plus kw-only `tg`, `tend`, `ndt`
  parameters on `make_s_full`. The TD primitives (`temporal_reflectance`,
  `temporal_fluence`, `temporal_gate_tot_path_len`, `temporal_gate_part_path_len`)
  use **picoseconds** internally (matching MATLAB exactly). `make_s_full` builds
  an `extra` kwargs dict per measurement that threads `fmod` (FD) or
  `tg`/`conv_t`/`conv_dt` (TD) through to the dispatch callable; existing entries
  absorb unused kwargs via `**_`.

- **`views.py`** — pure matplotlib, no Tkinter. `slice_s` (port of MATLAB `sliceS.m`) returns
  a 2D plane plus a `PlotParams` dataclass holding axis vectors and LaTeX labels. `make_color_limits`
  (port of `makeCL.m`) returns a quantile-based `(vmin, vmax)` plus a jet colormap with row 0
  forced black and row 99 forced white (saturated ends). `render_slice` draws `imshow + dashed contour`
  on a caller-provided `Axes` and inverts the y-axis when the vertical axis is depth (NIRS convention).
  Contour overlay is skipped when either dimension of the plane is degenerate (matplotlib requires
  `≥(2, 2)`).

- **`gui.py`** — Tkinter shell. `PARAM_CLASS` partitions every input parameter into `"cheap"`
  (slice axis/value, color quantiles, perturbation) vs `"expensive"` (everything that re-runs
  physics). The `MainWindow.recalculate()` button rebuilds `_cache: SensitivityResult`. Cheap
  changes call `_apply_cheap_change`, which re-slices off the cached `S` and re-convolves
  `Svox` with a new kernel for `pert` changes — no physics recompute. Expensive changes set
  the dirty flag; `revert()` restores the form to the inputs that produced the current plot.
  Session state persists to `./last_session.json` on close.
- `fmod` lives in **MHz** at the GUI boundary and **Hz** internally;
  conversion happens in `MainWindow.recalculate`. The `_fmod_entry` is
  state-driven by `type_str` (disabled when CW). Multi-row `rs`/`rd` use
  `_parse_float_matrix` which splits on `;`.
- v1.2 adds `tg` (gate window), `tend`, and `ndt` to the form. `tg` is in **ns**
  at the boundary and **ps** internally; conversion (×1000) also happens in
  `MainWindow.recalculate`. `_tg_entry` is enabled iff `type_str.endswith("_GI")`;
  `tend`/`ndt` sit behind a single "Override" checkbox so users normally inherit
  the MATLAB defaults (`tend=10000 ps`, `ndt=10000`) without typing them.

- **`__main__.py`** — argparse + Tk mainloop. `--smoke-test` flag is used by the test suite.

## Test fidelity convention

The Python port must be numerically equivalent to MATLAB. Tests load `.mat` fixtures via
`scipy.io.loadmat(squeeze_me=True)`; `tests/conftest.py` then **re-inflates the squeezed
single-row y-axis** so `S` is `(Nx, Ny=1, Nz)` instead of the MATLAB-collapsed `(Nx, Nz)`.
This matters: anything that produces a 3-D array must compare against the conftest's
re-shaped fixture, not the raw `.mat`.

Tolerances: `rtol=1e-10` for direct path-length / fluence formulas; relax to `rtol=1e-8` for
FFT-convolved `S` arrays. The `n2a` polynomial port has ~1.5e-12 relative drift from MATLAB
(7th-order polynomial, evaluation order differs) so its test uses 1e-10.

## MATLAB-fidelity principle

When porting from MATLAB:
- snake_case names that mirror the MATLAB symbol (`makeS` → `make_s`, `sliceS` → `slice_s`,
  `makeCL` → `make_color_limits`, `optProp` → `opt_prop`, `complexFluence` → `complex_fluence`).
- Preserve formula structure line-by-line. Don't refactor for Pythonic style unless it's
  behavior-preserving and clearly beneficial (e.g. struct → dataclass is fine).
- Coordinate conventions, quantile choices, contour positions, and colormap saturation
  match MATLAB.

## Reference paths (do not modify these)

- `/home/giles/GitHub/DOIT-Public/SensitivityCompendium/deps/` — canonical MATLAB sources.
  Read-only; `tests/fixtures/generate_fixtures.m` adds it to the MATLAB path to regenerate
  the reference `.mat`.
- `/home/giles/GitHub/DOIT-Toolbox/` — the larger MATLAB toolbox (read-only).
- `/home/giles/TuftsBox_local/DOIT/Writing/JIOHS/2023_sensReview/` — paper directory.

## Float-arithmetic gotcha

The `pert % dr == 0` "is multiple" check in `make_s_full` uses tolerance against the nearest
integer multiple of `dr` rather than Python's `%` operator: `1.0 % 0.1 == 0.0999...` in IEEE
floats and would otherwise reject legitimate inputs.

## Plan + spec

The full design and TDD-structured implementation plans live in `docs/superpowers/plans/`.
v1 was 19 atomic tasks; v1.1 was 19 tasks; v1.2 was a lighter 10-task combined spec+plan
(`2026-04-27-sensmaps-v1_2-implementation.md`). Each release reuses the dispatch table and
arrangement combinator scaffolding without restructuring.
