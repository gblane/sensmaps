# sensmaps v1.2 — Time-Domain Gated Intensity (TD\_\*\_GI)

**Status:** ready for implementation
**Author:** Giles + Claude (Opus 4.7)
**Date:** 2026-04-27
**Branch:** `v1.2-dev` (direct branch, no worktree)

This is a *combined spec + plan* — lighter process than v1.1's two-doc split, since
the scaffolding (dispatch table, optode expansion, arrangement combinators, GUI
patterns) is already in place.

---

## 1. Scope

Add 3 new measurement types: `TD_SD_GI`, `TD_SS_GI`, `TD_DS_GI` — time-domain
gated intensity for single-distance, single-slope, dual-slope arrangements.

**Explicitly out of scope** (deferred to later releases):
- `TD_*_DGI` (difference-of-gated-intensity)
- `TD_*_T` (mean time-of-flight, `temporalKthMoment` k=1)
- `TD_*_V` (variance, `temporalVar*`)
- Plain `TD_*_I` — does not exist in the MATLAB compendium; integrating
  TD reflectance over all t collapses to CW intensity (already covered).
- Monte-Carlo (`simTyp='MC'`) backend — DT only for v1.2.

After v1.2 the type combobox grows to **12 entries** (3 CW + 6 FD + 3 TD).

## 2. Physics layer (`physics.py`)

Three new functions, all DT-only ports of the matching `temporal*.m` files.
Internal time unit is **picoseconds (ps)** to match MATLAB exactly.

### 2.1 `temporal_reflectance(rs, rd, t, opt_prop)`

Port of `deps/temporalReflectance.m` (DT branch only, lines 70-88).

```python
def temporal_reflectance(rs, rd, t, opt_prop):
    """Time-resolved reflectance R(t) (1/(ps·mm²)).

    Parameters
    ----------
    rs : (Ns, 3) array — source coords (mm). Either Ns=1 or Nd=1.
    rd : (Nd, 3) array — detector coords (mm).
    t  : (Nt,) array — time samples (ps). Negative t → R=0.
    opt_prop : OpticalProperties

    Returns
    -------
    R : (max(Ns, Nd), Nt) array.
    """
```

Implementation mirrors MATLAB lines 80-88 verbatim. Negative `t` indices
return 0 (MATLAB lines 63-64, 182).

### 2.2 `temporal_fluence(rs, rd, t, opt_prop)`

Port of `deps/temporalFluence.m`. Same shape conventions as `temporal_reflectance`.
Implementation mirrors lines 65-68.

### 2.3 `temporal_gate_tot_path_len(rs, rd, tg, opt_prop, *, conv_t=10000.0, conv_dt=1.0)`

Port of `deps/temporalGateTotPathLen.m`.

- `tg`: shape `(2,)` — `[t_start, t_end]` in ps. (v1.2 supports a single gate;
  multi-gate `(2, Ng)` deferred.)
- Internal: builds `t = -conv_t : conv_dt : conv_t`, calls `temporal_reflectance`,
  finds index of nearest gate edges, evaluates `trapz(t·R) / trapz(R)` × v.
- Returns scalar `L` (mm).

### 2.4 `temporal_gate_part_path_len(rs, r, rd, V, tg, opt_prop, *, conv_t=10000.0, conv_dt=1.0)`

Port of `deps/temporalGatePartPathLen.m` — **FFT convolution branch only**
(lines 112-127), which is the path makeS.m exercises (`FFTconv=true` default).
No parfor.

- `rs`: (1, 3); `rd`: (1, 3); `r`: (Nr, 3) — voxel coords; `V`: scalar (mm³); `tg`: (2,).
- Builds `t = -conv_t : conv_dt : conv_t`, computes `PHI_si(t)` from
  `temporal_fluence(rs, r, t)`, `R_id(t)` from `temporal_reflectance(r, rd, t)`
  (only positive-t entries non-zero, mirroring MATLAB `posInds`).
- For each voxel: `conv = ifft(fft(PHI) * fft(R))` truncated to `len(posInds)`,
  then `num = sum(conv[j1:j2]) * conv_dt`. Final `l = (num / R_g) * V`.

Returns array shape `(Nr,)`. NaNs replaced with 0 in compute.py (compute layer).

### 2.5 Tests

Direct-port tests added to `tests/test_physics.py`, comparing against MATLAB
fixture scalars at fixed `(rs, rd, tg, opt_prop)`:

```python
def test_temporal_gate_tot_path_len_matches_matlab(td_sd_gi_fixture):
    ref = td_sd_gi_fixture
    L = temporal_gate_tot_path_len(ref["rs"], ref["rd"], ref["tg_ps"], op)
    assert_allclose(L, ref["L_scalar"], rtol=1e-10)
```

For partial path-length we'll use `rtol=1e-8` (FFT convolution introduces
~1e-9 drift relative to MATLAB's `ifft`).

## 3. Compute layer (`compute.py`)

### 3.1 Dispatch entry

```python
def _td_L(rs_i, rd_i, op, *, tg, conv_t, conv_dt, **_):
    return float(temporal_gate_tot_path_len(
        rs_i, rd_i, tg, op, conv_t=conv_t, conv_dt=conv_dt))

def _td_ll(rs_i, r_all, rd_i, V, op, *, tg, conv_t, conv_dt, **_):
    return temporal_gate_part_path_len(
        rs_i, r_all, rd_i, V, tg, op, conv_t=conv_t, conv_dt=conv_dt)

def _td_Y(*_a, **_kw):
    return 1.0

_PHYSICS_DISPATCH[("TD", "GI")] = (_td_L, _td_ll, _td_Y)
```

### 3.2 New `make_s_full` keyword args

```python
def make_s_full(
    type_str, rs, rd, opt_prop,
    xl, yl, zl, dr, pert, sim_typ="DT",
    *,
    fmod=None,
    tg=None,            # (2,) ps — required for TD_*_GI
    tend=None,          # ps — convolution window (default 10000.0)
    ndt=None,           # int — number of conv steps (default 10000)
):
```

**Validation** (mirrors v1.1's FD-fmod pattern, BEFORE dispatch lookup):

```python
if parsed.temporal == "TD":
    if tg is None:
        raise ValueError(f"tg is required for TD_*_* types (got {type_str!r})")
    tg = np.asarray(tg, dtype=float)
    if tg.shape != (2,):
        raise ValueError(f"tg must be shape (2,), got {tg.shape}")
    if tg[1] <= tg[0]:
        raise ValueError(f"tg[1] must be > tg[0], got tg={tg.tolist()}")
    if tend is None: tend = 10000.0
    if ndt  is None: ndt  = 10000
    if tend <= 0 or ndt <= 0:
        raise ValueError(f"tend and ndt must be positive (got tend={tend}, ndt={ndt})")
    conv_t  = tend
    conv_dt = tend / ndt
```

Then in the per-measurement loop, build `extra` kwargs:

```python
extra = {}
if parsed.temporal == "FD":  extra["omega"] = 2*np.pi*fmod
if parsed.temporal == "TD":  extra["tg"] = tg; extra["conv_t"] = conv_t; extra["conv_dt"] = conv_dt
```

…and thread `**extra` through `L_fn` / `ll_fn`. (Existing CW/FD `**_` swallow ignores them.)

### 3.3 `SensitivityResult` additions

```python
@dataclass
class SensitivityResult:
    # ...existing fields...
    tg:   np.ndarray | None = field(default=None, kw_only=True)  # (2,) ps
    tend: float       | None = field(default=None, kw_only=True)
    ndt:  int         | None = field(default=None, kw_only=True)
```

### 3.4 Tests

`tests/test_compute.py` — three regression tests, one per arrangement:

```python
def test_make_s_td_sd_gi_matches_matlab(td_sd_gi_fixture):
    ref = td_sd_gi_fixture
    S, _ = make_s(ref["type_str"], ref["rs"], ref["rd"], op,
                  ref["xl"], ref["yl"], ref["zl"], ref["dr"], ref["pert"],
                  tg=ref["tg_ps"])
    assert_allclose(S, ref["S"], rtol=1e-8, atol=1e-12)
```

(Default `tend=10000`, `ndt=10000` matches MATLAB defaults exactly.)

Plus the existing "rejects unsupported type" test still passes (TD GI now
*is* supported; will pick a fresh combo for the negative test or remove that
expectation entirely).

## 4. Views layer (`views.py`)

**No changes.** Slice/colorlimit logic is measurement-type-agnostic. The
LaTeX label generator already handles `S` per-type (returns `\mathcal{S}`).

## 5. GUI layer (`gui.py`)

### 5.1 New form fields

Three new rows between `fmod` and `pert`:

| Row | Widget | Default | State |
|---|---|---|---|
| `tg` | Entry, `[start; end]` ns | `1.0; 2.0` | enabled iff `type_str` ends in `_GI` |
| Override TD numerics | Checkbutton | unchecked | enabled iff `type_str` startswith `TD_` |
| `tend` (ns) | Entry | `10.0`, gray | enabled iff override checked |
| `ndt` | Entry (int) | `10000`, gray | enabled iff override checked |

Single override checkbox controls both `tend` and `ndt`. When unchecked,
`MainWindow.recalculate` passes `tend=None, ndt=None` to `make_s_full`
which falls back to MATLAB defaults regardless of entry contents.

### 5.2 PARAM_CLASS

```python
PARAM_CLASS["tg"]            = "expensive"
PARAM_CLASS["tend"]          = "expensive"
PARAM_CLASS["ndt"]           = "expensive"
PARAM_CLASS["td_override"]   = "expensive"
```

### 5.3 Defaults

```python
_DEFAULTS["tg"]          = [[1.0, 2.0]]   # ns, single-row matrix → flattened
_DEFAULTS["tend"]        = 10.0           # ns
_DEFAULTS["ndt"]         = 10000
_DEFAULTS["td_override"] = False
```

### 5.4 ns→ps boundary

In `MainWindow.recalculate`:

```python
type_str = values["type_str"]
fmod_hz = values["fmod"] * 1e6 if type_str.startswith("FD_") else None

if type_str.startswith("TD_"):
    tg_ps = np.asarray(values["tg"][0], dtype=float) * 1000.0
    if values["td_override"]:
        tend_ps = float(values["tend"]) * 1000.0
        ndt_val = int(values["ndt"])
    else:
        tend_ps = None
        ndt_val = None
else:
    tg_ps = tend_ps = ndt_val = None

result = make_s_full(..., fmod=fmod_hz, tg=tg_ps, tend=tend_ps, ndt=ndt_val)
```

### 5.5 State-driven entry enable/disable

In `_on_var_changed`:
- `_tg_entry`: enabled iff `type_str.endswith("_GI")`
- `_td_override_check`: enabled iff `type_str.startswith("TD_")`
- `_tend_entry`, `_ndt_entry`: enabled iff `type_str.startswith("TD_") AND td_override is True`

When disabled, entry text shown grayed via `state="disabled"`.

### 5.6 save_data / load_session

Archive: `tg=` (NaN-filled `(2,)` array if None), `tend=` (NaN if None),
`ndt=` (-1 if None), `td_override=` (bool).

`load_session` already uses try/except per-field — old v1.1 sessions load
cleanly with TD fields falling back to defaults.

### 5.7 Type combobox

Grows from 9 to 12 entries: existing 9 + `TD_SD_GI`, `TD_SS_GI`, `TD_DS_GI`.

## 6. Tests

### 6.1 New MATLAB fixtures

Extend `tests/fixtures/generate_fixtures.m` with:

```matlab
%% v1.2 fixtures — TD_*_GI
combos_tend = 10e3;   % ps
combos_ndt  = 10e3;
combos_tg   = [1000; 2000];  % ps (1-2 ns gate)

td_specs = {
    'td_sd_gi', 'TD_SD_GI', sd_rs, sd_rd;
    'td_ss_gi', 'TD_SS_GI', ss_rs, ss_rd;
    'td_ds_gi', 'TD_DS_GI', ds_rs, ds_rd;
};

for k = 1:size(td_specs, 1)
    name = td_specs{k,1}; type_str = td_specs{k,2};
    rs = td_specs{k,3};   rd = td_specs{k,4};
    [S, params, Svox] = makeS(type_str, rs, rd, opt_prop, ...
        'xl', combos_xl, 'yl', combos_yl, 'zl', combos_zl, ...
        'dr', combos_dr, 'pert', combos_pert, ...
        'tg', combos_tg, 'tend', combos_tend, 'ndt', combos_ndt);
    x = params.x; y = params.y; z = params.z;
    xl = combos_xl; yl = combos_yl; zl = combos_zl;
    dr = combos_dr; pert = combos_pert;
    tg_ps = combos_tg; tend_ps = combos_tend; ndt = combos_ndt;
    save(fullfile(here, [name, '.mat']), ...
        'nin','nout','musp','mua', 'dr','pert','xl','yl','zl', ...
        'rs','rd', 'tg_ps','tend_ps','ndt', ...
        'x','y','z','Svox','S','type_str');
    fprintf('Wrote %s.mat\n', name);
end
```

### 6.2 conftest.py

Add 3 named fixtures: `td_sd_gi_fixture`, `td_ss_gi_fixture`, `td_ds_gi_fixture`,
loading the three `.mat` files via the existing `_make_combo_fixture` factory.
Re-shape `S` to `(Nx, 1, Nz)` (existing fixture y-axis re-inflation logic
already handles this).

### 6.3 Expected new test count

- 3 TD GI compute regression tests
- 1 each for `temporal_reflectance`, `temporal_fluence`, `temporal_gate_tot_path_len`,
  `temporal_gate_part_path_len` (4 physics tests)
- 1 GUI test asserting the override checkbox toggles entry state
- 1 GUI test asserting tg-entry enable matches `_GI` types

Total: ~9 new tests, bringing the suite to ~83 tests.

## 7. Atomic task list

| # | Task | Layer | LoC est. |
|---|---|---|---|
| 1 | Generate TD MATLAB fixtures (extend `generate_fixtures.m`, run in MATLAB) | fixtures | ~30 |
| 2 | Port `temporal_reflectance` + `temporal_fluence` + tests | physics | ~60 |
| 3 | Port `temporal_gate_tot_path_len` + test | physics | ~40 |
| 4 | Port `temporal_gate_part_path_len` (FFT branch) + test | physics | ~70 |
| 5 | Wire `("TD","GI")` into `_PHYSICS_DISPATCH`; add tg/tend/ndt kwargs + validation; SensitivityResult fields; 3 regression tests | compute | ~80 |
| 6 | GUI form fields (tg/tend/ndt + override checkbox) + state-driven enable + ns→ps boundary | gui | ~120 |
| 7 | save_data + load_session for new fields | gui | ~20 |
| 8 | Update README, CLAUDE.md, pyproject.toml, `__init__.py` → 1.2.0 | docs | ~30 |
| 9 | Code-reviewer agent on full diff vs master; fix issues | review | — |
| 10 | Push branch + open PR | git | — |

Each task: failing test first → implement → verify pass → commit. Code-reviewer
checkpoints after task 4 (physics complete) and task 7 (full feature complete).

## 8. Migration notes

Old v1.1 `last_session.json` files load cleanly:
- New fields (`tg`, `tend`, `ndt`, `td_override`) absent → fall back to
  `_DEFAULTS` via the existing per-field try/except pattern.

Old v1.1 fixtures and tests untouched.

## 9. Out of scope (notes for v1.3)

- `TD_*_DGI` adds a second early gate `tgE`; would extend tg widget to a
  second row when `_DGI` selected.
- `TD_*_T` and `TD_*_V` use `temporalKthMom*` and `temporalVar*` —
  Y_fn returns `temporalKthMoment(...)` and `temporalVar(...)` resp. instead
  of 1.0. Combinator math reuses `_combine_ss/ds` exactly.
- `simTyp='MC'` backend bringing in `pmcx` — already a future extension point.
