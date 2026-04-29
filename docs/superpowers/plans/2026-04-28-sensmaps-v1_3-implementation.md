# sensmaps v1.3 — TD `DGI` / `T` / `V`

**Status:** ready for implementation
**Author:** Giles + Claude (Opus 4.7)
**Date:** 2026-04-28
**Branch:** `v1.3-dev` (direct branch, no worktree)

This is a *combined spec + plan*. v1.3 completes the ~30-combo MATLAB
compendium table by adding the remaining TD data types. Process matches
v1.2 (combined doc, code-reviewer at physics/compute/GUI checkpoints).

---

## 1. Scope

Add **7 measurement types** to the dispatch:

| Arrangement | DGI | T | V |
|---|---|---|---|
| SD | TD_SD_DGI | TD_SD_T | TD_SD_V |
| SS | (deferred) | TD_SS_T | TD_SS_V |
| DS | (deferred) | TD_DS_T | TD_DS_V |

After v1.3 the combobox grows from 12 → **19 entries**.

**SS/DS DGI deferred** to a later release because MATLAB `makeS.m`
restricts DGI to SD (`'DGI only supported for SD at this time'`,
`makeS.m:311`). Lifting this would mean computing reference fixtures
without a MATLAB ground truth.

**Explicitly out of scope:**
- Monte-Carlo backend (`simTyp='MC'`) — v3.
- Multi-gate (>2) DGI — paper only specifies pair (early/late).
- Three-slice projection / SNR thresholding — v1.4.

**Paper anchors** (`/home/giles/TuftsBox_local/.../JIOHS2023_senRev.pdf`):
- §3.2.3.1 Gated Intensity — Eq. 7: `t̄ = (t_start + t_end)/2`.
- Differences in Gated Intensity — Eq. 9-10: `(t̄)_gates`, `Δ(t̄)_gates`;
  paper restricts both gates to the same `Δt` (we do not enforce this — it
  is a user-side modeling choice; see GUI section).
- Variance section — `σ² = ⟨t²⟩ − ⟨t⟩²`.

**MATLAB anchors** (`DOIT-Public/SensitivityCompendium/deps/`):
- `temporalKthMoment.m` (closed-form, k∈{1,2,3,4})
- `temporalKthMomTotPathLen.m` (closed-form, two-moment combination)
- `temporalKthMomPartPathLen.m` (FFT-conv, k=1 or 2 used here)
- `temporalVar.m`, `temporalVarTotPathLen.m`, `temporalVarPartPathLen.m`
  (wrappers over KthMom for k=1 and k=2)
- DGI reuses `temporalGate*PathLen.m` (already ported in v1.2) — no new physics.

---

## 2. Physics layer (`physics.py`)

Six new functions. Internal time unit **picoseconds (ps)** (matches v1.2 / MATLAB).

### 2.1 `temporal_kth_moment(rs, rd, k, opt_prop)`

Port of `temporalKthMoment.m`. Closed-form analytic; returns ⟨tᵏ⟩ (ps^k).
Switch on k ∈ {1, 2, 3, 4} mirroring MATLAB `switch k`. Raises `ValueError`
for k>4 or k<1 (MATLAB error).

Same source-image trick as `complex_reflectance` (z0, zb, r1, r2, mueff).
Reuses `continuous_reflectance` and `n2A` from existing physics module.

### 2.2 `temporal_kth_mom_tot_path_len(rs, rd, k, opt_prop)`

Port of `temporalKthMomTotPathLen.m`. Closed-form; no convolution.

```
L = -(v · (t1Mom · tkMom − tk+1Mom)) / tkMom
```

Three calls to `temporal_kth_moment` (k=1, k, k+1).

### 2.3 `temporal_kth_mom_part_path_len(rs, r, rd, V, k, opt_prop, *, conv_t=10000.0, conv_dt=1.0)`

Port of `temporalKthMomPartPathLen.m` — **FFT-conv branch only** (lines 99-117).
Mirrors v1.2's `temporal_gate_part_path_len` structure: positive-t mask, rfft
+ workers + compact-form FFT, no parfor.

```
l = -(lC · tkMom − (V/RC) · trapz_t(t^k · conv(PHI, R))) / tkMom
```

Per-voxel; returns `(Nr,)` array. Reuses `continuous_part_path_len` and
`continuous_reflectance` from existing module.

**Perf-cap subtlety:** unlike GI we cannot trivially cap `conv_t` at a gate
boundary, because `t^k · conv(PHI, R)` integrates over the full positive-t
window. We keep `conv_t=10000 ps` default but apply rfft + workers + compact
form. Initial timing target: ≤2× the GI partial path length runtime.

### 2.4 `temporal_var(rs, rd, opt_prop)`

Port of `temporalVar.m`. Returns scalar `⟨t²⟩ − ⟨t⟩²` (ps²).

### 2.5 `temporal_var_tot_path_len(rs, rd, opt_prop)`

Port of `temporalVarTotPathLen.m`. Two `temporal_kth_moment` calls + two
`temporal_kth_mom_tot_path_len` calls:

```
L = (t2Mom · L_t2 − 2 · t1Mom² · L_t1) / V
```

### 2.6 `temporal_var_part_path_len(rs, r, rd, V, opt_prop, *, conv_t=10000.0, conv_dt=1.0)`

Port of `temporalVarPartPathLen.m`. Two `temporal_kth_mom_part_path_len`
calls (k=1 and k=2), combined as:

```
l = (t2Mom · l_t2 − 2 · t1Mom² · l_t1) / V_var
l[isnan] = 0
```

NaN handling mirrors MATLAB line 65.

### 2.7 Tests (`tests/test_physics.py`)

Six new direct-port tests against MATLAB scalars / per-voxel arrays from
`td_physics_refs.mat` (extended). Tolerances:
- Closed-form moments / Var / `*_tot_path_len`: `rtol=1e-10`.
- FFT-conv `*_part_path_len` (KthMom k=1/2, Var): `rtol=1e-8`.

---

## 3. Compute layer (`compute.py`)

### 3.1 New dispatch entries

Three new rows in `_PHYSICS_DISPATCH`:

```python
("TD", "T"): (
    lambda rs_i, rd_i, op, **_:
        temporal_kth_mom_tot_path_len(rs_i, rd_i, 1, op),
    lambda rs_i, r_all, rd_i, V, op, conv_t, conv_dt, **_:
        temporal_kth_mom_part_path_len(rs_i, r_all, rd_i, V, 1, op,
                                       conv_t=conv_t, conv_dt=conv_dt),
    lambda rs_i, rd_i, op, **_:                              # Y_fn = ⟨t⟩
        temporal_kth_moment(rs_i, rd_i, 1, op),
),
("TD", "V"): (
    lambda rs_i, rd_i, op, **_:
        temporal_var_tot_path_len(rs_i, rd_i, op),
    lambda rs_i, r_all, rd_i, V, op, conv_t, conv_dt, **_:
        temporal_var_part_path_len(rs_i, r_all, rd_i, V, op,
                                   conv_t=conv_t, conv_dt=conv_dt),
    lambda rs_i, rd_i, op, **_:                              # Y_fn = σ²
        temporal_var(rs_i, rd_i, op),
),
("TD", "DGI"): (
    lambda rs_i, rd_i, op, tg, tg2, conv_t, conv_dt, **_:
        temporal_gate_tot_path_len(rs_i, rd_i, tg2, op,
                                   conv_t=conv_t, conv_dt=conv_dt)
        - temporal_gate_tot_path_len(rs_i, rd_i, tg, op,
                                     conv_t=conv_t, conv_dt=conv_dt),
    lambda rs_i, r_all, rd_i, V, op, tg, tg2, conv_t, conv_dt, **_:
        temporal_gate_part_path_len(rs_i, r_all, rd_i, V, tg2, op,
                                    conv_t=conv_t, conv_dt=conv_dt)
        - temporal_gate_part_path_len(rs_i, r_all, rd_i, V, tg, op,
                                      conv_t=conv_t, conv_dt=conv_dt),
    lambda *_a, **_kw: 1.0,
),
```

DGI's L/ll lambdas explicitly take both `tg` and `tg2`. Existing entries
(GI, FD, CW) absorb the new `tg2` via `**_`.

### 3.2 `make_s_full` signature additions

```python
def make_s_full(..., tg=None, tend=None, ndt=None, tg2=None, ...):
```

`tg2` is kw-only and only consumed by DGI. The `extra` kwargs dict gains:
- For TD GI: `tg`, `conv_t`, `conv_dt` (unchanged from v1.2)
- For TD DGI: `tg`, `tg2`, `conv_t`, `conv_dt`
- For TD T: `conv_t`, `conv_dt`
- For TD V: `conv_t`, `conv_dt`

### 3.3 Validation

Extend the TD validation block:
- DGI requires both `tg` (2-tuple of floats) and `tg2` (2-tuple of floats).
- T and V require neither `tg` nor `tg2` (both must be None or are ignored).
- GI behavior unchanged.

DGI does **not** enforce `tg2[1]−tg2[0] == tg[1]−tg[0]` — paper's
"same Δt" constraint is exposed to the user but not policed.

### 3.4 `SensitivityResult` dataclass

Add `tg2: tuple[float, float] | None = None` (kw-only, like `tg`).

### 3.5 Tests (`tests/test_compute.py`)

- One end-to-end fixture comparison per arrangement × data-type (9 tests).
  Fixtures generated by extending `tests/fixtures/generate_fixtures.m`.
- Update existing `test_physics_dispatch_missing_pair_raises_via_make_s`
  if it relied on a now-supported pair.

---

## 4. GUI layer (`gui.py`)

### 4.1 `_VALID_TYPES`

Grow from 12 to 19 entries (add 7 TD combos: 1 DGI + 3 T + 3 V).

### 4.2 New `tg2` form field

- **Label:** `tg2 (early gate) [start; end] (ns)` — convention: `tg` is
  the late gate, `tg2` is the early gate (matches paper's `Δ(t̄)_gates =
  t̄ − t̄_early`, so the *difference* is `tg − tg2`).
- **Default:** `_DEFAULTS["tg2"] = [[0.5, 1.5]]` (early), with existing
  `tg` default kept at `[[1.0, 2.0]]` (late).
- **Parser:** reuse `_parse_float_matrix` with `n=2`.

### 4.3 State-driven enables (in `_on_var_changed`)

Replace the v1.2 `tg` enable line and add `tg2`:

```python
ends_gi   = type_str.endswith("_GI")    # GI only
ends_dgi  = type_str.endswith("_DGI")   # DGI only
self._tg_entry.config(state="normal" if (ends_gi or ends_dgi) else "disabled")
self._tg2_entry.config(state="normal" if ends_dgi else "disabled")
```

T and V: both `tg` and `tg2` greyed.
GI: `tg` enabled, `tg2` greyed (unchanged from v1.2).
DGI: both enabled.

### 4.4 ns→ps boundary in `recalculate`

```python
tg2_ps = None
if type_str.endswith("_DGI"):
    tg2_ps = np.asarray(values["tg2"][0], dtype=float) * 1000.0
# pass tg2=tg2_ps to make_s_full
```

`tg_ps` continues to be set whenever `_GI` *or* `_DGI`.

### 4.5 Session save/load

Add `tg2` to `save_data` (with NaN sentinel when not applicable). The
existing per-field try/except in `set_values` handles forward/backward
compatibility.

### 4.6 Tests (`tests/test_gui.py`)

- Combobox count: 21 (was 12).
- New TD recalculate test for DGI / T / V: assert `make_s_full` is called
  with `tg2` populated for DGI and `None` for T/V.
- `tg2` widget enable/disable matrix per type.

---

## 5. Fixture generation (`tests/fixtures/generate_fixtures.m`)

Extend the `td_specs` block to cover 9 new combos. New entries in
`td_physics_refs.mat`:

- `td_kth_mom_t1` (scalar, k=1) and `td_kth_mom_t2` (scalar, k=2)
- `td_kth_mom_tot_path_len_t1` (scalar)
- `td_kth_mom_part_path_len_t1` (Nr-vector, single test point cluster)
- `td_var` (scalar)
- `td_var_tot_path_len` (scalar)
- `td_var_part_path_len` (Nr-vector)

Reuse the existing `td_rs`, `td_rd`, `td_r_test`, `td_op` fixtures.

DGI compute fixtures: `td_specs` adds rows with both `tg` and `tg2` set.

`tests/conftest.py` extends `_COMBO_NAMES` from 3 to 12 names; adds new
keys to `td_physics_refs` session fixture.

---

## 6. Documentation

- `README.md`: bump status block to v1.3; control table adds `tg2` row;
  remove "v1.3" line from roadmap.
- `CLAUDE.md`: extend the TD-architecture note to mention DGI/T/V dispatch
  rows. Note that KthMom moments are closed-form (no convolution), unlike
  Gate.
- `pyproject.toml` + `__init__.py`: bump to `1.3.0`.

---

## 7. Atomic task list (TDD; ~12 tasks)

1. **Fixtures** — extend `generate_fixtures.m`; regenerate `.mat`; update
   conftest. *Verify*: existing tests still pass.
2. **`temporal_kth_moment`** — write test (closed-form), implement, pass.
3. **`temporal_kth_mom_tot_path_len`** — test, implement, pass.
4. **`temporal_kth_mom_part_path_len`** — test (FFT-conv), implement
   with rfft + workers + compact form, pass.
5. **`temporal_var` + `temporal_var_tot_path_len`** — combined task; test
   both, implement (thin wrappers), pass.
6. **`temporal_var_part_path_len`** — test, implement, pass.
   **Checkpoint: code-reviewer subagent on physics layer.**
7. **Dispatch entries `("TD", "T")` + `("TD", "V")`** — wire compute,
   `extra` dict, validation. End-to-end tests for 6 combos pass.
8. **Dispatch entry `("TD", "DGI")`** — gate-difference combine, `tg2`
   plumbing. End-to-end tests for 3 combos pass.
   **Checkpoint: code-reviewer on compute layer.**
9. **GUI form field `tg2`** + state-driven enables; widget tests pass.
10. **GUI ns→ps for `tg2`**; recalculate tests for DGI/T/V pass.
    **Checkpoint: code-reviewer on GUI layer.**
11. **Docs** — README, CLAUDE, version bump (`1.3.0`).
12. **Release** — merge PR, tag `v1.3.0`, push tag.

Each task = single commit on `v1.3-dev`. Smoke test (`pytest`,
`sensmaps --smoke-test`) green at every commit.

---

## 8. Open questions / risks

- **DGI sign convention.** Paper defines `Δ(t̄)_gates = t̄ − t̄_early`,
  i.e. late minus early. Plan implements `L_DGI = L(tg2_late) − L(tg2_early)`
  with `tg2 = early`, `tg = late`. If MATLAB uses the opposite convention,
  the fixture comparison will catch it on task 8.
- **Var-tot path length sign.** MATLAB formula has a `−` baked in via the
  KthMom-tot path length (which itself has a leading `−`); plan transcribes
  literally. Fixture compare on task 5 confirms.
- **Perf for `T` / `V`.** Each Var partial path length costs **2** FFT
  convolutions (k=1 and k=2). With v1.2 perf tricks the per-voxel cost
  should still be ≪1 s for typical grids; if not, evaluate caching the
  PHI/R FFTs across the two k-calls (premature optimization for now).
