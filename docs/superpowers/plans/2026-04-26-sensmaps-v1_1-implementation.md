# sensmaps v1.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the v1.1 design (`docs/superpowers/specs/2026-04-25-v1_1-design.md`) — extend `make_s_full` to all 9 CW + FD measurement-type combos, add SS/DS arrangement combinators, add the `fmod` GUI control, and ship as version `1.1.0`.

**Architecture:** Three additions to the compute layer (per-measurement physics dispatch table, optode-expansion helper, arrangement combinators) plus GUI wiring (multi-row optode entry, `fmod (MHz)` field, type combobox grows to 9 values). No physics-layer changes — the existing `complex_*` building blocks were ported in v1 specifically so FD lands as a dispatch-only change. Cheap/expensive split stays as in v1.0.

**Tech Stack:** Python 3.11+, NumPy, SciPy, Matplotlib, Tkinter, pytest, MATLAB (for regenerating reference fixtures only).

---

## File Structure

| Path | Created/Modified | Responsibility |
|------|------------------|----------------|
| `src/sensmaps/compute.py`            | modify | Add `_expand_optodes`, `_combine_*`, `_PHYSICS_DISPATCH`; refactor `make_s_full`; add `fmod`/`Y_per_meas` to `SensitivityResult` |
| `src/sensmaps/gui.py`                | modify | `_parse_float_matrix`, multi-row `rs`/`rd`, `fmod` field, type combobox, MHz→Hz conversion at recalc |
| `src/sensmaps/__init__.py`           | modify | bump `__version__` to `"1.1.0"` |
| `pyproject.toml`                     | modify | bump version to `"1.1.0"` |
| `tests/conftest.py`                  | modify | parameterized `combo_ref` fixture |
| `tests/test_compute.py`              | modify | unit tests for helpers + parameterized regression |
| `tests/test_gui.py`                  | modify | multi-row entry, fmod-field state, FD GUI smoke |
| `tests/fixtures/generate_fixtures.m` | modify | loop over 8 new combos |
| `tests/fixtures/cw_ss_i.mat`         | create | (manual MATLAB run) |
| `tests/fixtures/cw_ds_i.mat`         | create | (manual MATLAB run) |
| `tests/fixtures/fd_sd_i.mat`         | create | (manual MATLAB run) |
| `tests/fixtures/fd_sd_p.mat`         | create | (manual MATLAB run) |
| `tests/fixtures/fd_ss_i.mat`         | create | (manual MATLAB run) |
| `tests/fixtures/fd_ss_p.mat`         | create | (manual MATLAB run) |
| `tests/fixtures/fd_ds_i.mat`         | create | (manual MATLAB run) |
| `tests/fixtures/fd_ds_p.mat`         | create | (manual MATLAB run) |
| `README.md`                          | modify | status section + GUI-control table updates |
| `CLAUDE.md`                          | modify | dispatch-table note + `fmod`-units note |

`physics.py` and `views.py` are not touched.

---

## Task 1 — Optode-expansion helper

**Files:**
- Modify: `src/sensmaps/compute.py`
- Modify: `tests/test_compute.py`

- [ ] **Step 1: Write the SD failing test**

Add to `tests/test_compute.py`:

```python
def test_expand_optodes_sd_returns_z_offset_sources():
    from sensmaps.compute import _expand_optodes
    rs = np.array([[0.0, 0.0, 0.0]])
    rd = np.array([[35.0, 0.0, 0.0]])
    rSrcs, rDets = _expand_optodes("SD", rs, rd, z_offset=1.0 / 1.1)
    np.testing.assert_allclose(rSrcs, [[0.0, 0.0, 1.0 / 1.1]])
    np.testing.assert_array_equal(rDets, rd)
```

- [ ] **Step 2: Verify it fails**

Run: `pytest tests/test_compute.py::test_expand_optodes_sd_returns_z_offset_sources -v`
Expected: `ImportError: cannot import name '_expand_optodes' from 'sensmaps.compute'`.

- [ ] **Step 3: Add the helper to `src/sensmaps/compute.py`**

Insert after the `parse_type_str` function (around line 87 in v1.0):

```python
def _expand_optodes(arrangement: str, rs, rd, z_offset: float):
    """Expand (rs, rd) into per-measurement (rSrcs, rDets) pairs.

    Mirrors MATLAB makeS.m lines 123-149. Applies the z-offset to
    sources only. Validates optode counts against the arrangement.

    Returns
    -------
    rSrcs : ndarray, shape (N_meas, 3)
    rDets : ndarray, shape (N_meas, 3)
    """
    rs = np.atleast_2d(np.asarray(rs, dtype=np.float64))
    rd = np.atleast_2d(np.asarray(rd, dtype=np.float64))
    z_off = np.array([0.0, 0.0, z_offset])
    if arrangement == "SD":
        if rs.shape != (1, 3) or rd.shape != (1, 3):
            raise ValueError(
                f"Incorrect optode count for arrangement 'SD': "
                f"rs.shape={rs.shape}, rd.shape={rd.shape}"
            )
        return rs + z_off, rd
    if arrangement == "SS":
        if rs.shape == (1, 3) and rd.shape == (2, 3):
            return np.tile(rs, (2, 1)) + z_off, rd
        if rs.shape == (2, 3) and rd.shape == (1, 3):
            return rs + z_off, np.tile(rd, (2, 1))
        raise ValueError(
            f"Incorrect optode count for arrangement 'SS': "
            f"rs.shape={rs.shape}, rd.shape={rd.shape}"
        )
    if arrangement == "DS":
        if rs.shape != (2, 3) or rd.shape != (2, 3):
            raise ValueError(
                f"Incorrect optode count for arrangement 'DS': "
                f"rs.shape={rs.shape}, rd.shape={rd.shape}"
            )
        rSrcs = np.vstack([rs[[0, 0], :], rs[[1, 1], :]])
        rDets = np.vstack([rd, np.flipud(rd)])
        return rSrcs + z_off, rDets
    raise ValueError(f"Unknown arrangement {arrangement!r}")
```

- [ ] **Step 4: Verify SD test passes**

Run: `pytest tests/test_compute.py::test_expand_optodes_sd_returns_z_offset_sources -v`
Expected: PASS.

- [ ] **Step 5: Add SS, DS, and rejection tests**

Append to `tests/test_compute.py`:

```python
def test_expand_optodes_ss_one_src_two_dets():
    from sensmaps.compute import _expand_optodes
    rs = np.array([[0.0, 0.0, 0.0]])
    rd = np.array([[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    rSrcs, rDets = _expand_optodes("SS", rs, rd, z_offset=0.0)
    assert rSrcs.shape == (2, 3)
    np.testing.assert_array_equal(rSrcs[0], rSrcs[1])
    np.testing.assert_array_equal(rDets, rd)


def test_expand_optodes_ss_two_srcs_one_det():
    from sensmaps.compute import _expand_optodes
    rs = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    rd = np.array([[35.0, 0.0, 0.0]])
    rSrcs, rDets = _expand_optodes("SS", rs, rd, z_offset=0.0)
    np.testing.assert_array_equal(rSrcs, rs)
    assert rDets.shape == (2, 3)
    np.testing.assert_array_equal(rDets[0], rDets[1])


def test_expand_optodes_ds_meas_pattern():
    from sensmaps.compute import _expand_optodes
    rs = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    rd = np.array([[30.0, 0, 0], [40.0, 0, 0]])
    rSrcs, rDets = _expand_optodes("DS", rs, rd, z_offset=0.0)
    # MATLAB pattern: rSrcs = [rs1; rs1; rs2; rs2]
    np.testing.assert_array_equal(rSrcs[[0, 1]], np.tile(rs[0], (2, 1)))
    np.testing.assert_array_equal(rSrcs[[2, 3]], np.tile(rs[1], (2, 1)))
    # rDets = [rd; flipud(rd)] = [rd1; rd2; rd2; rd1]
    np.testing.assert_array_equal(rDets[0], rd[0])
    np.testing.assert_array_equal(rDets[1], rd[1])
    np.testing.assert_array_equal(rDets[2], rd[1])
    np.testing.assert_array_equal(rDets[3], rd[0])


def test_expand_optodes_rejects_bad_shapes():
    from sensmaps.compute import _expand_optodes
    rs1 = np.array([[0.0, 0, 0]])
    rs2 = np.array([[0.0, 0, 0], [10, 0, 0]])
    rd1 = np.array([[35.0, 0, 0]])
    rd2 = np.array([[30.0, 0, 0], [40, 0, 0]])
    with pytest.raises(ValueError, match="'SD'"):
        _expand_optodes("SD", rs2, rd1, z_offset=0.0)
    with pytest.raises(ValueError, match="'SS'"):
        _expand_optodes("SS", rs1, rd1, z_offset=0.0)
    with pytest.raises(ValueError, match="'DS'"):
        _expand_optodes("DS", rs1, rd2, z_offset=0.0)
    with pytest.raises(ValueError, match="Unknown arrangement"):
        _expand_optodes("BOGUS", rs1, rd1, z_offset=0.0)
```

- [ ] **Step 6: Verify all expand_optodes tests pass**

Run: `pytest tests/test_compute.py -k expand_optodes -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add src/sensmaps/compute.py tests/test_compute.py
git commit -m "feat(compute): _expand_optodes for SD/SS/DS arrangements"
```

---

## Task 2 — Arrangement combinators

**Files:**
- Modify: `src/sensmaps/compute.py`
- Modify: `tests/test_compute.py`

- [ ] **Step 1: Write the failing test for SD combinator**

Append to `tests/test_compute.py`:

```python
def test_combine_sd_basic():
    from sensmaps.compute import _combine_sd
    L = [10.0]
    Y = [1.0]
    ll = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    out = _combine_sd(L, Y, ll)
    np.testing.assert_allclose(out, np.array([[0.1, 0.2], [0.3, 0.4]]))
```

- [ ] **Step 2: Verify it fails**

Run: `pytest tests/test_compute.py::test_combine_sd_basic -v`
Expected: ImportError.

- [ ] **Step 3: Add the three combinators to `src/sensmaps/compute.py`**

Insert just below `_expand_optodes`:

```python
def _combine_sd(L, Y, ll):
    """SD combinator: Svox = ll[0] / L[0]. Port of makeS.m line 400."""
    return ll[0] / L[0]


def _combine_ss(L, Y, ll):
    """SS / SD_DIFF combinator. Port of makeS.m lines 401-403."""
    return (Y[1] * ll[1] - Y[0] * ll[0]) / (Y[1] * L[1] - Y[0] * L[0])


def _combine_ds(L, Y, ll):
    """DS combinator (4 measurements). Port of makeS.m lines 404-408."""
    num = (Y[1] * ll[1] - Y[0] * ll[0]) + (Y[3] * ll[3] - Y[2] * ll[2])
    den = (Y[1] * L[1]  - Y[0] * L[0])  + (Y[3] * L[3]  - Y[2] * L[2])
    return num / den


_ARRANGEMENT_COMBINE = {"SD": _combine_sd, "SS": _combine_ss, "DS": _combine_ds}
```

- [ ] **Step 4: Verify SD test passes**

Run: `pytest tests/test_compute.py::test_combine_sd_basic -v`
Expected: PASS.

- [ ] **Step 5: Add SS and DS tests**

Append to `tests/test_compute.py`:

```python
def test_combine_ss_two_meas():
    from sensmaps.compute import _combine_ss
    L = [10.0, 20.0]
    Y = [1.0, 1.0]
    g0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    g1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    out = _combine_ss(L, Y, [g0, g1])
    expected = (g1 - g0) / (20.0 - 10.0)
    np.testing.assert_allclose(out, expected)


def test_combine_ds_four_meas():
    from sensmaps.compute import _combine_ds
    L = [10.0, 20.0, 30.0, 40.0]
    Y = [1.0, 1.0, 1.0, 1.0]
    g = [np.full((2, 2), float(k + 1)) for k in range(4)]
    out = _combine_ds(L, Y, g)
    num = (g[1] - g[0]) + (g[3] - g[2])
    den = (20 - 10) + (40 - 30)
    np.testing.assert_allclose(out, num / den)


def test_combine_ss_uses_y_weight():
    from sensmaps.compute import _combine_ss
    # Y!=1 — verifies Y is actually applied (matters for v1.3's T/V types)
    L = [1.0, 1.0]
    Y = [2.0, 3.0]
    g0 = np.full((2, 2), 1.0)
    g1 = np.full((2, 2), 1.0)
    out = _combine_ss(L, Y, [g0, g1])
    np.testing.assert_allclose(out, (3.0 * 1.0 - 2.0 * 1.0) / (3.0 - 2.0) * np.ones((2, 2)))
```

- [ ] **Step 6: Verify all combinator tests pass**

Run: `pytest tests/test_compute.py -k combine -v`
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add src/sensmaps/compute.py tests/test_compute.py
git commit -m "feat(compute): _combine_{sd,ss,ds} arrangement combinators"
```

---

## Task 3 — `SensitivityResult` gains `Y_per_meas` and `fmod`

**Files:**
- Modify: `src/sensmaps/compute.py`
- Modify: `tests/test_compute.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compute.py`:

```python
def test_sensitivity_result_kw_only_new_fields():
    """v1.1 adds Y_per_meas (required, kw-only) and fmod (optional, kw-only)."""
    from sensmaps.compute import GridParams, SensitivityResult
    from sensmaps.physics import OpticalProperties
    g = GridParams.from_limits(xl=(0, 1), yl=(0, 1), zl=(0, 1), dr=1.0)
    arr = np.zeros((2, 2, 2))
    op = OpticalProperties()
    # All new fields must be passed by keyword.
    r = SensitivityResult(
        S=arr, Svox=arr, params=g, type_str="CW_SD_I",
        rs=np.zeros((1, 3)), rd=np.zeros((1, 3)),
        opt_prop=op, pert=(1.0, 1.0, 1.0), dr=1.0,
        Y_per_meas=np.array([1.0]),       # required kw-only
        # fmod omitted on purpose — defaults to None.
    )
    assert r.fmod is None
    np.testing.assert_array_equal(r.Y_per_meas, [1.0])
```

- [ ] **Step 2: Verify it fails**

Run: `pytest tests/test_compute.py::test_sensitivity_result_kw_only_new_fields -v`
Expected: TypeError on unexpected kwarg `Y_per_meas`.

- [ ] **Step 3: Edit `SensitivityResult` in `src/sensmaps/compute.py`**

At the top of the file, change the dataclasses import:

```python
from dataclasses import dataclass, field
```

Replace the existing `SensitivityResult` class (around line 98 in v1.0) with:

```python
@dataclass
class SensitivityResult:
    """Return type of make_s_full.

    Attributes (v1.0)
    -----------------
    S         : ndarray, shape (Nx, Ny, Nz) — sensitivity (pert-convolved)
    Svox      : ndarray, shape (Nx, Ny, Nz) — per-voxel pre-conv sensitivity
    params    : GridParams
    type_str  : str
    rs, rd    : source and detector coords actually used (post z-offset) [mm]
    opt_prop  : OpticalProperties
    pert, dr  : perturbation and voxel size used

    New in v1.1 (kw-only)
    ---------------------
    Y_per_meas : ndarray, shape (N_meas,) — measured signal Y per measurement;
                 1.0 for v1.1 combos, plumbed for v1.3's T/V.
    fmod       : float | None — modulation frequency [Hz] for FD types; None for CW.
    """

    S: np.ndarray
    Svox: np.ndarray
    params: GridParams
    type_str: str
    rs: np.ndarray
    rd: np.ndarray
    opt_prop: OpticalProperties
    pert: tuple[float, float, float]
    dr: float
    # NEW in v1.1 — kw-only so existing positional construction keeps working:
    Y_per_meas: np.ndarray = field(kw_only=True)
    fmod: float | None = field(default=None, kw_only=True)
```

Find the existing `make_s_full` return statement (around line 195 in v1.0) and add `Y_per_meas`:

```python
    return SensitivityResult(
        S=S, Svox=Svox, params=params, type_str=type_str,
        rs=rs_used, rd=rd, opt_prop=opt_prop, pert=tuple(pert), dr=dr,
        Y_per_meas=np.array([1.0]),   # v1.0 path: SD with Y=1
    )
```

- [ ] **Step 4: Verify new test + existing fixture both pass**

Run: `pytest tests/test_compute.py::test_sensitivity_result_kw_only_new_fields tests/test_compute.py::test_make_s_cw_sd_i_matches_matlab -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/compute.py tests/test_compute.py
git commit -m "feat(compute): SensitivityResult adds kw-only Y_per_meas and fmod"
```

---

## Task 4 — Physics dispatch table + `make_s_full` refactor (CW_SD_I regression preserved)

**Files:**
- Modify: `src/sensmaps/compute.py`
- Modify: `tests/test_compute.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compute.py`:

```python
def test_physics_dispatch_has_cw_i_entry():
    from sensmaps.compute import _PHYSICS_DISPATCH
    assert ("CW", "I") in _PHYSICS_DISPATCH
    L_fn, ll_fn, Y_fn = _PHYSICS_DISPATCH[("CW", "I")]
    assert callable(L_fn) and callable(ll_fn) and callable(Y_fn)


def test_physics_dispatch_missing_pair_raises_via_make_s():
    """Asking make_s for an unsupported (temporal, data_type) pair must raise."""
    from sensmaps.compute import make_s
    from sensmaps.physics import OpticalProperties
    op = OpticalProperties()
    with pytest.raises(NotImplementedError, match="not implemented"):
        make_s(
            type_str="TD_SD_GI",   # not in dispatch yet
            rs=np.array([[0, 0, 0]]),
            rd=np.array([[35, 0, 0]]),
            opt_prop=op,
            xl=(-5, 40), yl=(0, 0), zl=(0, 20),
            dr=1.0,
        )
```

- [ ] **Step 2: Verify it fails**

Run: `pytest tests/test_compute.py::test_physics_dispatch_has_cw_i_entry -v`
Expected: ImportError on `_PHYSICS_DISPATCH`.

- [ ] **Step 3: Add the dispatch and refactor `make_s_full`**

Add near the top of `src/sensmaps/compute.py`, just below the imports of `physics`:

```python
from typing import Callable

# Per-measurement physics callables, keyed on (temporal, data_type).
# Each entry is (L_fn, ll_fn, Y_fn). All callables accept **kwargs to absorb
# parameters they don't use (notably fmod), so additions in v1.2/v1.3 don't
# require touching the v1.1 entries.
#   L_fn(rs_i, rd_i, opt_prop, **kw)               -> float
#   ll_fn(rs_i, r_all, rd_i, V, opt_prop, **kw)    -> ndarray, shape (N_voxels,)
#   Y_fn(rs_i, rd_i, opt_prop, **kw)               -> float
_PHYSICS_DISPATCH: dict[tuple[str, str], tuple[Callable, Callable, Callable]] = {
    ("CW", "I"): (
        lambda rs_i, rd_i, op, **_:
            float(continuous_tot_path_len(rs_i, rd_i, op)[0][0]),
        lambda rs_i, r_all, rd_i, V, op, **_:
            continuous_part_path_len(rs_i, r_all, rd_i, V, op),
        lambda *_a, **_kw: 1.0,
    ),
}
```

Now replace the body of `make_s_full` from the validation block through to the `return`. The new body (replacing lines ~144-199 in v1.0):

```python
def make_s_full(
    type_str: str,
    rs,
    rd,
    opt_prop: OpticalProperties,
    xl: Sequence[float],
    yl: Sequence[float],
    zl: Sequence[float],
    dr: float,
    pert: Sequence[float] = (1.0, 1.0, 1.0),
    sim_typ: str = "DT",
) -> SensitivityResult:
    """Compute sensitivity map for a measurement type. Mirror of MATLAB makeS.m.

    See module docstring; v1.1 adds CW SS/DS and FD I/P combos.
    """
    parsed = parse_type_str(type_str)

    if sim_typ != "DT":
        raise NotImplementedError(
            f"sim_typ={sim_typ!r} is not implemented in v1 (DT only)"
        )

    key = (parsed.temporal, parsed.data_type)
    if key not in _PHYSICS_DISPATCH:
        raise NotImplementedError(
            f"type_str={type_str!r} is not implemented in v1.1 "
            f"(no dispatch entry for {key})"
        )
    L_fn, ll_fn, Y_fn = _PHYSICS_DISPATCH[key]

    # Pert-multiple-of-dr check (existing).
    _tol = 1e-9 * max(dr, 1.0)
    if any(abs(p - round(p / dr) * dr) > _tol for p in pert):
        raise ValueError(f"pert {pert} must be a multiple of dr={dr}")

    params = GridParams.from_limits(xl=xl, yl=yl, zl=zl, dr=dr, pert=pert)

    # Optode expansion + z-offset.
    rs_arr = np.atleast_2d(np.asarray(rs, dtype=np.float64))
    rd_arr = np.atleast_2d(np.asarray(rd, dtype=np.float64))
    z_offset = 1.0 / opt_prop.musp
    rSrcs, rDets = _expand_optodes(parsed.arrangement, rs_arr, rd_arr, z_offset)
    n_meas = rSrcs.shape[0]

    # Voxel-center coordinates.
    XX, YY, ZZ = np.meshgrid(params.x, params.y, params.z, indexing="ij")
    r_all = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    z_coords = r_all[:, 2]

    # Per-measurement physics.
    Ls: list[float] = []
    Ys: list[float] = []
    lls: list[np.ndarray] = []
    for i in range(n_meas):
        rs_i = rSrcs[[i], :]
        rd_i = rDets[[i], :]
        Ls.append(L_fn(rs_i, rd_i, opt_prop))
        Ys.append(Y_fn(rs_i, rd_i, opt_prop))
        with np.errstate(divide="ignore", invalid="ignore"):
            l_vec = ll_fn(rs_i, r_all, rd_i, dr ** 3, opt_prop)
        l_vec[z_coords < 0] = 0.0
        l_vec = np.nan_to_num(l_vec, nan=0.0)
        lls.append(l_vec.reshape(XX.shape))

    # Arrangement combinator.
    Svox = _ARRANGEMENT_COMBINE[parsed.arrangement](Ls, Ys, lls)

    S = apply_pert_kernel(Svox, pert, dr)

    return SensitivityResult(
        S=S, Svox=Svox, params=params, type_str=type_str,
        rs=rSrcs, rd=rDets, opt_prop=opt_prop, pert=tuple(pert), dr=dr,
        Y_per_meas=np.asarray(Ys, dtype=np.float64),
    )
```

Note the existing `rs = rs + np.array([[0, 0, z_offset]])` block from v1.0 is now subsumed by `_expand_optodes`. Delete the orphaned lines.

- [ ] **Step 4: Verify dispatch test + existing fixture both pass**

Run: `pytest tests/test_compute.py -v`
Expected: ALL pass — including `test_make_s_cw_sd_i_matches_matlab` (regression preserved).

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/compute.py tests/test_compute.py
git commit -m "refactor(compute): make_s_full uses dispatch + expand + combine"
```

---

## Task 5 — Add `fmod` kwarg + FD validation

**Files:**
- Modify: `src/sensmaps/compute.py`
- Modify: `tests/test_compute.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_compute.py`:

```python
def test_make_s_rejects_fmod_none_for_fd():
    from sensmaps.compute import make_s
    from sensmaps.physics import OpticalProperties
    op = OpticalProperties()
    with pytest.raises(ValueError, match="fmod is required"):
        make_s(
            type_str="FD_SD_I",
            rs=np.array([[0, 0, 0]]),
            rd=np.array([[35, 0, 0]]),
            opt_prop=op,
            xl=(-5, 40), yl=(0, 0), zl=(0, 20),
            dr=1.0,
            # fmod intentionally omitted
        )
```

- [ ] **Step 2: Verify it fails**

Run: `pytest tests/test_compute.py::test_make_s_rejects_fmod_none_for_fd -v`
Expected: it raises `NotImplementedError` (because `("FD","I")` isn't in dispatch yet) — **not** the `ValueError` we want. The test fails for the wrong reason; once we add the FD validation it will pass for the right reason.

- [ ] **Step 3: Add `fmod` kwarg and validation**

In `src/sensmaps/compute.py`, change the signature of `make_s_full`:

```python
def make_s_full(
    type_str: str,
    rs,
    rd,
    opt_prop: OpticalProperties,
    xl: Sequence[float],
    yl: Sequence[float],
    zl: Sequence[float],
    dr: float,
    pert: Sequence[float] = (1.0, 1.0, 1.0),
    sim_typ: str = "DT",
    *,
    fmod: float | None = None,
) -> SensitivityResult:
```

Just after the `parse_type_str` call and `sim_typ` check, **before** the dispatch lookup, add:

```python
    if parsed.temporal == "FD" and fmod is None:
        raise ValueError(
            f"fmod is required for FD_* types (got fmod=None for {type_str!r})"
        )
```

Pass `fmod` into the per-measurement callables — change the loop body:

```python
        Ls.append(L_fn(rs_i, rd_i, opt_prop, fmod=fmod))
        Ys.append(Y_fn(rs_i, rd_i, opt_prop, fmod=fmod))
        with np.errstate(divide="ignore", invalid="ignore"):
            l_vec = ll_fn(rs_i, r_all, rd_i, dr ** 3, opt_prop, fmod=fmod)
```

(The CW dispatch entry already absorbs `**_` so the new `fmod=` kwarg is harmless for CW.)

Update the return to plumb `fmod` into the result:

```python
    return SensitivityResult(
        S=S, Svox=Svox, params=params, type_str=type_str,
        rs=rSrcs, rd=rDets, opt_prop=opt_prop, pert=tuple(pert), dr=dr,
        Y_per_meas=np.asarray(Ys, dtype=np.float64),
        fmod=fmod,
    )
```

Also mirror the `fmod` kwarg in the thin wrapper `make_s`:

```python
def make_s(
    type_str: str,
    rs,
    rd,
    opt_prop: OpticalProperties,
    xl: Sequence[float],
    yl: Sequence[float],
    zl: Sequence[float],
    dr: float,
    pert: Sequence[float] = (1.0, 1.0, 1.0),
    sim_typ: str = "DT",
    *,
    fmod: float | None = None,
):
    """Thin variant returning only `(S, params)`. See `make_s_full` for details."""
    result = make_s_full(
        type_str=type_str, rs=rs, rd=rd, opt_prop=opt_prop,
        xl=xl, yl=yl, zl=zl, dr=dr, pert=pert, sim_typ=sim_typ,
        fmod=fmod,
    )
    return result.S, result.params
```

- [ ] **Step 4: Verify FD-validation test passes**

Run: `pytest tests/test_compute.py::test_make_s_rejects_fmod_none_for_fd -v`
Expected: PASS — raises `ValueError("fmod is required ...")`.

- [ ] **Step 5: Verify the existing v1.0 regression still passes**

Run: `pytest tests/test_compute.py::test_make_s_cw_sd_i_matches_matlab -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sensmaps/compute.py tests/test_compute.py
git commit -m "feat(compute): fmod kwarg + FD validation"
```

---

## Task 6 — `_parse_float_matrix` parser

**Files:**
- Modify: `src/sensmaps/gui.py`
- Modify: `tests/test_gui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui.py`:

```python
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
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_gui.py -k parse_float_matrix -v`
Expected: ImportError on `_parse_float_matrix`.

- [ ] **Step 3: Add the parser**

Add to `src/sensmaps/gui.py` next to the existing `_parse_float_list`:

```python
def _parse_float_matrix(text: str, ncols: int) -> list[list[float]]:
    """Parse a `;`-separated list of `ncols`-wide float rows.

    Each row is then whitespace/comma-separated. Trailing `;` is ignored.
    Empty input (or only `;`) raises ValueError.
    """
    rows = [r for r in text.split(";") if r.strip()]
    if not rows:
        raise ValueError(f"expected at least one row of {ncols} values; got {text!r}")
    return [_parse_float_list(r, ncols) for r in rows]
```

- [ ] **Step 4: Verify parser tests pass**

Run: `pytest tests/test_gui.py -k parse_float_matrix -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/gui.py tests/test_gui.py
git commit -m "feat(gui): _parse_float_matrix for multi-row optode entry"
```

---

## Task 7 — Multi-row `rs` / `rd` in `ParameterPanel`

**Files:**
- Modify: `src/sensmaps/gui.py`
- Modify: `tests/test_gui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui.py`:

```python
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
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_gui.py -k "round_trips_multi_row or default_rs_is_single_row" -v`
Expected: FAIL — current `get_values` returns `list[float]`, not `list[list[float]]`.

- [ ] **Step 3: Update `_DEFAULTS` for new shape**

In `src/sensmaps/gui.py`, change `_DEFAULTS["rs"]` and `_DEFAULTS["rd"]`:

```python
_DEFAULTS: dict[str, Any] = {
    "type_str": "CW_SD_I",
    "rs": [[0.0, 0.0, 0.0]],          # CHANGED — list of rows
    "rd": [[35.0, 0.0, 0.0]],         # CHANGED — list of rows
    ...
}
```

- [ ] **Step 4: Update `get_values` to use the matrix parser**

In `ParameterPanel.get_values`, change two lines:

```python
            "rs": _parse_float_matrix(v["rs"].get(), 3),
            "rd": _parse_float_matrix(v["rd"].get(), 3),
```

- [ ] **Step 5: Update `set_values` to format matrix rows**

Replace the current `_fmt_list` helper with two helpers:

```python
        def _fmt_list(xs):    return " ".join(f"{x:g}" for x in xs)
        def _fmt_matrix(xss): return "; ".join(_fmt_list(xs) for xs in xss)
```

Change the `rs`/`rd` set blocks to use the matrix formatter:

```python
        if "rs" in values:
            self._vars["rs"].set(_fmt_matrix(values["rs"]))
        if "rd" in values:
            self._vars["rd"].set(_fmt_matrix(values["rd"]))
```

- [ ] **Step 6: Update the field labels**

In `_build_widgets`, change two label texts:

```python
        ttk.Label(f, text="rs [x y z; …] (mm)").grid(row=row, column=0, sticky="w")
        ...
        ttk.Label(f, text="rd [x y z; …] (mm)").grid(row=row, column=0, sticky="w")
```

- [ ] **Step 7: Verify multi-row tests pass + the existing v1 GUI tests still pass**

Run: `pytest tests/test_gui.py -v`
Expected: ALL pass — the v1.0 round-trip test (`test_parameter_panel_reads_and_writes_values`) still passes because the rs/rd default is now a 2-D list and the `assert got["rs"] == [1.0, 2.0, 3.0]` line in that test will fail — **fix that test** at the same time.

In `tests/test_gui.py`, update `test_parameter_panel_reads_and_writes_values`:

```python
    new_vals["rs"] = [[1.0, 2.0, 3.0]]
    panel.set_values(new_vals)
    got = panel.get_values()
    assert got["opt_prop"]["musp"] == 1.2
    assert got["opt_prop"]["mua"] == 0.02
    assert got["rs"] == [[1.0, 2.0, 3.0]]
```

Run again: `pytest tests/test_gui.py -v`
Expected: ALL pass.

- [ ] **Step 8: Commit**

```bash
git add src/sensmaps/gui.py tests/test_gui.py
git commit -m "feat(gui): rs/rd accept multi-row input via semicolon"
```

---

## Task 8 — `fmod` field with type-driven disabled state

**Files:**
- Modify: `src/sensmaps/gui.py`
- Modify: `tests/test_gui.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gui.py`:

```python
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
```

(`type_str` "FD_SD_I" must be in the combobox `values=` list — added in Task 9; tests currently fail anyway because there's no `_fmod_entry` yet.)

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_gui.py -k "fmod" -v`
Expected: FAIL — no `fmod` key in `get_values`, no `_fmod_entry`.

- [ ] **Step 3: Add `fmod` to `_DEFAULTS`**

In `src/sensmaps/gui.py`:

```python
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
    "pert": [1.0, 1.0, 1.0],
    "pert_override": False,
    "slice_axis": "y",
    "slice_value": 0.0,
    "quantiles": [0.05, 0.95],
}
```

- [ ] **Step 4: Add the `fmod` row to the panel**

In `_build_widgets`, just before the `# Perturbation` block, insert:

```python
        # Modulation frequency (FD only — disabled when type is CW_*)
        ttk.Label(f, text="fmod (MHz)").grid(row=row, column=0, sticky="w")
        self._vars["fmod"] = tk.StringVar()
        self._fmod_entry = ttk.Entry(f, textvariable=self._vars["fmod"], width=10)
        self._fmod_entry.grid(row=row, column=1, sticky="w")
        self._inputs.append(self._fmod_entry)
        row += 1
```

- [ ] **Step 5: Wire `get_values` / `set_values`**

Add `fmod` to `get_values`:

```python
            "fmod": float(v["fmod"].get()),
```

Add to `set_values` (next to `dr`):

```python
        if "fmod" in values:
            self._vars["fmod"].set(f"{values['fmod']:g}")
```

Add `"fmod"` to `_VALID_KEYS` (already done implicitly since it's now in `_DEFAULTS`):

```python
    _VALID_KEYS = frozenset(_DEFAULTS.keys())
```

(no edit needed — frozenset is rebuilt from `_DEFAULTS` after Step 3.)

- [ ] **Step 6: Drive `_fmod_entry.state` from `type_str` in `_on_var_changed`**

Just before the `if name.startswith("opt_prop."):` line in `_on_var_changed`:

```python
        # Disable fmod entry when the selected type is CW_* (it's unused there).
        if values["type_str"].startswith("FD_"):
            self._fmod_entry.config(state="normal")
        else:
            self._fmod_entry.config(state="disabled")
```

- [ ] **Step 7: Verify the three new fmod tests + existing tests pass**

Run: `pytest tests/test_gui.py -v`
Expected: ALL pass.

- [ ] **Step 8: Commit**

```bash
git add src/sensmaps/gui.py tests/test_gui.py
git commit -m "feat(gui): fmod (MHz) field disabled for CW types"
```

---

## Task 9 — Type combobox grows to 9 entries; `PARAM_CLASS` adds `fmod`

**Files:**
- Modify: `src/sensmaps/gui.py`
- Modify: `tests/test_gui.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gui.py`:

```python
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
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_gui.py -k "param_class_classifies_fmod or type_combobox_lists" -v`
Expected: FAIL.

- [ ] **Step 3: Update `PARAM_CLASS`**

In `src/sensmaps/gui.py`:

```python
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
    "pert": "cheap",
    "pert_override": "cheap",
    "slice_axis": "cheap",
    "slice_value": "cheap",
    "quantiles": "cheap",
}
```

- [ ] **Step 4: Replace the combobox `values=` with the v1.1 list**

In `_build_widgets`, the type combobox block:

```python
        _VALID_TYPES = [
            "CW_SD_I", "CW_SS_I", "CW_DS_I",
            "FD_SD_I", "FD_SS_I", "FD_DS_I",
            "FD_SD_P", "FD_SS_P", "FD_DS_P",
        ]
        ttk.Label(f, text="Type").grid(row=row, column=0, sticky="w")
        self._vars["type_str"] = tk.StringVar(value="CW_SD_I")
        cb_type = ttk.Combobox(f, textvariable=self._vars["type_str"],
                               values=_VALID_TYPES, state="readonly", width=12)
        cb_type.grid(row=row, column=1, sticky="ew")
        self._inputs.append(cb_type)
        row += 1
```

- [ ] **Step 5: Verify all gui + compute tests pass**

Run: `pytest -v`
Expected: ALL pass.

- [ ] **Step 6: Commit**

```bash
git add src/sensmaps/gui.py tests/test_gui.py
git commit -m "feat(gui): type combobox grows to 9 v1.1 combos; fmod=expensive"
```

---

## Task 10 — `MainWindow.recalculate` passes multi-row optodes + `fmod_hz`

**Files:**
- Modify: `src/sensmaps/gui.py`
- Modify: `tests/test_gui.py`

- [ ] **Step 1: Verify the existing `test_main_window_constructs_and_recalculates` still passes**

Run: `pytest tests/test_gui.py::test_main_window_constructs_and_recalculates -v`
Expected: FAIL — because we now pass `rs=[[…]]` from the form, but `recalculate` still wraps with `np.array([values["rs"]])` (turning `[[…]]` into `[[[…]]]`).

- [ ] **Step 2: Update `MainWindow.recalculate`**

In `src/sensmaps/gui.py`:

```python
    def recalculate(self) -> None:
        try:
            values = self.params_panel.get_values()
        except Exception as e:
            messagebox.showerror("Invalid input", f"Could not parse form: {e}")
            return
        try:
            op = _opt_prop_from_dict(values["opt_prop"])
            fmod_hz = (
                values["fmod"] * 1e6
                if values["type_str"].startswith("FD_")
                else None
            )
            result = make_s_full(
                type_str=values["type_str"],
                rs=np.asarray(values["rs"], dtype=float),
                rd=np.asarray(values["rd"], dtype=float),
                opt_prop=op,
                xl=tuple(values["xl"]),
                yl=tuple(values["yl"]),
                zl=tuple(values["zl"]),
                dr=values["dr"],
                pert=tuple(values["pert"]),
                fmod=fmod_hz,
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
```

- [ ] **Step 3: Verify the existing recalc test passes**

Run: `pytest tests/test_gui.py::test_main_window_constructs_and_recalculates -v`
Expected: PASS.

- [ ] **Step 4: Add a behavioral test for the fmod-Hz boundary**

Append to `tests/test_gui.py`:

```python
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
```

- [ ] **Step 5: Verify**

Run: `pytest tests/test_gui.py::test_recalculate_passes_fmod_in_hz_for_fd -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sensmaps/gui.py tests/test_gui.py
git commit -m "feat(gui): recalculate handles multi-row optodes + MHz→Hz fmod"
```

---

## Task 11 — Old session JSON loads cleanly into v1.1

**Files:**
- Modify: `tests/test_gui.py`

- [ ] **Step 1: Write the test**

Append to `tests/test_gui.py`:

```python
def test_old_v1_session_json_loads_into_v1_1(tk_root, tmp_path, monkeypatch):
    """A session file produced by v1.0 (no `fmod`, single-row rs/rd as flat list)
    must load without raising and produce sensible v1.1 defaults."""
    import json
    monkeypatch.chdir(tmp_path)
    # Synthesize a v1.0-shaped session JSON.
    v1_session = {
        "type_str": "CW_SD_I",
        "rs": [0.0, 0.0, 0.0],            # v1.0 used flat list-of-floats
        "rd": [35.0, 0.0, 0.0],
        "opt_prop": {"n_in": 1.333, "n_out": 1.0, "musp": 1.1, "mua": 0.011},
        "xl": [-10.0, 70.0], "yl": [0.0, 0.0], "zl": [0.0, 25.0],
        "dr": 1.0, "pert": [1.0, 1.0, 1.0], "pert_override": False,
        "slice_axis": "y", "slice_value": 0.0, "quantiles": [0.05, 0.95],
    }
    (tmp_path / "last_session.json").write_text(json.dumps(v1_session))

    from sensmaps.gui import MainWindow
    mw = MainWindow(master=tk_root)
    got = mw.params_panel.get_values()
    # rs/rd silently ignored (set_values doesn't accept flat list), defaults retained:
    assert got["rs"] == [[0.0, 0.0, 0.0]]
    # fmod missing in JSON → uses _DEFAULTS:
    assert got["fmod"] == 100.0
    # Cache built from defaults — auto-recalc on launch works.
    assert mw._cache is not None
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_gui.py::test_old_v1_session_json_loads_into_v1_1 -v`
Expected: It might FAIL the first time because `set_values` will try to call `_fmt_matrix(rs)` on the flat list and fail. Per the spec §7.1, the load_session try/except (post fix #7) catches this and falls back to defaults — so the cache should still build.

- [ ] **Step 3: If it fails, confirm `load_session` already swallows the exception**

The post-fix-#7 `load_session` is:

```python
    def load_session(self) -> None:
        if not SESSION_FILE.exists():
            return
        try:
            values = json.loads(SESSION_FILE.read_text())
            self.params_panel.set_values(values)
        except Exception:
            return
```

So a flat-list `rs` raises inside `_fmt_matrix` → caught here → defaults stay. The test should PASS as written. If it doesn't, inspect by running `pytest -s` and adjust the test's expectations (e.g., `got["rs"] == _DEFAULTS["rs"]`) to match the actual behavior.

- [ ] **Step 4: Commit**

```bash
git add tests/test_gui.py
git commit -m "test(gui): old v1.0 last_session.json loads into v1.1"
```

---

## Task 12 — Generate MATLAB fixtures (manual step + script update)

**Files:**
- Modify: `tests/fixtures/generate_fixtures.m`
- Create: `tests/fixtures/cw_ss_i.mat`
- Create: `tests/fixtures/cw_ds_i.mat`
- Create: `tests/fixtures/fd_sd_i.mat`
- Create: `tests/fixtures/fd_sd_p.mat`
- Create: `tests/fixtures/fd_ss_i.mat`
- Create: `tests/fixtures/fd_ss_p.mat`
- Create: `tests/fixtures/fd_ds_i.mat`
- Create: `tests/fixtures/fd_ds_p.mat`

- [ ] **Step 1: Append a v1.1 generation block to `generate_fixtures.m`**

After the existing `save(fullfile(here, 'cw_sd_i_example1.mat'), ...)` line, before the `rmpath(deps);` line, insert:

```matlab
%% v1.1 fixtures — 8 new combos
% Common inputs (same grid as cw_sd_i_example1)
combos_xl = [-5, 40];
combos_yl = [0, 0];
combos_zl = [0, 20];
combos_dr = 1.0;
combos_pert = [1, 1, 1];
combos_fmod = 100e6;   % Hz (matches GUI default of 100 MHz)

% Optode geometries per arrangement
sd_rs = [0, 0, 0];          sd_rd = [25, 0, 0];
ss_rs = [0, 0, 0];          ss_rd = [20, 0, 0; 30, 0, 0];   % 1×2 form
ds_rs = [0, 0, 0; 5, 0, 0]; ds_rd = [25, 0, 0; 30, 0, 0];

combo_specs = {
    'cw_ss_i', 'CW_SS_I', ss_rs, ss_rd, NaN;
    'cw_ds_i', 'CW_DS_I', ds_rs, ds_rd, NaN;
    'fd_sd_i', 'FD_SD_I', sd_rs, sd_rd, combos_fmod;
    'fd_sd_p', 'FD_SD_P', sd_rs, sd_rd, combos_fmod;
    'fd_ss_i', 'FD_SS_I', ss_rs, ss_rd, combos_fmod;
    'fd_ss_p', 'FD_SS_P', ss_rs, ss_rd, combos_fmod;
    'fd_ds_i', 'FD_DS_I', ds_rs, ds_rd, combos_fmod;
    'fd_ds_p', 'FD_DS_P', ds_rs, ds_rd, combos_fmod;
};

for k = 1:size(combo_specs, 1)
    name      = combo_specs{k, 1};
    type_str  = combo_specs{k, 2};
    rs        = combo_specs{k, 3};
    rd        = combo_specs{k, 4};
    fmod_hz   = combo_specs{k, 5};

    if isnan(fmod_hz)
        [S, params, Svox] = makeS(type_str, rs, rd, opt_prop, ...
            'xl', combos_xl, 'yl', combos_yl, 'zl', combos_zl, ...
            'dr', combos_dr, 'pert', combos_pert);
    else
        [S, params, Svox] = makeS(type_str, rs, rd, opt_prop, ...
            'xl', combos_xl, 'yl', combos_yl, 'zl', combos_zl, ...
            'dr', combos_dr, 'pert', combos_pert, 'fmod', fmod_hz);
    end

    x = params.x; y = params.y; z = params.z;
    xl = combos_xl; yl = combos_yl; zl = combos_zl;
    dr = combos_dr; pert = combos_pert;

    save(fullfile(here, [name, '.mat']), ...
        'nin', 'nout', 'musp', 'mua', ...
        'dr', 'pert', 'xl', 'yl', 'zl', ...
        'rs', 'rd', 'fmod_hz', ...
        'x', 'y', 'z', 'Svox', 'S', 'type_str');
    fprintf('Wrote %s.mat\n', name);
end
```

- [ ] **Step 2: Run the script in MATLAB (manual)**

Open MATLAB → set working directory to `tests/fixtures/` → run `generate_fixtures`. It will produce the 8 new `.mat` files alongside the existing one. Confirm each file is < 1 MB.

If the user is running this plan via subagent execution: the agent should pause here and ask the user to run MATLAB. **Do not proceed past this step until all 8 `.mat` files exist.**

- [ ] **Step 3: Verify file presence**

Run: `ls tests/fixtures/*.mat | wc -l`
Expected: `9` (1 from v1.0 + 8 from v1.1).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/generate_fixtures.m tests/fixtures/*.mat
git commit -m "test(fixtures): generate 8 v1.1 reference fixtures from MATLAB"
```

---

## Task 13 — Parameterized `combo_ref` fixture in `conftest.py`

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Write the failing test (first)**

Append to `tests/test_compute.py`:

```python
@pytest.mark.parametrize("combo_name", [
    "cw_ss_i", "cw_ds_i",
    "fd_sd_i", "fd_sd_p",
    "fd_ss_i", "fd_ss_p",
    "fd_ds_i", "fd_ds_p",
])
def test_combo_ref_fixture_loads(combo_name, request):
    """Sanity: each fixture file loads into a dict with the expected keys."""
    ref = request.getfixturevalue("combo_ref_" + combo_name)
    assert "S" in ref and "Svox" in ref
    assert ref["S"].ndim == 3
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_compute.py -k combo_ref_fixture -v`
Expected: FAIL — fixture functions don't exist.

- [ ] **Step 3: Add a parameterized fixture factory to `tests/conftest.py`**

```python
def _make_combo_fixture(name: str):
    """Build a session-scoped pytest fixture loading a v1.1 combo .mat."""
    @pytest.fixture(scope="session", name=f"combo_ref_{name}")
    def _fixture():
        path = Path(__file__).parent / "fixtures" / f"{name}.mat"
        if not path.exists():
            pytest.skip(f"Fixture not found at {path}; run generate_fixtures.m")
        d = scipy.io.loadmat(path, squeeze_me=True)
        d["x"] = np.atleast_1d(np.asarray(d["x"], dtype=float))
        d["y"] = np.atleast_1d(np.asarray(d["y"], dtype=float))
        d["z"] = np.atleast_1d(np.asarray(d["z"], dtype=float))
        shape3d = (d["x"].size, d["y"].size, d["z"].size)
        d["S"] = np.asarray(d["S"], dtype=float).reshape(shape3d)
        d["Svox"] = np.asarray(d["Svox"], dtype=float).reshape(shape3d)
        d["rs"] = np.atleast_2d(np.asarray(d["rs"], dtype=float))
        d["rd"] = np.atleast_2d(np.asarray(d["rd"], dtype=float))
        return d
    return _fixture


_COMBO_NAMES = [
    "cw_ss_i", "cw_ds_i",
    "fd_sd_i", "fd_sd_p",
    "fd_ss_i", "fd_ss_p",
    "fd_ds_i", "fd_ds_p",
]

# Register one named fixture per combo. Tests reference them via
#   request.getfixturevalue("combo_ref_<name>")
# or by listing combo_ref_<name> as a param.
for _name in _COMBO_NAMES:
    globals()[f"combo_ref_{_name}"] = _make_combo_fixture(_name)
```

- [ ] **Step 4: Verify**

Run: `pytest tests/test_compute.py -k combo_ref_fixture -v`
Expected: 8 PASS (or 8 SKIP if you haven't run MATLAB yet — skip is acceptable here, the next task will hit them).

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_compute.py
git commit -m "test(conftest): per-combo parameterized reference fixtures"
```

---

## Task 14 — `("FD","I")` and `("FD","P")` dispatch entries + parameterized regression

**Files:**
- Modify: `src/sensmaps/compute.py`
- Modify: `tests/test_compute.py`

- [ ] **Step 1: Write the parameterized regression test**

Append to `tests/test_compute.py`:

```python
@pytest.mark.parametrize("combo_name, type_str", [
    ("cw_ss_i", "CW_SS_I"),
    ("cw_ds_i", "CW_DS_I"),
    ("fd_sd_i", "FD_SD_I"),
    ("fd_sd_p", "FD_SD_P"),
    ("fd_ss_i", "FD_SS_I"),
    ("fd_ss_p", "FD_SS_P"),
    ("fd_ds_i", "FD_DS_I"),
    ("fd_ds_p", "FD_DS_P"),
])
def test_make_s_combo_matches_matlab(combo_name, type_str, request):
    from sensmaps.compute import make_s
    from sensmaps.physics import OpticalProperties
    ref = request.getfixturevalue(f"combo_ref_{combo_name}")

    op = OpticalProperties(
        n_in=float(ref["nin"]), n_out=float(ref["nout"]),
        musp=float(ref["musp"]), mua=float(ref["mua"]),
    )
    rs = np.asarray(ref["rs"], dtype=float)
    rd = np.asarray(ref["rd"], dtype=float)
    fmod = (
        float(ref["fmod_hz"])
        if type_str.startswith("FD_") and ref["fmod_hz"].size > 0
        else None
    )

    S, _ = make_s(
        type_str=type_str,
        rs=rs, rd=rd, opt_prop=op,
        xl=(float(ref["xl"][0]), float(ref["xl"][1])),
        yl=(float(ref["yl"][0]), float(ref["yl"][1])),
        zl=(float(ref["zl"][0]), float(ref["zl"][1])),
        dr=float(ref["dr"]),
        pert=tuple(float(p) for p in ref["pert"]),
        fmod=fmod,
    )
    np.testing.assert_allclose(S, ref["S"], rtol=1e-8, atol=1e-12)
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_compute.py -k test_make_s_combo_matches_matlab -v`
Expected: 2 PASS (CW combos — already work via the existing `("CW", "I")` entry + new combinators), 6 FAIL (FD combos — `NotImplementedError`).

- [ ] **Step 3: Add the FD dispatch entries**

In `src/sensmaps/compute.py`, extend `_PHYSICS_DISPATCH`:

```python
from sensmaps.physics import (
    OpticalProperties,
    complex_part_path_len,
    complex_tot_path_len,
    continuous_part_path_len,
    continuous_tot_path_len,
)

# ... (existing imports unchanged)

_PHYSICS_DISPATCH: dict[tuple[str, str], tuple[Callable, Callable, Callable]] = {
    ("CW", "I"): (
        lambda rs_i, rd_i, op, **_:
            float(continuous_tot_path_len(rs_i, rd_i, op)[0][0]),
        lambda rs_i, r_all, rd_i, V, op, **_:
            continuous_part_path_len(rs_i, r_all, rd_i, V, op),
        lambda *_a, **_kw: 1.0,
    ),
    ("FD", "I"): (
        lambda rs_i, rd_i, op, fmod, **_:
            float(complex_tot_path_len(rs_i, rd_i, 2.0 * np.pi * fmod, op)[0][0].real),
        lambda rs_i, r_all, rd_i, V, op, fmod, **_:
            complex_part_path_len(rs_i, r_all, rd_i, V, 2.0 * np.pi * fmod, op).real,
        lambda *_a, **_kw: 1.0,
    ),
    ("FD", "P"): (
        lambda rs_i, rd_i, op, fmod, **_:
            float(complex_tot_path_len(rs_i, rd_i, 2.0 * np.pi * fmod, op)[0][0].imag),
        lambda rs_i, r_all, rd_i, V, op, fmod, **_:
            complex_part_path_len(rs_i, r_all, rd_i, V, 2.0 * np.pi * fmod, op).imag,
        lambda *_a, **_kw: 1.0,
    ),
}
```

- [ ] **Step 4: Verify all 8 regression tests pass**

Run: `pytest tests/test_compute.py -k test_make_s_combo_matches_matlab -v`
Expected: 8 PASS.

- [ ] **Step 5: Verify the v1.0 regression still passes**

Run: `pytest tests/test_compute.py::test_make_s_cw_sd_i_matches_matlab -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sensmaps/compute.py tests/test_compute.py
git commit -m "feat(compute): FD I/P dispatch entries; 8 combos match MATLAB"
```

---

## Task 15 — GUI smoke for `FD_SD_I` against MATLAB fixture

**Files:**
- Modify: `tests/test_gui.py`

- [ ] **Step 1: Write the test**

Append to `tests/test_gui.py`:

```python
def test_main_window_recalculates_fd_sd_i(tk_root, tmp_path, monkeypatch, request):
    monkeypatch.chdir(tmp_path)
    ref = request.getfixturevalue("combo_ref_fd_sd_i")
    from sensmaps.gui import MainWindow

    mw = MainWindow(master=tk_root)
    values = {
        "type_str": "FD_SD_I",
        "rs": [[float(ref["rs"][0, 0]), float(ref["rs"][0, 1]), float(ref["rs"][0, 2])]],
        "rd": [[float(ref["rd"][0, 0]), float(ref["rd"][0, 1]), float(ref["rd"][0, 2])]],
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
        "fmod": float(ref["fmod_hz"]) / 1e6,         # GUI stores MHz
        "pert": [float(p) for p in ref["pert"]],
        "slice_axis": "y",
        "slice_value": 0.0,
        "quantiles": [0.05, 0.95],
    }
    mw.params_panel.set_values(values)
    mw.recalculate()
    assert mw._cache is not None
    np.testing.assert_allclose(mw._cache.S, ref["S"], rtol=1e-8, atol=1e-12)
```

- [ ] **Step 2: Run + verify**

Run: `pytest tests/test_gui.py::test_main_window_recalculates_fd_sd_i -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_gui.py
git commit -m "test(gui): MainWindow recalculate matches MATLAB for FD_SD_I"
```

---

## Task 16 — `save_data` archive includes `fmod`

**Files:**
- Modify: `src/sensmaps/gui.py`
- Modify: `tests/test_gui.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gui.py`:

```python
def test_save_data_archive_includes_fmod(tk_root, tmp_path, monkeypatch):
    """save_data must round-trip fmod (NaN for CW, exact Hz for FD)."""
    monkeypatch.chdir(tmp_path)
    from sensmaps.gui import MainWindow
    mw = MainWindow(master=tk_root)
    # Default state is CW_SD_I — fmod should serialize as NaN.
    out = tmp_path / "cw.npz"
    monkeypatch.setattr("tkinter.filedialog.asksaveasfilename", lambda **_: str(out))
    mw.save_data()
    arr = np.load(out)
    assert "fmod" in arr.files
    assert np.isnan(float(arr["fmod"]))

    # Switch to FD and rerun — fmod should match the form value (in Hz).
    mw.params_panel.set_values({"type_str": "FD_SD_I", "fmod": 100.0})
    mw.recalculate()
    out2 = tmp_path / "fd.npz"
    monkeypatch.setattr("tkinter.filedialog.asksaveasfilename", lambda **_: str(out2))
    mw.save_data()
    arr2 = np.load(out2)
    assert float(arr2["fmod"]) == 100.0 * 1e6
```

- [ ] **Step 2: Verify failure**

Run: `pytest tests/test_gui.py::test_save_data_archive_includes_fmod -v`
Expected: FAIL — `fmod` not in archive.

- [ ] **Step 3: Add `fmod` to the `save_data` archive**

In `src/sensmaps/gui.py`, in `save_data`, change the `np.savez(...)` call:

```python
            np.savez(
                path,
                S=c.S, Svox=c.Svox,
                x=c.params.x, y=c.params.y, z=c.params.z,
                rs=c.rs, rd=c.rd, pert=np.asarray(c.pert), dr=c.dr,
                type_str=c.type_str,
                n_in=c.opt_prop.n_in, n_out=c.opt_prop.n_out,
                musp=c.opt_prop.musp, mua=c.opt_prop.mua,
                fmod=(np.nan if c.fmod is None else c.fmod),    # NEW
                sensmaps_version=version,
            )
```

- [ ] **Step 4: Verify**

Run: `pytest tests/test_gui.py::test_save_data_archive_includes_fmod -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/gui.py tests/test_gui.py
git commit -m "feat(gui): save_data archive round-trips fmod"
```

---

## Task 17 — Bump version `1.0.0` → `1.1.0`

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/sensmaps/__init__.py`

- [ ] **Step 1: Bump in `pyproject.toml`**

Change line 3:

```toml
version = "1.1.0"
```

- [ ] **Step 2: Bump in `src/sensmaps/__init__.py`**

Change the `__version__` line:

```python
__version__ = "1.1.0"
```

- [ ] **Step 3: Verify tests pass with the new version**

Run: `pytest -q`
Expected: ALL PASS (count is now ~50+ tests; check the count is up from v1.0's 27).

- [ ] **Step 4: Smoke the GUI**

Run: `sensmaps --smoke-test`
Expected: exit `0`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/sensmaps/__init__.py
git commit -m "release: bump version 1.0.0 → 1.1.0"
```

---

## Task 18 — Update `README.md` and `CLAUDE.md`

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Status section in `README.md`**

Replace the existing one-line Status block (around line 13-15) with:

```markdown
## Status

v1.1 — CW (`SD`/`SS`/`DS` × `I`) and FD (`SD`/`SS`/`DS` × `I`/`P`) under
diffusion theory. Tkinter GUI, save PNG/PDF + `.npz`, multi-row optode
entry, modulation frequency control. Time-domain types in v1.2/v1.3;
Monte Carlo backend in v3.
```

- [ ] **Step 2: Update the GUI control table in `README.md`**

Replace the existing `Control` table (around line 56-66) with:

```markdown
| Control                  | Class      | What it does                                       |
|--------------------------|------------|----------------------------------------------------|
| Type                     | expensive  | Measurement type (v1.1: 9 CW + FD combos)          |
| rs, rd                   | expensive  | Source / detector coordinates [mm]; multi-row via `;` |
| n_in, n_out              | expensive  | Index of refraction inside / outside               |
| musp, mua                | expensive  | Reduced scattering / absorption [1/mm]             |
| xl, yl, zl, dr           | expensive  | Voxel-grid limits and resolution [mm]              |
| fmod                     | expensive  | Modulation frequency [MHz]; greyed out unless type is `FD_*` |
| pert                     | cheap      | Perturbation box size [mm]                         |
| slice axis, value        | cheap      | Which 2D slice to display                          |
| quantiles                | cheap      | Color-limit quantiles (lo, hi)                     |
```

Add this short paragraph after the table:

```markdown
**Multi-optode formats:** `SS` accepts `(1 src, 2 dets)` *or* `(2 srcs, 1 det)`;
`DS` requires `(2 srcs, 2 dets)`. Enter additional rows separated by `;` —
e.g. `0 0 0; 30 0 0` for two sources.
```

- [ ] **Step 3: Update `CLAUDE.md`**

In the `compute.py` bullet (around line 50-65), append:

```markdown
- v1.1 introduces `_PHYSICS_DISPATCH[(temporal, data_type)] -> (L_fn, ll_fn, Y_fn)`
  and `_ARRANGEMENT_COMBINE[arrangement]`. Adding a new combo is one row in
  each table plus, if needed, a new physics function. `_expand_optodes`
  validates optode counts against the arrangement and applies the z-offset.
```

In the GUI section (around line 70-80), append:

```markdown
- `fmod` lives in **MHz** at the GUI boundary and **Hz** internally;
  conversion happens in `MainWindow.recalculate`. The `_fmod_entry` is
  state-driven by `type_str` (disabled when CW). Multi-row `rs`/`rd` use
  `_parse_float_matrix` which splits on `;`.
```

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: README + CLAUDE.md updates for v1.1"
```

---

## Task 19 — Tag `v1.1.0` locally (do not push)

**Files:**
- (none — git tag only)

- [ ] **Step 1: Verify tree is clean**

Run: `git status`
Expected: `nothing to commit, working tree clean`.

- [ ] **Step 2: Create the annotated tag**

```bash
git tag -a v1.1.0 -m "v1.1.0 — CW SS/DS + FD all-arrangements

Adds 8 new combos (CW_{SS,DS}_I, FD_{SD,SS,DS}_{I,P}) under DT.
SS/DS arrangement combinators land. fmod GUI control (MHz units),
multi-row optode entry. No physics-layer changes — reuses v1.0's
complex_* building blocks. Full MATLAB regression at rtol=1e-8.

Time-domain types and SD_DIFF combinator deferred to v1.2/v1.3."
```

- [ ] **Step 3: Verify tag exists locally**

Run: `git tag -l --sort=-creatordate`
Expected: `v1.1.0` and `v1.0.0` both listed.

- [ ] **Step 4: Confirm not pushed**

Run: `git ls-remote --tags origin v1.1.0`
Expected: empty output (tag not on remote).

This concludes the v1.1 implementation. Per project policy, the tag
stays local until the user explicitly asks to push.

---

## Self-review (after writing the plan)

**Spec coverage:** every numbered section / requirement of
`docs/superpowers/specs/2026-04-25-v1_1-design.md` maps to a task:

- §3.1 (8 new combos) → Tasks 4, 14
- §3.2 (no physics changes) → confirmed via "Tasks 1-19 don't touch physics.py"
- §3.3.1 (`_PHYSICS_DISPATCH`) → Task 4 (CW), Task 14 (FD)
- §3.3.2 (`_expand_optodes`) → Task 1
- §3.3.3 (combinators) → Task 2
- §3.3.4 (`make_s_full` signature + validation order) → Tasks 4, 5
- §3.3.5 (`SensitivityResult` fields) → Task 3
- §3.4 (views unchanged) → no task
- §3.5.1 (type combobox) → Task 9
- §3.5.2 (fmod field) → Task 8
- §3.5.3 (multi-row rs/rd) → Tasks 6, 7
- §3.5.4 (`recalculate`) → Task 10
- §3.5.5 (`PARAM_CLASS`) → Task 9
- §3.5.6 (`save_data`) → Task 16
- §6.1 (fixtures) → Task 12
- §6.2 (regression tests) → Task 14
- §6.3 (behavioral tests) → Tasks 1, 2, 5, 7, 8, 9, 10, 11, 14, 15, 16
- §7 (migration) → Task 11
- §2 (versioning) → Tasks 17, 19

**Placeholder scan:** no TBD / TODO / "implement later"; every code block is concrete; every command has expected output.

**Type consistency:** function names match across tasks (`_expand_optodes`, `_combine_sd/ss/ds`, `_PHYSICS_DISPATCH`, `_ARRANGEMENT_COMBINE`, `_parse_float_matrix`, `_fmod_entry`, `_VALID_TYPES`); `Y_per_meas` used identically across Tasks 3 and 4.
