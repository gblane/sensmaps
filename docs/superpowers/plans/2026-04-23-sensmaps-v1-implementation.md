# sensmaps v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build v1 of `sensmaps` — an interactive Python/Tkinter GUI that computes and plots 2D slice maps of the Jacobian `S = ∂Y/∂μₐ(r)` for the `CW_SD_I` (continuous-wave, single-distance, intensity) measurement type using diffusion-theory.

**Architecture:** `src/sensmaps/` package with four layers: `physics.py` (pure NumPy analytical formulas ported line-by-line from MATLAB), `compute.py` (top-level `make_s` dispatcher), `views.py` (slicing + matplotlib rendering), `gui.py` (Tkinter window with cheap/expensive parameter classification). Regression-tested against MATLAB-generated `.mat` fixtures.

**Tech Stack:** Python 3.11+, NumPy, SciPy (`scipy.io.loadmat`, `scipy.signal.fftconvolve`), Matplotlib (with `FigureCanvasTkAgg`), Tkinter (stdlib), pytest, hatchling build backend.

**Working directory for all tasks:** `/home/giles/GitHub/sensmaps`.

---

## Task 1: Project scaffolding

**Files:**
- Create: `/home/giles/GitHub/sensmaps/.gitignore`
- Create: `/home/giles/GitHub/sensmaps/pyproject.toml`
- Create: `/home/giles/GitHub/sensmaps/README.md` (stub — replaced in Task 18)
- Create: `/home/giles/GitHub/sensmaps/src/sensmaps/__init__.py`
- Create: `/home/giles/GitHub/sensmaps/src/sensmaps/__main__.py`

- [ ] **Step 1: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/
build/
dist/

# Runtime artifacts
last_session.json
*.npz

# Large local test fixtures (not committed)
tests/fixtures/CW_l_L.mat
tests/fixtures/DSsetSDSSDSsen.mat

# Editors
.vscode/
.idea/
*.swp
.DS_Store
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "sensmaps"
version = "0.1.0"
description = "Interactive GUI for 2D sensitivity maps in diffuse optical imaging"
readme = "README.md"
requires-python = ">=3.11"
authors = [{ name = "Giles Blaney" }]
license = { text = "TBD" }
dependencies = [
  "numpy>=1.24",
  "scipy>=1.11",
  "matplotlib>=3.7",
]

[project.optional-dependencies]
dev = ["pytest>=8"]
mc  = ["pmcx"]

[project.scripts]
sensmaps = "sensmaps.__main__:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/sensmaps"]
```

- [ ] **Step 3: Create `src/sensmaps/__init__.py` (minimal; public API is populated in Task 12)**

```python
"""sensmaps — Interactive GUI for diffuse-optical sensitivity maps.

The public API is populated at the end of Task 12 (once the physics, compute,
and views modules exist). Until then, import from the submodules directly
(e.g. `from sensmaps.physics import n2a`).
"""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create `src/sensmaps/__main__.py` (stub; fleshed out in Task 17)**

```python
"""Entry point. `python -m sensmaps` or the `sensmaps` console-script."""


def main() -> int:
    print("sensmaps (stub): GUI launch is wired up in Task 17.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Create `README.md` (stub only — replaced by the real README in Task 18)**

`pyproject.toml` references `README.md` via `readme = "README.md"`, and the hatchling backend fails `pip install -e .` when that file is missing. Create a one-line stub now; Task 18 overwrites it with the real README:

```markdown
# sensmaps

Interactive GUI for 2D sensitivity maps in diffuse optical imaging. See the implementation plan in `docs/superpowers/plans/` during development. Full README lands at Task 18.
```

- [ ] **Step 6: Create a venv, install the package editable**

Run:
```bash
cd /home/giles/GitHub/sensmaps
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```
Expected: `pip install` succeeds; the last lines include `Successfully installed sensmaps-0.1.0 ...`.

- [ ] **Step 7: Smoke-check install via the console script**

The console-script calls `main` which does not import the package layers, so it works before the other tasks:
```bash
sensmaps
```
Expected: prints `sensmaps (stub): GUI launch is wired up in Task 17.` and exits 0.

- [ ] **Step 8: Commit**

```bash
git add .gitignore pyproject.toml README.md src/sensmaps/__init__.py src/sensmaps/__main__.py
git commit -m "feat: project scaffolding (pyproject, package layout, entry stub)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Fixture generator (MATLAB reference)

**Files:**
- Create: `/home/giles/GitHub/sensmaps/tests/fixtures/generate_fixtures.m`
- User-action generates: `/home/giles/GitHub/sensmaps/tests/fixtures/cw_sd_i_example1.mat`

This task produces the `.mat` file that all physics and compute tests check against.

- [ ] **Step 1: Create `tests/fixtures/generate_fixtures.m`**

```matlab
% generate_fixtures.m
% Regenerate reference data for sensmaps v1 Python tests.
%
% Run this script in MATLAB with the working directory set to
%   tests/fixtures
% It will write cw_sd_i_example1.mat alongside itself.
%
% Requirements: DOIT-Public/SensitivityCompendium/deps on MATLAB path
%   relative to this repo: ../../../DOIT-Public/SensitivityCompendium/deps

clear; clc;

%% Locate the compendium deps
here = fileparts(mfilename('fullpath'));
deps = fullfile(here, '..', '..', '..', ...
    'DOIT-Public', 'SensitivityCompendium', 'deps');
assert(isfolder(deps), ...
    'Expected DOIT-Public compendium deps at: %s', deps);
addpath(deps);

%% Canonical inputs (smaller than example1_DT.m to keep fixture under 1 MB)
nin  = 1.333;
nout = 1.0;
musp = 1.1;
mua  = 0.011;

opt_prop = struct('nin', nin, 'nout', nout, 'musp', musp, 'mua', mua);

dr   = 1.0;          % mm
pert = [1, 1, 1];    % mm
xl   = [-5, 40];     % mm
yl   = [0, 0];       % mm
zl   = [0, 20];      % mm
rho  = 25.0;         % mm

rs = [0, 0, 1/musp];       % source (with z-offset)
rd = [rho, 0, 0];           % detector

%% n2A scalar references (sweep both sides of 1)
n_in_vals = [1.0, 1.333, 1.4, 1.5, 0.8];
A_vals = arrayfun(@(n) n2A(n, 1.0), n_in_vals);

%% Single-point references at r_test, omega=0 (CW)
r_test = [10, 0, 5];
phi_test_cw = complexFluence(rs, r_test, 0, opt_prop);
R_test_cw   = complexReflectance(rs, rd, 0, opt_prop);
[L_test_cw, ~] = complexTotPathLen(rs, rd, 0, opt_prop);
l_test_cw   = complexPartPathLen(rs, r_test, rd, dr^3, 0, opt_prop);

%% Full end-to-end CW_SD_I — mirrors makeS.m
x = xl(1):dr:xl(2);
y = yl(1):dr:yl(2);
z = zl(1):dr:zl(2);

[YY, XX, ZZ] = meshgrid(y, x, z);
r_all = [XX(:), YY(:), ZZ(:)];

[L_scalar, ~] = complexTotPathLen(rs, rd, 0, opt_prop);
L = real(L_scalar);

l_vec = complexPartPathLen(rs, r_all, rd, dr^3, 0, opt_prop);
l_vec(isnan(l_vec)) = 0;
l_vec = real(l_vec);
ll = reshape(l_vec, size(XX));

Svox = ll / L;

H = ones(pert / dr);
S = convn(Svox, H, 'same');

%% Save fixture — individual vars (structs load awkwardly via scipy.io.loadmat)
save(fullfile(here, 'cw_sd_i_example1.mat'), ...
    'nin', 'nout', 'musp', 'mua', ...
    'dr', 'pert', 'xl', 'yl', 'zl', 'rho', ...
    'rs', 'rd', ...
    'n_in_vals', 'A_vals', ...
    'r_test', 'phi_test_cw', 'R_test_cw', 'L_test_cw', 'l_test_cw', ...
    'x', 'y', 'z', 'Svox', 'S');

rmpath(deps);

fprintf('Wrote %s\n', fullfile(here, 'cw_sd_i_example1.mat'));
```

- [ ] **Step 2: User-action — run the script in MATLAB**

Tell the user (Giles) to run the script:
```
Open MATLAB, cd to /home/giles/GitHub/sensmaps/tests/fixtures,
run: generate_fixtures
```
This will populate `cw_sd_i_example1.mat` in-place.

- [ ] **Step 3: Verify the fixture file exists and is reasonably sized**

Run:
```bash
ls -lh /home/giles/GitHub/sensmaps/tests/fixtures/cw_sd_i_example1.mat
```
Expected: file exists, size well under 1 MB (grid is 46×1×21 = 966 voxels; a handful of float64 arrays this size totals ~50–100 KB).

- [ ] **Step 4: Spot-check the fixture loads in Python**

Run:
```bash
cd /home/giles/GitHub/sensmaps
source .venv/bin/activate
python -c "
import scipy.io
d = scipy.io.loadmat('tests/fixtures/cw_sd_i_example1.mat', squeeze_me=True)
print({k: (type(v).__name__, getattr(v, 'shape', None)) for k, v in d.items() if not k.startswith('__')})
"
```
Expected output: prints a dict including `'S': ('ndarray', (46, 1, 21))`, `'x': ('ndarray', (46,))`, `'n_in_vals': ('ndarray', (5,))`, etc.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures/generate_fixtures.m tests/fixtures/cw_sd_i_example1.mat
git commit -m "test: add MATLAB-generated reference fixture for CW_SD_I

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Physics — `OpticalProperties` dataclass and `n2a`

**Files:**
- Create: `/home/giles/GitHub/sensmaps/src/sensmaps/physics.py`
- Create: `/home/giles/GitHub/sensmaps/tests/conftest.py`
- Create: `/home/giles/GitHub/sensmaps/tests/__init__.py`
- Create: `/home/giles/GitHub/sensmaps/tests/test_physics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/__init__.py` (empty, marks `tests` as a package) and `tests/conftest.py`:

```python
"""Shared pytest fixtures."""
from pathlib import Path

import pytest
import scipy.io


@pytest.fixture(scope="session")
def cw_sd_i_ref() -> dict:
    """Load the MATLAB reference fixture as a plain dict."""
    path = Path(__file__).parent / "fixtures" / "cw_sd_i_example1.mat"
    if not path.exists():
        pytest.skip(f"Reference fixture not found at {path}; run generate_fixtures.m")
    return scipy.io.loadmat(path, squeeze_me=True)
```

Create `tests/test_physics.py`:

```python
"""Regression tests for sensmaps.physics vs. MATLAB reference fixture."""
import numpy as np
import pytest

from sensmaps.physics import OpticalProperties, n2a


def test_optical_properties_defaults():
    op = OpticalProperties()
    assert op.n_in == 1.333
    assert op.n_out == 1.0
    assert op.musp == 1.1
    assert op.mua == 0.011


def test_n2a_matches_matlab(cw_sd_i_ref):
    for n_in, a_ref in zip(cw_sd_i_ref["n_in_vals"], cw_sd_i_ref["A_vals"]):
        a = n2a(float(n_in), 1.0)
        np.testing.assert_allclose(a, a_ref, rtol=1e-12, atol=0)


def test_n2a_unit_case_is_one():
    assert n2a(1.0, 1.0) == 1.0
```

- [ ] **Step 2: Run tests; verify they fail**

Run:
```bash
cd /home/giles/GitHub/sensmaps
source .venv/bin/activate
pytest tests/test_physics.py -v
```
Expected: `ModuleNotFoundError: No module named 'sensmaps.physics'` (or similar ImportError).

- [ ] **Step 3: Implement `src/sensmaps/physics.py`**

```python
"""Analytical diffusion-theory formulas.

All functions are ports of MATLAB code from
DOIT-Public/SensitivityCompendium/deps/. Variable names mirror the
MATLAB source (snake_case) and the formulas follow line-by-line.

Coordinate conventions:
    rs — source coordinates, shape (1, 3) or (3,) — [x, y, z] in mm
    rd — detector coordinates, shape (1, 3) or (3,) — [x, y, z] in mm
    r  — voxel-center coordinates, shape (N, 3) — [x, y, z] in mm
"""
from __future__ import annotations

from dataclasses import dataclass


C_MM_PER_SEC = 2.99792458e11  # speed of light in vacuum, mm/sec


@dataclass
class OpticalProperties:
    """Optical properties of a semi-infinite homogeneous medium.

    Defaults match the paper example (Blaney et al. 2024, JIOHS).
    """

    n_in: float = 1.333
    n_out: float = 1.0
    musp: float = 1.1   # reduced scattering, 1/mm
    mua: float = 0.011  # absorption, 1/mm
    g: float = 0.9      # anisotropy (used by MC backend in v3)


def n2a(n_in: float, n_out: float) -> float:
    """Index-of-refraction mismatch parameter A. Port of n2A.m."""
    dan12 = n_in / n_out
    if dan12 > 1:
        return (
            504.332889
            - 2641.00214 * dan12
            + 5923.699064 * dan12**2
            - 7376.355814 * dan12**3
            + 5507.53041 * dan12**4
            - 2463.357945 * dan12**5
            + 610.956547 * dan12**6
            - 64.8047 * dan12**7
        )
    if dan12 < 1:
        return (
            3.084635
            - 6.531194 * dan12
            + 8.357854 * dan12**2
            - 5.082751 * dan12**3
            + 1.171382 * dan12**4
        )
    return 1.0
```

- [ ] **Step 4: Run tests; verify they pass**

Run:
```bash
pytest tests/test_physics.py -v
```
Expected: all three tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/physics.py tests/__init__.py tests/conftest.py tests/test_physics.py
git commit -m "feat(physics): OpticalProperties dataclass and n2a port

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Physics — `complex_fluence`

**Files:**
- Modify: `/home/giles/GitHub/sensmaps/src/sensmaps/physics.py`
- Modify: `/home/giles/GitHub/sensmaps/tests/test_physics.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_physics.py`:

```python
def _opt_prop_from_ref(ref) -> OpticalProperties:
    return OpticalProperties(
        n_in=float(ref["nin"]),
        n_out=float(ref["nout"]),
        musp=float(ref["musp"]),
        mua=float(ref["mua"]),
    )


def test_complex_fluence_cw_matches_matlab(cw_sd_i_ref):
    from sensmaps.physics import complex_fluence
    ref = cw_sd_i_ref
    op = _opt_prop_from_ref(ref)
    rs = np.asarray(ref["rs"], dtype=float).reshape(1, 3)
    r_test = np.asarray(ref["r_test"], dtype=float).reshape(1, 3)
    phi_ref = complex(ref["phi_test_cw"])

    phi = complex_fluence(rs, r_test, 0.0, op)
    assert phi.shape == (1,)
    np.testing.assert_allclose(phi[0], phi_ref, rtol=1e-10, atol=0)
```

- [ ] **Step 2: Run; verify fail**

Run:
```bash
pytest tests/test_physics.py::test_complex_fluence_cw_matches_matlab -v
```
Expected: `ImportError: cannot import name 'complex_fluence' ...`.

- [ ] **Step 3: Implement `complex_fluence`**

Append to `src/sensmaps/physics.py`:

```python
import numpy as np


def _unpack_source(rs):
    """Canonicalize rs to a (1,3) float array; return (x0, y0, z0)."""
    rs = np.atleast_2d(np.asarray(rs, dtype=np.float64))
    if rs.shape != (1, 3):
        raise ValueError(f"rs must be (1,3); got shape {rs.shape}")
    return rs, float(rs[0, 0]), float(rs[0, 1]), float(rs[0, 2])


def complex_fluence(rs, r, omega: float, opt_prop: OpticalProperties):
    """Complex fluence at positions r from a source at rs. Port of complexFluence.m.

    Parameters
    ----------
    rs        : array-like, shape (1, 3) or (3,)   — source coords [mm]
    r         : array-like, shape (N, 3) or (3,)   — detector-volume coords [mm]
    omega     : float                              — angular modulation freq [rad/sec]
    opt_prop  : OpticalProperties

    Returns
    -------
    phi : complex ndarray, shape (N,)  — fluence [1/mm^2]
    """
    rs, x0, y0, z0 = _unpack_source(rs)
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))

    v = C_MM_PER_SEC / opt_prop.n_in
    a_mismatch = n2a(opt_prop.n_in, opt_prop.n_out)
    D = 1.0 / (3.0 * opt_prop.musp)
    zb = -2.0 * a_mismatch * D

    mueff = np.sqrt(opt_prop.mua / D - 1j * omega / (v * D))

    rsp = np.array([[x0, y0, -z0 + 2.0 * zb]])

    r1 = np.linalg.norm(r - rs, axis=1)
    r2 = np.linalg.norm(r - rsp, axis=1)

    return (np.exp(-mueff * r1) / r1 - np.exp(-mueff * r2) / r2) / (4.0 * np.pi * D)
```

- [ ] **Step 4: Run; verify pass**

Run:
```bash
pytest tests/test_physics.py -v
```
Expected: all tests pass (4 now, including the new one).

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/physics.py tests/test_physics.py
git commit -m "feat(physics): complex_fluence port

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Physics — `complex_reflectance`

**Files:**
- Modify: `/home/giles/GitHub/sensmaps/src/sensmaps/physics.py`
- Modify: `/home/giles/GitHub/sensmaps/tests/test_physics.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_physics.py`:

```python
def test_complex_reflectance_cw_matches_matlab(cw_sd_i_ref):
    from sensmaps.physics import complex_reflectance
    ref = cw_sd_i_ref
    op = _opt_prop_from_ref(ref)
    rs = np.asarray(ref["rs"], dtype=float).reshape(1, 3)
    rd = np.asarray(ref["rd"], dtype=float).reshape(1, 3)
    R_ref = complex(ref["R_test_cw"])

    R = complex_reflectance(rs, rd, 0.0, op)
    assert R.shape == (1,)
    np.testing.assert_allclose(R[0], R_ref, rtol=1e-10, atol=0)
```

- [ ] **Step 2: Run; verify fail**

```bash
pytest tests/test_physics.py::test_complex_reflectance_cw_matches_matlab -v
```
Expected: ImportError for `complex_reflectance`.

- [ ] **Step 3: Implement `complex_reflectance`**

Append to `src/sensmaps/physics.py`:

```python
def complex_reflectance(rs, rd, omega: float, opt_prop: OpticalProperties):
    """Complex reflectance for source-detector pairs. Port of complexReflectance.m.

    Parameters
    ----------
    rs        : array-like, shape (1, 3) — source [mm]
    rd        : array-like, shape (N, 3) or (3,) — detector coords [mm]
    omega     : float                     — rad/sec
    opt_prop  : OpticalProperties

    Returns
    -------
    R : complex ndarray, shape (N,) — reflectance [1/mm^2]
    """
    rs, x0, y0, z0 = _unpack_source(rs)
    rd = np.atleast_2d(np.asarray(rd, dtype=np.float64))

    v = C_MM_PER_SEC / opt_prop.n_in
    a_mismatch = n2a(opt_prop.n_in, opt_prop.n_out)
    D = 1.0 / (3.0 * opt_prop.musp)
    zb = -2.0 * a_mismatch * D

    mueff = np.sqrt(opt_prop.mua / D - 1j * omega / (v * D))

    rsp = np.array([[x0, y0, -z0 + 2.0 * zb]])

    r1 = np.linalg.norm(rd - rs, axis=1)
    r2 = np.linalg.norm(rd - rsp, axis=1)

    return (
        z0 * (1.0 / r1 + mueff) * np.exp(-mueff * r1) / r1**2
        + (z0 - 2.0 * zb) * (1.0 / r2 + mueff) * np.exp(-mueff * r2) / r2**2
    ) / (4.0 * np.pi)
```

- [ ] **Step 4: Run; verify pass**

```bash
pytest tests/test_physics.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/physics.py tests/test_physics.py
git commit -m "feat(physics): complex_reflectance port

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Physics — `complex_tot_path_len`

**Files:**
- Modify: `/home/giles/GitHub/sensmaps/src/sensmaps/physics.py`
- Modify: `/home/giles/GitHub/sensmaps/tests/test_physics.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_physics.py`:

```python
def test_complex_tot_path_len_cw_matches_matlab(cw_sd_i_ref):
    from sensmaps.physics import complex_tot_path_len
    ref = cw_sd_i_ref
    op = _opt_prop_from_ref(ref)
    rs = np.asarray(ref["rs"], dtype=float).reshape(1, 3)
    rd = np.asarray(ref["rd"], dtype=float).reshape(1, 3)
    L_ref = complex(ref["L_test_cw"])

    L, R = complex_tot_path_len(rs, rd, 0.0, op)
    assert L.shape == (1,)
    np.testing.assert_allclose(L[0], L_ref, rtol=1e-10, atol=0)
    # At omega=0 the imaginary part is numerically zero
    assert abs(L[0].imag) < 1e-12
```

- [ ] **Step 2: Run; verify fail**

```bash
pytest tests/test_physics.py::test_complex_tot_path_len_cw_matches_matlab -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `complex_tot_path_len`**

Append to `src/sensmaps/physics.py`:

```python
def complex_tot_path_len(rs, rd, omega: float, opt_prop: OpticalProperties):
    """Complex total path length and reflectance. Port of complexTotPathLen.m.

    Returns
    -------
    L : complex ndarray, shape (N,) — total path length [mm]
    R : complex ndarray, shape (N,) — reflectance [1/mm^2]
    """
    rs, x0, y0, z0 = _unpack_source(rs)
    rd = np.atleast_2d(np.asarray(rd, dtype=np.float64))

    v = C_MM_PER_SEC / opt_prop.n_in
    a_mismatch = n2a(opt_prop.n_in, opt_prop.n_out)
    D = 1.0 / (3.0 * opt_prop.musp)
    zb = -2.0 * a_mismatch * D

    mueff = np.sqrt(opt_prop.mua / D - 1j * omega / (v * D))

    rsp = np.array([[x0, y0, -z0 + 2.0 * zb]])

    r1 = np.linalg.norm(rd - rs, axis=1)
    r2 = np.linalg.norm(rd - rsp, axis=1)

    R = complex_reflectance(rs, rd, omega, opt_prop)

    L = (
        (z0 / r1) * np.exp(-mueff * r1)
        + ((z0 - 2.0 * zb) / r2) * np.exp(-mueff * r2)
    ) / (8.0 * np.pi * D * R)

    return L, R
```

- [ ] **Step 4: Run; verify pass**

```bash
pytest tests/test_physics.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/physics.py tests/test_physics.py
git commit -m "feat(physics): complex_tot_path_len port

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Physics — `complex_part_path_len`

**Files:**
- Modify: `/home/giles/GitHub/sensmaps/src/sensmaps/physics.py`
- Modify: `/home/giles/GitHub/sensmaps/tests/test_physics.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_physics.py`:

```python
def test_complex_part_path_len_cw_scalar_matches_matlab(cw_sd_i_ref):
    from sensmaps.physics import complex_part_path_len
    ref = cw_sd_i_ref
    op = _opt_prop_from_ref(ref)
    rs = np.asarray(ref["rs"], dtype=float).reshape(1, 3)
    rd = np.asarray(ref["rd"], dtype=float).reshape(1, 3)
    r_test = np.asarray(ref["r_test"], dtype=float).reshape(1, 3)
    dr = float(ref["dr"])
    l_ref = complex(ref["l_test_cw"])

    l = complex_part_path_len(rs, r_test, rd, dr**3, 0.0, op)
    assert l.shape == (1,)
    np.testing.assert_allclose(l[0], l_ref, rtol=1e-10, atol=0)
```

- [ ] **Step 2: Run; verify fail**

```bash
pytest tests/test_physics.py::test_complex_part_path_len_cw_scalar_matches_matlab -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `complex_part_path_len`**

Append to `src/sensmaps/physics.py`:

```python
def complex_part_path_len(rs, r, rd, V: float, omega: float,
                          opt_prop: OpticalProperties):
    """Complex partial path length per voxel. Port of complexPartPathLen.m.

    Parameters
    ----------
    rs       : (1, 3)   — source [mm]
    r        : (N, 3)   — voxel centers [mm]
    rd       : (1, 3)   — detector [mm]
    V        : float    — voxel volume [mm^3]
    omega    : float    — rad/sec
    opt_prop : OpticalProperties

    Returns
    -------
    l : complex ndarray, shape (N,) — partial path length [mm]
    """
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))

    phi_rs_r = complex_fluence(rs, r, omega, opt_prop)       # (N,)
    R_r_rd  = complex_reflectance(r, rd, omega, opt_prop)    # broadcast N source-voxels to 1 detector
    R_rs_rd = complex_reflectance(rs, rd, omega, opt_prop)   # (1,)

    return (phi_rs_r * R_r_rd * V) / R_rs_rd
```

Note: `complex_reflectance(r, rd, …)` treats each row of `r` as a source. `_unpack_source` enforces a single-row source, so the call would fail on N > 1. Update `complex_reflectance` to accept `rs` of shape `(N, 3)` with one-detector broadcast by generalizing `_unpack_source` OR write a small inline loop.

Cleanest: broaden `_unpack_source` and `complex_reflectance` to accept `rs` of shape `(N, 3)`. Replace `_unpack_source` with this generalized version:

```python
def _split_source(rs):
    """Return (rs, x0, y0, z0) where each of x0,y0,z0 is an ndarray of shape (N,)."""
    rs = np.atleast_2d(np.asarray(rs, dtype=np.float64))
    if rs.shape[1] != 3:
        raise ValueError(f"rs must have shape (N,3); got {rs.shape}")
    return rs, rs[:, 0], rs[:, 1], rs[:, 2]
```

Update `complex_fluence`, `complex_reflectance`, `complex_tot_path_len` to use `_split_source` and:
- Operate with broadcasting on `rs` (N_s, 3) and `rd`/`r` (N_r, 3) where MATLAB's convention is: one of them is length 1, the other is length N.
- Specifically: r1, r2 are vector differences. If `rs` is (1,3) and `r`/`rd` is (N,3), r1 = `||rd - rs||` is (N,). If `rs` is (N,3) and `rd` is (1,3), r1 is (N,). This works with numpy broadcasting if you subtract an (N,3) from a (1,3) or vice versa — no changes needed.

Replace the whole `complex_fluence`, `complex_reflectance`, `complex_tot_path_len` definitions with versions that use `_split_source` instead of `_unpack_source`, and delete `_unpack_source`:

```python
def _split_source(rs):
    rs = np.atleast_2d(np.asarray(rs, dtype=np.float64))
    if rs.shape[1] != 3:
        raise ValueError(f"rs must have shape (N,3); got {rs.shape}")
    return rs, rs[:, 0], rs[:, 1], rs[:, 2]


def complex_fluence(rs, r, omega: float, opt_prop: OpticalProperties):
    """Complex fluence at r from source rs. Port of complexFluence.m."""
    rs, x0, y0, z0 = _split_source(rs)
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))

    v = C_MM_PER_SEC / opt_prop.n_in
    a_mismatch = n2a(opt_prop.n_in, opt_prop.n_out)
    D = 1.0 / (3.0 * opt_prop.musp)
    zb = -2.0 * a_mismatch * D

    mueff = np.sqrt(opt_prop.mua / D - 1j * omega / (v * D))

    rsp = np.column_stack([x0, y0, -z0 + 2.0 * zb])  # shape (N_s, 3)

    r1 = np.linalg.norm(r - rs, axis=1)
    r2 = np.linalg.norm(r - rsp, axis=1)

    return (np.exp(-mueff * r1) / r1 - np.exp(-mueff * r2) / r2) / (4.0 * np.pi * D)


def complex_reflectance(rs, rd, omega: float, opt_prop: OpticalProperties):
    """Complex reflectance for source-detector pairs. Port of complexReflectance.m."""
    rs, x0, y0, z0 = _split_source(rs)
    rd = np.atleast_2d(np.asarray(rd, dtype=np.float64))

    v = C_MM_PER_SEC / opt_prop.n_in
    a_mismatch = n2a(opt_prop.n_in, opt_prop.n_out)
    D = 1.0 / (3.0 * opt_prop.musp)
    zb = -2.0 * a_mismatch * D

    mueff = np.sqrt(opt_prop.mua / D - 1j * omega / (v * D))

    rsp = np.column_stack([x0, y0, -z0 + 2.0 * zb])

    r1 = np.linalg.norm(rd - rs, axis=1)
    r2 = np.linalg.norm(rd - rsp, axis=1)

    return (
        z0 * (1.0 / r1 + mueff) * np.exp(-mueff * r1) / r1**2
        + (z0 - 2.0 * zb) * (1.0 / r2 + mueff) * np.exp(-mueff * r2) / r2**2
    ) / (4.0 * np.pi)


def complex_tot_path_len(rs, rd, omega: float, opt_prop: OpticalProperties):
    """Complex total path length and reflectance. Port of complexTotPathLen.m."""
    rs, x0, y0, z0 = _split_source(rs)
    rd = np.atleast_2d(np.asarray(rd, dtype=np.float64))

    v = C_MM_PER_SEC / opt_prop.n_in
    a_mismatch = n2a(opt_prop.n_in, opt_prop.n_out)
    D = 1.0 / (3.0 * opt_prop.musp)
    zb = -2.0 * a_mismatch * D

    mueff = np.sqrt(opt_prop.mua / D - 1j * omega / (v * D))

    rsp = np.column_stack([x0, y0, -z0 + 2.0 * zb])

    r1 = np.linalg.norm(rd - rs, axis=1)
    r2 = np.linalg.norm(rd - rsp, axis=1)

    R = complex_reflectance(rs, rd, omega, opt_prop)

    L = (
        (z0 / r1) * np.exp(-mueff * r1)
        + ((z0 - 2.0 * zb) / r2) * np.exp(-mueff * r2)
    ) / (8.0 * np.pi * D * R)

    return L, R
```

Delete the old `_unpack_source` function.

- [ ] **Step 4: Run; verify all physics tests pass**

```bash
pytest tests/test_physics.py -v
```
Expected: all five tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/physics.py tests/test_physics.py
git commit -m "feat(physics): complex_part_path_len and N-source broadcasting

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Physics — `continuous_*` wrappers

**Files:**
- Modify: `/home/giles/GitHub/sensmaps/src/sensmaps/physics.py`
- Modify: `/home/giles/GitHub/sensmaps/tests/test_physics.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_physics.py`:

```python
def test_continuous_wrappers_return_real_and_match_cw(cw_sd_i_ref):
    from sensmaps.physics import (
        continuous_fluence,
        continuous_part_path_len,
        continuous_reflectance,
        continuous_tot_path_len,
    )
    ref = cw_sd_i_ref
    op = _opt_prop_from_ref(ref)
    rs = np.asarray(ref["rs"], dtype=float).reshape(1, 3)
    rd = np.asarray(ref["rd"], dtype=float).reshape(1, 3)
    r_test = np.asarray(ref["r_test"], dtype=float).reshape(1, 3)
    dr = float(ref["dr"])

    phi = continuous_fluence(rs, r_test, op)
    R = continuous_reflectance(rs, rd, op)
    L, R2 = continuous_tot_path_len(rs, rd, op)
    l = continuous_part_path_len(rs, r_test, rd, dr**3, op)

    # Real dtype
    assert np.isrealobj(phi)
    assert np.isrealobj(R)
    assert np.isrealobj(L)
    assert np.isrealobj(l)

    # Values match the MATLAB CW reference
    np.testing.assert_allclose(phi[0], float(ref["phi_test_cw"].real), rtol=1e-10)
    np.testing.assert_allclose(R[0], float(ref["R_test_cw"].real), rtol=1e-10)
    np.testing.assert_allclose(L[0], float(ref["L_test_cw"].real), rtol=1e-10)
    np.testing.assert_allclose(l[0], float(ref["l_test_cw"].real), rtol=1e-10)
```

- [ ] **Step 2: Run; verify fail**

```bash
pytest tests/test_physics.py::test_continuous_wrappers_return_real_and_match_cw -v
```
Expected: ImportError for `continuous_*`.

- [ ] **Step 3: Implement the wrappers**

Append to `src/sensmaps/physics.py`:

```python
def continuous_fluence(rs, r, opt_prop: OpticalProperties):
    """CW fluence — wrapper for complex_fluence at omega=0. Returns real ndarray."""
    return complex_fluence(rs, r, 0.0, opt_prop).real


def continuous_reflectance(rs, rd, opt_prop: OpticalProperties):
    """CW reflectance — wrapper for complex_reflectance at omega=0. Returns real ndarray."""
    return complex_reflectance(rs, rd, 0.0, opt_prop).real


def continuous_tot_path_len(rs, rd, opt_prop: OpticalProperties):
    """CW total path length and reflectance — wrapper at omega=0. Returns real arrays."""
    L, R = complex_tot_path_len(rs, rd, 0.0, opt_prop)
    return L.real, R.real


def continuous_part_path_len(rs, r, rd, V: float, opt_prop: OpticalProperties):
    """CW partial path length — wrapper for complex_part_path_len at omega=0."""
    return complex_part_path_len(rs, r, rd, V, 0.0, opt_prop).real
```

- [ ] **Step 4: Run; verify pass**

```bash
pytest tests/test_physics.py -v
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/physics.py tests/test_physics.py
git commit -m "feat(physics): continuous_* CW wrappers (omega=0, real-valued)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Compute — `GridParams` dataclass and type-string parsing

**Files:**
- Create: `/home/giles/GitHub/sensmaps/src/sensmaps/compute.py`
- Create: `/home/giles/GitHub/sensmaps/tests/test_compute.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_compute.py`:

```python
"""Tests for sensmaps.compute — dispatcher, parsing, make_s."""
import numpy as np
import pytest


def test_grid_params_from_limits():
    from sensmaps.compute import GridParams
    g = GridParams.from_limits(xl=(-5, 5), yl=(0, 0), zl=(0, 10), dr=1.0)
    np.testing.assert_array_equal(g.x, np.arange(-5, 6, 1.0))
    np.testing.assert_array_equal(g.y, np.array([0.0]))
    np.testing.assert_array_equal(g.z, np.arange(0, 11, 1.0))


def test_parse_type_str_valid():
    from sensmaps.compute import parse_type_str
    t = parse_type_str("CW_SD_I")
    assert t.temporal == "CW"
    assert t.arrangement == "SD"
    assert t.data_type == "I"


def test_parse_type_str_invalid():
    from sensmaps.compute import parse_type_str
    with pytest.raises(ValueError, match="AA_BB_C"):
        parse_type_str("bogus")
```

- [ ] **Step 2: Run; verify fail**

```bash
pytest tests/test_compute.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/sensmaps/compute.py`**

```python
"""Top-level dispatcher: make_s(type_str, rs, rd, opt_prop, **kw).

Mirrors MATLAB makeS.m: parses the AA_BB_C type string, builds a voxel
grid, dispatches to physics functions, applies the perturbation
convolution, and returns (S, params).

v1 only implements CW_SD_I in DT mode. The dispatch-table scaffolding is
laid out to make v2 additions a single-row change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class GridParams:
    """Voxel-grid coordinate axes and valid limits (mirror of MATLAB params struct)."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    xl_valid: tuple[float, float]
    yl_valid: tuple[float, float]
    zl_valid: tuple[float, float]

    @classmethod
    def from_limits(
        cls,
        xl: Sequence[float],
        yl: Sequence[float],
        zl: Sequence[float],
        dr: float,
        pert: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> "GridParams":
        """Build axis vectors from (min, max) limits and voxel size dr."""
        x = np.arange(xl[0], xl[1] + dr / 2, dr)
        y = np.arange(yl[0], yl[1] + dr / 2, dr)
        z = np.arange(zl[0], zl[1] + dr / 2, dr)
        # Clamp single-point axes when xl[0] == xl[1]
        if y.size == 0:
            y = np.array([yl[0]])
        if x.size == 0:
            x = np.array([xl[0]])
        if z.size == 0:
            z = np.array([zl[0]])
        if np.allclose(pert, (1.0, 1.0, 1.0)):
            xl_valid = (float(x[0]), float(x[-1]))
            yl_valid = (float(y[0]), float(y[-1]))
            zl_valid = (float(z[0]), float(z[-1]))
        else:
            xl_valid = (float(x[0] + pert[0] / 2), float(x[-1] - pert[0] / 2))
            yl_valid = (float(y[0] + pert[1] / 2), float(y[-1] - pert[1] / 2))
            zl_valid = (float(z[0] + pert[2] / 2), float(z[-1] - pert[2] / 2))
        return cls(x=x, y=y, z=z, xl_valid=xl_valid, yl_valid=yl_valid, zl_valid=zl_valid)


@dataclass(frozen=True)
class ParsedType:
    temporal: str      # one of "CW", "FD", "TD"
    arrangement: str   # one of "SD", "SS", "DS"
    data_type: str     # one of "I", "GI", "DGI", "P", "T", "V"


_TEMPORALS = {"CW", "FD", "TD"}
_ARRANGEMENTS = {"SD", "SS", "DS"}
_DATA_TYPES = {"I", "GI", "DGI", "P", "T", "V"}


def parse_type_str(type_str: str) -> ParsedType:
    """Parse an AA_BB_C measurement-type string."""
    parts = type_str.upper().split("_")
    if len(parts) != 3:
        raise ValueError(
            f"type_str must be AA_BB_C format; got {type_str!r}"
        )
    temporal, arrangement, data_type = parts
    if temporal not in _TEMPORALS:
        raise ValueError(f"Unknown temporal {temporal!r}; expected one of {_TEMPORALS}")
    if arrangement not in _ARRANGEMENTS:
        raise ValueError(f"Unknown arrangement {arrangement!r}; expected one of {_ARRANGEMENTS}")
    if data_type not in _DATA_TYPES:
        raise ValueError(f"Unknown data_type {data_type!r}; expected one of {_DATA_TYPES}")
    return ParsedType(temporal=temporal, arrangement=arrangement, data_type=data_type)
```

- [ ] **Step 4: Run; verify pass**

```bash
pytest tests/test_compute.py -v
```
Expected: three tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/compute.py tests/test_compute.py
git commit -m "feat(compute): GridParams dataclass and type-string parser

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: Compute — `make_s` for CW_SD_I (end-to-end vs. fixture)

**Files:**
- Modify: `/home/giles/GitHub/sensmaps/src/sensmaps/compute.py`
- Modify: `/home/giles/GitHub/sensmaps/tests/test_compute.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_compute.py`:

```python
def test_make_s_cw_sd_i_matches_matlab(cw_sd_i_ref):
    from sensmaps.compute import make_s
    from sensmaps.physics import OpticalProperties
    ref = cw_sd_i_ref
    op = OpticalProperties(
        n_in=float(ref["nin"]),
        n_out=float(ref["nout"]),
        musp=float(ref["musp"]),
        mua=float(ref["mua"]),
    )
    rs_input = np.asarray(ref["rs"], dtype=float).reshape(1, 3) - np.array([[0, 0, 1 / op.musp]])
    rd = np.asarray(ref["rd"], dtype=float).reshape(1, 3)

    S, params = make_s(
        type_str="CW_SD_I",
        rs=rs_input,
        rd=rd,
        opt_prop=op,
        xl=(float(ref["xl"][0]), float(ref["xl"][1])),
        yl=(float(ref["yl"][0]), float(ref["yl"][1])),
        zl=(float(ref["zl"][0]), float(ref["zl"][1])),
        dr=float(ref["dr"]),
        pert=tuple(float(p) for p in ref["pert"]),
    )
    S_ref = np.asarray(ref["S"], dtype=float)
    assert S.shape == S_ref.shape
    np.testing.assert_allclose(S, S_ref, rtol=1e-8, atol=1e-12)

    # Grid axes match
    np.testing.assert_allclose(params.x, np.asarray(ref["x"], dtype=float))
    np.testing.assert_allclose(params.y, np.asarray(ref["y"], dtype=float))
    np.testing.assert_allclose(params.z, np.asarray(ref["z"], dtype=float))


def test_make_s_rejects_unsupported_type():
    from sensmaps.compute import make_s
    from sensmaps.physics import OpticalProperties
    with pytest.raises(NotImplementedError, match="CW_SD_I"):
        make_s(
            type_str="FD_SS_P",
            rs=np.array([[0, 0, 0]]),
            rd=np.array([[25, 0, 0]]),
            opt_prop=OpticalProperties(),
            xl=(-5, 40), yl=(0, 0), zl=(0, 20), dr=1.0,
        )
```

Note the `rs_input` subtracts the z-offset `1/musp` that `make_s` adds internally, so the fixture's `rs` (which was stored pre-added) lines up.

Wait — look back at `generate_fixtures.m`: it builds `rs = [0, 0, 1/musp]` and saves that. So the fixture has the offset baked in. Our `make_s` should NOT add it a second time. But MATLAB `makeS.m` adds `+[0,0,z0]` internally. So when the user calls `make_s(…, rs=[0,0,0])` the offset is applied. When we test against a fixture whose `rs` already has the offset, we need to pass `rs` without the offset. The easiest way: the test passes `rs_input = rs - [0,0,1/musp]` so `make_s` re-adds it and hits the same working rs.

Keep the test as written above (it subtracts `1/musp` from `ref["rs"]` before passing to `make_s`).

- [ ] **Step 2: Run; verify fail**

```bash
pytest tests/test_compute.py::test_make_s_cw_sd_i_matches_matlab -v
```
Expected: AttributeError / ImportError for `make_s`.

- [ ] **Step 3: Implement `make_s_full` and `make_s`**

Append to `src/sensmaps/compute.py`:

```python
from scipy.signal import fftconvolve

from sensmaps.physics import (
    OpticalProperties,
    continuous_part_path_len,
    continuous_tot_path_len,
)


@dataclass
class SensitivityResult:
    """Return type of make_s_full.

    Attributes
    ----------
    S         : ndarray, shape (Nx, Ny, Nz) — sensitivity (pert-convolved)
    Svox      : ndarray, shape (Nx, Ny, Nz) — per-voxel pre-conv sensitivity
    params    : GridParams
    type_str  : str
    rs, rd    : source and detector coords actually used (post z-offset) [mm]
    opt_prop  : OpticalProperties
    pert, dr  : perturbation and voxel size used
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

    This is the "thick" variant: it returns a SensitivityResult that
    includes Svox (pre-convolution) and the resolved inputs, so callers
    (notably the GUI) can re-convolve without recomputing the physics.

    Parameters
    ----------
    type_str  : "CW_SD_I" in v1; other values raise NotImplementedError.
    rs        : (1, 3) source coordinate (before z-offset is applied)
    rd        : (1, 3) detector coordinate
    opt_prop  : OpticalProperties
    xl, yl, zl: (min, max) grid limits [mm]
    dr        : voxel side length [mm]
    pert      : (px, py, pz) perturbation box size [mm]
    sim_typ   : "DT" in v1; "MC" reserved for v3.
    """
    parsed = parse_type_str(type_str)

    if sim_typ != "DT":
        raise NotImplementedError(
            f"sim_typ={sim_typ!r} is not implemented in v1 (DT only)"
        )
    if not (parsed.temporal == "CW" and parsed.arrangement == "SD"
            and parsed.data_type == "I"):
        raise NotImplementedError(
            f"type_str={type_str!r} is not implemented in v1 (only CW_SD_I)"
        )
    if pert[0] % dr or pert[1] % dr or pert[2] % dr:
        raise ValueError(f"pert {pert} must be a multiple of dr={dr}")

    params = GridParams.from_limits(xl=xl, yl=yl, zl=zl, dr=dr, pert=pert)

    # z-offset the source by 1/musp (DT convention)
    rs = np.atleast_2d(np.asarray(rs, dtype=np.float64))
    rd = np.atleast_2d(np.asarray(rd, dtype=np.float64))
    z_offset = 1.0 / opt_prop.musp
    rs_used = rs + np.array([[0, 0, z_offset]])

    # Build voxel-center coordinate matrix (Nx, Ny, Nz, 3)
    XX, YY, ZZ = np.meshgrid(params.x, params.y, params.z, indexing="ij")
    r_all = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    # Total path length (scalar for SD)
    L, _ = continuous_tot_path_len(rs_used, rd, opt_prop)
    L_scalar = float(L[0])

    # Partial path length per voxel
    l_vec = continuous_part_path_len(rs_used, r_all, rd, dr ** 3, opt_prop)
    l_vec = np.nan_to_num(l_vec, nan=0.0)
    ll = l_vec.reshape(XX.shape)

    Svox = ll / L_scalar

    # Perturbation convolution
    kernel_shape = tuple(int(round(p / dr)) for p in pert)
    H = np.ones(kernel_shape, dtype=np.float64)
    S = fftconvolve(Svox, H, mode="same")

    return SensitivityResult(
        S=S, Svox=Svox, params=params, type_str=type_str,
        rs=rs_used, rd=rd, opt_prop=opt_prop, pert=tuple(pert), dr=dr,
    )


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
):
    """Thin variant returning only `(S, params)`. See `make_s_full` for details."""
    result = make_s_full(
        type_str=type_str, rs=rs, rd=rd, opt_prop=opt_prop,
        xl=xl, yl=yl, zl=zl, dr=dr, pert=pert, sim_typ=sim_typ,
    )
    return result.S, result.params
```

- [ ] **Step 4: Run; verify pass**

```bash
pytest tests/test_compute.py -v
```
Expected: all 5 compute tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/compute.py tests/test_compute.py
git commit -m "feat(compute): make_s CW_SD_I with DT + end-to-end fixture test

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 11: Views — `slice_s`

**Files:**
- Create: `/home/giles/GitHub/sensmaps/src/sensmaps/views.py`
- Create: `/home/giles/GitHub/sensmaps/tests/test_views.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_views.py`:

```python
"""Tests for sensmaps.views."""
import numpy as np
import pytest


@pytest.fixture
def tiny_result(cw_sd_i_ref):
    """Build a SensitivityResult-like namespace from the fixture."""
    from sensmaps.compute import GridParams
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
    return S, params


def test_slice_s_y_axis_shape_and_labels(tiny_result):
    from sensmaps.views import slice_s
    S, params = tiny_result
    S_plane, pp = slice_s(S, params, axis="y", value=0.0)
    # S shape (Nx, Ny, Nz) → slice at y=0 gives (Nx, Nz), then transpose → (Nz, Nx)
    assert S_plane.shape == (params.z.size, params.x.size)
    assert pp.horz_label == "$x$ (mm)"
    assert pp.vert_label == "$z$ (mm)"
    np.testing.assert_array_equal(pp.horz_axis, params.x)
    np.testing.assert_array_equal(pp.vert_axis, params.z)


def test_slice_s_rejects_bad_axis(tiny_result):
    from sensmaps.views import slice_s
    S, params = tiny_result
    with pytest.raises(ValueError, match="axis"):
        slice_s(S, params, axis="bogus", value=0.0)


def test_slice_s_snaps_to_nearest_axis_value(tiny_result):
    from sensmaps.views import slice_s
    S, params = tiny_result
    S_plane_a, _ = slice_s(S, params, axis="y", value=0.0)
    S_plane_b, _ = slice_s(S, params, axis="y", value=0.3)  # snaps to 0
    np.testing.assert_array_equal(S_plane_a, S_plane_b)
```

- [ ] **Step 2: Run; verify fail**

```bash
pytest tests/test_views.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/sensmaps/views.py`**

```python
"""Slicing, colormap, and rendering utilities. Mirror MATLAB sliceS / makeCL."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlotParams:
    """Axis metadata for a 2D slice. Mirror of MATLAB plotParams struct."""

    horz_axis: np.ndarray
    horz_label: str
    vert_axis: np.ndarray
    vert_label: str


def slice_s(S: np.ndarray, params, axis: str, value: float):
    """Slice a 3D S array for 2D plotting. Port of sliceS.m.

    Parameters
    ----------
    S     : ndarray, shape (Nx, Ny, Nz)
    params: GridParams with axes .x, .y, .z
    axis  : "x", "y", or "z" — the axis to hold fixed
    value : float — value along `axis` to slice at (snaps to nearest voxel)

    Returns
    -------
    S_plane   : 2D ndarray — oriented for imshow (vert axis × horz axis)
    plot_params : PlotParams
    """
    axis = axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"axis must be one of 'x', 'y', 'z'; got {axis!r}")

    axis_vec = getattr(params, axis)
    idx = int(np.argmin(np.abs(axis_vec - value)))

    if axis == "x":
        plane = S[idx, :, :].T
        pp = PlotParams(
            horz_axis=params.y, horz_label="$y$ (mm)",
            vert_axis=params.z, vert_label="$z$ (mm)",
        )
    elif axis == "y":
        plane = S[:, idx, :].T
        pp = PlotParams(
            horz_axis=params.x, horz_label="$x$ (mm)",
            vert_axis=params.z, vert_label="$z$ (mm)",
        )
    else:  # z
        plane = S[:, :, idx].T
        pp = PlotParams(
            horz_axis=params.x, horz_label="$x$ (mm)",
            vert_axis=params.y, vert_label="$y$ (mm)",
        )
    return plane, pp
```

- [ ] **Step 4: Run; verify pass**

```bash
pytest tests/test_views.py -v
```
Expected: three tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/views.py tests/test_views.py
git commit -m "feat(views): slice_s port with 2D slice orientation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 12: Views — `make_color_limits`

**Files:**
- Modify: `/home/giles/GitHub/sensmaps/src/sensmaps/views.py`
- Modify: `/home/giles/GitHub/sensmaps/tests/test_views.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_views.py`:

```python
def test_make_color_limits_defaults():
    from sensmaps.views import make_color_limits
    rng = np.random.default_rng(0)
    x = rng.normal(size=10000)
    clim, cmap = make_color_limits(x)
    assert clim == (float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95)))
    # cmap has 100 rows, first row black, last row white
    assert cmap.shape == (100, 4)
    np.testing.assert_array_equal(cmap[0, :3], [0, 0, 0])
    np.testing.assert_array_equal(cmap[-1, :3], [1, 1, 1])


def test_make_color_limits_custom_quantiles():
    from sensmaps.views import make_color_limits
    x = np.arange(100).astype(float)
    clim, _ = make_color_limits(x, quantiles=(0.10, 0.90))
    np.testing.assert_allclose(clim[0], np.quantile(x, 0.10))
    np.testing.assert_allclose(clim[1], np.quantile(x, 0.90))
```

- [ ] **Step 2: Run; verify fail**

```bash
pytest tests/test_views.py::test_make_color_limits_defaults -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `make_color_limits`**

Append to `src/sensmaps/views.py`:

```python
import matplotlib.pyplot as _plt


def make_color_limits(x, quantiles=(0.05, 0.95)):
    """Quantile-based color limits and saturated-ends jet colormap. Port of makeCL.m.

    Parameters
    ----------
    x         : array-like — values whose quantiles set the color limits
    quantiles : (lo, hi) floats in [0, 1]

    Returns
    -------
    clim : (vmin, vmax) tuple of floats
    cmap : ndarray of shape (100, 4) — jet with row 0 black and row 99 white
    """
    arr = np.asarray(x).ravel()
    clim = (float(np.quantile(arr, quantiles[0])), float(np.quantile(arr, quantiles[1])))
    jet = _plt.get_cmap("jet")
    cmap = jet(np.linspace(0, 1, 100))
    cmap[0, :3] = 0.0     # first row black
    cmap[-1, :3] = 1.0    # last row white
    return clim, cmap
```

- [ ] **Step 4: Run; verify pass**

```bash
pytest tests/test_views.py -v
```
Expected: all views tests pass.

- [ ] **Step 5: Populate the public API in `src/sensmaps/__init__.py`**

Every public symbol referenced here is now defined (Task 3: `OpticalProperties`; Task 9: `GridParams`; Task 10: `make_s`; Task 11: `slice_s`; Task 12: `make_color_limits`), so replace the minimal `__init__.py` from Task 1 with:

```python
"""sensmaps — Interactive GUI for diffuse-optical sensitivity maps.

Public API (stable from v1):
    make_s             — compute S for a measurement type
    slice_s            — slice a 3D S array for 2D plotting
    make_color_limits  — compute saturated colormap and limits
    OpticalProperties  — dataclass of optical properties
    GridParams         — dataclass of voxel-grid parameters

More symbols are importable from sensmaps.physics / compute / views / gui
but are not part of the v1 public-API guarantee.
"""

from sensmaps.compute import GridParams, make_s
from sensmaps.physics import OpticalProperties
from sensmaps.views import make_color_limits, slice_s

__version__ = "0.1.0"

__all__ = [
    "GridParams",
    "OpticalProperties",
    "__version__",
    "make_color_limits",
    "make_s",
    "slice_s",
]
```

Sanity-check the import:
```bash
python -c "import sensmaps; print(sensmaps.__version__, sensmaps.__all__)"
```
Expected: `0.1.0 ['GridParams', 'OpticalProperties', '__version__', 'make_color_limits', 'make_s', 'slice_s']`.

- [ ] **Step 6: Commit**

```bash
git add src/sensmaps/__init__.py src/sensmaps/views.py tests/test_views.py
git commit -m "feat(views): make_color_limits and populate public API in __init__

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 13: Views — `render_slice`

**Files:**
- Modify: `/home/giles/GitHub/sensmaps/src/sensmaps/views.py`
- Modify: `/home/giles/GitHub/sensmaps/tests/test_views.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_views.py`:

```python
def test_render_slice_draws_image_contour_labels(tiny_result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sensmaps.views import make_color_limits, render_slice, slice_s
    S, params = tiny_result
    S_plane, pp = slice_s(S, params, axis="y", value=0.0)
    clim, cmap = make_color_limits(S)

    fig, ax = plt.subplots()
    artists = render_slice(ax, S_plane, pp, clim, cmap)

    # Expected artists: AxesImage + contour collections + axis labels
    from matplotlib.image import AxesImage
    has_image = any(isinstance(a, AxesImage) for a in ax.get_images())
    assert has_image
    assert ax.get_xlabel() == pp.horz_label
    assert ax.get_ylabel() == pp.vert_label
    plt.close(fig)
```

- [ ] **Step 2: Run; verify fail**

```bash
pytest tests/test_views.py::test_render_slice_draws_image_contour_labels -v
```
Expected: ImportError for `render_slice`.

- [ ] **Step 3: Implement `render_slice`**

Append to `src/sensmaps/views.py`:

```python
from matplotlib.colors import ListedColormap


def render_slice(ax, S_plane, plot_params, clim, cmap, contour_alpha=0.5):
    """Draw a 2D slice with dashed contour overlay on a matplotlib Axes.

    Port of the imagesc + contour pattern from MATLAB example1_DT.m.

    Parameters
    ----------
    ax           : matplotlib.axes.Axes
    S_plane      : 2D ndarray (vert × horz) from slice_s
    plot_params  : PlotParams from slice_s
    clim         : (vmin, vmax)
    cmap         : (N, 4) RGBA array OR a matplotlib Colormap
    contour_alpha: float in [0, 1] for the overlay gray level

    Returns
    -------
    dict with keys "image", "contour", "colorbar" — the matplotlib artists.
    """
    ax.clear()

    cmap_obj = cmap if hasattr(cmap, "__call__") else ListedColormap(cmap)

    extent = (
        plot_params.horz_axis[0], plot_params.horz_axis[-1],
        plot_params.vert_axis[0], plot_params.vert_axis[-1],
    )
    image = ax.imshow(
        S_plane,
        extent=extent, origin="lower", aspect="equal",
        vmin=clim[0], vmax=clim[1], cmap=cmap_obj,
        interpolation="nearest",
    )
    fig = ax.figure
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(r"$\mathcal{S}$")

    # Contour overlay at colorbar tick values
    levels = colorbar.get_ticks()
    contour = ax.contour(
        plot_params.horz_axis, plot_params.vert_axis, S_plane,
        levels=levels, linestyles="--",
        colors=[(contour_alpha, contour_alpha, contour_alpha)],
        linewidths=0.8,
    )

    ax.set_xlabel(plot_params.horz_label)
    ax.set_ylabel(plot_params.vert_label)
    ax.set_aspect("equal", adjustable="box")

    return {"image": image, "contour": contour, "colorbar": colorbar}
```

- [ ] **Step 4: Run; verify pass**

```bash
pytest tests/test_views.py -v
```
Expected: all views tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/views.py tests/test_views.py
git commit -m "feat(views): render_slice with imshow + dashed contour overlay

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 14: GUI — `PlotCanvas`

**Files:**
- Create: `/home/giles/GitHub/sensmaps/src/sensmaps/gui.py`
- Create: `/home/giles/GitHub/sensmaps/tests/test_gui.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui.py`:

```python
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
```

- [ ] **Step 2: Run; verify fail**

```bash
pytest tests/test_gui.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/sensmaps/gui.py` (PlotCanvas only for now)**

```python
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
```

- [ ] **Step 4: Run; verify pass**

```bash
pytest tests/test_gui.py -v
```
Expected: `test_plot_canvas_constructs_and_shows_slice` passes (or is skipped on headless CI).

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/gui.py tests/test_gui.py
git commit -m "feat(gui): PlotCanvas embedding matplotlib in Tk frame

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 15: GUI — `ParameterPanel`

**Files:**
- Modify: `/home/giles/GitHub/sensmaps/src/sensmaps/gui.py`
- Modify: `/home/giles/GitHub/sensmaps/tests/test_gui.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_gui.py`:

```python
def test_parameter_panel_reads_and_writes_values(tk_root):
    from sensmaps.gui import ParameterPanel
    panel = ParameterPanel(master=tk_root)

    # Default values should round-trip through get_values
    defaults = panel.get_values()
    assert defaults["type_str"] == "CW_SD_I"
    assert defaults["slice_axis"] in ("x", "y", "z")
    assert isinstance(defaults["rs"], list)
    assert len(defaults["rs"]) == 3
    assert isinstance(defaults["opt_prop"], dict)
    assert "musp" in defaults["opt_prop"]

    # Round-trip set_values → get_values
    new_vals = dict(defaults)
    new_vals["opt_prop"] = dict(defaults["opt_prop"], musp=1.2, mua=0.02)
    new_vals["rs"] = [1.0, 2.0, 3.0]
    panel.set_values(new_vals)
    got = panel.get_values()
    assert got["opt_prop"]["musp"] == 1.2
    assert got["opt_prop"]["mua"] == 0.02
    assert got["rs"] == [1.0, 2.0, 3.0]


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
```

- [ ] **Step 2: Run; verify fail**

```bash
pytest tests/test_gui.py::test_parameter_panel_reads_and_writes_values -v
```
Expected: ImportError for `ParameterPanel`.

- [ ] **Step 3: Implement `ParameterPanel`**

Append to `src/sensmaps/gui.py`:

```python
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
```

- [ ] **Step 4: Run; verify pass**

```bash
pytest tests/test_gui.py -v
```
Expected: all GUI tests pass (or are skipped on headless systems).

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/gui.py tests/test_gui.py
git commit -m "feat(gui): ParameterPanel with default values and subscribe hook

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 16: GUI — `MainWindow` wiring, buttons, save, session persistence

**Files:**
- Modify: `/home/giles/GitHub/sensmaps/src/sensmaps/gui.py`
- Modify: `/home/giles/GitHub/sensmaps/tests/test_gui.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_gui.py`:

```python
def test_main_window_constructs_and_recalculates(tk_root, tmp_path, monkeypatch, cw_sd_i_ref):
    """Constructs MainWindow, triggers a recalc, verifies S was computed and cached."""
    monkeypatch.chdir(tmp_path)  # so last_session.json writes into tmp_path
    from sensmaps.gui import MainWindow
    ref = cw_sd_i_ref
    mw = MainWindow(master=tk_root)
    # Set the form to values that match the fixture
    values = {
        "type_str": "CW_SD_I",
        "rs": [0.0, 0.0, 0.0],
        "rd": [float(ref["rho"]), 0.0, 0.0],
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
    vals["rs"] = [1.0, 2.0, 3.0]
    mw1.params_panel.set_values(vals)
    mw1.save_session()

    mw2 = MainWindow(master=tk_root)
    mw2.load_session()
    got = mw2.params_panel.get_values()
    assert got["opt_prop"]["musp"] == 1.7
    assert got["rs"] == [1.0, 2.0, 3.0]


def test_revert_restores_to_last_computed(tk_root, tmp_path, monkeypatch, cw_sd_i_ref):
    monkeypatch.chdir(tmp_path)
    from sensmaps.gui import MainWindow
    ref = cw_sd_i_ref
    mw = MainWindow(master=tk_root)
    values = {
        "type_str": "CW_SD_I",
        "rs": [0.0, 0.0, 0.0],
        "rd": [float(ref["rho"]), 0.0, 0.0],
        "opt_prop": {"n_in": 1.333, "n_out": 1.0, "musp": 1.1, "mua": 0.011},
        "xl": [-5.0, 40.0], "yl": [0.0, 0.0], "zl": [0.0, 20.0],
        "dr": 1.0, "pert": [1.0, 1.0, 1.0],
        "slice_axis": "y", "slice_value": 0.0, "quantiles": [0.05, 0.95],
    }
    mw.params_panel.set_values(values)
    mw.recalculate()
    # Mutate an expensive field
    dirty = dict(values); dirty["rs"] = [5.0, 5.0, 5.0]
    mw.params_panel.set_values(dirty)
    assert mw.is_dirty
    mw.revert()
    got = mw.params_panel.get_values()
    assert got["rs"] == [0.0, 0.0, 0.0]
    assert not mw.is_dirty
```

- [ ] **Step 2: Run; verify fail**

```bash
pytest tests/test_gui.py -v
```
Expected: ImportError for `MainWindow`.

- [ ] **Step 3: Implement `MainWindow`**

Append to `src/sensmaps/gui.py`:

```python
import json
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np

from sensmaps.compute import make_s_full
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

        self.plot_canvas = PlotCanvas(master=right)
        self.plot_canvas.widget.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Action bar
        bar = ttk.Frame(master)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._dirty_label = ttk.Label(bar, text="● in sync", foreground="green")
        self._dirty_label.pack(side=tk.LEFT, padx=8)
        ttk.Button(bar, text="Recalculate", command=self.recalculate
                   ).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(bar, text="Revert", command=self.revert
                   ).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(bar, text="Save Figure…", command=self.save_figure
                   ).pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(bar, text="Save Data…", command=self.save_data
                   ).pack(side=tk.LEFT, padx=4, pady=4)

        # State
        self._cache = None   # SensitivityResult or None
        self._last_inputs: dict | None = None
        self._dirty = False

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def _set_dirty(self, flag: bool) -> None:
        self._dirty = flag
        if flag:
            self._dirty_label.config(text="● form changed — click Recalculate",
                                     foreground="orange")
        else:
            self._dirty_label.config(text="● in sync", foreground="green")

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
        S = self._cache.S
        # Pert is cheap: re-conv Svox with new kernel before slicing
        if name == "pert" and list(values["pert"]) != list(self._cache.pert):
            from scipy.signal import fftconvolve
            dr = self._cache.dr
            kernel_shape = tuple(int(round(p / dr)) for p in values["pert"])
            H = np.ones(kernel_shape, dtype=np.float64)
            S = fftconvolve(self._cache.Svox, H, mode="same")
            # Update cache's S and pert
            self._cache.S = S
            self._cache.pert = tuple(values["pert"])
        self.plot_canvas.show(
            S=S,
            params=self._cache.params,
            axis=values["slice_axis"],
            value=values["slice_value"],
            quantiles=tuple(values["quantiles"]),
        )

    def recalculate(self) -> None:
        values = self.params_panel.get_values()
        op = _opt_prop_from_dict(values["opt_prop"])
        result = make_s_full(
            type_str=values["type_str"],
            rs=np.array([values["rs"]]),
            rd=np.array([values["rd"]]),
            opt_prop=op,
            xl=tuple(values["xl"]),
            yl=tuple(values["yl"]),
            zl=tuple(values["zl"]),
            dr=values["dr"],
            pert=tuple(values["pert"]),
        )
        self._cache = result
        self._last_inputs = values  # deep-enough for the revert round-trip
        self._set_dirty(False)
        self.plot_canvas.show(
            S=result.S, params=result.params,
            axis=values["slice_axis"], value=values["slice_value"],
            quantiles=tuple(values["quantiles"]),
        )

    def revert(self) -> None:
        if self._last_inputs is None:
            return
        self.params_panel.set_values(self._last_inputs)
        self._set_dirty(False)

    def save_figure(self) -> None:
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")],
        )
        if not path:
            return
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
        except Exception:
            return
        self.params_panel.set_values(values)
```

- [ ] **Step 4: Run; verify pass**

```bash
pytest tests/test_gui.py -v
```
Expected: all GUI tests pass (or skip on headless).

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/gui.py tests/test_gui.py
git commit -m "feat(gui): MainWindow with cheap/expensive routing, save, and session

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 17: Entry point — flesh out `__main__.py`

**Files:**
- Modify: `/home/giles/GitHub/sensmaps/src/sensmaps/__main__.py`
- Modify: `/home/giles/GitHub/sensmaps/tests/test_gui.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_gui.py`:

```python
def test_main_smoke_constructs_and_exits(tk_root, tmp_path, monkeypatch):
    """Ensure main() can construct a MainWindow and shut down cleanly."""
    monkeypatch.chdir(tmp_path)
    from sensmaps.__main__ import main
    # --smoke-test: construct, start mainloop, schedule a 100ms destroy, exit 0
    exit_code = main(argv=["--smoke-test"])
    assert exit_code == 0
```

- [ ] **Step 2: Run; verify fail**

```bash
pytest tests/test_gui.py::test_main_smoke_constructs_and_exits -v
```
Expected: main doesn't accept `argv`, raises TypeError.

- [ ] **Step 3: Rewrite `src/sensmaps/__main__.py`**

Replace the stub with:

```python
"""Entry point. `python -m sensmaps` or the `sensmaps` console-script."""
from __future__ import annotations

import argparse
import sys
import tkinter as tk

from sensmaps.gui import MainWindow


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sensmaps")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build the main window, tick Tk once, destroy, and exit 0.",
    )
    args = parser.parse_args(argv)

    root = tk.Tk()
    root.title("sensmaps")
    root.geometry("1100x750")
    window = MainWindow(master=root)
    window.load_session()

    if args.smoke_test:
        root.update()
        root.destroy()
        return 0

    # Persist session on close
    def _on_close() -> None:
        window.save_session()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run; verify pass**

```bash
pytest tests/test_gui.py -v
```
Expected: all tests pass.

Also manually sanity-check (optional, outside the test run):
```bash
sensmaps --smoke-test
```
Expected: exits 0.

- [ ] **Step 5: Commit**

```bash
git add src/sensmaps/__main__.py tests/test_gui.py
git commit -m "feat: __main__ with --smoke-test flag and session persistence on close

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 18: README

**Files:**
- Modify: `/home/giles/GitHub/sensmaps/README.md` (replaces the stub written in Task 1)

- [ ] **Step 1: Overwrite `README.md`**

```markdown
# sensmaps

Interactive Python/Tkinter GUI for exploring 2D slice maps of the Jacobian

$$\mathcal{S}(\vec{r}) = \frac{\partial Y / \partial \mu_a(\vec{r})}{\partial Y / \partial \mu_{a,\text{homogeneous}}}$$

of a detected optical signal `Y` with respect to local absorption changes — "sensitivity to absorption change" maps for diffuse optical spectroscopy and imaging.

This is the Python companion to the MATLAB [`SensitivityCompendium`](https://github.com/DOIT-Lab/DOIT-Public/tree/main/SensitivityCompendium) that accompanies:

> G. Blaney, A. Sassaroli, and S. Fantini, *Spatial sensitivity to absorption changes for various near-infrared spectroscopy methods: A compendium review*, J. Innov. Opt. Health Sci. **17**(04), 2430001 (2024). [DOI](https://doi.org/10.1142/S1793545824300015)

## Status

v1 — CW single-distance intensity (`CW_SD_I`) under diffusion theory. Tkinter GUI, save PNG/PDF + `.npz`. More measurement types in v2; Monte Carlo backend in v3.

## Install

### venv (zero prerequisites beyond Python 3.11+)

```bash
git clone https://github.com/gblane/sensmaps.git
cd sensmaps
python -m venv .venv
source .venv/bin/activate
pip install -e .
sensmaps                     # or: python -m sensmaps
```

### conda

```bash
conda create -n sensmaps python=3.11
conda activate sensmaps
git clone https://github.com/gblane/sensmaps.git
cd sensmaps
pip install -e .
sensmaps
```

### uv

```bash
git clone https://github.com/gblane/sensmaps.git
cd sensmaps
uv venv
source .venv/bin/activate
uv pip install -e .
sensmaps
```

## Using the GUI

| Control             | Class      | What it does                                 |
|---------------------|------------|----------------------------------------------|
| Type                | expensive  | Measurement type (v1: `CW_SD_I`)             |
| rs, rd              | expensive  | Source / detector coordinates [mm]           |
| n_in, n_out         | expensive  | Index of refraction inside / outside         |
| musp, mua           | expensive  | Reduced scattering / absorption [1/mm]       |
| xl, yl, zl, dr      | expensive  | Voxel-grid limits and resolution [mm]        |
| pert                | cheap      | Perturbation box size [mm]                   |
| slice axis, value   | cheap      | Which 2D slice to display                    |
| quantiles           | cheap      | Color-limit quantiles (lo, hi)               |

- **cheap** params redraw the plot live.
- **expensive** params mark the form dirty; click **Recalculate** to rerun the physics.
- **Revert** restores the form to the inputs that produced the currently-displayed map.
- **Save Figure…** writes PNG / PDF / SVG (by extension).
- **Save Data…** writes an `.npz` containing `S`, `Svox`, grid axes, and all inputs.

The last session is saved to `./last_session.json` on exit and reloaded on launch. Delete that file to reset to defaults.

## Development

```bash
pip install -e .[dev]
pytest
```

Regression tests compare the Python port to MATLAB reference values stored as `.mat` fixtures in `tests/fixtures/`. To regenerate them, run `tests/fixtures/generate_fixtures.m` in MATLAB.

## License

TBD — will be added before public release.

## Authorship

Giles Blaney, with development assistance from Claude Code.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with install, usage, and paper citation

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 19: Final verification

**Files:** none.

- [ ] **Step 1: Run the full test suite**

```bash
cd /home/giles/GitHub/sensmaps
source .venv/bin/activate
pytest -v
```
Expected: all physics / compute / views / gui tests pass (or skip on headless).

- [ ] **Step 2: Manual smoke-launch of the GUI**

```bash
sensmaps
```
Expected: window opens, parameter panel populated with defaults, click **Recalculate**, a 2D slice renders in the plot pane. Close the window; verify `./last_session.json` was written. Re-launch; verify the session reloaded.

- [ ] **Step 3: Push**

```bash
git push origin master
```
Expected: remote is up-to-date.

---

## Summary of tasks

1. Project scaffolding.
2. MATLAB fixture generator + reference `.mat`.
3. Physics: `OpticalProperties`, `n2a`.
4. Physics: `complex_fluence`.
5. Physics: `complex_reflectance`.
6. Physics: `complex_tot_path_len`.
7. Physics: `complex_part_path_len` + N-source broadcasting.
8. Physics: `continuous_*` wrappers.
9. Compute: `GridParams`, `parse_type_str`.
10. Compute: `make_s` / `make_s_full` end-to-end vs. fixture.
11. Views: `slice_s`.
12. Views: `make_color_limits`.
13. Views: `render_slice`.
14. GUI: `PlotCanvas`.
15. GUI: `ParameterPanel`.
16. GUI: `MainWindow` wiring, buttons, save, session.
17. Entry point: `__main__.py`.
18. README.
19. Final verification + push.
