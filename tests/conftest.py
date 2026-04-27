"""Shared pytest fixtures."""
from pathlib import Path

import numpy as np
import pytest
import scipy.io



@pytest.fixture(scope="session")
def cw_sd_i_ref() -> dict:
    """Load the MATLAB reference fixture as a plain dict.

    `scipy.io.loadmat(..., squeeze_me=True)` strips length-1 axes, which
    collapses this fixture's single-row y-axis: `y` comes back as a Python
    scalar and `S`/`Svox` come back as (Nx, Nz) instead of (Nx, 1, Nz).
    Restore the original 3-D shapes so tests can compare directly against
    `make_s`/`make_s_full` output, which is always (Nx, Ny, Nz).
    """
    path = Path(__file__).parent / "fixtures" / "cw_sd_i_example1.mat"
    if not path.exists():
        pytest.skip(f"Reference fixture not found at {path}; run generate_fixtures.m")
    d = scipy.io.loadmat(path, squeeze_me=True)
    d["x"] = np.atleast_1d(np.asarray(d["x"], dtype=float))
    d["y"] = np.atleast_1d(np.asarray(d["y"], dtype=float))
    d["z"] = np.atleast_1d(np.asarray(d["z"], dtype=float))
    shape3d = (d["x"].size, d["y"].size, d["z"].size)
    d["S"] = np.asarray(d["S"], dtype=float).reshape(shape3d)
    d["Svox"] = np.asarray(d["Svox"], dtype=float).reshape(shape3d)
    return d


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
