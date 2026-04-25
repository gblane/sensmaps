"""Top-level dispatcher: make_s(type_str, rs, rd, opt_prop, **kw).

Mirrors MATLAB makeS.m: parses the AA_BB_C type string, builds a voxel
grid, dispatches to physics functions, applies the perturbation
convolution, and returns (S, params).

v1 only implements CW_SD_I in DT mode. The dispatch-table scaffolding is
laid out to make v2 additions a single-row change.
"""
from __future__ import annotations

from dataclasses import dataclass
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

    # Partial path length per voxel. The diffusion-theory kernel diverges at
    # r1=0 (voxel coincident with source), producing NaN there — handled by
    # nan_to_num below; the resulting RuntimeWarnings are not informative.
    with np.errstate(divide="ignore", invalid="ignore"):
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
