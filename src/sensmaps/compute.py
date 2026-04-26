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


def _combine_sd(L, Y, ll):
    """SD combinator: Svox = ll[0] / L[0]. Port of makeS.m line 400.

    Y cancels for SD (single measurement), so it is not used.
    """
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


from scipy.signal import fftconvolve

from sensmaps.physics import (
    OpticalProperties,
    continuous_part_path_len,
    continuous_tot_path_len,
)


def apply_pert_kernel(
    Svox: np.ndarray, pert: Sequence[float], dr: float
) -> np.ndarray:
    """Convolve `Svox` with a uniform box kernel sized `pert` (in mm)."""
    kernel_shape = tuple(int(round(p / dr)) for p in pert)
    H = np.ones(kernel_shape, dtype=np.float64)
    return fftconvolve(Svox, H, mode="same")


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
    # `pert % dr == 0` looks right but is a float-arithmetic trap
    # (1.0 % 0.1 == 0.09999...). Compare against the nearest integer multiple
    # of dr instead, with a small relative tolerance.
    _tol = 1e-9 * max(dr, 1.0)
    if any(abs(p - round(p / dr) * dr) > _tol for p in pert):
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
    
    # Values with z < 0 should not be calculated (set to 0)
    z_coords = r_all[:, 2]
    l_vec[z_coords < 0] = 0.0

    l_vec = np.nan_to_num(l_vec, nan=0.0)
    ll = l_vec.reshape(XX.shape)

    Svox = ll / L_scalar

    S = apply_pert_kernel(Svox, pert, dr)

    return SensitivityResult(
        S=S, Svox=Svox, params=params, type_str=type_str,
        rs=rs_used, rd=rd, opt_prop=opt_prop, pert=tuple(pert), dr=dr,
        Y_per_meas=np.array([1.0]),   # v1.0 path: SD with Y=1
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
