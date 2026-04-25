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
