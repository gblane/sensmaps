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
