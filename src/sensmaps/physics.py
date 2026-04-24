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

import numpy as np


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
