"""Analytical diffusion-theory formulas.

All functions are ports of MATLAB code from
DOIT-Public/SensitivityCompendium/deps/. Variable names mirror the
MATLAB source (snake_case) and the formulas follow line-by-line.

Coordinate conventions:
    rs — source coordinates, shape (N_s, 3) — [x, y, z] in mm
    rd — detector coordinates, shape (N_d, 3) or (3,) — [x, y, z] in mm
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


def _split_source(rs):
    """Canonicalize rs to a (N,3) float array; return (rs, x0, y0, z0) where
    x0, y0, z0 are length-N vectors. Permits N=1 (scalar broadcast) or N=N_r
    (one source per row)."""
    rs = np.atleast_2d(np.asarray(rs, dtype=np.float64))
    if rs.shape[1] != 3:
        raise ValueError(f"rs must have shape (N,3); got {rs.shape}")
    return rs, rs[:, 0], rs[:, 1], rs[:, 2]


def complex_fluence(rs, r, omega: float, opt_prop: OpticalProperties):
    """Complex fluence at positions r from a source at rs. Port of complexFluence.m."""
    rs, x0, y0, z0 = _split_source(rs)
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))

    v = C_MM_PER_SEC / opt_prop.n_in
    a_mismatch = n2a(opt_prop.n_in, opt_prop.n_out)
    D = 1.0 / (3.0 * opt_prop.musp)
    zb = -2.0 * a_mismatch * D

    mueff = np.sqrt(opt_prop.mua / D - 1j * omega / (v * D))

    rsp = np.column_stack([x0, y0, -z0 + 2.0 * zb])

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
    """Complex total path length and reflectance. Port of complexTotPathLen.m.

    Returns
    -------
    L : complex ndarray, shape (N,) — total path length [mm]
    R : complex ndarray, shape (N,) — reflectance [1/mm^2]
    """
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
    R_r_rd = complex_reflectance(r, rd, omega, opt_prop)     # (N,) — N voxel sources to 1 detector
    R_rs_rd = complex_reflectance(rs, rd, omega, opt_prop)   # (1,)

    return (phi_rs_r * R_r_rd * V) / R_rs_rd
