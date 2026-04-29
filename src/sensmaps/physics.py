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


C_MM_PER_PS = C_MM_PER_SEC * 1e-12  # 0.299792458 mm/ps


def temporal_reflectance(rs, rd, t, opt_prop: OpticalProperties):
    """Time-resolved reflectance R(t). Port of temporalReflectance.m (DT branch).

    Parameters
    ----------
    rs : (Ns, 3) — source coords [mm]; either Ns=1 or Nd=1.
    rd : (Nd, 3) — detector coords [mm].
    t  : (Nt,)   — time samples [ps]. Entries with t<=0 yield R=0.
    opt_prop : OpticalProperties

    Returns
    -------
    R : (max(Ns, Nd), Nt) array — temporal reflectance [1/(ps·mm^2)].
    """
    rs, x0, y0, z0 = _split_source(rs)
    rd = np.atleast_2d(np.asarray(rd, dtype=np.float64))
    t  = np.asarray(t, dtype=np.float64).ravel()

    if rs.shape[0] > 1 and rd.shape[0] > 1:
        raise ValueError("Cannot use multiple sources and multiple detectors")

    v = C_MM_PER_PS / opt_prop.n_in
    A = n2a(opt_prop.n_in, opt_prop.n_out)
    D = 1.0 / (3.0 * opt_prop.musp)
    zb = -2.0 * A * D
    mua = opt_prop.mua

    rsp = np.column_stack([x0, y0, -z0 + 2.0 * zb])
    r1 = np.linalg.norm(rd - rs, axis=1)   # (N,)
    r2 = np.linalg.norm(rd - rsp, axis=1)  # (N,)

    N = max(rs.shape[0], rd.shape[0])
    R = np.zeros((N, t.size), dtype=np.float64)
    pos = t > 0
    if np.any(pos):
        tp = t[pos][np.newaxis, :]                     # (1, Np)
        z0c = np.atleast_1d(z0)[:, np.newaxis]         # (Ns, 1) — broadcasts to N
        r1c = r1[:, np.newaxis]
        r2c = r2[:, np.newaxis]
        prefactor = np.exp(-mua * v * tp) / (
            (4.0 * np.pi * D * v) ** 1.5 * tp**2.5
        )
        bracket = (
            z0c * np.exp(-r1c**2 / (4.0 * D * v * tp))
            + (z0c - 2.0 * zb) * np.exp(-r2c**2 / (4.0 * D * v * tp))
        )
        R[:, pos] = 0.5 * prefactor * bracket
    return R


def temporal_fluence(rs, r, t, opt_prop: OpticalProperties):
    """Time-resolved fluence Φ(t). Port of temporalFluence.m.

    Parameters
    ----------
    rs : (Ns, 3) — source coords [mm]; either Ns=1 or Nr=1.
    r  : (Nr, 3) — interior position coords [mm].
    t  : (Nt,)   — time samples [ps]. Entries with t<=0 yield Φ=0.
    opt_prop : OpticalProperties

    Returns
    -------
    PHI : (max(Ns, Nr), Nt) array — fluence [1/(ps·mm^2)].
    """
    rs, x0, y0, z0 = _split_source(rs)
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    t = np.asarray(t, dtype=np.float64).ravel()

    if rs.shape[0] > 1 and r.shape[0] > 1:
        raise ValueError("Cannot use multiple sources and multiple positions")

    v = C_MM_PER_PS / opt_prop.n_in
    A = n2a(opt_prop.n_in, opt_prop.n_out)
    D = 1.0 / (3.0 * opt_prop.musp)
    zb = -2.0 * A * D
    mua = opt_prop.mua

    rsp = np.column_stack([x0, y0, -z0 + 2.0 * zb])
    r1 = np.linalg.norm(r - rs, axis=1)
    r2 = np.linalg.norm(r - rsp, axis=1)

    N = max(rs.shape[0], r.shape[0])
    PHI = np.zeros((N, t.size), dtype=np.float64)
    pos = t > 0
    if np.any(pos):
        tp = t[pos][np.newaxis, :]
        r1c = r1[:, np.newaxis]
        r2c = r2[:, np.newaxis]
        prefactor = (v * np.exp(-mua * v * tp)) / (4.0 * np.pi * D * v * tp) ** 1.5
        bracket = (
            np.exp(-r1c**2 / (4.0 * D * v * tp))
            - np.exp(-r2c**2 / (4.0 * D * v * tp))
        )
        PHI[:, pos] = prefactor * bracket
    return PHI


def temporal_kth_moment(rs, rd, k: int, opt_prop: OpticalProperties):
    """Kth moment of t for the temporal point-spread function: ⟨t^k⟩ [ps^k].

    Port of `temporalKthMoment.m`. Closed-form analytic for k ∈ {1, 2, 3, 4}.
    """
    if int(k) not in (1, 2, 3, 4):
        raise ValueError(f"k must be in {{1,2,3,4}}; got {k}")
    k = int(k)

    rs, x0, y0, z0 = _split_source(rs)
    rd = np.atleast_2d(np.asarray(rd, dtype=np.float64))

    if rs.shape[0] > 1 and rd.shape[0] > 1:
        raise ValueError("Cannot use multiple sources and multiple detectors")

    v = C_MM_PER_PS / opt_prop.n_in
    A = n2a(opt_prop.n_in, opt_prop.n_out)
    D = 1.0 / (3.0 * opt_prop.musp)
    zb = -2.0 * A * D
    mua = opt_prop.mua
    mueff = np.sqrt(mua / D)

    rsp = np.column_stack([x0, y0, -z0 + 2.0 * zb])
    r1 = np.linalg.norm(rd - rs, axis=1)
    r2 = np.linalg.norm(rd - rsp, axis=1)

    z0a = np.atleast_1d(z0).astype(np.float64)
    R_C = continuous_reflectance(rs, rd, opt_prop)

    if k == 1:
        out = (
            (z0a / r1) * np.exp(-mueff * r1)
            + ((z0a - 2.0 * zb) / r2) * np.exp(-mueff * r2)
        ) / (8.0 * np.pi * v * D * R_C)
    elif k == 2:
        out = (
            z0a * np.exp(-mueff * r1)
            + (z0a - 2.0 * zb) * np.exp(-mueff * r2)
        ) / (16.0 * np.pi * (v * D) ** 2 * mueff * R_C)
    elif k == 3:
        out = (
            z0a * (r1 + 1.0 / mueff) * np.exp(-mueff * r1)
            + (z0a - 2.0 * zb) * (r2 + 1.0 / mueff) * np.exp(-mueff * r2)
        ) / (32.0 * np.pi * D**2 * v**3 * mua * R_C)
    else:  # k == 4
        out = (
            z0a * (r1**2 + 3.0 * r1 / mueff + 3.0 / mueff**2) * np.exp(-mueff * r1)
            + (z0a - 2.0 * zb) * (r2**2 + 3.0 * r2 / mueff + 3.0 / mueff**2) * np.exp(-mueff * r2)
        ) / (64.0 * np.pi * D ** 2.5 * mua ** 1.5 * v**4 * R_C)
    return out


def temporal_kth_mom_tot_path_len(rs, rd, k: int, opt_prop: OpticalProperties):
    """Total path length L for kth moment of t. Port of temporalKthMomTotPathLen.m.

    Closed-form via three calls to `temporal_kth_moment`. Effective k ∈ {1, 2, 3}
    because the formula needs ⟨t^(k+1)⟩, and `temporal_kth_moment` caps k at 4.
    """
    v = C_MM_PER_PS / opt_prop.n_in
    t1 = temporal_kth_moment(rs, rd, 1, opt_prop)
    tk = temporal_kth_moment(rs, rd, int(k), opt_prop)
    tkp1 = temporal_kth_moment(rs, rd, int(k) + 1, opt_prop)
    return -(v * (t1 * tk - tkp1)) / tk


def temporal_gate_tot_path_len(rs, rd, tg, opt_prop: OpticalProperties,
                                *, conv_t: float = 10000.0,
                                conv_dt: float = 1.0):
    """Gated total path length L for time-domain GI. Port of temporalGateTotPathLen.m.

    Parameters
    ----------
    rs : (1, 3) — source coords [mm].
    rd : (1, 3) — detector coords [mm].
    tg : (2,)   — [t_start, t_end] gate edges [ps].
    opt_prop : OpticalProperties
    conv_t  : float — half-width of convolution time window [ps] (default 10000).
    conv_dt : float — convolution time step [ps] (default 1).

    Returns
    -------
    L : float — gated total path length [mm].
    """
    tg = np.asarray(tg, dtype=np.float64).ravel()
    if tg.size != 2:
        raise ValueError(f"tg must have 2 elements, got {tg.size}")

    # Values of R(t) for t > tg[1] never enter the gate integral, so cap the
    # effective time window at tg[1]. Bit-identical to the full window.
    conv_t_eff = min(float(conv_t), float(tg[1]))

    v = C_MM_PER_PS / opt_prop.n_in
    t = np.arange(-conv_t_eff, conv_t_eff + conv_dt / 2.0, conv_dt)  # mirror MATLAB colon
    R_t = temporal_reflectance(rs, rd, t, opt_prop)          # (1, Nt)

    i1 = int(np.argmin(np.abs(t - tg[0])))
    i2 = int(np.argmin(np.abs(t - tg[1])))
    sl = slice(i1, i2 + 1)

    num = np.trapezoid(v * t[sl] * R_t[:, sl], t[sl], axis=1)
    R_g = np.trapezoid(R_t[:, sl], t[sl], axis=1)
    return float((num / R_g)[0])


def temporal_gate_part_path_len(rs, r, rd, V: float, tg,
                                 opt_prop: OpticalProperties,
                                 *, conv_t: float = 10000.0,
                                 conv_dt: float = 1.0):
    """Gated partial path length per voxel. Port of temporalGatePartPathLen.m
    (FFT-convolution branch, no parfor; matches makeS.m default invocation).

    Parameters
    ----------
    rs : (1, 3)  — source coords [mm].
    r  : (Nr, 3) — voxel centers [mm].
    rd : (1, 3)  — detector coords [mm].
    V  : float   — voxel volume [mm^3].
    tg : (2,)    — [t_start, t_end] gate edges [ps].
    opt_prop : OpticalProperties
    conv_t, conv_dt : convolution window/step [ps].

    Returns
    -------
    l : (Nr,) — gated partial path length [mm].
    """
    tg = np.asarray(tg, dtype=np.float64).ravel()
    if tg.size != 2:
        raise ValueError(f"tg must have 2 elements, got {tg.size}")
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))

    # PHI(t') and R(t') for t' > tg[1] never enter the gate integral; cap the
    # effective window at tg[1] for a bit-identical but much faster computation.
    conv_t_eff = min(float(conv_t), float(tg[1]))
    t = np.arange(-conv_t_eff, conv_t_eff + conv_dt / 2.0, conv_dt)
    pos = t > 0
    n_pos = int(np.sum(pos))
    t_pos = t[pos]

    # Source→detector reflectance for normalization (gate integral)
    Rsd_t = temporal_reflectance(rs, rd, t, opt_prop)[0]      # (Nt,)
    i1 = int(np.argmin(np.abs(t - tg[0])))
    i2 = int(np.argmin(np.abs(t - tg[1])))
    Rsd_g = np.trapezoid(Rsd_t[i1:i2 + 1], t[i1:i2 + 1])

    # Source→voxel fluence and voxel→detector reflectance on positive-t only.
    PHI_pos = temporal_fluence(rs, r, t_pos, opt_prop)            # (Nr, n_pos)
    R_pos   = temporal_reflectance(r, rd, t_pos, opt_prop)        # (Nr, n_pos)

    # Linear convolution via real-FFT on the compact (positive-t-only) signals.
    # MATLAB's `tmp = ifft(fft(PHIsi).*fft(Rid)); tmp(1:n_pos)` (with PHIsi/Rid
    # zero-padded to length Nt) is equivalent to:
    #   conv_PR[:, 0]  = 0  (gate point at conv_dt — empty integration interval)
    #   conv_PR[:, k]  = (PHI_pos * R_pos)[k-1]   for k >= 1
    # by shifting the leading-zero offset out of the FFT (halves the transform
    # length, ~3× speedup at the default ndt=10000).
    import scipy.fft as _sfft
    M = _sfft.next_fast_len(2 * n_pos - 1, real=True)
    fft_PHI = _sfft.rfft(PHI_pos, n=M, axis=1, workers=-1)
    fft_R   = _sfft.rfft(R_pos,   n=M, axis=1, workers=-1)
    c_compact = _sfft.irfft(fft_PHI * fft_R, n=M, axis=1, workers=-1)
    conv_PR = np.zeros((r.shape[0], n_pos), dtype=np.float64)
    conv_PR[:, 1:] = c_compact[:, : n_pos - 1]

    # Map gate edges into the positive-only axis and integrate (rectangular sum × dt)
    j1 = int(np.argmin(np.abs(t[i1] - t_pos)))
    j2 = int(np.argmin(np.abs(t[i2] - t_pos)))
    num = np.sum(conv_PR[:, j1:j2 + 1], axis=1) * conv_dt

    return (num / Rsd_g) * V


def temporal_kth_mom_part_path_len(rs, r, rd, V: float, k: int,
                                    opt_prop: OpticalProperties,
                                    *, conv_t: float = 10000.0,
                                    conv_dt: float = 1.0):
    """Partial path length per voxel for the kth moment ⟨t^k⟩.

    Port of `temporalKthMomPartPathLen.m` (FFT-conv branch, no parfor).
    """
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    k = int(k)

    t = np.arange(-float(conv_t), float(conv_t) + conv_dt / 2.0, conv_dt)
    pos = t > 0
    n_pos = int(np.sum(pos))
    t_pos = t[pos]

    tk = float(np.asarray(temporal_kth_moment(rs, rd, k, opt_prop)).ravel()[0])
    RC = float(np.asarray(continuous_reflectance(rs, rd, opt_prop)).ravel()[0])
    lC = continuous_part_path_len(rs, r, rd, V, opt_prop)

    PHI_pos = temporal_fluence(rs, r, t_pos, opt_prop)
    R_pos   = temporal_reflectance(r, rd, t_pos, opt_prop)

    import scipy.fft as _sfft
    M = _sfft.next_fast_len(2 * n_pos - 1, real=True)
    fft_PHI = _sfft.rfft(PHI_pos, n=M, axis=1, workers=-1)
    fft_R   = _sfft.rfft(R_pos,   n=M, axis=1, workers=-1)
    c_compact = _sfft.irfft(fft_PHI * fft_R, n=M, axis=1, workers=-1)
    conv_PR = np.zeros((r.shape[0], n_pos), dtype=np.float64)
    conv_PR[:, 1:] = c_compact[:, : n_pos - 1]

    weighted = (t_pos ** k) * conv_PR
    integral = np.sum(weighted, axis=1) * (conv_dt ** 2)
    d = lC * tk - (V / RC) * integral
    return -d / tk


def temporal_var(rs, rd, opt_prop: OpticalProperties):
    """Variance of t for the temporal point-spread function: ⟨t²⟩ − ⟨t⟩² [ps²].

    Port of `temporalVar.m`.
    """
    t1 = temporal_kth_moment(rs, rd, 1, opt_prop)
    t2 = temporal_kth_moment(rs, rd, 2, opt_prop)
    return t2 - t1 ** 2


def temporal_var_tot_path_len(rs, rd, opt_prop: OpticalProperties):
    """Total path length L for variance. Port of `temporalVarTotPathLen.m`.

    L = (⟨t²⟩·L₂ − 2·⟨t⟩²·L₁) / (⟨t²⟩ − ⟨t⟩²)
    """
    t1 = temporal_kth_moment(rs, rd, 1, opt_prop)
    t2 = temporal_kth_moment(rs, rd, 2, opt_prop)
    Vvar = t2 - t1 ** 2
    L1 = temporal_kth_mom_tot_path_len(rs, rd, 1, opt_prop)
    L2 = temporal_kth_mom_tot_path_len(rs, rd, 2, opt_prop)
    return (t2 * L2 - 2.0 * t1 ** 2 * L1) / Vvar


def temporal_var_part_path_len(rs, r, rd, V: float,
                                opt_prop: OpticalProperties,
                                *, conv_t: float = 10000.0,
                                conv_dt: float = 1.0):
    """Partial path length per voxel for variance. Port of `temporalVarPartPathLen.m`.

    l = (⟨t²⟩·l₂ − 2·⟨t⟩²·l₁) / (⟨t²⟩ − ⟨t⟩²); NaNs → 0.
    """
    t1_arr = np.asarray(temporal_kth_moment(rs, rd, 1, opt_prop)).ravel()[0]
    t2_arr = np.asarray(temporal_kth_moment(rs, rd, 2, opt_prop)).ravel()[0]
    Vvar = t2_arr - t1_arr ** 2
    l1 = temporal_kth_mom_part_path_len(rs, r, rd, V, 1, opt_prop,
                                         conv_t=conv_t, conv_dt=conv_dt)
    l2 = temporal_kth_mom_part_path_len(rs, r, rd, V, 2, opt_prop,
                                         conv_t=conv_t, conv_dt=conv_dt)
    out = (t2_arr * l2 - 2.0 * t1_arr ** 2 * l1) / Vvar
    out = np.where(np.isnan(out), 0.0, out)
    return out
