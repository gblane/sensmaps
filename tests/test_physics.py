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
        np.testing.assert_allclose(a, a_ref, rtol=1e-10, atol=0)


def test_n2a_unit_case_is_one():
    assert n2a(1.0, 1.0) == 1.0


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
    np.testing.assert_allclose(phi[0], float(complex(ref["phi_test_cw"]).real), rtol=1e-10)
    np.testing.assert_allclose(R[0], float(complex(ref["R_test_cw"]).real), rtol=1e-10)
    np.testing.assert_allclose(L[0], float(complex(ref["L_test_cw"]).real), rtol=1e-10)
    np.testing.assert_allclose(l[0], float(complex(ref["l_test_cw"]).real), rtol=1e-10)
