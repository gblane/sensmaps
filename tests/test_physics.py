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
