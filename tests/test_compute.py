"""Tests for sensmaps.compute — dispatcher, parsing, make_s."""
import numpy as np
import pytest


def test_grid_params_from_limits():
    from sensmaps.compute import GridParams
    g = GridParams.from_limits(xl=(-5, 5), yl=(0, 0), zl=(0, 10), dr=1.0)
    np.testing.assert_array_equal(g.x, np.arange(-5, 6, 1.0))
    np.testing.assert_array_equal(g.y, np.array([0.0]))
    np.testing.assert_array_equal(g.z, np.arange(0, 11, 1.0))


def test_parse_type_str_valid():
    from sensmaps.compute import parse_type_str
    t = parse_type_str("CW_SD_I")
    assert t.temporal == "CW"
    assert t.arrangement == "SD"
    assert t.data_type == "I"


def test_parse_type_str_invalid():
    from sensmaps.compute import parse_type_str
    with pytest.raises(ValueError, match="AA_BB_C"):
        parse_type_str("bogus")
