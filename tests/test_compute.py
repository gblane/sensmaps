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


def test_make_s_cw_sd_i_matches_matlab(cw_sd_i_ref):
    from sensmaps.compute import make_s
    from sensmaps.physics import OpticalProperties
    ref = cw_sd_i_ref
    op = OpticalProperties(
        n_in=float(ref["nin"]),
        n_out=float(ref["nout"]),
        musp=float(ref["musp"]),
        mua=float(ref["mua"]),
    )
    rs_input = np.asarray(ref["rs"], dtype=float).reshape(1, 3) - np.array([[0, 0, 1 / op.musp]])
    rd = np.asarray(ref["rd"], dtype=float).reshape(1, 3)

    S, params = make_s(
        type_str="CW_SD_I",
        rs=rs_input,
        rd=rd,
        opt_prop=op,
        xl=(float(ref["xl"][0]), float(ref["xl"][1])),
        yl=(float(ref["yl"][0]), float(ref["yl"][1])),
        zl=(float(ref["zl"][0]), float(ref["zl"][1])),
        dr=float(ref["dr"]),
        pert=tuple(float(p) for p in ref["pert"]),
    )
    S_ref = np.asarray(ref["S"], dtype=float)
    assert S.shape == S_ref.shape
    np.testing.assert_allclose(S, S_ref, rtol=1e-8, atol=1e-12)

    # Grid axes match
    np.testing.assert_allclose(params.x, np.asarray(ref["x"], dtype=float))
    np.testing.assert_allclose(params.y, np.asarray(ref["y"], dtype=float))
    np.testing.assert_allclose(params.z, np.asarray(ref["z"], dtype=float))


def test_expand_optodes_sd_returns_z_offset_sources():
    from sensmaps.compute import _expand_optodes
    rs = np.array([[0.0, 0.0, 0.0]])
    rd = np.array([[35.0, 0.0, 0.0]])
    rSrcs, rDets = _expand_optodes("SD", rs, rd, z_offset=1.0 / 1.1)
    np.testing.assert_allclose(rSrcs, [[0.0, 0.0, 1.0 / 1.1]])
    np.testing.assert_array_equal(rDets, rd)


def test_expand_optodes_ss_one_src_two_dets():
    from sensmaps.compute import _expand_optodes
    rs = np.array([[0.0, 0.0, 0.0]])
    rd = np.array([[20.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    z = 1.0 / 1.1
    rSrcs, rDets = _expand_optodes("SS", rs, rd, z_offset=z)
    assert rSrcs.shape == (2, 3)
    np.testing.assert_array_equal(rSrcs[0], rSrcs[1])
    np.testing.assert_allclose(rSrcs[0], [0.0, 0.0, z])  # offset on tiled sources
    np.testing.assert_array_equal(rDets, rd)             # detectors untouched


def test_expand_optodes_ss_two_srcs_one_det():
    from sensmaps.compute import _expand_optodes
    rs = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    rd = np.array([[35.0, 0.0, 0.0]])
    rSrcs, rDets = _expand_optodes("SS", rs, rd, z_offset=0.0)
    np.testing.assert_array_equal(rSrcs, rs)
    assert rDets.shape == (2, 3)
    np.testing.assert_array_equal(rDets[0], rDets[1])
    np.testing.assert_array_equal(rDets[0], rd[0])


def test_expand_optodes_ds_meas_pattern():
    from sensmaps.compute import _expand_optodes
    rs = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    rd = np.array([[30.0, 0, 0], [40.0, 0, 0]])
    rSrcs, rDets = _expand_optodes("DS", rs, rd, z_offset=0.0)
    # MATLAB pattern: rSrcs = [rs1; rs1; rs2; rs2]
    np.testing.assert_array_equal(rSrcs[[0, 1]], np.tile(rs[0], (2, 1)))
    np.testing.assert_array_equal(rSrcs[[2, 3]], np.tile(rs[1], (2, 1)))
    # rDets = [rd; flipud(rd)] = [rd1; rd2; rd2; rd1]
    np.testing.assert_array_equal(rDets[0], rd[0])
    np.testing.assert_array_equal(rDets[1], rd[1])
    np.testing.assert_array_equal(rDets[2], rd[1])
    np.testing.assert_array_equal(rDets[3], rd[0])


def test_expand_optodes_rejects_bad_shapes():
    from sensmaps.compute import _expand_optodes
    rs1 = np.array([[0.0, 0, 0]])
    rs2 = np.array([[0.0, 0, 0], [10, 0, 0]])
    rd1 = np.array([[35.0, 0, 0]])
    rd2 = np.array([[30.0, 0, 0], [40, 0, 0]])
    with pytest.raises(ValueError, match="'SD'"):
        _expand_optodes("SD", rs2, rd1, z_offset=0.0)
    with pytest.raises(ValueError, match="'SS'"):
        _expand_optodes("SS", rs1, rd1, z_offset=0.0)
    with pytest.raises(ValueError, match="'DS'"):
        _expand_optodes("DS", rs1, rd2, z_offset=0.0)
    with pytest.raises(ValueError, match="Unknown arrangement"):
        _expand_optodes("BOGUS", rs1, rd1, z_offset=0.0)


def test_make_s_rejects_unsupported_type():
    from sensmaps.compute import make_s
    from sensmaps.physics import OpticalProperties
    with pytest.raises(NotImplementedError, match="CW_SD_I"):
        make_s(
            type_str="FD_SS_P",
            rs=np.array([[0, 0, 0]]),
            rd=np.array([[25, 0, 0]]),
            opt_prop=OpticalProperties(),
            xl=(-5, 40), yl=(0, 0), zl=(0, 20), dr=1.0,
        )


def test_combine_sd_basic():
    from sensmaps.compute import _combine_sd
    L = [10.0]
    Y = [1.0]
    ll = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    out = _combine_sd(L, Y, ll)
    np.testing.assert_allclose(out, np.array([[0.1, 0.2], [0.3, 0.4]]))


def test_combine_ss_two_meas():
    from sensmaps.compute import _combine_ss
    L = [10.0, 20.0]
    Y = [1.0, 1.0]
    g0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    g1 = np.array([[5.0, 6.0], [7.0, 8.0]])
    out = _combine_ss(L, Y, [g0, g1])
    expected = (g1 - g0) / (20.0 - 10.0)
    np.testing.assert_allclose(out, expected)


def test_combine_ds_four_meas():
    from sensmaps.compute import _combine_ds
    L = [10.0, 20.0, 30.0, 40.0]
    Y = [1.0, 1.0, 1.0, 1.0]
    g = [np.full((2, 2), float(k + 1)) for k in range(4)]
    out = _combine_ds(L, Y, g)
    num = (g[1] - g[0]) + (g[3] - g[2])
    den = (20 - 10) + (40 - 30)
    np.testing.assert_allclose(out, num / den)


def test_combine_ss_uses_y_weight():
    """Y!=1 — verifies Y is actually applied (matters for v1.3's T/V types)."""
    from sensmaps.compute import _combine_ss
    L = [1.0, 1.0]
    Y = [2.0, 3.0]
    g0 = np.full((2, 2), 1.0)
    g1 = np.full((2, 2), 1.0)
    out = _combine_ss(L, Y, [g0, g1])
    np.testing.assert_allclose(out, (3.0 * 1.0 - 2.0 * 1.0) / (3.0 - 2.0) * np.ones((2, 2)))


def test_combine_ds_uses_y_weight():
    """DS analogue of the SS Y-weight test: confirms Y is applied across all 4 measurements."""
    from sensmaps.compute import _combine_ds
    L = [1.0, 1.0, 1.0, 1.0]
    Y = [2.0, 3.0, 5.0, 7.0]
    g0 = np.full((2, 2), 1.0)
    g1 = np.full((2, 2), 1.0)
    g2 = np.full((2, 2), 1.0)
    g3 = np.full((2, 2), 1.0)
    out = _combine_ds(L, Y, [g0, g1, g2, g3])
    expected_num = (3.0 * 1.0 - 2.0 * 1.0) + (7.0 * 1.0 - 5.0 * 1.0)  # 1 + 2 = 3
    expected_den = (3.0 * 1.0 - 2.0 * 1.0) + (7.0 * 1.0 - 5.0 * 1.0)  # 1 + 2 = 3
    np.testing.assert_allclose(out, np.full((2, 2), expected_num / expected_den))


def test_sensitivity_result_kw_only_new_fields():
    """v1.1 adds Y_per_meas (required, kw-only) and fmod (optional, kw-only)."""
    from sensmaps.compute import GridParams, SensitivityResult
    from sensmaps.physics import OpticalProperties
    g = GridParams.from_limits(xl=(0, 1), yl=(0, 1), zl=(0, 1), dr=1.0)
    arr = np.zeros((2, 2, 2))
    op = OpticalProperties()
    # All new fields must be passed by keyword.
    r = SensitivityResult(
        S=arr, Svox=arr, params=g, type_str="CW_SD_I",
        rs=np.zeros((1, 3)), rd=np.zeros((1, 3)),
        opt_prop=op, pert=(1.0, 1.0, 1.0), dr=1.0,
        Y_per_meas=np.array([1.0]),       # required kw-only
        # fmod omitted on purpose — defaults to None.
    )
    assert r.fmod is None
    np.testing.assert_array_equal(r.Y_per_meas, [1.0])
