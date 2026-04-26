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
