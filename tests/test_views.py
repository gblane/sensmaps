"""Tests for sensmaps.views."""
import numpy as np
import pytest


@pytest.fixture
def tiny_result(cw_sd_i_ref):
    """Build a SensitivityResult-like namespace from the fixture."""
    from sensmaps.compute import GridParams
    x = np.asarray(cw_sd_i_ref["x"], dtype=float)
    y = np.asarray(cw_sd_i_ref["y"], dtype=float)
    z = np.asarray(cw_sd_i_ref["z"], dtype=float)
    S = np.asarray(cw_sd_i_ref["S"], dtype=float)
    params = GridParams(
        x=x, y=y, z=z,
        xl_valid=(float(x[0]), float(x[-1])),
        yl_valid=(float(y[0]), float(y[-1])),
        zl_valid=(float(z[0]), float(z[-1])),
    )
    return S, params


def test_slice_s_y_axis_shape_and_labels(tiny_result):
    from sensmaps.views import slice_s
    S, params = tiny_result
    S_plane, pp = slice_s(S, params, axis="y", value=0.0)
    # S shape (Nx, Ny, Nz) → slice at y=0 gives (Nx, Nz), then transpose → (Nz, Nx)
    assert S_plane.shape == (params.z.size, params.x.size)
    assert pp.horz_label == "$x$ (mm)"
    assert pp.vert_label == "$z$ (mm)"
    np.testing.assert_array_equal(pp.horz_axis, params.x)
    np.testing.assert_array_equal(pp.vert_axis, params.z)


def test_slice_s_rejects_bad_axis(tiny_result):
    from sensmaps.views import slice_s
    S, params = tiny_result
    with pytest.raises(ValueError, match="axis"):
        slice_s(S, params, axis="bogus", value=0.0)


def test_slice_s_snaps_to_nearest_axis_value(tiny_result):
    from sensmaps.views import slice_s
    S, params = tiny_result
    S_plane_a, _ = slice_s(S, params, axis="y", value=0.0)
    S_plane_b, _ = slice_s(S, params, axis="y", value=0.3)  # snaps to 0
    np.testing.assert_array_equal(S_plane_a, S_plane_b)


def test_make_color_limits_defaults():
    from sensmaps.views import make_color_limits
    rng = np.random.default_rng(0)
    x = rng.normal(size=10000)
    clim, cmap = make_color_limits(x)
    assert clim == (float(np.quantile(x, 0.05)), float(np.quantile(x, 0.95)))
    # cmap has 100 rows, first row black, last row white
    assert cmap.shape == (100, 4)
    np.testing.assert_array_equal(cmap[0, :3], [0, 0, 0])
    np.testing.assert_array_equal(cmap[-1, :3], [1, 1, 1])


def test_make_color_limits_custom_quantiles():
    from sensmaps.views import make_color_limits
    x = np.arange(100).astype(float)
    clim, _ = make_color_limits(x, quantiles=(0.10, 0.90))
    np.testing.assert_allclose(clim[0], np.quantile(x, 0.10))
    np.testing.assert_allclose(clim[1], np.quantile(x, 0.90))
