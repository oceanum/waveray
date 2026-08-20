"""Wind forcing against SWAN's source-term formulations in closed form.

Flat-bottom domains give straight backward rays of known length, so the
Komen et al. (1984) exponential gain and the Cavaleri & Malanotte-Rizzoli
(1981) linear growth integrate to closed forms the operator must reproduce.
Expected values are recomputed here from the SWAN formulas with the
constants written out, pinning the implementation to the reference.
"""

import numpy as np
import pytest
import xarray as xr

from waveray.bathymetry import LocalGrid
from waveray.dispersion import GRAV, group_speed, phase_speed
from waveray.model import SiteModel
from waveray.operator import TransferOperator, build_operator
from waveray.wind import (
    RHO_AIR_WATER,
    WindField,
    as_wind_field,
    drag_coefficient,
    friction_velocity,
)

DIRS = np.arange(0.0, 360.0, 10.0)
F = 0.1  # Hz
DEPTH = 20.0
U10 = 20.0
LENGTH = 5_000.0  # target at the centre -> straight ray to the west edge


def flat_grid(depth=DEPTH, size=10_000.0, n=101):
    x = np.linspace(-size / 2, size / 2, n)
    y = np.linspace(-size / 2, size / 2, n)
    return LocalGrid(x=x, y=y, depth=np.full((n, n), depth))


def build(grid, **kwargs):
    kwargs.setdefault("target_xy", (0.0, 0.0))
    kwargs.setdefault("boundary_xy", np.array([[-5_000.0, 0.0]]))
    kwargs.setdefault("freqs", np.array([F]))
    kwargs.setdefault("dirs", DIRS)
    kwargs.setdefault("nsub", 9)
    kwargs.setdefault("cf_jonswap", None)
    return build_operator(grid, **kwargs)


def probe_270():
    """Unit density in the coming-from-west boundary bin."""
    e_b = np.zeros((1, 1, DIRS.size))
    e_b[0, 0, DIRS == 270.0] = 1.0
    return e_b


def komen_gain(f, depth, u10, length):
    """exp(B L / cg) for a following wind over a straight flat-bottom ray."""
    omega = 2.0 * np.pi * f
    c = float(phase_speed(omega, np.array(depth)))
    cg = float(group_speed(omega, np.array(depth)))
    us = float(friction_velocity(u10))
    b = max(0.0, 0.25 * RHO_AIR_WATER * (28.0 * us / c - 1.0)) * omega
    return np.exp(b / cg * length)


# --------------------------------------------------------------------- #
def test_drag_law_zijlema_2012():
    """Cd = (0.55 + 2.97 U~ - 1.49 U~^2) 1e-3, U~ = U10 / 31.5 (SWAN default)."""
    assert np.isclose(drag_coefficient(0.0), 0.55e-3)
    assert np.isclose(drag_coefficient(31.5), (0.55 + 2.97 - 1.49) * 1e-3)
    u = 20.0
    ut = u / 31.5
    assert np.isclose(drag_coefficient(u), (0.55 + 2.97 * ut - 1.49 * ut**2) * 1e-3)
    assert np.isclose(friction_velocity(u), np.sqrt(drag_coefficient(u)) * u)
    assert friction_velocity(0.0) == 0.0
    # the fit is clipped at zero far beyond its validity range
    assert drag_coefficient(100.0) == 0.0


def test_swan_density_ratio_constant():
    assert RHO_AIR_WATER == 0.00125  # SWAN PWIND(9)


def test_uniform_wind_direction_convention():
    """Wind direction is coming-from nautical: from the west -> u* points east."""
    grid = flat_grid(n=11)
    west = WindField.uniform(grid, 10.0, 270.0)
    assert np.all(west.usx > 0) and np.allclose(west.usy, 0.0, atol=1e-12)
    north = WindField.uniform(grid, 10.0, 0.0)
    assert np.allclose(north.usx, 0.0, atol=1e-12) and np.all(north.usy < 0)


def test_following_wind_matches_analytic_komen_gain():
    """Uniform following wind: operator gain equals exp(B L / cg)."""
    grid = flat_grid()
    op_free = build(grid)
    op_wind = build(grid, wind=(U10, 270.0))
    e_b = probe_270()
    ratio = op_wind.apply(e_b).sum() / op_free.apply(e_b).sum()
    expected = komen_gain(F, DEPTH, U10, LENGTH)
    assert expected > 1.05  # the case must exercise real growth
    assert np.isclose(ratio, expected, rtol=2e-2), (ratio, expected)


def test_opposing_and_cross_wind_do_not_grow():
    """Komen input is max[0, ...]: no growth against or across the wind."""
    grid = flat_grid()
    op_free = build(grid)
    e_b = probe_270()
    base = op_free.apply(e_b).sum()
    for wdir in (90.0, 0.0):  # opposing, cross
        op = build(grid, wind=(U10, wdir))
        assert np.isclose(op.apply(e_b).sum() / base, 1.0, rtol=1e-12)


def test_gridded_constant_components_equal_uniform():
    """A constant u10/v10 Dataset must reproduce the uniform tuple exactly."""
    grid = flat_grid()
    xs = np.linspace(-6_000.0, 6_000.0, 13)
    ys = np.linspace(-6_000.0, 6_000.0, 11)
    shape = (ys.size, xs.size)
    wind_ds = xr.Dataset(
        {"u10": (("y", "x"), np.full(shape, U10)), "v10": (("y", "x"), np.zeros(shape))},
        coords={"x": xs, "y": ys},
    )
    op_u = build(grid, wind=(U10, 270.0))
    op_g = build(grid, wind=wind_ds)
    assert np.allclose(op_u.T, op_g.T, rtol=0.0, atol=1e-14)


def test_gridded_wspd_wdir_equals_uniform():
    grid = flat_grid()
    xs = np.linspace(-6_000.0, 6_000.0, 13)
    ys = np.linspace(-6_000.0, 6_000.0, 11)
    shape = (ys.size, xs.size)
    wind_ds = xr.Dataset(
        {"wspd": (("y", "x"), np.full(shape, U10)), "wdir": (("y", "x"), np.full(shape, 270.0))},
        coords={"x": xs, "y": ys},
    )
    op_u = build(grid, wind=(U10, 270.0))
    op_g = build(grid, wind=wind_ds)
    assert np.allclose(op_u.T, op_g.T, rtol=0.0, atol=1e-12)


def test_gridded_lonlat_on_geographic_grid():
    """lon/lat wind interpolates onto a grid built from a geographic bathy."""
    lon = np.linspace(4.0, 4.2, 41)
    lat = np.linspace(52.0, 52.1, 41)
    bathy = xr.DataArray(
        np.full((lat.size, lon.size), -DEPTH),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
    )
    grid = LocalGrid.from_dataarray(bathy, positive="up")
    wlon = np.linspace(3.8, 4.4, 7)
    wlat = np.linspace(51.9, 52.2, 5)
    shape = (wlat.size, wlon.size)
    wind_ds = xr.Dataset(
        {"u10": (("lat", "lon"), np.full(shape, U10)), "v10": (("lat", "lon"), np.zeros(shape))},
        coords={"lon": wlon, "lat": wlat},
    )
    wf = WindField.from_dataset(grid, wind_ds)
    ref = WindField.uniform(grid, U10, 270.0)
    assert np.allclose(wf.usx, ref.usx, atol=1e-12)
    assert np.allclose(wf.usy, ref.usy, atol=1e-12)


def test_spatially_varying_wind_orders_growth():
    """Wind over the eastern half only grows a west-approach ray less than
    wind everywhere, and more than no wind."""
    grid = flat_grid()
    xs = grid.x
    ys = grid.y
    u10 = np.where(xs[None, :] >= 0.0, U10, 0.0) * np.ones((ys.size, 1))
    half = xr.Dataset(
        {"u10": (("y", "x"), u10), "v10": (("y", "x"), np.zeros_like(u10))},
        coords={"x": xs, "y": ys},
    )
    e_b = probe_270()
    base = build(grid).apply(e_b).sum()
    r_half = build(grid, wind=half).apply(e_b).sum() / base
    r_full = build(grid, wind=(U10, 270.0)).apply(e_b).sum() / base
    assert r_full > 1.05
    assert 1.0 - 1e-9 <= r_half < 0.5 * (r_full - 1.0) + 1.0


def test_agrow_seeds_analytic_linear_growth():
    """Zero boundary energy + agrow: E0 equals the closed-form integral of
    the Cavaleri-MR term q/r (exp(r L) - 1) / (c cg)."""
    grid = flat_grid()
    op = build(grid, wind=(U10, 270.0), agrow=True)
    assert op.E0 is not None and op.E0.shape == (1, DIRS.size)

    omega = 2.0 * np.pi * F
    c = float(phase_speed(omega, np.array(DEPTH)))
    cg = float(group_speed(omega, np.array(DEPTH)))
    us = float(friction_velocity(U10))
    r = 0.25 * RHO_AIR_WATER * (28.0 * us / c - 1.0) * omega / cg
    sig_pm = 2.0 * np.pi * 0.13 * GRAV / (28.0 * us)
    h = np.exp(-((omega / sig_pm) ** -4))
    a = 1.5e-3 / (2.0 * np.pi * GRAV**2) * us**4 * h * (2.0 * np.pi * np.pi / 180.0)
    e0_expected = (c * a / r) * (np.exp(r * LENGTH) - 1.0) / (c * cg)

    e0 = op.E0[0, DIRS == 270.0][0]
    assert np.isclose(e0, e0_expected, rtol=3e-2), (e0, e0_expected)
    # no seed against the wind
    assert op.E0[0, DIRS == 90.0][0] == 0.0
    # apply() adds the seed to a zero boundary spectrum
    out = op.apply(np.zeros((1, 1, DIRS.size)))
    assert np.allclose(out[0], op.E0)


def test_agrow_with_friction_matches_closed_form():
    """agrow + JONSWAP friction: E0 follows q/r (exp(r L) - 1) with the NET
    rate r = (B - D_fric)/cg — both sources share one path exponent."""
    grid = flat_grid()
    cf = 0.038
    op = build(grid, wind=(U10, 270.0), agrow=True, cf_jonswap=cf)

    omega = 2.0 * np.pi * F
    from waveray.dispersion import wavenumber

    k = float(wavenumber(omega, np.array(DEPTH)))
    c = float(phase_speed(omega, np.array(DEPTH)))
    cg = float(group_speed(omega, np.array(DEPTH)))
    us = float(friction_velocity(U10))
    b = 0.25 * RHO_AIR_WATER * (28.0 * us / c - 1.0) * omega
    d_fric = cf * omega**2 / (GRAV**2 * np.sinh(k * DEPTH) ** 2)
    r = (b - d_fric) / cg
    sig_pm = 2.0 * np.pi * 0.13 * GRAV / (28.0 * us)
    h = np.exp(-((omega / sig_pm) ** -4))
    a = 1.5e-3 / (2.0 * np.pi * GRAV**2) * us**4 * h * (2.0 * np.pi * np.pi / 180.0)
    e0_expected = (c * a / r) * (np.exp(r * LENGTH) - 1.0) / (c * cg)

    e0 = op.E0[0, DIRS == 270.0][0]
    assert np.isclose(e0, e0_expected, rtol=3e-2), (e0, e0_expected)


def test_growth_clip_bounds_the_runaway():
    """High frequency over a long fetch: unbalanced wind input would grow
    without limit, so max_growth must cap it, warn, and stay finite."""
    grid = flat_grid(size=20_000.0, n=101)
    hi_freqs = np.array([0.5, 0.9])
    with pytest.warns(UserWarning, match="max_growth"):
        op = build_operator(
            grid,
            target_xy=(0.0, 0.0),
            boundary_xy=np.array([[-10_000.0, 0.0]]),
            freqs=hi_freqs,
            dirs=DIRS,
            nsub=3,
            cf_jonswap=None,
            wind=(U10, 270.0),
            agrow=True,
            max_growth=100.0,
        )
    assert op.attrs["growth_clipped_fraction"] > 0.0
    assert np.all(np.isfinite(op.T)) and np.all(np.isfinite(op.E0))
    # ccg ratio is ~1 on flat bathymetry, so the transfer coefficient of a
    # direction bin cannot exceed the gain ceiling by more than interpolation
    assert op.T.max() <= 100.0 * 1.05


def test_growth_clip_does_not_bind_on_short_paths():
    """The default ceiling must leave the verified analytic case untouched."""
    grid = flat_grid()
    op = build(grid, wind=(U10, 270.0))
    assert op.attrs["growth_clipped_fraction"] == 0.0


def test_agrow_requires_wind():
    with pytest.raises(ValueError, match="agrow"):
        build(flat_grid(n=21), agrow=True)


def test_wind_time_axis_rejected():
    grid = flat_grid(n=21)
    xs = np.linspace(-6_000.0, 6_000.0, 5)
    ys = np.linspace(-6_000.0, 6_000.0, 5)
    wind_ds = xr.Dataset(
        {
            "u10": (("time", "y", "x"), np.full((2, ys.size, xs.size), U10)),
            "v10": (("time", "y", "x"), np.zeros((2, ys.size, xs.size))),
        },
        coords={"x": xs, "y": ys, "time": [0, 1]},
    )
    with pytest.raises(ValueError, match="single snapshot"):
        build(grid, wind=wind_ds)


def test_as_wind_field_rejects_unknown_type():
    with pytest.raises(TypeError, match="wind must be"):
        as_wind_field(42, flat_grid(n=11))


def test_operator_netcdf_roundtrip_with_wind(tmp_path):
    grid = flat_grid(n=41)
    op = build(grid, nsub=3, wind=(U10, 270.0), agrow=True)
    assert op.attrs["wind_source"] == "uniform"
    assert op.attrs["wind_speed"] == U10
    assert op.attrs["agrow"] == 1

    path = str(tmp_path / "op.nc")
    op.to_netcdf(path)
    back = TransferOperator.from_netcdf(path)
    assert np.allclose(op.T, back.T)
    assert back.E0 is not None and np.allclose(op.E0, back.E0)
    assert back.attrs["wind_source"] == "uniform"
    assert back.attrs["agrow"] == 1


def test_sitemodel_wind_passthrough_transforms_zero_spectrum():
    """SiteModel.build(wind=..., agrow=True) seeds wind sea through transform()."""
    grid = flat_grid()
    model = SiteModel.build(
        bathy=grid,
        target=(0.0, 0.0),
        boundary_points=[(-5_000.0, 0.0)],
        freqs=np.array([F]),
        dirs=DIRS,
        cf_jonswap=None,
        nsub=5,
        wind=(U10, 270.0),
        agrow=True,
    )
    efth = xr.DataArray(
        np.zeros((3, 1, DIRS.size)),
        dims=("time", "freq", "dir"),
        coords={"time": np.arange(3), "freq": [F], "dir": DIRS},
    )
    out = model.transform(efth)
    assert float(out.sum()) > 0.0
    assert out.dims == ("time", "freq", "dir")
