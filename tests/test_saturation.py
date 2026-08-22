"""Wind-sea saturation cap: shape, inertness, and the fetch geometry.

The cap's *skill* is measured against SWAN in tests/test_validation_swan.py.
These tests pin the properties that must hold regardless of calibration.
"""

import numpy as np
import pytest
import xarray as xr

from waveray import SiteModel
from waveray.bathymetry import LocalGrid
from waveray.breaking import dir_resolution, hm0
from waveray.saturation import (
    SATURATION_K,
    apply_saturation,
    jonswap_fetch_limited,
    pm_spectrum,
    saturation_cap,
    upwind_fetch,
)
from waveray.wind import friction_velocity, u10_from_u_star

FREQS = np.linspace(0.04, 0.5, 40)
DIRS = np.arange(0.0, 360.0, 10.0)


def flat_grid(depth=30.0, size=20_000.0, n=81):
    x = np.linspace(-size / 2, size / 2, n)
    y = np.linspace(-size / 2, size / 2, n)
    return LocalGrid(x=x, y=y, depth=np.full((n, n), depth))


# ------------------------------------------------------------------ #
# Spectral references
# ------------------------------------------------------------------ #
def test_pm_peak_sits_at_the_swan_reference_frequency():
    """PM peaks at f_p = 0.13 g / (28 u*), the same reference AGROW uses."""
    us = 0.6
    e = pm_spectrum(FREQS, us)
    fp_expected = 0.13 * 9.81 / (28.0 * us)
    assert abs(FREQS[e.argmax()] - fp_expected) < 0.02
    assert np.all(np.isfinite(e))


def test_pm_scales_with_wind_and_decays_as_f_minus_5():
    strong = pm_spectrum(FREQS, 0.8)
    weak = pm_spectrum(FREQS, 0.4)
    assert strong.sum() > weak.sum()
    # well above the peak the tail must follow f^-5
    hi = FREQS > 0.35
    slope = np.polyfit(np.log(FREQS[hi]), np.log(strong[hi]), 1)[0]
    assert -5.2 < slope < -4.8, slope


def test_jonswap_matches_the_fetch_limited_growth_law():
    """Integrated Hs must track the JONSWAP fetch law within its own scatter."""
    u10, fetch = 12.0, 20_000.0
    e = jonswap_fetch_limited(FREQS, u10, fetch)
    m0 = np.trapezoid(e, FREQS)
    hs = 4.0 * np.sqrt(m0)
    xhat = 9.81 * fetch / u10**2
    hs_law = 4.0 * np.sqrt(1.6e-7 * xhat * u10**4 / 9.81**2)
    assert 0.7 * hs_law < hs < 1.4 * hs_law, (hs, hs_law)


def test_longer_fetch_gives_more_energy_at_a_lower_peak():
    short = jonswap_fetch_limited(FREQS, 12.0, 2_000.0)
    long = jonswap_fetch_limited(FREQS, 12.0, 50_000.0)
    assert np.trapezoid(long, FREQS) > np.trapezoid(short, FREQS)
    assert FREQS[long.argmax()] < FREQS[short.argmax()]


def test_cap_is_the_larger_of_the_two_references():
    u10 = 12.0
    us = float(friction_velocity(u10))
    cap = saturation_cap(FREQS, u10, us, 5_000.0)
    pm = pm_spectrum(FREQS, us)
    js = SATURATION_K * jonswap_fetch_limited(FREQS, u10, 5_000.0)
    assert np.allclose(cap, np.maximum(pm, js))
    # each reference wins somewhere: PM in the far tail, JONSWAP near its peak
    assert np.any(cap == pm) and np.any(cap > pm)


# ------------------------------------------------------------------ #
# Applying the cap
# ------------------------------------------------------------------ #
def _swell(hs=2.0, fp=0.1, spread=25.0):
    ef = np.exp(-1.25 * (FREQS / fp) ** -4) * FREQS**-5.0
    gth = np.exp(-0.5 * (((DIRS - 270.0 + 180) % 360 - 180) / spread) ** 2)
    efth = ef[:, None] * gth[None, :]
    return efth * (hs / hm0(efth[None], FREQS, DIRS)[0]) ** 2


def test_cap_is_inert_on_a_swell_below_saturation():
    """A 2 m / 10 s swell is nowhere near the wind-sea ceiling."""
    efth = _swell()
    out, scale = apply_saturation(efth, FREQS, DIRS, u10=12.0, u_star=0.459, fetch=15_000.0)
    assert np.allclose(out, efth)
    assert np.all(scale == 1.0)


def test_cap_bites_on_an_inflated_tail():
    """Inflate the high-frequency tail the way unbalanced growth does."""
    efth = _swell()
    blown = efth.copy()
    blown[FREQS > 0.25] *= 1e6
    out, scale = apply_saturation(blown, FREQS, DIRS, u10=12.0, u_star=0.459, fetch=15_000.0)
    assert np.all(scale <= 1.0 + 1e-12)
    assert scale.min() < 1e-3, "the cap barely engaged on a 1e6 tail"
    # the swell peak must survive untouched
    peak = FREQS.argmin() + np.argmax(efth.sum(axis=1))
    assert np.isclose(out[peak].sum(), efth[peak].sum(), rtol=1e-9)
    # and the result must be finite and smaller
    assert np.all(np.isfinite(out)) and out.sum() < blown.sum()


def test_cap_preserves_directional_shape_within_a_frequency():
    efth = _swell()
    blown = efth * 1e4
    out, _ = apply_saturation(blown, FREQS, DIRS, u10=12.0, u_star=0.459, fetch=5_000.0)
    i = np.argmax(blown.sum(axis=1))
    a = blown[i] / blown[i].sum()
    b = out[i] / out[i].sum()
    assert np.allclose(a, b), "the cap distorted the directional distribution"


def test_scale_is_reported_per_frequency_over_leading_dims():
    efth = np.stack([_swell(), _swell() * 1e5])  # (2, nf, ndir)
    out, scale = apply_saturation(efth, FREQS, DIRS, u10=12.0, u_star=0.459, fetch=5_000.0)
    assert scale.shape == (2, FREQS.size)
    assert np.all(scale[0] == 1.0)  # untouched
    assert scale[1].min() < 1.0  # capped
    assert out.shape == efth.shape


# ------------------------------------------------------------------ #
# Fetch geometry
# ------------------------------------------------------------------ #
def test_fetch_runs_upwind_to_the_domain_edge():
    """Wind from the west: the fetch is the distance west to the edge."""
    grid = flat_grid(size=20_000.0)
    fetch = upwind_fetch(grid, 0.0, 0.0, wind_dir=270.0)
    assert 9_000.0 < fetch < 10_100.0, fetch


def test_fetch_direction_follows_the_wind():
    grid = flat_grid(size=20_000.0)
    near_west = upwind_fetch(grid, -8_000.0, 0.0, wind_dir=270.0)  # close to west edge
    far_west = upwind_fetch(grid, 8_000.0, 0.0, wind_dir=270.0)  # far from it
    assert near_west < 3_000.0 < far_west


def test_fetch_is_blocked_by_land():
    """An island upwind must shorten the fetch."""
    n = 81
    x = np.linspace(-10_000.0, 10_000.0, n)
    y = np.linspace(-10_000.0, 10_000.0, n)
    xx, yy = np.meshgrid(x, y)
    depth = np.where(np.hypot(xx + 4_000.0, yy) < 1_500.0, -1.0, 30.0)
    grid = LocalGrid(x=x, y=y, depth=depth)
    blocked = upwind_fetch(grid, 0.0, 0.0, wind_dir=270.0)  # island to the west
    open_water = upwind_fetch(grid, 0.0, 0.0, wind_dir=90.0)  # clear to the east
    assert blocked < 3_000.0, blocked
    assert open_water > 9_000.0, open_water


def test_u10_inverts_the_drag_law():
    for u in (0.5, 5.0, 12.0, 25.0, 40.0):
        assert np.isclose(u10_from_u_star(float(friction_velocity(u))), u, atol=1e-4)
    assert u10_from_u_star(0.0) == 0.0


# ------------------------------------------------------------------ #
# Wiring through the model
# ------------------------------------------------------------------ #
def _model(**kw):
    grid = flat_grid()
    return SiteModel.build(
        bathy=grid,
        target=(0.0, 0.0),
        boundary_points=[(-10_000.0, 0.0)],
        freqs=FREQS,
        dirs=DIRS,
        cf_jonswap=None,
        nsub=3,
        **kw,
    )


def test_operator_records_the_wind_state_needed_by_the_cap():
    m = _model(wind=(12.0, 270.0), agrow=True)
    a = m.operator.attrs
    assert np.isclose(a["u10_target"], 12.0, atol=1e-3)
    assert np.isclose(a["wind_dir_target"], 270.0, atol=1e-6)
    assert a["wind_fetch"] > 5_000.0
    assert np.isclose(a["u_star_target"], float(friction_velocity(12.0)), rtol=1e-6)


def test_no_wind_operator_carries_no_cap_state():
    m = _model()
    assert "u10_target" not in m.operator.attrs
    efth = xr.DataArray(_swell(), dims=("freq", "dir"), coords={"freq": FREQS, "dir": DIRS})
    out = m.transform(efth, breaking=False)
    assert "wind_saturation_scale_min" not in out.attrs


def test_saturation_can_be_switched_off():
    m = _model(wind=(20.0, 270.0), agrow=True)
    efth = xr.DataArray(_swell(), dims=("freq", "dir"), coords={"freq": FREQS, "dir": DIRS})
    on = m.transform(efth, breaking=False)
    off = m.transform(efth, breaking=False, saturation=False)
    assert float(off.sum()) > float(on.sum()), "the cap did nothing at 20 m/s"
    assert "wind_saturation_scale_min" in on.attrs
    assert on.attrs["wind_saturation_scale_min"] <= 1.0


def test_saturation_survives_the_netcdf_roundtrip(tmp_path):
    m = _model(wind=(12.0, 270.0), agrow=True)
    path = str(tmp_path / "m.nc")
    m.to_netcdf(path)
    back = SiteModel.from_netcdf(path)
    for key in ("u10_target", "u_star_target", "wind_fetch"):
        assert np.isclose(back.operator.attrs[key], m.operator.attrs[key])
    efth = xr.DataArray(_swell(), dims=("freq", "dir"), coords={"freq": FREQS, "dir": DIRS})
    assert np.allclose(
        back.transform(efth, breaking=False).values, m.transform(efth, breaking=False).values
    )


@pytest.mark.parametrize("u10", [5.0, 12.0, 20.0])
def test_capped_spectrum_never_exceeds_the_ceiling(u10):
    m = _model(wind=(u10, 270.0), agrow=True)
    efth = xr.DataArray(_swell() * 1e3, dims=("freq", "dir"), coords={"freq": FREQS, "dir": DIRS})
    out = m.transform(efth, breaking=False).values
    a = m.operator.attrs
    cap = saturation_cap(FREQS, a["u10_target"], a["u_star_target"], a["wind_fetch"])
    ef = out.sum(axis=-1) * dir_resolution(DIRS)
    assert np.all(ef <= cap * (1.0 + 1e-9)), (ef / cap).max()
