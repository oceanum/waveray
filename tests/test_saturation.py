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


def test_cap_is_the_fetch_limited_spectrum_alone():
    """One reference, scaled. PM is kept only as a diagnostic.

    An earlier design capped at ``max(PM, k*JONSWAP)``; PM never governed the
    tail for any fetch this code can produce (the crossover is ~57 000 km at
    12 m/s), so its only effect was to clip *below* the wind-sea peak, which
    is where swell lives.
    """
    u10, fetch = 12.0, 5_000.0
    cap = saturation_cap(FREQS, u10, fetch)
    assert np.allclose(cap, SATURATION_K * jonswap_fetch_limited(FREQS, u10, fetch))
    # and it is everywhere above PM at and beyond its own peak, which is why
    # the PM branch was redundant
    fp = FREQS[cap.argmax()]
    above = FREQS >= fp
    assert np.all(cap[above] > pm_spectrum(FREQS, 0.459)[above])


# ------------------------------------------------------------------ #
# Applying the cap
# ------------------------------------------------------------------ #
def _swell(hs=2.0, fp=0.1, spread=25.0):
    ef = np.exp(-1.25 * (FREQS / fp) ** -4) * FREQS**-5.0
    gth = np.exp(-0.5 * (((DIRS - 270.0 + 180) % 360 - 180) / spread) ** 2)
    efth = ef[:, None] * gth[None, :]
    return efth * (hs / hm0(efth[None], FREQS, DIRS)[0]) ** 2


def test_cap_only_touches_the_increment_never_the_swell():
    """The cap sees only what the wind added; swell passes through untouched."""
    swell = _swell()
    increment = np.zeros_like(swell)  # wind added nothing
    out, scale = apply_saturation(swell, increment, FREQS, DIRS, u10=12.0, fetch=15_000.0)
    assert np.array_equal(out, swell)
    assert np.all(scale == 1.0)


def test_a_long_swell_under_a_light_breeze_survives_end_to_end():
    """The case that broke the previous design, driven through the real path.

    A unit test on ``apply_saturation`` cannot catch this: with the increment
    formulation ``base + increment*scale >= base`` holds for any scale, so such
    a test passes by construction. The regression only shows up end to end,
    where the cap sees a spectrum the operator actually produced — under the
    old total-capping design this ratio was 0.29.
    """
    swell14 = xr.DataArray(
        _swell(hs=2.0, fp=1 / 14.0), dims=("freq", "dir"), coords={"freq": FREQS, "dir": DIRS}
    )
    nowind = float(
        hm0(
            _model().transform(swell14, breaking=False).values[None],
            FREQS,
            DIRS,
        )[0]
    )
    capped = float(
        hm0(
            _model(wind=(8.0, 270.0), agrow=True)
            .transform(swell14, breaking=False, saturation=True)
            .values[None],
            FREQS,
            DIRS,
        )[0]
    )
    assert capped >= 0.95 * nowind, (
        f"the cap removed swell energy: {capped:.4f} against {nowind:.4f} with no wind"
    )


def test_cap_bites_on_a_runaway_increment():
    """An inflated wind increment must be cut back to the ceiling."""
    swell = _swell()
    increment = swell * 1e6
    out, scale = apply_saturation(swell, increment, FREQS, DIRS, u10=12.0, fetch=5_000.0)
    assert np.all(scale <= 1.0 + 1e-12)
    assert scale.min() < 1e-3, "the cap barely engaged on a 1e6 increment"
    assert np.all(np.isfinite(out))
    # `out >= swell` is true by construction for any scale in [0, 1], so it is
    # not asserted here; the end-to-end test above is what can actually fail.


def test_cap_preserves_directional_shape_within_a_frequency():
    swell = _swell()
    increment = swell * 1e4
    out, _ = apply_saturation(swell, increment, FREQS, DIRS, u10=12.0, fetch=5_000.0)
    # pick a frequency where the cap actually leaves something behind
    i = int(np.argmax((out - swell).sum(axis=1)))
    a = increment[i] / increment[i].sum()
    b = (out[i] - swell[i]) / (out[i] - swell[i]).sum()
    assert np.allclose(a, b, atol=1e-9), "the cap distorted the directional distribution"


def test_scale_is_reported_per_frequency_over_leading_dims():
    swell = _swell()
    increment = np.stack([np.zeros_like(swell), swell * 1e5])  # (2, nf, ndir)
    base = np.stack([swell, swell])
    out, scale = apply_saturation(base, increment, FREQS, DIRS, u10=12.0, fetch=5_000.0)
    assert scale.shape == (2, FREQS.size)
    assert np.all(scale[0] == 1.0)  # nothing added -> nothing capped
    assert scale[1].min() < 1.0
    assert out.shape == base.shape


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


def test_saturation_is_opt_in_and_reduces_a_strong_wind_case():
    m = _model(wind=(20.0, 270.0), agrow=True)
    efth = xr.DataArray(_swell(), dims=("freq", "dir"), coords={"freq": FREQS, "dir": DIRS})
    default = m.transform(efth, breaking=False)
    on = m.transform(efth, breaking=False, saturation=True)
    off = m.transform(efth, breaking=False, saturation=False)
    # off by default: the closure is empirical, so it must be asked for
    assert np.allclose(default.values, off.values)
    assert float(on.sum()) < float(off.sum()), "the cap did nothing at 20 m/s"
    assert "wind_saturation_scale_min" in on.attrs
    assert on.attrs["wind_saturation_scale_min"] <= 1.0


def test_calm_wind_transforms_exactly_like_no_wind():
    """Regression for the design that annihilated a spectrum at wind=(0, ...).

    tests/test_wind.py checks T and E0; this checks what the user sees.
    """
    efth = xr.DataArray(_swell(), dims=("freq", "dir"), coords={"freq": FREQS, "dir": DIRS})
    nowind = _model().transform(efth, breaking=False, saturation=False).values
    calm = (
        _model(wind=(0.0, 270.0), agrow=True)
        .transform(efth, breaking=False, saturation=True)
        .values
    )
    assert np.allclose(calm, nowind, rtol=0, atol=1e-12)


def test_wind_never_removes_energy_from_a_swell():
    """Across wind speed and swell period, the capped result must not fall
    below the no-wind answer — the invariant the old design violated in 7 of
    9 configurations."""
    efth_nowind = xr.DataArray(_swell(), dims=("freq", "dir"), coords={"freq": FREQS, "dir": DIRS})
    base = float(hm0(_model().transform(efth_nowind, breaking=False).values[None], FREQS, DIRS)[0])
    for u10 in (4.0, 8.0, 16.0):
        for fp in (1 / 14.0, 1 / 10.0):
            sp = xr.DataArray(
                _swell(fp=fp), dims=("freq", "dir"), coords={"freq": FREQS, "dir": DIRS}
            )
            b = float(hm0(_model().transform(sp, breaking=False).values[None], FREQS, DIRS)[0])
            w = float(
                hm0(
                    _model(wind=(u10, 270.0), agrow=True)
                    .transform(sp, breaking=False, saturation=True)
                    .values[None],
                    FREQS,
                    DIRS,
                )[0]
            )
            assert w >= b - 1e-9, f"U10={u10} fp={fp:.3f}: wind removed energy ({w} < {b})"
    assert base > 0


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


def test_saturation_k_survives_the_netcdf_roundtrip(tmp_path):
    """A site-tuned k must not silently reset to the default on reload."""
    m = _model(wind=(12.0, 270.0), agrow=True)
    m.saturation_k = 7.5
    path = str(tmp_path / "k.nc")
    m.to_netcdf(path)
    assert SiteModel.from_netcdf(path).saturation_k == 7.5


def test_saturation_k_can_be_set_at_build_time():
    m = _model(wind=(12.0, 270.0), agrow=True, saturation_k=5.0)
    assert m.saturation_k == 5.0


def test_explicit_fetch_overrides_the_geometric_one():
    """The geometric fetch is domain-bounded; a known fetch must win."""
    geometric = _model(wind=(12.0, 270.0), agrow=True)
    given = _model(wind=(12.0, 270.0), agrow=True, fetch=250_000.0)
    assert geometric.operator.attrs["fetch_source"] == "geometric"
    assert given.operator.attrs["fetch_source"] == "given"
    assert given.operator.attrs["wind_fetch"] == 250_000.0
    assert given.operator.attrs["wind_fetch"] > geometric.operator.attrs["wind_fetch"]
    # a longer fetch means a higher ceiling, so more of the increment survives
    efth = xr.DataArray(_swell(), dims=("freq", "dir"), coords={"freq": FREQS, "dir": DIRS})
    a = float(
        hm0(geometric.transform(efth, breaking=False, saturation=True).values[None], FREQS, DIRS)[0]
    )
    b = float(
        hm0(given.transform(efth, breaking=False, saturation=True).values[None], FREQS, DIRS)[0]
    )
    assert b >= a


@pytest.mark.parametrize("u10", [5.0, 12.0, 20.0])
def test_capped_increment_never_exceeds_the_ceiling(u10):
    """Whatever the wind adds must sit under the fetch-limited ceiling."""
    m = _model(wind=(u10, 270.0), agrow=True)
    efth = xr.DataArray(_swell() * 1e3, dims=("freq", "dir"), coords={"freq": FREQS, "dir": DIRS})
    capped = m.transform(efth, breaking=False, saturation=True).values
    nowind = m.operator.apply_nowind(
        efth.expand_dims({"site": 1}).transpose("site", "freq", "dir").values
    )
    a = m.operator.attrs
    cap = saturation_cap(FREQS, a["u10_target"], a["wind_fetch"])
    added = (capped - nowind).sum(axis=-1) * dir_resolution(DIRS)
    # Absolute tolerance tied to the spectrum's own magnitude, not to the cap:
    # far below the wind-sea peak the ceiling falls to ~1e-13, and `added` there
    # is pure round-off of a spectrum three orders of magnitude larger. A purely
    # relative test would be comparing noise against noise.
    tol = 1e-9 * float((nowind.sum(axis=-1) * dir_resolution(DIRS)).max())
    assert np.all(added <= cap * (1.0 + 1e-6) + tol), float((added - cap).max())
