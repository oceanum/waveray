"""SiteModel end-to-end on synthetic geographic bathymetry."""

import numpy as np
import pytest
import xarray as xr

from nearshore_transform import LocalGrid, SiteModel

FREQS = np.linspace(0.04, 0.25, 12)
DIRS = np.arange(0.0, 360.0, 15.0)


@pytest.fixture(scope="module")
def geographic_beach():
    """Plane beach as a geographic elevation DataArray (GEBCO-like)."""
    lon = np.linspace(114.3, 114.6, 121)  # ~30 km
    lat = np.linspace(-28.9, -28.7, 91)  # ~22 km
    # depth 40 m at west edge shoaling to 0 at east edge
    depth = np.tile(40.0 * (1.0 - (lon - lon[0]) / (lon[-1] - lon[0])), (lat.size, 1))
    return xr.DataArray(
        -depth,  # elevation convention (positive up)
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )


@pytest.fixture(scope="module")
def built_model(geographic_beach):
    return SiteModel.build(
        bathy=geographic_beach,
        target=(114.55, -28.8),  # ~6.7 m depth
        boundary_points=[(114.3, -28.85), (114.3, -28.75)],
        freqs=FREQS,
        dirs=DIRS,
        positive="up",
        nsub=5,
    )


def boundary_spectra(nt=4, hs=2.0):
    """(time, site, freq, dir) DataArray, smooth swell from the west."""
    ef = np.exp(-0.5 * ((FREQS - 0.08) / 0.02) ** 2)
    dd = np.exp(-0.5 * (((DIRS - 270.0 + 180.0) % 360.0 - 180.0) / 25.0) ** 2)
    e = ef[:, None] * dd[None, :]
    ddir = 360.0 / DIRS.size
    m0 = np.trapezoid(e.sum(axis=-1) * ddir, FREQS)
    e = e * (hs**2 / 16.0) / m0
    vals = np.tile(e, (nt, 2, 1, 1)) * np.linspace(0.5, 2.0, nt)[:, None, None, None]
    return xr.DataArray(
        vals,
        dims=("time", "site", "freq", "dir"),
        coords={"time": np.arange(nt), "freq": FREQS, "dir": DIRS},
        attrs={"units": "m2 Hz-1 deg-1"},
    )


def test_transform_shapes_and_coords(built_model):
    efth = boundary_spectra()
    out = built_model.transform(efth)
    assert out.dims == ("time", "freq", "dir")
    assert out.shape == (4, FREQS.size, DIRS.size)
    assert np.allclose(out["freq"].values, FREQS)
    assert float(out.max()) > 0.0


def test_transform_produces_plausible_shoaling(built_model):
    """Small swell (no breaking): nearshore Hs within a plausible factor of
    offshore Hs — energy arrives and is not wildly amplified."""
    efth = boundary_spectra(nt=1, hs=1.0)
    out = built_model.transform(efth, breaking=False)
    ddir = 360.0 / DIRS.size
    m0 = np.trapezoid(out.values[0].sum(axis=-1) * ddir, FREQS)
    hs_near = 4.0 * np.sqrt(m0)
    assert 0.3 < hs_near < 1.6  # sheltering can lose energy; shoaling can add some


def test_breaking_engages_for_big_seas(built_model):
    efth = boundary_spectra(nt=1, hs=8.0)
    unbroken = built_model.transform(efth, breaking=False)
    broken = built_model.transform(efth, breaking=True)
    assert float(broken.sum()) <= float(unbroken.sum())
    d = built_model.operator.depth_target
    ddir = 360.0 / DIRS.size
    m0 = np.trapezoid(broken.values[0].sum(axis=-1) * ddir, FREQS)
    assert 4.0 * np.sqrt(m0) <= 0.9 * d + 0.01  # miche cap is below ~0.9d here


def test_missing_site_dim_rejected_for_multi_bp(built_model):
    efth = boundary_spectra().isel(site=0, drop=True)
    with pytest.raises(ValueError, match="site"):
        built_model.transform(efth)


def test_reordered_boundary_sites_rejected(built_model):
    """Sites carrying lon/lat coords in the wrong order must be rejected."""
    efth = boundary_spectra(nt=1)
    good = efth.assign_coords(lon=("site", [114.3, 114.3]), lat=("site", [-28.85, -28.75]))
    assert built_model.transform(good) is not None
    swapped = efth.assign_coords(lon=("site", [114.3, 114.3]), lat=("site", [-28.75, -28.85]))
    with pytest.raises(ValueError, match="boundary points"):
        built_model.transform(swapped)


def test_mismatched_freqs_rejected(built_model):
    efth = boundary_spectra().assign_coords(freq=FREQS * 1.5)
    with pytest.raises(ValueError, match="freq"):
        built_model.transform(efth)


def test_model_netcdf_roundtrip(built_model, tmp_path):
    path = str(tmp_path / "model.nc")
    built_model.to_netcdf(path)
    loaded = SiteModel.from_netcdf(path)
    efth = boundary_spectra(nt=2)
    a = built_model.transform(efth)
    b = loaded.transform(efth)
    assert np.allclose(a.values, b.values)
    assert loaded.gamma == built_model.gamma


def test_local_grid_roundtrip_geography(geographic_beach):
    grid = LocalGrid.from_dataarray(geographic_beach, positive="up")
    x, y = grid.to_local(np.array([114.45]), np.array([-28.8]))
    d = grid.sample_depth(x, y)
    # analytic: halfway across the 40->0 beach -> ~20 m
    assert np.isclose(d[0], 20.0, atol=1.0)
