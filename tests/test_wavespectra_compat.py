"""Spectral input and output must be wavespectra-compatible.

wavespectra is a hard dependency, so its conventions are the contract: the
``.spec`` accessor must work on anything waveray returns, and integrated
parameters must agree exactly with wavespectra's own.
"""

import numpy as np
import pytest
import wavespectra  # noqa: F401  (registers the .spec accessor)
import xarray as xr

from waveray import SiteModel, to_specdataset
from waveray.breaking import dir_resolution, freq_resolution, hm0

FREQS = np.linspace(0.04, 0.25, 12)
DIRS = np.arange(0.0, 360.0, 15.0)
# The Datamesh hindcasts ship a wrapped, descending dir coordinate.
DIRS_NON_MONOTONIC = np.roll(DIRS, 7)[::-1]


def spectrum(dirs, nt=3, hs=2.0):
    ef = np.exp(-0.5 * ((FREQS - 0.08) / 0.02) ** 2)
    dd = np.exp(-0.5 * (((dirs - 270.0 + 180.0) % 360.0 - 180.0) / 25.0) ** 2)
    e = ef[:, None] * dd[None, :]
    e = e * (hs**2 / 16.0) / (hm0(e, FREQS, dirs) ** 2 / 16.0)
    return xr.DataArray(
        np.tile(e, (nt, 2, 1, 1)),
        dims=("time", "site", "freq", "dir"),
        coords={"time": np.arange(nt), "freq": FREQS, "dir": dirs},
        name="efth",
    )


@pytest.fixture(scope="module")
def bathy():
    lon = np.linspace(114.3, 114.6, 81)
    lat = np.linspace(-28.9, -28.7, 61)
    depth = np.tile(40.0 * (1 - (lon - lon[0]) / (lon[-1] - lon[0])), (lat.size, 1))
    return xr.DataArray(-depth, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def build(bathy, dirs):
    return SiteModel.build(
        bathy=bathy,
        target=(114.55, -28.8),
        boundary_points=[(114.3, -28.85), (114.3, -28.75)],
        freqs=FREQS,
        dirs=dirs,
        positive="up",
        nsub=3,
    )


def test_resolution_helpers_match_wavespectra():
    da = spectrum(DIRS, nt=1).isel(time=0, site=0)
    assert np.allclose(freq_resolution(FREQS), da.spec.df.values)
    assert np.isclose(dir_resolution(DIRS), da.spec.dd)


def test_hm0_equals_wavespectra_hs_exactly():
    """Our quadrature must be wavespectra's, not merely close to it."""
    da = spectrum(DIRS).isel(site=0)
    ours = hm0(da.values, FREQS, DIRS)
    theirs = da.spec.hs(tail=False).values
    assert np.allclose(ours, theirs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dirs", [DIRS, DIRS_NON_MONOTONIC], ids=["sorted", "non-monotonic"])
def test_output_supports_spec_accessor(bathy, dirs):
    out = build(bathy, dirs).transform(spectrum(dirs))

    assert out.name == "efth"
    assert out.attrs["units"] == "m2 s degree-1"
    assert out["freq"].attrs["units"] == "Hz"
    assert out["dir"].attrs["standard_name"] == "sea_surface_wave_from_direction"

    # the accessor works, and agrees with our own integrated parameter
    hs_spec = out.spec.hs(tail=False).values
    hs_ours = hm0(out.values, FREQS, dirs)
    assert np.allclose(hs_spec, hs_ours, rtol=1e-12, atol=1e-12)

    # other wavespectra statistics are finite and physical
    assert np.all(np.isfinite(out.spec.tp().values))
    dpm = out.spec.dpm().values
    assert np.all((dpm >= 0.0) & (dpm <= 360.0))


def test_dataset_input_accepted(bathy):
    """A wavespectra SpecDataset (Dataset holding efth) is valid input."""
    model = build(bathy, DIRS)
    da = spectrum(DIRS)
    from_da = model.transform(da)
    from_ds = model.transform(da.to_dataset())
    assert np.allclose(from_da.values, from_ds.values)

    with pytest.raises(ValueError, match="efth"):
        model.transform(da.rename("spectra").to_dataset())


def test_to_specdataset_roundtrips_through_netcdf(bathy, tmp_path):
    out = build(bathy, DIRS).transform(spectrum(DIRS))
    ds = to_specdataset(out)
    assert "efth" in ds
    assert np.allclose(ds.spec.hs(tail=False).values, out.spec.hs(tail=False).values)

    path = tmp_path / "spec.nc"
    ds.to_netcdf(path)
    with xr.open_dataset(path) as reloaded:
        assert np.allclose(reloaded.spec.hs(tail=False).values, out.spec.hs(tail=False).values)


def test_single_frequency_moment_matches_wavespectra_convention():
    """wavespectra uses df = 1.0 for a single frequency; trapezoid would give 0."""
    f1 = np.array([0.1])
    e = np.ones((1, DIRS.size))
    assert freq_resolution(f1) == np.array([1.0])
    assert hm0(e, f1, DIRS) > 0.0
