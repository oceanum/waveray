# Quickstart

A complete downscaling, from an offshore hindcast to a nearshore point.

```python
import numpy as np
import wavespectra  # registers the .spec accessor
from waveray import SiteModel, fetch_datamesh_bathymetry

# 1. Local bathymetry around the site (GEBCO is elevation: positive up)
bathy = fetch_datamesh_bathymetry(
    "gebco_2023", bbox=(3.85, 52.00, 4.55, 52.50), positive="up"
)

# 2. Offshore boundary spectra: efth(time, site, freq, dir) from your hindcast.
#    `site` must be ordered to match `boundary_points` below.
efth_boundary = ...   # xarray DataArray, wavespectra convention

# 3. Build the transfer operator once (~10 s)
model = SiteModel.build(
    bathy=bathy,
    target=(4.416, 52.240),                       # lon, lat of the target
    boundary_points=[(3.85, 52.20), (3.85, 52.25),  # the hindcast output sites
                     (4.10, 52.45), (4.40, 52.35)],
    freqs=efth_boundary.freq.values,
    dirs=efth_boundary.dir.values,
    positive="up",
)

# 4. Transform the whole hindcast in one call (~10,000 spectra/s)
efth_near = model.transform(efth_boundary, tide=None)

# 5. The result is a wavespectra spectrum
print(efth_near.spec.hs())    # significant wave height
print(efth_near.spec.tp())    # peak period
print(efth_near.spec.dpm())   # mean direction at the peak

# 6. Persist the operator; rebuilding is never needed
model.to_netcdf("noordwijk.nc")
model = SiteModel.from_netcdf("noordwijk.nc")
```

## What just happened

`SiteModel.build` traced rays backward from the target across every
(frequency, direction) bin of your spectrum, recording for each one where it
left the domain and how much its energy density changed on the way. That is
the operator. `transform` contracts it against your boundary spectra with a
single `einsum`, then applies the depth-limited breaking cap at the target.

## Choosing the pieces

- **Domain (`bbox`)** — draw it so its edge passes through (or near) your
  hindcast's output sites. Rays stop at the domain edge and pick up the
  boundary spectra there.
- **`boundary_points`** — the hindcast sites, in the same order as the `site`
  dimension of `efth`. waveray checks this ordering when the spectra carry
  `lon`/`lat` coordinates, and raises rather than silently mixing them up.
- **`target`** — any wet point on the bathymetry grid. Coarse grids do not
  resolve the beach step, so nudge seaward until `sample_depth` returns a
  sensible depth (see [User guide](usage.md#choosing-a-target)).

## Without Datamesh

Any `xarray.DataArray` with 1-D lon/lat (or x/y) coordinates works:

```python
import xarray as xr
from waveray import LocalGrid, SiteModel

bathy = xr.open_dataarray("my_survey_grid.nc")   # depth or elevation
grid = LocalGrid.from_dataarray(bathy, positive="down")   # "down" = depths
model = SiteModel.build(bathy=grid, target=(lon, lat), ...)
```

Next: [Concepts](concepts.md) for what is inside the operator, the
[User guide](usage.md) for the practical knobs, or [Working with Oceanum
Datamesh](datamesh.md) if that is where your data lives.
