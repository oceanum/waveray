# User guide

## Bathymetry

waveray works on a `LocalGrid`: a regular depth grid projected onto a local
tangent plane in metres. Depth is **positive down**; land is any node at or
above the waterline.

```python
from waveray import LocalGrid, fetch_datamesh_bathymetry
import xarray as xr

# From Oceanum Datamesh (needs the `datamesh` extra + DATAMESH_TOKEN)
grid = fetch_datamesh_bathymetry("gebco_2023", bbox=(3.85, 52.0, 4.55, 52.5), positive="up")

# From any DataArray with 1-D lon/lat (or latitude/longitude, or y/x) coords
grid = LocalGrid.from_dataarray(xr.open_dataarray("survey.nc"), positive="down")
```

`positive="up"` for **elevation** sources (GEBCO: seabed is negative);
`positive="down"` for **depth** sources. Getting this backwards turns your
ocean into a mountain — check with `grid.depth[~grid.land].max()`.

Useful attributes and methods:

```python
grid.bounds          # (xmin, xmax, ymin, ymax) in metres
grid.spacing         # (dx, dy) median spacing in metres
grid.land            # boolean mask
grid.to_local(lon, lat)     # degrees -> local metres
grid.to_lonlat(x, y)        # local metres -> degrees
grid.sample_depth(x, y)     # bilinear depth at local points
```

### Resolution matters

Ray blocking is a depth test, so **a feature the grid does not resolve cannot
shelter anything**. A 300 m islet on a 1 km grid is smoothed away by the
bilinear depth sampling and is nearly transparent to rays. GEBCO (~450 m) will
not shelter you behind a small reef. Use a survey grid where reef or breakwater
sheltering matters.

## Choosing a domain and boundary points

Rays terminate at the **edge of the bathymetry grid you supply**, and boundary
spectra are interpolated by each ray's exit position along that perimeter.
Therefore:

- Draw the `bbox` so its edge passes through, or close to, your hindcast's
  spectral output sites. The transfer coefficient uses the depth at the ray's
  exit point, so a boundary site sitting well inside the domain will be treated
  as if it were on the edge.
- Both geometries work: a target **nested inside a ring** of output sites, or a
  target **inshore of a line** of them (the classic transect along the 20 m
  contour). In the second case the coast behind the target closes the problem
  naturally — shoreward rays ground out, which is correct.
- Order `boundary_points` to match the `site` dimension of your spectra. When
  the spectra carry `lon`/`lat` coordinates, `transform` verifies this and
  raises rather than silently mixing sites up.

## Choosing a target

Any wet grid point (depth ≥ `d_min`, default 0.3 m). On coarse bathymetry the
beach step is unresolved, so nudge seaward until the depth is sensible:

```python
import numpy as np

lon, lat = 4.42, 52.24
for _ in range(40):
    x, y = grid.to_local(np.array([lon]), np.array([lat]))
    if grid.sample_depth(x, y)[0] >= 3.0:
        break
    lon -= 0.002       # walk offshore
```

Each target is its own operator (~10 s to build). A berth line or transect is
just a loop; each operator persists to netCDF.

## Building the operator

```python
model = SiteModel.build(
    bathy=grid,                 # LocalGrid or DataArray
    target=(lon, lat),
    boundary_points=[(lon1, lat1), (lon2, lat2), ...],
    freqs=efth.freq.values,
    dirs=efth.dir.values,       # must be uniformly spaced over the full circle
    gamma=0.73,                 # breaking coefficient
    breaking_method="miche",    # or "gamma"
    # ray tracing kwargs, forwarded to build_operator:
    nsub=7,                     # sub-rays per direction bin (smooths caustics)
    ds=None,                    # ray step [m]; default min(grid spacing)/3
    d_min=0.3,                  # grounding depth [m]
    cf_jonswap=0.038,           # JONSWAP friction; None disables it
)
```

Check the build afterwards:

```python
op = model.operator
op.depth_target                    # depth at the target [m]
op.attrs["landed_fraction"]        # rays that grounded (sheltering)
op.attrs["lost_fraction"]          # rays that never resolved — should be ~0
```

A non-zero `lost_fraction` means rays are circulating without exiting or
grounding; raise `max_steps` or check for a pathological bathymetry.

## Transforming spectra

```python
efth_near = model.transform(efth_boundary, tide=None, breaking=True)
```

- `efth_boundary` — `DataArray` with dims `(..., [site,] freq, dir)`, or any
  Dataset holding an `efth` variable (a wavespectra `SpecDataset` works).
  The `site` dimension is required when there is more than one boundary point.
- `tide` — water level offset in metres added to the target depth for the
  breaking cap. Scalar, array, or `DataArray` matching the leading dimensions.
- `breaking` — set `False` to get the pure linear transformation.

The result is a wavespectra spectrum; see [wavespectra
interoperability](wavespectra.md).

### Tide

The operator is built at a **fixed water level** (mean sea level as given by
your bathymetry). Tide only modulates the breaking cap:

```python
efth_near = model.transform(efth_boundary, tide=tide_series)   # metres, per timestep
```

If refraction at your site is strongly tide-dependent (large tidal range
relative to depth), build one operator per tide stage and interpolate between
them, rather than relying on this.

### Breaking

```python
model.gamma = 0.8                  # tune per site against observations
model.breaking_method = "gamma"    # simple Hmax = gamma * d
efth_unbroken = model.transform(efth_boundary, breaking=False)
```

Inspect what breaking did:

```python
efth_near.attrs["breaking"]              # e.g. "miche gamma=0.73"
efth_near.attrs["breaking_scale_min"]    # smallest energy scale factor applied
```

## Persisting operators

```python
model.to_netcdf("berth3.nc")
model = SiteModel.from_netcdf("berth3.nc")
```

The netCDF holds the matrix, the spectral grid, the boundary point positions
and the breaking settings. This is what an operational pipeline ships: build
offline, load at runtime, transform each forecast cycle.

## Exporting rays for inspection

The same rays that build the operator can be written as GeoJSON — one
`MultiLineString` per (frequency, direction) bin, with period, direction,
per-ray status and friction decay in the properties.

```python
from waveray import ray_paths_geojson
import numpy as np

x, y = grid.to_local(np.array([lon]), np.array([lat]))
gj = ray_paths_geojson(
    grid, (float(x[0]), float(y[0])),
    freqs=[0.06, 0.09, 0.14],
    dirs=np.arange(0, 360, 15),
    nsub=1,            # 1 clean line per bin
    stride=5,          # keep every 5th vertex
    path="rays.geojson",
)
```

Coordinates are WGS84 when the grid has a geographic origin, otherwise local
metres. Drop the file into QGIS to see the arrival cone: from a shallow target
only a narrow band of offshore directions physically reaches the site — the
rest ground out.

## Performance notes

- Operator build cost scales with `nsub × ndirs × nfreqs` rays and the domain
  crossing length. ~10 s is typical for a 200×150 grid, 32 frequencies and 36
  directions.
- `transform` is a single `einsum`; a month of hourly spectra takes ~100 ms.
- Everything is pure NumPy. There is no compilation step and no wave-model
  binary.
