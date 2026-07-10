# Working with Oceanum Datamesh

[Datamesh](https://docs.oceanum.io/datamesh/overview) is Oceanum's data platform.
waveray needs two things from it — **bathymetry** for the local grid, and
**offshore spectra** for the boundary condition — and both come from the same
Python client.

Datamesh is optional. waveray works with any `xarray` bathymetry and any
wavespectra-readable spectrum from any source. This page is for when you do use
it.

## Setup

```bash
pip install "waveray[datamesh]"     # pulls in the `oceanum` package
export DATAMESH_TOKEN="..."         # your Datamesh token
```

Get a token from your [Oceanum account](https://docs.oceanum.io/get-started/account).
Every client call reads `DATAMESH_TOKEN` from the environment; you can also pass
one explicitly (`Connector(token=...)`, or `fetch_datamesh_bathymetry(..., token=...)`).

> [!WARNING]
> Treat the token as a credential. Keep it in the environment or a secrets
> manager — never commit it, and never leave it in a notebook cell output.

```python
from oceanum.datamesh import Connector

conn = Connector()          # reads $DATAMESH_TOKEN, talks to https://datamesh.oceanum.io
```

## Finding data

Browse [datasets.oceanum.io](https://datasets.oceanum.io/) or the
[Datamesh UI](https://docs.oceanum.io/datamesh/ui), or search the catalog from
Python:

```python
cat = conn.get_catalog(search="wave hindcast spectra")
for ds in cat:
    print(ds.id, "-", ds.name)

meta = conn.get_datasource("oceanum_wave_dutch_era5_v1_spec")
print(meta.tstart, meta.tend, meta.coordinates, meta.geom)
```

`get_catalog(search=..., timefilter=..., geofilter=..., limit=...)` filters the
catalog; `get_datasource(id)` returns the full metadata record (`id`, `name`,
`description`, `geom`, `tstart`, `tend`, `tags`, `labels`, `coordinates`,
`dataschema`, `details`, …). Check `coordinates` before querying — it tells you
what the time, x/y, frequency and direction dimensions are called.

### Datasources used in this project

These are the ones the bundled examples and notebooks actually query.

| Datasource | What it is |
|---|---|
| `gebco_2023`, `gebco_2025` | Global 15-arcsecond bathymetry. **Elevation**, positive up. |
| `oceanum_wave_dutch_era5_v1_spec` | SWAN 1 km Dutch waters, hourly spectra, 1979–present |
| `oceanum_wave_ec_abrol500m_spec_nowcast` | SWAN 500 m Abrolhos (WA) spectra, nowcast |
| `oceanum_wave_weuro_era5_v1_spec` | SWAN 5 km Western Europe spectra |
| `oceanum_wave_waddenzee_nora3_v1_spec` | SWAN ~500 m Waddenzee spectra |
| `oceanum_wave_glob05_era5_v1_spec` | Global WW3 0.5° spectra, 26,962 sites, 1979–present |
| `oceanum_wave_glob05_gfs_spec_nowcast` | Global WW3 0.5° spectra, GFS-forced nowcast |

Regional SWAN spectra make the best boundary condition, because they already
contain the shelf-scale transformation. A global WW3 datasource works when
nothing regional exists, but see
[Limitations](limitations.md#your-boundary-spectra-bound-your-skill) — the
directional accuracy of your boundary spectra bounds your nearshore skill.

## Bathymetry

`fetch_datamesh_bathymetry` is a thin convenience wrapper: it issues a bbox
query, takes the first data variable unless you name one, and returns a
[`LocalGrid`](api.md#localgrid).

```python
from waveray import fetch_datamesh_bathymetry

grid = fetch_datamesh_bathymetry(
    "gebco_2023",
    bbox=(3.85, 52.00, 4.55, 52.50),   # (west, south, east, north) degrees
    positive="up",                      # GEBCO is elevation: seabed negative
)
print(grid.x.size, "x", grid.y.size, "nodes at ~", grid.spacing[0], "m")
```

> [!IMPORTANT]
> `positive="up"` for **elevation** sources (GEBCO), `positive="down"` for
> **depth** sources. Getting it backwards turns your ocean into a mountain and
> every ray will immediately ground. Sanity check with
> `grid.depth[~grid.land].max()`.

Equivalent by hand, if you want the raw Dataset:

```python
ds = conn.query({
    "datasource": "gebco_2023",
    "geofilter": {"type": "bbox", "geom": [3.85, 52.00, 4.55, 52.50]},
})
grid = LocalGrid.from_dataarray(ds["elevation"], positive="up")
```

GEBCO is ~450 m at these latitudes. That is fine for open-coast refraction and
too coarse to resolve a reef, an islet or a breakwater — see
[Limitations](limitations.md#sheltering-is-binary-and-only-as-good-as-the-bathymetry).
Where sheltering matters, query a survey-resolution regional grid instead.

## Spectra

A spectral query needs a **geofilter** (which sites) and a **timefilter** (which
times).

```python
ds = conn.query({
    "datasource": "oceanum_wave_dutch_era5_v1_spec",
    "geofilter": {"type": "bbox", "geom": [3.85, 52.00, 4.55, 52.50]},
    "timefilter": {"times": ["2023-12-15T00:00:00Z", "2024-01-15T00:00:00Z"]},
})
efth = ds["efth"]          # (time, site, freq, dir), wavespectra convention
```

The result already speaks waveray's language: `efth` with `freq` in Hz and `dir`
as coming-from nautical degrees. Pass it straight to
[`SiteModel.transform`](api.md#sitemodeltransform-xrdataarray).

### Query reference

The `query` dict is validated against `oceanum.datamesh.query.Query`. The fields
that matter here:

| Field | Purpose |
|---|---|
| `datasource` | the datasource id (required) |
| `geofilter` | `{"type": "bbox", "geom": [w, s, e, n]}` or `{"type": "feature", "geom": <GeoJSON Feature>}` |
| `timefilter` | `{"times": [start, end]}` — also accepts ISO-8601 durations, e.g. `"-P7D"` for "seven days ago" |
| `variables` | list of variables to return, to trim the payload |
| `limit` | cap the number of records |
| `crs` | reproject the result |

Geofilter types are `bbox` and `feature`; the `interp` option is `nearest`
(default) or `linear`. Timefilter types are `range`, `series` and `trajectory`.
`conn.query(..., use_dask=True)` returns a lazily-loaded Dataset, and
`cache_timeout=<seconds>` caches the response locally.

Full details: [Datamesh Python client](https://docs.oceanum.io/datamesh/integrations/python)
and the [oceanum-python API reference](https://oceanum-python.readthedocs.io/en/latest/api.html).

### Selecting boundary points

waveray needs the boundary spectra **in the same order** as the
`boundary_points` you pass to `SiteModel.build`. A robust recipe — probe one
timestep to find the sites, choose an offshore arc, then fetch only those sites:

```python
import numpy as np

# 1. one cheap timestep to discover the site positions
probe = conn.query({
    "datasource": SPEC,
    "geofilter": {"type": "bbox", "geom": list(BBOX)},
    "timefilter": {"times": [T0, T0]},
})
lons = np.atleast_1d(probe["lon"].values)
lats = np.atleast_1d(probe["lat"].values)

# 2. keep sites deep enough to sit outside the last-stage transformation,
#    spread by azimuth around the target
xs, ys = grid.to_local(lons, lats)
deep = np.flatnonzero(grid.sample_depth(xs, ys) >= 16.0)
az = np.arctan2(ys[deep] - ty, xs[deep] - tx)
picks = deep[np.argsort(az)][np.linspace(0, deep.size - 1, 6).astype(int)]
bpts = [(float(lons[i]), float(lats[i])) for i in picks]

# 3. fetch the full time series for just those sites, one small query each
import xarray as xr
parts = []
for lon, lat in bpts:
    d = conn.query({
        "datasource": SPEC,
        "geofilter": {"type": "bbox",
                      "geom": [lon - 0.004, lat - 0.004, lon + 0.004, lat + 0.004]},
        "timefilter": {"times": TIMES},
    })
    sd = next(x for x in d["efth"].dims if x not in ("time", "freq", "dir"))
    parts.append(d["efth"].isel({sd: 0}).rename({sd: "site"}))
efth_b = xr.concat(parts, dim="site").transpose("time", "site", "freq", "dir")
```

Several small per-site queries beat one big bbox query: you get exactly the
sites you asked for, in the order you asked for them, and you do not download a
month of spectra for sites you will discard.

If the spectra carry `lon`/`lat` coordinates, `transform` **verifies** they
match the boundary points used at build time and raises rather than silently
mixing sites up.

## Gotchas seen in real Datamesh data

These are not hypothetical — each one cost time in this project.

> [!WARNING]
> **The `dir` coordinate is not monotonic.** `oceanum_wave_dutch_era5_v1_spec`
> ships directions running `265° → 5°`, then wrapping `355° → 275°`. waveray
> sorts internally with circular interpolation and returns the spectrum on your
> original axis, so this is invisible to you — *unless* you plot it yourself
> with `pcolormesh` on polar axes, which requires a monotonic theta and will
> otherwise smear energy across the circle. Sort before plotting.

**The site dimension is not always called `site`.** Discover it rather than
assume it:

```python
site_dim = next(d for d in ds["efth"].dims if d not in ("time", "freq", "dir"))
```

**Don't assume coordinate shapes.** A one-site bbox still returns length-1
arrays and keeps the `site` dimension (verified on the Dutch and Abrolhos
spectra), but `np.atleast_1d(ds["lon"].values)` is a cheap guard that keeps the
same code working if a driver ever collapses the dimension.

**Spectral sites near a bbox edge may fall outside your bathymetry grid.**
GEBCO coordinates are cell centres, so the grid's usable extent is slightly
inside the bbox you requested. Keep a margin (~2 km) when picking interior sites,
or `LocalGrid` will reject the target as outside its bounds.

**Datasource ids encode the model, forcing and product.** Read them as
`oceanum_<var>_<domain><res>_<forcing>_<grid|spec>_<nowcast|forecast>`; a
`_spec` datasource holds 2-D spectra at sites, a `_grid` datasource holds
integrated parameters on a grid. You want `_spec` for waveray's input.

## A complete example

[`examples/demo_holland.py`](https://github.com/oceanum/waveray/blob/main/examples/demo_holland.py)
does all of the above end to end — bathymetry, site discovery, boundary
selection, transformation and validation against SWAN's own nearshore answer.
The same material with plots is in
[`notebooks/01_holland_downscaling.ipynb`](https://github.com/oceanum/waveray/blob/main/notebooks/01_holland_downscaling.ipynb).

Run it with:

```bash
uv sync --extra datamesh
DATAMESH_TOKEN=... uv run python examples/demo_holland.py
```

## Reference

| Resource | Link |
|---|---|
| Datamesh overview | <https://docs.oceanum.io/datamesh/overview> |
| Datamesh Python client | <https://docs.oceanum.io/datamesh/integrations/python> |
| Datamesh UI | <https://docs.oceanum.io/datamesh/ui> |
| Getting an account and token | <https://docs.oceanum.io/get-started/account> |
| Select-and-export tutorial | <https://docs.oceanum.io/tutorials/select-and-export> |
| Dataset catalogue | <https://datasets.oceanum.io/> |
| `oceanum` Python package | <https://pypi.org/project/oceanum/> · [source](https://github.com/oceanum-io/oceanum-python) |
| `oceanum-python` API docs | <https://oceanum-python.readthedocs.io/en/latest/api.html> |
