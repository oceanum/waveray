# API reference

Everything in this page is importable from the top-level package unless noted.

```python
from waveray import (
    LocalGrid, SiteModel, TransferOperator,
    build_operator, fetch_datamesh_bathymetry, ray_paths_geojson,
    set_wavespectra_attrs, to_specdataset,
)
```

---

## `SiteModel`

A built nearshore transformation for one target point. Fields: `operator`
(`TransferOperator`), `gamma` (float), `breaking_method` (str).

### `SiteModel.build(...) -> SiteModel`

```python
SiteModel.build(
    bathy,                      # xr.DataArray | LocalGrid
    target,                     # (lon, lat), or (x, y) for a non-geographic LocalGrid
    boundary_points,            # list[(lon, lat)] | ndarray (K, 2)
    freqs,                      # ndarray (nf,)  [Hz]
    dirs,                       # ndarray (ndir,) coming-from nautical degrees, uniform
    positive="down",            # "up" for elevation bathymetry (GEBCO)
    gamma=0.73,
    breaking_method="miche",    # "miche" | "gamma"
    **ray_kwargs,               # forwarded to build_operator
) -> SiteModel
```

Traces rays once and assembles the transfer operator. `ray_kwargs` are `nsub`,
`ds`, `max_steps`, `d_min`, `cf_jonswap`.

### `SiteModel.transform(...) -> xr.DataArray`

```python
model.transform(
    efth,                # xr.DataArray | xr.Dataset with an 'efth' variable
    site_dim="site",
    tide=None,           # float | ndarray | DataArray, metres
    breaking=True,
) -> xr.DataArray        # dims (..., freq, dir), wavespectra-compatible
```

Raises `ValueError` if the `freq`/`dir` coordinates disagree with the operator,
if the `site` dimension is missing when `K > 1`, or if the spectra's `lon`/`lat`
coordinates do not match the boundary points used at build time.

Result attributes: `target_x`, `target_y`, `depth_target`, `breaking`,
`breaking_scale_min`, plus wavespectra CF attributes.

### `SiteModel.to_netcdf(path)` / `SiteModel.from_netcdf(path) -> SiteModel`

Persist and reload a built operator, including breaking settings.

---

## `LocalGrid`

Regular depth grid on a local tangent plane. Depth positive down; land is
`depth <= 0` or NaN.

```python
LocalGrid(x, y, depth, lon0=None, lat0=None)
LocalGrid.from_dataarray(da, positive="down") -> LocalGrid
```

| Member | Description |
|---|---|
| `.bounds` | `(xmin, xmax, ymin, ymax)` metres |
| `.spacing` | `(dx, dy)` median spacing, metres |
| `.land` | boolean mask |
| `.depth` | `(ny, nx)` depths, metres positive down |
| `.to_local(lon, lat)` | degrees → local metres |
| `.to_lonlat(x, y)` | local metres → degrees |
| `.sample_depth(x, y)` | bilinear depth at local points |

### `fetch_datamesh_bathymetry(...) -> LocalGrid`

```python
fetch_datamesh_bathymetry(
    datasource,                 # e.g. "gebco_2023"
    bbox,                       # (west, south, east, north) degrees
    variable=None,              # defaults to the first data variable
    positive="up",              # GEBCO convention
    token=None,                 # defaults to $DATAMESH_TOKEN
) -> LocalGrid
```

Requires the `datamesh` extra.

---

## `build_operator(...) -> TransferOperator`

The low-level entry point; `SiteModel.build` wraps it.

```python
build_operator(
    grid,                  # LocalGrid
    target_xy,             # (x, y) in grid metres
    boundary_xy,           # ndarray (K, 2) in grid metres
    freqs, dirs,
    nsub=7,                # sub-rays per direction bin
    ds=None,               # ray step [m]; default min(spacing)/3
    max_steps=None,        # default ~1.5 domain perimeters
    d_min=0.3,             # grounding depth [m]
    cf_jonswap=0.038,      # JONSWAP friction; None disables
) -> TransferOperator
```

## `TransferOperator`

| Member | Description |
|---|---|
| `.T` | `(nf, ndir_t, K, ndir_b)` transfer matrix |
| `.freq`, `.dir_t`, `.dir_b` | spectral grid |
| `.bp_x`, `.bp_y` | boundary point positions, metres |
| `.target_x`, `.target_y`, `.depth_target` | target position and depth |
| `.n_boundary` | `K` |
| `.attrs` | `nsub`, `ds`, `d_min`, `max_steps`, `cf_jonswap`, `lost_fraction`, `landed_fraction` |
| `.apply(efth)` | contract with `(..., K, nf, ndir_b)` → `(..., nf, ndir_t)` |
| `.to_netcdf(path)` / `.from_netcdf(path)` | persistence |
| `.to_dataset()` / `.from_dataset(ds)` | xarray round-trip |

---

## `ray_paths_geojson(...) -> dict`

```python
ray_paths_geojson(
    grid, target_xy, freqs, dirs,
    lonlat=None,        # default True when the grid has a geographic origin
    nsub=1,
    ds=None, max_steps=None, d_min=0.3, cf_jonswap=0.038,
    stride=5,           # keep every stride-th vertex
    path=None,          # also write the FeatureCollection here
) -> dict
```

One `MultiLineString` Feature per `(freq, dir)` bin. Properties: `freq`,
`period`, `dir`, `wavelength_at_target`, `status` (per sub-ray: `exited`,
`landed`, `lost`), `mean_friction_decay`.

---

## `waveray.spectra`

```python
set_wavespectra_attrs(efth) -> xr.DataArray   # stamp CF attrs in place
to_specdataset(efth) -> xr.Dataset            # wrap for wavespectra writers
```

See [wavespectra interoperability](wavespectra.md).

---

## `waveray.breaking`

```python
from waveray.breaking import hm0, hm0_max, apply_breaking, spectral_moment

hm0(efth, freqs, dirs)                    # == efth.spec.hs(tail=False)
spectral_moment(efth, freqs, dirs, n=0)   # wavespectra quadrature
hm0_max(depth_total, fm=None, gamma=0.73, method="miche")
apply_breaking(efth, freqs, dirs, depth, tide=0.0, gamma=0.73, method="miche")
    -> (efth_capped, scale)
freq_resolution(freqs)   # == SpecArray.df
dir_resolution(dirs)     # == SpecArray.dd
```

## `waveray.dispersion`

```python
from waveray.dispersion import wavenumber, phase_speed, group_speed, ccg

wavenumber(omega, depth)     # solves omega^2 = g k tanh(kd) to machine precision
phase_speed(omega, depth, k=None)
group_speed(omega, depth, k=None)
ccg(omega, depth)            # the ray invariant factor c * cg
```

## `waveray.rays`

```python
from waveray.rays import SpeedField, trace_backward, RayFan
from waveray.rays import STATUS_EXITED, STATUS_LANDED, STATUS_LOST

SpeedField.build(grid, omega, d_min, cf_jonswap=None)
trace_backward(field, x0, y0, theta0, ds, max_steps, d_min=0.3, record_paths=False) -> RayFan
```

`RayFan` carries `status`, `x`, `y`, `theta`, `atten`, and (when
`record_paths=True`) `paths`: a list of `(m_i, 2)` local-metre polylines.

## Direction conventions

```python
from waveray.operator import dir_to_theta, theta_to_dir
```

`dir_to_theta` maps coming-from nautical degrees to going-to mathematical
radians; `theta_to_dir` inverts it. Both are `(270 − x) mod 360` in their
respective units, and are self-inverse.
