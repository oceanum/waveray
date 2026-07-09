# waveray

Fast last-stage nearshore spectral wave transformation: precomputed backward
ray-traced linear transfer operators over local bathymetry, with parametric
depth-limited wave breaking at the target.

Built to downscale directional wave spectra from SWAN (or WW3) hindcasts and
nowcasts through the final nearshore transformation to a target site — decades
of hourly spectra in seconds, no wave model in the runtime loop.

## Install

```bash
pip install waveray                # core
pip install "waveray[datamesh]"    # + Oceanum Datamesh bathymetry/spectra access
```

## Documentation

| Guide | Read it for |
|---|---|
| [Installation](docs/installation.md) | Install, extras, environment |
| [Quickstart](docs/quickstart.md) | A working example in 20 lines |
| [Concepts](docs/concepts.md) | What the operator is, and the physics inside it |
| [User guide](docs/usage.md) | Bathymetry, boundary points, tide, breaking, persistence, ray export |
| [wavespectra interop](docs/wavespectra.md) | Spectral conventions and the `.spec` accessor |
| [Validation](docs/validation.md) | Measured skill against parent SWAN models |
| [Limitations](docs/limitations.md) | What waveray does **not** model |
| [API reference](docs/api.md) | Every public function and its arguments |

Start at [`docs/index.md`](docs/index.md).

## Method

Setup, once per site:

1. Pull local bathymetry (e.g. from an Oceanum Datamesh datasource) into a
   `LocalGrid` on a local tangent plane.
2. For every (frequency, direction) bin, trace ray bundles **backward** from
   the target over the 2D bathymetry (O'Reilly & Guza style spectral
   refraction). Each sub-ray records where and in what direction it exits the
   domain, or whether it runs aground (island/headland sheltering).
3. Assemble a linear operator `T[f, dir_t, bp, dir_b]` mapping **K boundary
   spectra** (the SWAN output points around the domain perimeter —
   alongshore inhomogeneity is handled by exit-point interpolation) to the
   target spectrum. The per-ray coefficient is the exact linear invariant
   `E(f, theta) * c * cg = const`, which reproduces Snell refraction and
   energy-flux shoaling identically (validated against analytic plane-beach
   solutions in the test suite).

Runtime, per hindcast:

4. `E_t = einsum(T, E_b)` over all timesteps at once.
5. Depth-limited breaking as a nonlinear post-step: transformed Hm0 is capped
   at a Miche-type (Battjes-Janssen) or `gamma * d` limit with optional
   tide-modulated depth; the cap scales the spectrum proportionally, matching
   how SWAN distributes surf-breaking dissipation over the spectrum.

The stationary, linear-propagation assumption (no temporal nonlinear
evolution, no quadruplets/triads) is deliberate: over a last-mile nearshore
domain the propagation time is minutes and the transformation is dominated by
refraction, shoaling, sheltering and breaking.

## Quickstart

```python
import numpy as np
from waveray import SiteModel, fetch_datamesh_bathymetry

bathy = fetch_datamesh_bathymetry(
    "gebco_2025", bbox=(114.35, -28.95, 114.65, -28.60), positive="up"
)
model = SiteModel.build(
    bathy=bathy,
    target=(114.58, -28.77),                     # lon, lat
    boundary_points=[(114.35, -28.90), (114.35, -28.65)],  # SWAN output sites
    freqs=efth_boundary.freq.values,
    dirs=efth_boundary.dir.values,
)
efth_near = model.transform(efth_boundary, tide=tide_series)   # (time, freq, dir)
model.to_netcdf("geraldton_berth.nc")            # reuse without rebuilding
```

`efth` follows the wavespectra convention: dims `(time, [site,] freq, dir)`,
`dir` = coming-from nautical degrees, density per Hz per degree. The `site`
dimension (K boundary points, same order as `boundary_points`) is required
when K > 1.

Ray paths can be exported for inspection (QGIS, EIDOS, any web map) as a
GeoJSON FeatureCollection — one MultiLineString per (freq, dir) bin, with
period, direction, per-ray status and friction decay in the properties:

```python
from waveray import ray_paths_geojson

x, y = bathy_grid.to_local(lon, lat)
ray_paths_geojson(bathy_grid, (x[0], y[0]), freqs=[0.06, 0.1],
                  dirs=np.arange(0, 360, 15), path="rays.geojson")
```

## Notebooks

Two executed, illustrated notebooks (plots included in the committed output):

| Notebook | What it shows |
|---|---|
| [`notebooks/01_holland_downscaling.ipynb`](notebooks/01_holland_downscaling.ipynb) | End-to-end downscaling of a SWAN 1 km hindcast to a point off **Noordwijk aan Zee** through storms Pia and Henk: bathymetry, ray geometry and the arrival cone, offshore vs nearshore Hs, the directional spectrum before/after, validation against SWAN (r ≈ 0.99), and what depth-limited breaking contributes. |
| [`notebooks/02_abrolhos_validation.ipynb`](notebooks/02_abrolhos_validation.ipynb) | Held-out validation on a **reef-fronted WA coast**, and an ablation isolating JONSWAP bottom friction (Hm0 bias +1.11 m → +0.18 m when friction is on). |

```bash
uv sync --extra datamesh --extra notebooks
DATAMESH_TOKEN=... uv run jupyter lab notebooks/
```

## Limitations (v0)

- No diffraction: accuracy degrades inside harbours / behind breakwaters, and
  island shadows are sharper than reality (rays block, they do not leak).
- Island / headland sheltering *is* included — a ray grounding on any land
  carries no energy — but blocking is binary and only as good as the
  bathymetry: a feature smaller than ~2 grid cells is smoothed away by the
  bilinear depth sampling and will not shelter (GEBCO at ~450 m cannot
  shelter behind a small reef or islet). See `tests/test_island.py`.
- Bottom friction IS included (JONSWAP, integrated along ray paths) and is ON
  by default with the SWAN swell coefficient `cf_jonswap=0.038`; pass
  `cf_jonswap=None` for pure refraction + shoaling. No triad interactions.
- Breaking is an endpoint cap, not accumulated dissipation along the approach
  — appropriate at berths and outside the inner surf zone; tune `gamma` per
  site against observations.
- The operator is built at a fixed water level; tide only modulates the
  breaking cap. For strongly tide-dependent refraction, build operators per
  tide stage.
- Boundary spectra quality bounds achievable skill — nested-model literature
  consistently finds offshore directional accuracy, not nearshore resolution,
  is the limiting factor.

## Development

```bash
uv sync                     # or: uv sync --extra datamesh
uv run pytest               # physics-validation suite
uv run ruff check . && uv run ruff format --check .
```
