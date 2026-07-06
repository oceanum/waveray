# nearshore-transform

Fast last-stage nearshore spectral wave transformation: precomputed backward
ray-traced linear transfer operators over local bathymetry, with parametric
depth-limited wave breaking at the target.

Built to downscale directional wave spectra from SWAN (or WW3) hindcasts and
nowcasts through the final nearshore transformation to a target site — decades
of hourly spectra in seconds, no wave model in the runtime loop.

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
from nearshore_transform import SiteModel, fetch_datamesh_bathymetry

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

## Limitations (v0)

- No diffraction: accuracy degrades inside harbours / behind breakwaters.
- No bottom friction or triad interactions along the path.
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
