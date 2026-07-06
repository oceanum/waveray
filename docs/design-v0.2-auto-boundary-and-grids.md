# v0.2 design scope: automatic boundary definition & gridded output

Status: scoped 2026-07-06, not implemented. Two independent extensions.

## 1. Automatic ray-origin boundary from target + bathymetry

Today the user supplies the bathy bbox and the boundary spectra points, and
rays terminate on the bbox perimeter. Proposal: derive all of that from the
target point and the bathymetry alone.

### Physical criterion

Backward rays should stop where the *remaining* transformation upstream is
negligible, i.e. where the water is deep enough that refraction/shoaling of
the longest resolved wave is weak. Criterion: `kd >= kd_min` (default 1.2,
where cg is within ~5% of deep water and ray curvature is negligible) for the
lowest frequency of interest `f_min`:

    d_bc = depth such that k(f_min, d_bc) * d_bc = kd_min

e.g. f_min = 0.04 Hz -> d_bc ~ 180 m; f_min = 0.06 Hz -> d_bc ~ 80 m.
User-overridable (`d_bc=...`) because SWAN output lines are often shallower;
when the SWAN sites are the boundary the *sites'* depth wins — the criterion
then only validates the choice (warn when sites are shallower than d_bc).

### Mechanics

1. **Depth-threshold ray stopping** (core change, trivial): rays gain a third
   stop condition `depth >= d_bc` (STATUS_EXITED_DEEP) alongside bbox exit
   and grounding. The "ray-origin boundary" is then implicitly the d_bc
   contour — no polyline geometry needed, one comparison per step.
2. **Expanding bathy fetch**: starting from the target, fetch the Datamesh
   bathy in a growing bbox (e.g. 10 km steps, capped at `max_domain_km`)
   until every non-landed backward test-ray (coarse fan at f_min) reaches
   d_bc or land. Guarantees the domain contains its own boundary.
3. **Boundary-condition interpolation without a perimeter**: with stopping on
   a contour, the perimeter parameterisation no longer applies. Replace with
   inverse-distance weighting over the k=2..3 nearest boundary sites at each
   ray's stop point. Simpler, handles scattered site clouds, degenerates to
   nearest-site, and reproduces the current behaviour when sites bracket the
   exits. (This also answers "arbitrary interior SWAN points" exactly.)
4. **Auto site discovery** (optional, needs a spec datasource id): query the
   datasource for sites with depth >= ~0.8 d_bc inside the domain, keep those
   spanning the angular window of actual ray exits, warn on angular gaps
   > ~30 deg of exit density.

### API sketch

```python
model = SiteModel.auto(
    target=(114.596, -28.777),
    bathy_datasource="gebco_2025",
    spec_datasource="oceanum_wave_ec_abrol500m_spec_nowcast",  # or boundary_points=[...]
    f_min=0.04,
    d_bc=None,            # derived from f_min unless given
    max_domain_km=60.0,
)
```

Effort: ~1 day. Risks: directions where d_bc is unreachable (enclosed water)
fall back to bbox exit + nearest site; disconnected deep regions are handled
naturally by IDW; GEBCO landmask quality near harbours remains the dominant
input error (allow regional bathy datasource override).

## 2. Gridded output (Hs variation over the domain)

The operator is per-target. Options for a full grid, in order of pragmatism:

**A. Batched per-node backward build (recommended first step).** The
per-frequency SpeedFields are shared across targets, and the ray integrator
is already vectorised — so trace *all nodes' rays in one lockstep batch* per
frequency. With reduced settings for mapping (nsub=3, coarse dir bins,
frequency subset), estimated <0.5 s/node one-off, embarrassingly parallel;
a 50x50 wet-node berth-area grid ~ tens of minutes once, then every timestep
of the hindcast is one einsum for the whole grid (the grid operator is just
a stack of point operators). Produces full spectra per node -> Hs, Tp, Dp
maps. Persist as one netcdf; publish maps to Datamesh/EIDOS as a layer.
API sketch: `GridModel.build(bathy, targets=grid_or_mask, ...)` reusing
`build_operator` internals with a `targets` batch dimension.

**B. Forward ray shooting per timestep.** Shoot rays *forward* from the
boundary sampling the actual boundary spectrum of that timestep, accumulate
energy flux onto grid cells. One sweep gives the whole map; cost scales with
ray count not node count; caustic noise needs many rays + smoothing. Right
choice for on-demand animation frames over large domains; wrong choice for
long hindcast statistics (cost x timesteps).

**C. Mini stationary spectral sweep (SWAN-lite / rompy + SWAN).** For exact
physics including accumulated breaking over the grid, run a stationary SWAN
(rompy-configured) on the local domain per selected timestep. Seconds-to-
minutes per map; the honest benchmark for A/B, and the fallback where
diffraction/triads matter. Not for per-timestep hindcast use.

**D. Sparse nodes + conditioned interpolation.** Build A on a few hundred
smart nodes (stratified by depth/exposure), interpolate the *transfer ratio*
(Hs_local / Hs_boundary) over the grid. Cheapest for visualisation; hides
fine reef structure.

Recommendation: implement A (GridModel batch build) — it reuses ~all existing
code, keeps the einsum runtime model, and covers the berth-area mapping use
case; keep C as validation. B only if large-domain animations become a
product need.
