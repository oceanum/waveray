# Wind forcing

waveray can integrate SWAN's wind input source term along every backward ray
path, so the transformed spectra include the wave growth the wind would have
produced across the downscale domain. The wind is supplied at **build time**
— either a single uniform value or a gridded field — and is baked into the
operator like the bathymetry.

```python
model = SiteModel.build(
    bathy=grid,
    target=(lon, lat),
    boundary_points=[...],
    freqs=freqs, dirs=dirs,
    wind=(12.0, 225.0),      # U10 [m/s], coming-from nautical degrees
    agrow=True,              # optional: seed locally generated wind sea
)
```

## When it matters

The boundary spectra of a downscale already carry the offshore wind sea, so
over a short fetch the wind input is usually a modest correction. It becomes
important when:

- the domain has an appreciable fetch of its own (a few km or more) aligned
  with a strong wind, so the sea keeps growing between the boundary and the
  target;
- part of the target's directional exposure is *not* covered by the boundary
  — a fetch opening across a bay or sound — so some direction bins arrive
  with no boundary energy at all. Only the linear growth term (`agrow=True`)
  can seed energy in those bins: exponential growth multiplies what is
  already there, and multiplying zero stays zero.

## Formulations

The implementation follows the SWAN technical documentation (GEN3 KOMEN
defaults). Wind input is `S_in = A + B E` with `sigma` the radian frequency,
`theta` the wave direction and `theta_w` the wind direction (both going-to
here; the API uses coming-from nautical degrees for both).

### Exponential growth — Komen et al. (1984)

```text
B = max[0, 0.25 (rho_a / rho_w) (28 u*/c cos(theta - theta_w) - 1)] sigma
```

with `rho_a / rho_w = 0.00125` (SWAN's value) and `c` the local phase speed
— growth strengthens in shallow water as `c` drops. `B E` is linear in `E`,
exactly like JONSWAP bottom friction, so it folds into the same per-ray path
exponent: along a backward ray the invariant `F = E c cg` evolves as

```text
dF/ds = (B - D_friction) / cg * F        =>      F ~ exp(+/- path integral)
```

and the operator coefficient simply carries `exp(-atten)` with
`atten = integral (D_friction - B)/cg ds`. A following wind makes `atten`
negative (net gain); an opposing or cross wind contributes nothing (the
`max[0, ...]` cutoff).

### Linear growth — Cavaleri & Malanotte-Rizzoli (1981), `agrow=True`

```text
A = 1.5e-3 / (2 pi g^2) * (u* max[0, cos(theta - theta_w)])^4 * H
H = exp(-(sigma / sigma_PM)^-4)
sigma_PM = 2 pi * 0.13 g / (28 u*)
```

This is SWAN's `AGROW` term with Tolman's (1992) filter `H`, which cuts
growth below the Pierson–Moskowitz frequency of a fully developed sea. `A`
does not depend on `E`, so it cannot live in the multiplicative operator;
instead it is integrated along each ray into an **additive spectrum**
`TransferOperator.E0[freq, dir]`:

```text
E0 = integral  c A  * (net gain from the source point to the target)  ds / (c cg)_target
```

`apply()` adds `E0` after the linear transfer. Two consequences worth
knowing:

- **Blocked rays still generate.** A ray that grounds on an island carries
  no boundary energy, but the wind blowing over its wet path still makes
  waves that reach the target — `E0` includes them.
- **Units become absolute.** `T` is built from density *ratios* and works in
  any consistent units, but `E0` is an absolute density. With `agrow=True`
  the spectra you transform must be in m² / Hz / deg (the wavespectra
  convention).

### The growth ceiling — `max_growth`

> [!WARNING]
> Wind input on its own is **unbounded**. In SWAN it is balanced by
> whitecapping and quadruplet interactions; both are nonlinear in `E` and
> cannot live in a spectrum-independent linear operator. SWAN will not even
> run the configuration this package implements: it raises a level-2 error
> for a third-generation wind without quadruplets, and when forced past it
> the run never converges and returns zero.

The runaway is worst at high frequency, where `cg` is small so a ray spends
a long time under the wind: at 0.9 Hz with U10 = 15 m/s, `exp(B·L/cg)` over
5 km is about `e³⁶`. Left alone that produces nonsense — an early build of
this feature returned Hs = 7×10²⁴ m at 5 km fetch — and eventually `inf`.

`build_operator(..., max_growth=100.0)` (the default) floors the growth
exponent so no single ray path can gain more than a hundredfold in energy.
When it binds you get a warning naming the affected fraction, and the
operator records it:

```python
op.attrs["growth_clipped_fraction"]   # 0.0 when the ceiling never bit
```

A non-zero fraction means the case is outside the regime where input-only
wind forcing is meaningful. Shorten the domain, drop the high-frequency
bins, or reduce the wind — do not simply raise the ceiling. Pass
`max_growth=None` to disable it entirely (expect overflow).

### Friction velocity — Zijlema et al. (2012)

`u*` is computed from U10 with SWAN's default drag law (since SWAN 41.01):

```text
Cd = (0.55 + 2.97 U~ - 1.49 U~^2) * 1e-3,   U~ = U10 / 31.5 m/s
u*^2 = Cd U10^2
```

The fit peaks near 31.5 m/s and is clipped at zero far beyond its validity
range. `drag_coefficient(u10)` and `friction_velocity(u10)` are exported if
you want the numbers.

## Supplying the wind

`wind` accepts three forms (anything else raises `TypeError`):

### Uniform value

```python
wind=(12.0, 225.0)   # U10 [m/s], coming-from nautical degrees
```

The meteorological convention matches the package's wave directions: 225°
means wind *from* the south-west.

### Gridded field

```python
wind=wind_ds         # xarray.Dataset, one snapshot
```

The Dataset must hold a **single snapshot** (the operator is stationary — a
`time` axis with more than one entry raises; a length-1 axis is squeezed)
with either

- eastward/northward 10-m components under one of the variable-name pairs
  `u10/v10`, `ugrd10m/vgrd10m`, `uwnd/vwnd`, `u/v` (case-insensitive), or
- speed and coming-from direction as `wspd`/`wdir`,

on 1-D coordinates named `lon`/`lat` (or `longitude`/`latitude`) for
geographic grids, or `x`/`y` in local metres for non-geographic `LocalGrid`s.
Descending coordinates are handled. The components are interpolated
bilinearly onto the bathymetry grid nodes; points outside the wind grid take
the nearest edge value, so make the wind grid at least as large as the
domain. Direction is interpolated through the vector components, so it is
well behaved across the 360°/0° wrap.

```python
# e.g. an ERA5 snapshot from Datamesh
wind_ds = conn.query({
    "datasource": "era5_wind10m",
    "timefilter": {"times": ["2024-01-02T12:00:00"]},
    "geofilter": {"type": "bbox", "geom": [114.2, -29.0, 114.8, -28.5]},
})
model = SiteModel.build(..., wind=wind_ds)
```

### Prebuilt `WindField`

```python
from waveray import WindField
wf = WindField.uniform(grid, 12.0, 225.0)      # or WindField(usx=..., usy=...)
model = SiteModel.build(..., wind=wf)
```

`WindField` stores the friction-velocity vector `(usx, usy)` on the
LocalGrid nodes (going-to math convention, x east / y north). Build one
directly if your wind comes from a source the Dataset reader does not cover.

## What the operator records

```python
op.attrs["wind_source"]   # "none" | "uniform" | "gridded" | "windfield"
op.attrs["wind_speed"]    # uniform wind only
op.attrs["wind_dir"]      # uniform wind only
op.attrs["agrow"]         # 0 | 1
op.attrs["growth_clipped_fraction"]   # rays that hit the max_growth ceiling
op.E0                     # (nf, ndir) additive spectrum, or None
```

Everything round-trips through `to_netcdf` / `from_netcdf`, so a persisted
operator keeps its wind physics.

## Caveats

- **Stationary.** The wind is fixed at build time. If wind input matters at
  your site and the wind varies, build one operator per wind condition (a
  handful of speed/direction classes is usually enough) and pick per
  timestep — the same pattern as per-tide-stage operators.
- **Input term only.** SWAN balances `S_in` against whitecapping and
  quadruplet interactions; both are nonlinear in `E` and cannot live in a
  linear operator. Over the short fetches this package targets the
  imbalance is small; over tens of kilometres of strong following wind the
  sea will overgrow. The depth-limited breaking cap still bounds the total
  energy in shallow water.
- **No sheltering of the wind itself.** The wind field is applied as given;
  waveray does not modify it for land shadows or stability.

## Validation

**Analytic.** `tests/test_wind.py` pins the implementation to closed forms on
flat-bottom domains, where straight rays make the path integrals analytic:
the operator gain equals `exp(B L / cg)` for a following wind,
opposing/cross winds leave the operator bit-for-bit unchanged, and the
`agrow` seed equals `q/r (exp(rL) - 1) / (c cg)` with and without friction.
A constant gridded Dataset reproduces the uniform tuple exactly, and
`wind=None` is verified identical to the pre-wind operator.

**Against SWAN.** `tests/test_validation_swan.py` runs stationary SWAN 41.51A
in the official `delftwaves/swan` docker image on identical boundary spectra
(see [Validation](validation.md#against-swan-4151a-docker)). Measured with a
12 m/s wind:

| Case | waveray / SWAN Hs |
|---|---|
| Swell + wind, plane beach, 15 km fetch | 1.03 – 1.25 |
| Wind sea from calm, `f ≤ 0.4 Hz`, 1–5 km fetch | 0.54 – 0.70 |

The two rows bracket the physics that is missing. With swell already present
the operator **over**-predicts, because SWAN's whitecapping dissipates while
waveray only adds. Generating a sea from calm it **under**-predicts, because
without quadruplets there is no downshifting to move energy into the peak.
Both errors grow with fetch — which is the practical answer to "how long a
fetch can I trust this over?": a few kilometres, on the resolved swell
frequencies.

## References

- Cavaleri, L. and P. Malanotte-Rizzoli (1981). Wind wave prediction in
  shallow water: theory and applications. *J. Geophys. Res.* 86, 10961–10973.
- Komen, G.J., S. Hasselmann and K. Hasselmann (1984). On the existence of a
  fully developed wind-sea spectrum. *J. Phys. Oceanogr.* 14, 1271–1285.
- Tolman, H.L. (1992). Effects of numerics on the physics in a
  third-generation wind-wave model. *J. Phys. Oceanogr.* 22, 1095–1111.
- Zijlema, M., G.Ph. van Vledder and L.H. Holthuijsen (2012). Bottom
  friction and wind drag for wave models. *Coastal Engineering* 65, 19–26.
- [SWAN scientific and technical documentation](https://swanmodel.sourceforge.io/online_doc/swantech/swantech.html),
  "Input by wind".
