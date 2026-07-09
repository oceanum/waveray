# Concepts

## The conservation law

For a stationary, linear wave field without currents, the directional variance
density obeys, along a wave ray,

```
E(f, θ) · c · cg = constant
```

where `c` is the phase speed and `cg` the group speed at that frequency and
depth. Frequency is conserved along the ray (the medium is steady), so a ray
carries a single frequency and the transformation is **frequency-diagonal**:
no energy moves between frequencies.

This is exact linear physics. It reproduces Snell's law of refraction and
energy-flux (Green's law) shoaling identically — both are consequences of the
same invariant — which is why the test suite can check the operator against
closed-form plane-beach solutions rather than against another model.

## Backward ray tracing

Rays obey, with arclength `s` and propagation direction `θ`:

```
dx/ds     = cos θ
dy/ds     = sin θ
dθ/ds     = ( sin θ · ∂c/∂x − cos θ · ∂c/∂y ) / c
```

Rays are reversible, so waveray integrates these equations *backward* from the
target with a vectorised RK4 scheme (all rays of one frequency march in
lockstep). Each ray ends in one of three states:

| State | Meaning | Contribution |
|---|---|---|
| `exited` | left through the domain boundary | picks up the boundary spectrum at its exit point and direction |
| `landed` | depth fell below `d_min` — ran aground | **zero** (this is island and headland sheltering) |
| `lost` | still inside after `max_steps` | zero; the fraction is reported in `operator.attrs["lost_fraction"]` |

Backward tracing is what makes the method cheap: you trace only the rays that
actually reach your target, not a whole domain's worth.

## The transfer operator

Assembling those rays gives a matrix

```
T[f, θ_target, k, θ_boundary]
```

mapping `K` boundary spectra to the target spectrum:

```
E_target[f, θ_t] = Σ_k Σ_θb  T[f, θ_t, k, θ_b] · E_boundary[k, f, θ_b]
```

Each of the `nsub` sub-rays fired per direction bin contributes a coefficient
`(c·cg)_exit / (c·cg)_target · exp(−friction) / nsub`, distributed onto the two
bracketing boundary points (by position along the domain perimeter) and the two
bracketing boundary direction bins. Averaging sub-rays smooths caustics.

Three things live inside `T`:

- **Refraction** — where each ray came *from*, in direction
- **Shoaling** — the `c·cg` ratio between exit and target
- **Sheltering** — grounded rays contribute nothing
- **Bottom friction** — a JONSWAP decay integrated along each ray path; it is
  linear in energy, so it folds into the coefficient without breaking
  linearity (see below)

Because `T` depends only on the bathymetry and the spectral discretisation, it
is computed once and reused for every timestep. That is the entire performance
story: **a wave model becomes a matrix multiply.**

## Bottom friction

The JONSWAP formulation dissipates energy at a rate proportional to `E`:

```
S_bf = − C_b · σ² / ( g² · sinh²(kd) ) · E
```

Linear in `E`, so along a ray the invariant `I = E·c·cg` decays as

```
I(s) = I(0) · exp( − ∫ C_b σ² / ( g² sinh²(kd) · cg ) ds )
```

waveray accumulates that exponent during the backward trace and folds
`exp(−∫…)` straight into the ray's coefficient. It is **on by default** with
the SWAN swell coefficient `cf_jonswap = 0.038`; pass `cf_jonswap=None` for
pure refraction and shoaling. On shallow shelves this is the leading
error term — see [Validation](validation.md).

## Breaking is the exception

Depth-induced breaking dissipates energy at a rate that depends on the **total**
transformed energy, so it cannot be a fixed linear coefficient. waveray applies
it as a nonlinear post-step at the target: if the transformed `Hm0` exceeds a
depth-limited maximum, the whole spectrum is scaled down by `(Hmax/H)²`.

Two limits are available:

- `gamma`: `Hm0_max = γ · d` — classic shallow-water saturation
- `miche` (default): `Hm0_max = (0.88/k̄) · tanh(γ · k̄ · d / 0.88)` — the
  Miche-type limit of Battjes & Janssen (1978), with `k̄` the wavenumber of the
  energy-weighted mean frequency. Adds steepness limiting in deeper water and
  reduces to `γ·d` in the shallow limit.

The default `γ = 0.73` is SWAN's. Proportional scaling matches how SWAN
distributes surf-breaking dissipation across the spectrum, so the endpoint is
consistent with the parent model — but the *history* of dissipation along the
approach is not modelled. See [Limitations](limitations.md).

## Why stationary is enough

Over a last-mile domain (tens of kilometres) the wave propagation time is
minutes, far shorter than the timescale on which the offshore sea state
changes. The transformation is dominated by refraction, shoaling, sheltering
and breaking — all instantaneous, geometry-driven processes. Temporal nonlinear
spectral evolution (quadruplet and triad interactions, wind input) matters over
fetches and hours, not over the last few kilometres. Dropping it is what buys
four orders of magnitude in speed, at a cost the [validation](validation.md)
quantifies.
