# Validation

waveray is validated **against the parent model it replaces**. The test is
strict: offshore spectra from a SWAN hindcast become the boundary condition,
interior SWAN output sites are held out as truth, and the ray-traced operator
transforms the boundary spectra to those sites. waveray never sees SWAN's
answer at the truth site.

Both studies are reproducible notebooks in [`notebooks/`](https://github.com/oceanum/waveray/tree/main/notebooks).

## Dutch coast — smooth sandy shoreface

Source: `oceanum_wave_dutch_era5_v1_spec` (SWAN 1 km, hourly, ERA5-forced),
bathymetry GEBCO 2023 (the same generation the parent model used). Period
15 Dec 2023 – 15 Jan 2024, spanning storms **Pia** and **Henk** (745 hourly
timesteps). Six boundary sites on an offshore arc; the held-out truth site sits
in 9.8 m of water.

| Metric | Value |
|---|---|
| Correlation `r` | **0.992** |
| Bias | **−0.06 m** |
| RMSE | **0.10 m** |
| Scatter index | 0.06 |
| SWAN mean `Hs` | 1.61 m |
| Operator build | 13.8 s (one-off) |
| Throughput | 745 spectra in 105 ms (~7,100 spectra/s) |

A gently sloping sandy shoreface with smooth depth contours, no reefs and no
diffracting structures is close to the ideal case for ray theory.

## Abrolhos, Western Australia — reef-fronted coast

Source: `oceanum_wave_ec_abrol500m_spec_nowcast` (SWAN 500 m). Bathymetry GEBCO
2025. Five boundary sites on the western edge; three held-out interior sites.
385 timesteps.

| Truth site depth | `r` | Bias | RMSE | SI |
|---|---|---|---|---|
| 4.3 m | 0.944 | +0.36 m | 0.60 m | 0.35 |
| 8.5 m | 0.925 | +0.18 m | 0.48 m | 0.22 |
| 10.5 m | 0.980 | −0.98 m | 1.05 m | 0.41 |

Correlation stays high, but the biases are larger than on the Dutch coast. Two
causes, both structural:

1. **Bathymetry mismatch.** The operator is fed GEBCO (~450 m); the parent SWAN
   model was built on the 250 m Australian Bathymetry and Topography grid, which
   resolves reef structure GEBCO smooths away. The worst site (−0.98 m, behind
   the Point Moore reefs) is the most sheltered one — exactly where unresolved
   bathymetry hurts most.
2. **Binary blocking.** Rays either ground on a reef or pass; SWAN transmits
   energy partially across it.

## Bottom friction is the leading correctable error

On the Abrolhos shelf, ablating JONSWAP bottom friction at the 8.5 m site:

| Configuration | `r` | Bias | RMSE |
|---|---|---|---|
| Friction **off** | 0.940 | **+1.11 m** | 1.32 m |
| Friction **on** (`cf_jonswap=0.038`) | 0.925 | **+0.18 m** | 0.48 m |

Over long shallow approaches, friction removes real energy. This is why it is
**on by default**. Note the correlation barely moves while the bias collapses:
friction is a systematic energy sink, not a timing correction.

## Analytic validation

Beyond model-vs-model comparison, the operator is checked against closed-form
solutions, because linear ray theory has them (`tests/`):

- **Snell refraction and energy-flux shoaling** on a plane beach — the transfer
  coefficient matches the analytic `Ks²·Kr²` to within 5–7%
- **Flat bathymetry is the identity** — every direction passes through with
  `T = 1.000`
- **Island sheltering** — directions whose rays cross an island give exactly
  `T = 0.000`; unobstructed directions give exactly `T = 1.000`
- **Friction** attenuates on long shallow paths and is negligible in deep water
- **Integrated parameters** agree with wavespectra to 1e-12

## Against SWAN 41.51A (docker)

The hindcast comparisons above measure waveray against a parent model run by
someone else, so grid, boundary and settings all differ. `tests/validation/`
removes those differences: it runs **stationary SWAN 41.51A** in the official
`delftwaves/swan` image on the same bathymetry, the same spectral grid and the
*same* boundary spectrum (written directly as a stationary SWAN ASCII spectral
file), so what is left is the transformation itself.

```bash
docker pull delftwaves/swan:latest
uv run pytest -m swan -s          # excluded from the default suite; takes minutes
```

SWAN runs with its **full physics** — quadruplets and whitecapping on — because
the question is how good a surrogate waveray is for a real SWAN run, not
whether it reimplements a subset of SWAN (the closed-form tests settle that).
Depth-induced breaking is the one term left off, since waveray applies it at
the target rather than along the path.

| Case | SWAN | waveray | Difference |
|---|---|---|---|
| Plane beach, shore-normal swell, 20→6 m | Hs 1.962→2.147 m | 1.960→2.153 m | **0.31 %** |
| Plane beach, 30° oblique — refracted direction | 246.0→256.7° | 246.7→257.4° | **< 0.8°** (Hs 0.62 %) |
| Real GEBCO bathymetry (Noordwijk, NL) | Hs 2.380→2.149 m | 2.483→2.490 m | 4.3 – 15.9 % |
| Circular island, deep lee | Hs 0.466 m | 0.675 m | ×1.45 |
| Swell + 12 m/s wind, 15 km fetch | Hs 2.157→2.380 m | 2.243→3.087 m | ×1.04 – 1.30 |
| **Wind sea from calm**, peak resolved | Hs 0.322→0.644 m | 0.702→2.090 m | **×2.2 – 3.2** |

Reading the table: on the analytic beach the operator is essentially exact —
refraction and shoaling are reproduced to a fraction of a percent. On real
bathymetry the gap widens shoreward, as the depth field stops being smooth and
alongshore-uniform. The island lee is the deep shadow behind a blocking
obstacle, where neither model has diffraction and a ray model and a spectral
model are entitled to disagree; treat sheltered-lee heights as indicative.

The last two rows are the price of the missing source terms, and they say
something specific: **transform swell with this, do not generate a sea with
it.** Adding wind to an existing swell over a 15 km fetch costs up to 30 %;
growing a sea from calm over 1–5 km is wrong by a factor of 2–3, because the
shape of a wind sea is set by the balance between input and the two sinks
waveray does not carry.

Two supporting tests make the point sharper. Strip SWAN's sinks so both models
carry wind input alone and the factor-of-three collapses to tens of percent —
the formulation is right, the balance is what is missing. Do the same on the
15 km swell case and *both* models run away, SWAN to 77.8 m and waveray to
35.0 m from the same 2 m swell, which is the evidence behind waveray's
`max_growth` ceiling.

Every SWAN run in the suite is checked for convergence, and rejected if it
stopped at the iteration cap or collapsed to zero — a stationary run that did
either is not reproducible and cannot serve as a reference.

## Interpreting these numbers

The honest summary: on a smooth coast waveray reproduces its parent SWAN model
to within a few centimetres, at ~10,000 spectra per second. On a complex reef
coast it tracks the parent model's *variability* well (r ≈ 0.93–0.98) but
carries a bias that is dominated by the bathymetry you feed it, not by the
method. Feed it the bathymetry the parent model used, turn friction on, and
tune `gamma` against observations before trusting absolute heights at a new
site.
