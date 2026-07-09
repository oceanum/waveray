# Limitations

waveray is a fast surrogate for the last nested domain of a spectral wave
model. These are the things it deliberately does not do. Read this before
trusting it at a new site.

## No diffraction

Rays block; real waves leak. Energy does not bend into geometric shadows.

- **Harbour interiors and lee-of-breakwater**: accuracy degrades. If you need
  wave heights inside a breakwatered basin, you need a phase-resolving or
  diffraction-capable model (SWASH, XBeach, or a Boussinesq model).
- **Island shadows** come out sharper than reality. For an island with a
  shoaling rim, refraction *around* the rim is captured and is usually the
  larger of the two lee-filling mechanisms; for a steep-sided island the
  modelled lee will be too quiet.

## Sheltering is binary, and only as good as the bathymetry

A ray either grounds or it does not. There is no partial transmission over a
reef that SWAN would let some energy across. And a feature smaller than about
two grid cells is smoothed away by the bilinear depth sampling and will not
shelter at all — measured: a 1500 m island blocks completely on a 500 m grid,
but a 300 m islet on a 1 km grid leaves ~89% of the energy through. GEBCO at
~450 m **cannot** shelter behind a small reef or islet.

## Breaking is an endpoint cap

Depth-limited breaking is applied at the target as a proportional cap on `Hm0`,
not as dissipation accumulated along the approach. A wave train that broke on
an offshore bar and re-formed in the trough behind it will be over-predicted:
waveray sees only the depth at the destination.

This is appropriate at berths, buoy sites, and generally outside the inner surf
zone. It is not a surf-zone model. Tune `gamma` per site against observations.

## Stationary and linear

No temporal nonlinear evolution: no quadruplet or triad interactions, no wind
input, no whitecapping along the path. Frequencies do not exchange energy — the
operator is frequency-diagonal. Over a last-mile domain this is a good
approximation (see [Concepts](concepts.md#why-stationary-is-enough)); over a
fetch it is not.

## Fixed water level

The operator is built at the water level implied by your bathymetry. Tide
modulates only the breaking cap, not the refraction. Where the tidal range is
large compared with the depth — a drying estuary, a shallow tidal flat —
build one operator per tide stage and interpolate.

## No currents

The ray equations used here have no current term. Strong tidal jets or river
plumes refract waves, and that is not modelled.

## Your boundary spectra bound your skill

The nested-model literature is consistent on this: nearshore error in sheltered
regions is dominated by the **directional accuracy of the offshore boundary
spectra**, not by nearshore resolution. A boundary spectrum that has the right
total `Hs` but the wrong directional distribution will produce a confidently
wrong nearshore answer, because refraction sorts energy by direction. Check
your boundary source before blaming the transformation.

## Caustics

Where rays converge (behind a focusing shoal), linear ray theory predicts
infinite energy density. waveray fires `nsub` sub-rays per direction bin and
averages them, which smooths caustics rather than resolving them. Increase
`nsub` where focusing matters; treat very large local amplification with
suspicion.

## When to use something else

| Situation | Use |
|---|---|
| Inside a harbour, behind a breakwater | SWASH, XBeach, Boussinesq |
| Surf-zone morphodynamics | XBeach |
| Wave–current interaction matters | SWAN or SCHISM+WWM with currents |
| Generation over a fetch | SWAN, WAVEWATCH III |
| A full nearshore field, not points | SWAN — or see the gridded-output design note in `docs/design-v0.2-auto-boundary-and-grids.md` |
| Last-mile transformation to points, fast, many timesteps | **waveray** |
