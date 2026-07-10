# waveray documentation

Fast last-stage nearshore spectral wave transformation.

waveray turns offshore directional wave spectra — from a SWAN or WAVEWATCH III
hindcast, or any wavespectra-readable source — into spectra at a nearshore
point, by precomputing a **linear transfer operator** with backward ray
tracing over local bathymetry. Build once per site (seconds), then transform
decades of hourly spectra at ~10,000 spectra/second. No wave model runs in the
loop.

## Where to start

| Guide | Read it for |
|---|---|
| [Installation](installation.md) | Install, extras, environment |
| [Quickstart](quickstart.md) | A working example in 20 lines |
| [Concepts](concepts.md) | What the operator is, and the physics it contains |
| [User guide](usage.md) | Bathymetry, boundary points, tide, breaking, persistence, ray export |
| [wavespectra interoperability](wavespectra.md) | Input/output conventions and the `.spec` accessor |
| [Validation](validation.md) | Measured skill against parent SWAN models |
| [Limitations](limitations.md) | What waveray does **not** model, and when not to use it |
| [API reference](api.md) | Every public function and its arguments |

Two executed notebooks with plots live in [`notebooks/`](https://github.com/oceanum/waveray/tree/main/notebooks):
an end-to-end Dutch-coast downscaling, and a reef-coast validation study.

## The idea in one paragraph

For a stationary, linear wave field the directional spectral density obeys a
conservation law along a wave ray: `E(f, θ) · c · cg` is constant. So if you
trace rays *backward* from a target point across the bathymetry, each one tells
you where on the domain boundary its energy came from, from which direction,
and by what factor it was amplified. Do that for every (frequency, direction)
bin and you have a matrix `T` mapping boundary spectra to the target spectrum.
The matrix is fixed for a site, so applying it to a whole hindcast is one
`einsum`. Refraction, shoaling, island sheltering and bottom friction all live
inside `T`; only depth-limited breaking, which is nonlinear in energy, is
applied afterwards as a cap.

## Scope

waveray is built for the **last mile**: from an offshore spectral boundary
(typically the 20–100 m contour, where a regional model's output points sit)
to a nearshore target such as a port berth, a buoy site, or a monitoring
station. It is not a replacement for SWAN — it is a fast surrogate for SWAN's
final nested domain, validated against SWAN itself.

Licensed under Apache-2.0. Source: <https://github.com/oceanum/waveray>.
