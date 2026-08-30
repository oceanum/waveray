"""Wind-sea saturation applied at the target point.

Wind input is a source term with no sink: the operator carries the Komen
growth but not the whitecapping and quadruplet interactions that balance it
in a spectral model, so the wind contribution grows without limit. The damage
is concentrated in the high-frequency tail, where the group velocity is small
and a ray therefore spends a long time under the wind — at 0.4 Hz over a
15 km fetch the gain ``exp(B L / cg)`` reaches ~1e6, and it is the *boundary*
spectrum's tail being amplified, not the locally seeded sea, that dominates
the error.

Rather than model the missing sinks, this module bounds their outcome: the
energy the wind *adds* cannot exceed what the same wind would raise over the
fetch available, which is the fetch-limited (JONSWAP) spectrum.

Crucially the cap is applied to the wind **increment**

    increment = (T_wind - T_nowind) . E_b  +  E0

and never to the total. Propagated swell is therefore exempt by construction,
and a calm wind reduces exactly to the no-wind answer. An earlier version
capped the total against a Pierson-Moskowitz reference and destroyed swell:
both spectral references roll off as ``exp(-1.25 (f/f_p)^-4)`` below the
wind-sea peak, so any swell whose peak sits below it was clipped to nothing —
a 14 s swell under an 8 m/s breeze lost 71 % of its height.

``SATURATION_K`` scales the fetch-limited reference. It is an empirical
constant fitted against stationary SWAN 41.51A with its full physics, **not**
one of SWAN's source terms, and it does not hold equally at all wind speeds:
see the calibration table in the wind forcing guide before relying on it.
"""

from __future__ import annotations

import numpy as np

from .breaking import dir_resolution
from .dispersion import GRAV

#: Phillips constant of the fully developed spectrum.
ALPHA_PM = 0.0081

#: Multiplier on the fetch-limited spectrum. Fitted against full-physics
#: SWAN at 8, 12 and 18 m/s: k=4 minimises the worst case across both wind
#: regimes (27.5 %, against 40.7 % at k=2 and 46.5 % at k=9). See the wind
#: forcing guide for the per-regime envelope — a single constant cannot
#: serve all wind speeds, because the correction needed grows ~3x from
#: 8 to 18 m/s.
SATURATION_K = 4.0


def pm_spectrum(freqs: np.ndarray, u_star: float) -> np.ndarray:
    """Fully developed Pierson-Moskowitz ``E(f)`` [m^2/Hz] for friction velocity.

    The peak frequency uses the same reference as SWAN's linear-growth filter,
    ``f_p = 0.13 g / (28 u*)``, so the cap is anchored to a quantity the
    package already computes.
    """
    freqs = np.asarray(freqs, dtype=float)
    u_star = max(float(u_star), 1e-6)
    fp = 0.13 * GRAV / (28.0 * u_star)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        shape = np.exp(-1.25 * (freqs / fp) ** -4)
        return ALPHA_PM * GRAV**2 * (2.0 * np.pi) ** -4 * freqs**-5.0 * shape


def jonswap_fetch_limited(freqs: np.ndarray, u10: float, fetch: float) -> np.ndarray:
    """Fetch-limited JONSWAP ``E(f)`` [m^2/Hz] (Hasselmann et al. 1973)."""
    freqs = np.asarray(freqs, dtype=float)
    u10 = max(float(u10), 1e-6)
    xhat = max(GRAV * float(fetch) / u10**2, 1e-6)
    fp = 3.5 * (GRAV / u10) * xhat**-0.33
    alpha = 0.076 * xhat**-0.22
    sigma = np.where(freqs <= fp, 0.07, 0.09)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        peak = np.exp(-((freqs - fp) ** 2) / (2.0 * sigma**2 * fp**2))
        shape = np.exp(-1.25 * (freqs / fp) ** -4)
        return alpha * GRAV**2 * (2.0 * np.pi) ** -4 * freqs**-5.0 * shape * 3.3**peak


def saturation_cap(
    freqs: np.ndarray,
    u10: float,
    fetch: float,
    k: float = SATURATION_K,
) -> np.ndarray:
    """Per-frequency ceiling [m^2/Hz] on the energy the wind may add.

    The fetch-limited spectrum alone: it already tends to the fully developed
    form at long fetch, so it self-bounds. (A Pierson-Moskowitz branch was
    tried and removed — for every fetch this code can produce, ``k`` times the
    fetch-limited spectrum exceeds PM at and above its peak, so PM only ever
    governed *below* the peak, which is where swell lives.)
    """
    return k * jonswap_fetch_limited(freqs, u10, fetch)


def apply_saturation(
    base: np.ndarray,
    increment: np.ndarray,
    freqs: np.ndarray,
    dirs: np.ndarray,
    u10: float,
    fetch: float,
    k: float = SATURATION_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap the wind-added part of a transformed spectrum.

    ``base`` is the spectrum the same rays give with the wind term omitted,
    and ``increment`` is what the wind adds on top; the result is
    ``base + increment * scale``. Taking the two separately rather than the
    total is deliberate: when a runaway makes the increment dwarf the base —
    the case the cap exists for — recovering the base by subtraction cancels
    to noise and can even produce negative energies.

    Only the increment is tested against the ceiling, so whatever the wind did
    not add passes through untouched and swell is exempt by construction.

    Note the ceiling falls away below the wind-sea peak, where the
    fetch-limited spectrum is negligible. Wind amplification of a swell longer
    than the local wind sea is therefore suppressed rather than bounded. That
    is conservative — the result can never fall below ``base`` — and it only
    bites for a wind strong enough to force a swell (28 u* > c), which is
    outside the regime this closure was fitted for.

    Returns ``(capped, scale)`` with ``scale`` (<= 1) the per-frequency factor
    applied to the increment.
    """
    base = np.asarray(base, dtype=float)
    increment = np.asarray(increment, dtype=float)
    cap = saturation_cap(freqs, u10, fetch, k=k)
    ef = increment.sum(axis=-1) * dir_resolution(dirs)  # (..., nf) added by the wind
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(ef > cap, cap / np.maximum(ef, 1e-300), 1.0)
    scale = np.where(np.isfinite(scale), scale, 1.0)
    return base + increment * scale[..., None], scale


def upwind_fetch(
    grid,
    x: float,
    y: float,
    wind_dir: float,
    d_min: float = 0.3,
    step: float | None = None,
    max_fetch: float = 500_000.0,
) -> float:
    """Over-water distance upwind of a point [m].

    Marches from ``(x, y)`` into the wind (``wind_dir`` is coming-from
    nautical degrees) until the depth falls below ``d_min`` or the domain
    edge is reached. This is the fetch available to generate a local wind
    sea; it is bounded by the domain, so a target in a large open domain
    reports the domain crossing rather than a true geographic fetch.
    """
    theta = np.deg2rad((270.0 - float(wind_dir)) % 360.0)  # going-to, math
    dx, dy = -np.cos(theta), -np.sin(theta)  # upwind
    if step is None:
        step = min(grid.spacing) / 2.0
    xmin, xmax, ymin, ymax = grid.bounds
    px, py, travelled = float(x), float(y), 0.0
    while travelled < max_fetch:
        px, py = px + dx * step, py + dy * step
        if not (xmin <= px <= xmax and ymin <= py <= ymax):
            break
        if grid.sample_depth(np.array([px]), np.array([py]))[0] < d_min:
            break
        travelled += step
    return max(travelled, step)
