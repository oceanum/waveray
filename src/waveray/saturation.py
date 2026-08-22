"""Wind-sea saturation applied at the target point.

Wind input is a source term with no sink: the operator carries the Komen
growth but not the whitecapping and quadruplet interactions that balance it
in a spectral model, so the transformed spectrum grows without limit. The
damage is concentrated in the high-frequency tail, where the group velocity
is small and a ray therefore spends a long time under the wind — at 0.4 Hz
over a 15 km fetch the gain ``exp(B L / cg)`` reaches ~1e6, and it is the
*boundary* spectrum's tail being amplified, not the locally seeded sea, that
dominates the error.

Rather than model the missing sinks, this module imposes their outcome. A
wind sea cannot exceed

- the **fully developed** (Pierson-Moskowitz) spectrum for its own wind, which
  bounds the tail and sits well above any swell peak, nor
- a multiple of the **fetch-limited** (JONSWAP) spectrum for the fetch
  actually available, which is what permits the peak overshoot of a young,
  locally generated sea — a plain PM cap clips that and costs a factor of two.

The cap is the larger of the two, applied to the directionally-integrated
E(f) with each frequency's directional distribution scaled proportionally.
This is the same nonlinear post-step pattern as the depth-limited breaking
cap, and it is inert on a spectrum that carries no wind sea.

``SATURATION_K = 3.0`` was calibrated against stationary SWAN 41.51A with its
full physics (see ``tests/test_validation_swan.py``); it is the value that
minimises the worst-case error across both wind regimes. This is an
empirical closure, not one of SWAN's source terms.
"""

from __future__ import annotations

import numpy as np

from .breaking import dir_resolution
from .dispersion import GRAV

#: Phillips constant of the fully developed spectrum.
ALPHA_PM = 0.0081

#: Multiplier on the fetch-limited spectrum, calibrated against SWAN.
SATURATION_K = 3.0


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
    u_star: float,
    fetch: float,
    k: float = SATURATION_K,
) -> np.ndarray:
    """Per-frequency ceiling on E(f) [m^2/Hz] for a wind sea."""
    return np.maximum(pm_spectrum(freqs, u_star), k * jonswap_fetch_limited(freqs, u10, fetch))


def apply_saturation(
    efth: np.ndarray,
    freqs: np.ndarray,
    dirs: np.ndarray,
    u10: float,
    u_star: float,
    fetch: float,
    k: float = SATURATION_K,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap ``efth(..., nf, ndir)`` at the wind-sea saturation level.

    Returns ``(efth_capped, scale)`` where ``scale`` is the per-frequency
    factor applied (<= 1), broadcast over the leading dimensions.
    """
    efth = np.asarray(efth, dtype=float)
    cap = saturation_cap(freqs, u10, u_star, fetch, k=k)
    ef = efth.sum(axis=-1) * dir_resolution(dirs)  # (..., nf) m^2/Hz
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(ef > cap, cap / np.maximum(ef, 1e-300), 1.0)
    scale = np.where(np.isfinite(scale), scale, 1.0)
    return efth * scale[..., None], scale


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
