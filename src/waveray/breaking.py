"""Depth-limited breaking applied at the target point.

The linear transfer operator cannot contain breaking (dissipation depends on
the total transformed energy), so it is applied as a nonlinear post-step: if
the transformed Hm0 exceeds a depth-limited maximum, the whole spectrum is
scaled down proportionally. Proportional scaling matches how SWAN distributes
Battjes-Janssen surf breaking over the spectrum (dissipation proportional to
E(f, theta)), so this is consistent with the parent-model physics at the
endpoint, though it neglects dissipation history along the approach.

Two cap formulations:

- ``gamma``: Hm0_max = gamma * d  (classic shallow-water saturation)
- ``miche``: Hm0_max = (0.88 / k_m) * tanh(gamma * k_m * d / 0.88), the
  Miche-type limit used by Battjes-Janssen (1978); k_m is the wavenumber of
  the energy-weighted mean frequency. Includes steepness limiting in deeper
  water and reduces to gamma * d in the shallow limit.

The default gamma = 0.73 follows the SWAN default. Tune per site.
"""

from __future__ import annotations

import numpy as np

from .dispersion import wavenumber


def spectral_moment(
    efth: np.ndarray, freqs: np.ndarray, dirs: np.ndarray, n: int = 0
) -> np.ndarray:
    """n-th frequency moment of efth(..., nf, ndir) [density per Hz per deg]."""
    efth = np.asarray(efth, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    ddir = 360.0 / dirs.size  # uniform bins enforced upstream
    ef = efth.sum(axis=-1) * ddir  # (..., nf)
    return np.trapezoid(ef * freqs**n, freqs, axis=-1)


def hm0(efth: np.ndarray, freqs: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Significant wave height Hm0 = 4 sqrt(m0) of efth(..., nf, ndir)."""
    return 4.0 * np.sqrt(spectral_moment(efth, freqs, dirs, n=0))


def hm0_max(
    depth_total: np.ndarray,
    fm: np.ndarray | None = None,
    gamma: float = 0.73,
    method: str = "miche",
) -> np.ndarray:
    """Depth-limited maximum Hm0 [m] for total water depth ``depth_total``."""
    depth_total = np.maximum(np.asarray(depth_total, dtype=float), 1e-3)
    if method == "gamma":
        return gamma * depth_total
    if method == "miche":
        if fm is None:
            raise ValueError("miche cap needs the mean frequency fm")
        k = wavenumber(2.0 * np.pi * np.asarray(fm, dtype=float), depth_total)
        return (0.88 / k) * np.tanh(gamma * k * depth_total / 0.88)
    raise ValueError(f"unknown breaking method {method!r}")


def apply_breaking(
    efth: np.ndarray,
    freqs: np.ndarray,
    dirs: np.ndarray,
    depth: float,
    tide: np.ndarray | float = 0.0,
    gamma: float = 0.73,
    method: str = "miche",
) -> tuple[np.ndarray, np.ndarray]:
    """Cap efth(..., nf, ndir) at the depth-limited Hm0.

    ``tide`` is a water-level offset [m] added to ``depth``; scalar or an
    array broadcastable over the leading dimensions of efth.

    Returns (efth_capped, scale) where scale (<= 1) is the energy scale
    factor applied at each leading index.
    """
    efth = np.asarray(efth, dtype=float)
    m0 = spectral_moment(efth, freqs, dirs, n=0)
    m1 = spectral_moment(efth, freqs, dirs, n=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fm = np.where(m0 > 0, m1 / np.maximum(m0, 1e-300), np.nan)
    h = 4.0 * np.sqrt(m0)

    d_tot = np.asarray(depth, dtype=float) + np.asarray(tide, dtype=float)
    hmax = hm0_max(np.broadcast_to(d_tot, h.shape), fm=fm, gamma=gamma, method=method)

    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where((h > 0) & (h > hmax), (hmax / np.maximum(h, 1e-12)) ** 2, 1.0)
    scale = np.where(np.isfinite(scale), scale, 1.0)
    return efth * scale[..., None, None], scale
