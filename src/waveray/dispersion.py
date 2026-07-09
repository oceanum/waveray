"""Linear wave dispersion relations, vectorised.

Angular frequency ``omega = 2*pi*f`` [rad/s], water depth ``d`` [m, positive
down], wavenumber ``k`` [rad/m]. All functions accept and return numpy arrays
of any broadcastable shape.
"""

from __future__ import annotations

import numpy as np

GRAV = 9.81

# Depths below this are treated as this value to keep the solver finite.
_DEPTH_FLOOR = 1e-3


def wavenumber(omega: np.ndarray, depth: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Solve the linear dispersion relation ``omega**2 = g k tanh(k d)`` for k.

    Uses the explicit approximation of Guo (2002) as the initial guess,
    polished with Newton iterations. Accurate to machine precision across
    deep, intermediate and shallow regimes.
    """
    omega = np.asarray(omega, dtype=float)
    depth = np.maximum(np.asarray(depth, dtype=float), _DEPTH_FLOOR)

    x = np.maximum(omega * np.sqrt(depth / GRAV), 1e-8)
    kd = x**2 * (1.0 - np.exp(-(x**2.5))) ** (-0.4)
    k = kd / depth

    for _ in range(50):
        kd = k * depth
        t = np.tanh(kd)
        f = GRAV * k * t - omega**2
        dfdk = GRAV * (t + kd * (1.0 - t**2))
        dk = f / dfdk
        k = k - dk
        if np.all(np.abs(dk) <= tol * np.maximum(k, 1e-12)):
            break
    return k


def phase_speed(omega: np.ndarray, depth: np.ndarray, k: np.ndarray | None = None) -> np.ndarray:
    """Phase speed c = omega / k [m/s]."""
    if k is None:
        k = wavenumber(omega, depth)
    return omega / k


def group_speed(omega: np.ndarray, depth: np.ndarray, k: np.ndarray | None = None) -> np.ndarray:
    """Group speed cg = n * c with n = 1/2 (1 + 2kd / sinh 2kd) [m/s]."""
    if k is None:
        k = wavenumber(omega, depth)
    depth = np.maximum(np.asarray(depth, dtype=float), _DEPTH_FLOOR)
    kd = np.minimum(k * depth, 25.0)  # n -> 1/2 beyond this; avoids sinh overflow
    n = 0.5 * (1.0 + 2.0 * kd / np.sinh(2.0 * kd))
    return n * omega / k


def ccg(omega: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """The ray invariant factor c * cg.

    For a stationary linear wave field without currents, the directional
    spectral density transforms along a ray as ``E(f, theta) * c * cg =
    const`` (conservation of wave action in phase space).
    """
    k = wavenumber(omega, depth)
    return phase_speed(omega, depth, k=k) * group_speed(omega, depth, k=k)
