"""Backward ray tracing over a local bathymetry grid.

For a stationary, linear wave field without currents, rays for a given
absolute frequency follow (arclength s, propagation direction theta measured
counter-clockwise from +x in radians):

    dx/ds     =  cos(theta)
    dy/ds     =  sin(theta)
    dtheta/ds =  (sin(theta) * dc/dx - cos(theta) * dc/dy) / c

with c(x, y) the phase speed at that frequency. Rays are reversible, so the
path from a target point back to the domain boundary is found by integrating
the same equations with a negated right-hand side ("backward").

Everything is vectorised: all rays of one frequency advance in lockstep with
an active mask (RK4). This keeps a full operator build in pure numpy at a
few seconds per site without a numba/compilation dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bathymetry import LocalGrid, bilinear
from .dispersion import GRAV, group_speed, phase_speed, wavenumber

STATUS_EXITED = 0  # left the domain through the boundary -> picks up boundary energy
STATUS_LANDED = 1  # ran aground (depth below threshold) -> blocked, zero energy
STATUS_LOST = 2  # still inside after max_steps -> treated as blocked


@dataclass
class SpeedField:
    """Phase speed, its gradient, and optional friction decay rate for one
    frequency on a LocalGrid."""

    grid: LocalGrid
    c: np.ndarray
    dcdx: np.ndarray
    dcdy: np.ndarray
    fric: np.ndarray | None = None  # spatial decay rate [1/m] of E*c*cg

    @classmethod
    def build(
        cls, grid: LocalGrid, omega: float, d_min: float, cf_jonswap: float | None = None
    ) -> SpeedField:
        # Land / very shallow nodes get the floor depth so that c decreases
        # smoothly toward shore and rays are stopped by the depth check.
        depth = np.maximum(grid._depth_filled, d_min)
        k = wavenumber(omega, depth)
        c = phase_speed(omega, depth, k=k)
        dcdy, dcdx = np.gradient(c, grid.y, grid.x)
        fric = None
        if cf_jonswap is not None:
            # JONSWAP bottom friction S_bf = -C_b sigma^2 / (g^2 sinh^2 kd) E
            # is linear in E, so along a ray the invariant decays as
            # exp(-integral of C_b sigma^2 / (g^2 sinh^2(kd) cg) ds).
            cg = group_speed(omega, depth, k=k)
            kd = np.minimum(k * depth, 25.0)
            fric = cf_jonswap * omega**2 / (GRAV**2 * np.sinh(kd) ** 2 * cg)
        return cls(grid=grid, c=c, dcdx=dcdx, dcdy=dcdy, fric=fric)

    def sample(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        g = self.grid
        return (
            bilinear(self.c, g.x, g.y, x, y),
            bilinear(self.dcdx, g.x, g.y, x, y),
            bilinear(self.dcdy, g.x, g.y, x, y),
        )


@dataclass
class RayFan:
    """Exit state of a batch of backward-traced rays."""

    status: np.ndarray  # (n,) int8, STATUS_* above
    x: np.ndarray  # (n,) exit / stop x [m]
    y: np.ndarray  # (n,) exit / stop y [m]
    theta: np.ndarray  # (n,) propagation direction at exit [rad, math convention]
    atten: np.ndarray  # (n,) path-integrated friction decay exponent (>= 0)
    paths: list[np.ndarray] | None = None  # per-ray (m_i, 2) local-metre polylines


def _rhs(
    field: SpeedField, x: np.ndarray, y: np.ndarray, th: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward ray RHS (negated forward equations)."""
    c, cx, cy = field.sample(x, y)
    return -np.cos(th), -np.sin(th), -(np.sin(th) * cx - np.cos(th) * cy) / c


def trace_backward(
    field: SpeedField,
    x0: float,
    y0: float,
    theta0: np.ndarray,
    ds: float,
    max_steps: int,
    d_min: float = 0.3,
    record_paths: bool = False,
) -> RayFan:
    """Trace rays backward from (x0, y0) with propagation directions theta0.

    Rays stop when they leave the grid bounds (STATUS_EXITED, exit point
    clipped to the boundary), when the local depth falls below ``d_min``
    (STATUS_LANDED), or after ``max_steps`` (STATUS_LOST).

    With ``record_paths=True`` the full trajectory of every ray is recorded
    and returned in ``RayFan.paths`` (list of (m_i, 2) local-metre arrays,
    ordered from the start point outward).
    """
    grid = field.grid
    xmin, xmax, ymin, ymax = grid.bounds

    theta0 = np.atleast_1d(np.asarray(theta0, dtype=float))
    n = theta0.size
    x = np.full(n, float(x0))
    y = np.full(n, float(y0))
    th = theta0.copy()
    atten = np.zeros(n)
    status = np.full(n, STATUS_LOST, dtype=np.int8)
    active = np.ones(n, dtype=bool)
    history: list[np.ndarray] | None = None
    stop_step = np.full(n, 0, dtype=np.int64)
    if record_paths:
        history = [np.column_stack([x, y])]

    if not (xmin <= x0 <= xmax and ymin <= y0 <= ymax):
        raise ValueError(f"start point ({x0}, {y0}) outside grid bounds {grid.bounds}")
    if grid.sample_depth(np.array([x0]), np.array([y0]))[0] < d_min:
        raise ValueError(f"start point ({x0}, {y0}) is dry (depth < {d_min} m)")

    for _ in range(max_steps):
        if not active.any():
            break
        xa, ya, tha = x[active], y[active], th[active]

        # RK4 step of length ds
        k1x, k1y, k1t = _rhs(field, xa, ya, tha)
        k2x, k2y, k2t = _rhs(field, xa + 0.5 * ds * k1x, ya + 0.5 * ds * k1y, tha + 0.5 * ds * k1t)
        k3x, k3y, k3t = _rhs(field, xa + 0.5 * ds * k2x, ya + 0.5 * ds * k2y, tha + 0.5 * ds * k2t)
        k4x, k4y, k4t = _rhs(field, xa + ds * k3x, ya + ds * k3y, tha + ds * k3t)
        xn = xa + ds / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        yn = ya + ds / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
        tn = tha + ds / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)

        # Exits: clip the segment (xa, ya) -> (xn, yn) to the grid bbox.
        out = (xn < xmin) | (xn > xmax) | (yn < ymin) | (yn > ymax)
        if out.any():
            t = np.ones(out.sum())
            xo, yo = xa[out], ya[out]
            dx_, dy_ = xn[out] - xo, yn[out] - yo
            for lim, p, dp, side in (
                (xmin, xo, dx_, -1.0),
                (xmax, xo, dx_, 1.0),
                (ymin, yo, dy_, -1.0),
                (ymax, yo, dy_, 1.0),
            ):
                cross = (dp * side) > 0
                tt = np.where(cross, (lim - p) / np.where(dp == 0, np.inf, dp), np.inf)
                t = np.minimum(t, np.clip(tt, 0.0, 1.0))
            xn[out] = xo + t * dx_
            yn[out] = yo + t * dy_
            tn[out] = tha[out] + t * (tn[out] - tha[out])

        if field.fric is not None:
            # accumulate friction decay at the segment midpoint; step length
            # equals ds except for the clipped exit segments
            step = np.full(xa.size, ds)
            if out.any():
                step[out] = np.hypot(xn[out] - xa[out], yn[out] - ya[out])
            rate = bilinear(field.fric, grid.x, grid.y, 0.5 * (xa + xn), 0.5 * (ya + yn))
            atten[active] += rate * step

        x[active], y[active], th[active] = xn, yn, tn

        # Landed: local depth below threshold.
        depth_n = grid.sample_depth(xn, yn)
        landed = depth_n < d_min

        idx = np.flatnonzero(active)
        status[idx[out]] = STATUS_EXITED
        status[idx[landed & ~out]] = STATUS_LANDED
        active[idx[out | landed]] = False
        if history is not None:
            history.append(np.column_stack([x, y]))
            stop_step[idx[out | landed]] = len(history) - 1
            stop_step[active] = len(history) - 1

    paths = None
    if history is not None:
        traj = np.stack(history)  # (nsteps+1, n, 2)
        paths = [traj[: stop_step[i] + 1, i, :] for i in range(n)]
    return RayFan(status=status, x=x, y=y, theta=th, atten=atten, paths=paths)
