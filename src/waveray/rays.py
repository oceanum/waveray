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

from dataclasses import dataclass, field

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
    seg: np.ndarray | None = None  # (n,) crossed boundary-line segment, -1 if none
    u: np.ndarray | None = None  # (n,) fractional position along the crossed segment


@dataclass
class BoundaryLine:
    """A polyline (open) or polygon (closed ring) through the boundary sites.

    Sites are taken in the supplied order; a backward ray stops at its *first*
    crossing of this geometry (nearest the target). Segment ``j`` joins site
    ``seg_a[j]`` to site ``seg_b[j]``; the crossing's fractional position along
    that segment gives the alongshore interpolation weight between the two
    sites, replacing the bbox-perimeter parameterisation for interior sites.
    """

    x: np.ndarray  # (K,) site x [m]
    y: np.ndarray  # (K,) site y [m]
    closed: bool = False
    seg_a: np.ndarray = field(init=False)  # (M,) start-site index per segment
    seg_b: np.ndarray = field(init=False)  # (M,) end-site index per segment

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.shape != self.y.shape or self.x.ndim != 1:
            raise ValueError("boundary line x and y must be matching 1D arrays")
        k = self.x.size
        if self.closed and k < 3:
            raise ValueError("a closed boundary ring needs at least 3 sites")
        if not self.closed and k < 2:
            raise ValueError("a boundary line needs at least 2 sites")
        a = np.arange(k if self.closed else k - 1)
        self.seg_a = a
        self.seg_b = (a + 1) % k

    @property
    def n_seg(self) -> int:
        return int(self.seg_a.size)


def _first_line_crossing(
    px: np.ndarray, py: np.ndarray, qx: np.ndarray, qy: np.ndarray, line: BoundaryLine
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """First intersection of ray steps (p->q) with the boundary line.

    Vectorised over the A ray steps and M line segments. Returns, per ray:
    ``t`` (parameter along p->q of the nearest crossing, ``inf`` if none),
    ``seg`` (index of the crossed segment, ``-1`` if none) and ``u`` (fractional
    position along that segment, 0 at ``seg_a`` and 1 at ``seg_b``).
    """
    ax, ay = line.x[line.seg_a], line.y[line.seg_a]  # (M,)
    bx, by = line.x[line.seg_b], line.y[line.seg_b]
    rx = (qx - px)[:, None]  # (A, 1) ray-step vector
    ry = (qy - py)[:, None]
    sx = (bx - ax)[None, :]  # (1, M) segment vector
    sy = (by - ay)[None, :]
    denom = rx * sy - ry * sx  # (A, M)
    qpx = ax[None, :] - px[:, None]  # (A, M) segment-start minus ray-start
    qpy = ay[None, :] - py[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qpx * sy - qpy * sx) / denom
        u = (qpx * ry - qpy * rx) / denom
    valid = (denom != 0.0) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)
    t = np.where(valid, t, np.inf)
    seg = np.argmin(t, axis=1)  # (A,) nearest valid crossing
    tmin = np.take_along_axis(t, seg[:, None], axis=1)[:, 0]
    umin = np.take_along_axis(np.where(valid, u, 0.0), seg[:, None], axis=1)[:, 0]
    has = np.isfinite(tmin)
    return tmin, np.where(has, seg, -1), umin


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
    boundary_line: BoundaryLine | None = None,
) -> RayFan:
    """Trace rays backward from (x0, y0) with propagation directions theta0.

    Rays stop when they leave the grid bounds (STATUS_EXITED, exit point
    clipped to the boundary), when the local depth falls below ``d_min``
    (STATUS_LANDED), or after ``max_steps`` (STATUS_LOST).

    With ``boundary_line`` given, a ray also stops (STATUS_EXITED) at its first
    crossing of that polyline/ring — whichever comes first, the line crossing
    or the bbox exit. The crossed segment and fractional position are recorded
    in ``RayFan.seg`` / ``RayFan.u``; rays that reach the bbox without crossing
    the line keep ``seg == -1`` (they "escaped" the line's ends).

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
    seg_out = np.full(n, -1, dtype=np.int64)
    u_out = np.zeros(n)
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
        dx_, dy_, dth_ = xn - xa, yn - ya, tn - tha

        # Stop parameter t in [0, 1] along (xa, ya) -> (xn, yn): the earliest of
        # a grid-bbox exit and (optionally) a boundary-line crossing.
        out = (xn < xmin) | (xn > xmax) | (yn < ymin) | (yn > ymax)
        t_bbox = np.full(xa.size, np.inf)
        if out.any():
            tt = np.ones(xa.size)
            for lim, p, dp, side in (
                (xmin, xa, dx_, -1.0),
                (xmax, xa, dx_, 1.0),
                (ymin, ya, dy_, -1.0),
                (ymax, ya, dy_, 1.0),
            ):
                cross = (dp * side) > 0
                cand = np.where(cross, (lim - p) / np.where(dp == 0, np.inf, dp), np.inf)
                tt = np.minimum(tt, np.clip(cand, 0.0, 1.0))
            t_bbox = np.where(out, tt, np.inf)

        if boundary_line is not None:
            t_line, seg_i, u_i = _first_line_crossing(xa, ya, xn, yn, boundary_line)
        else:
            t_line = np.full(xa.size, np.inf)

        ts = np.minimum(t_bbox, t_line)
        stopped = np.isfinite(ts)
        line_stop = stopped & (t_line <= t_bbox)
        ts = np.where(stopped, ts, 1.0)
        xn = xa + ts * dx_
        yn = ya + ts * dy_
        tn = tha + ts * dth_

        if field.fric is not None:
            # accumulate friction decay at the segment midpoint; step length
            # equals ds except for the clipped stop segments
            step = np.full(xa.size, ds)
            if stopped.any():
                step[stopped] = np.hypot(xn[stopped] - xa[stopped], yn[stopped] - ya[stopped])
            rate = bilinear(field.fric, grid.x, grid.y, 0.5 * (xa + xn), 0.5 * (ya + yn))
            atten[active] += rate * step

        x[active], y[active], th[active] = xn, yn, tn

        # Landed: local depth below threshold.
        depth_n = grid.sample_depth(xn, yn)
        landed = depth_n < d_min

        idx = np.flatnonzero(active)
        status[idx[stopped]] = STATUS_EXITED
        status[idx[landed & ~stopped]] = STATUS_LANDED
        if boundary_line is not None:
            seg_out[idx[line_stop]] = seg_i[line_stop]
            u_out[idx[line_stop]] = u_i[line_stop]
        active[idx[stopped | landed]] = False
        if history is not None:
            history.append(np.column_stack([x, y]))
            stop_step[idx[stopped | landed]] = len(history) - 1
            stop_step[active] = len(history) - 1

    paths = None
    if history is not None:
        traj = np.stack(history)  # (nsteps+1, n, 2)
        paths = [traj[: stop_step[i] + 1, i, :] for i in range(n)]
    seg = seg_out if boundary_line is not None else None
    u = u_out if boundary_line is not None else None
    return RayFan(status=status, x=x, y=y, theta=th, atten=atten, paths=paths, seg=seg, u=u)
