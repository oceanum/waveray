"""Spectral transfer operators built from backward ray fans.

The operator ``T[f, j, k, l]`` maps a stack of K boundary directional spectra
``E_b[k, f, l]`` (wavespectra convention: dir = coming-from, nautical degrees,
density per Hz per degree) to the target spectrum:

    E_t[f, j] = sum_k sum_l T[f, j, k, l] * E_b[k, f, l]

It is frequency-diagonal (linear, stationary: no energy exchange between
frequencies) and encodes, per target (freq, dir) bin:

- refraction: the backward-ray direction mapping theta_target -> theta_exit
- shoaling: the ray invariant E(f, theta) * c * cg = const
- island / headland sheltering: rays that run aground contribute nothing
- alongshore boundary inhomogeneity: each sub-ray interpolates the boundary
  spectra at its own exit point on the domain perimeter

Directions: nautical coming-from degrees at the API; math going-to radians
internally. ``theta_math_deg = (270 - dir_nautical) % 360`` and the inverse
is the same expression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from ._version import __version__ as _pkg_version
from .bathymetry import LocalGrid
from .dispersion import ccg
from .rays import STATUS_EXITED, STATUS_LANDED, STATUS_LOST, SpeedField, trace_backward


def dir_to_theta(dir_nautical_deg: np.ndarray) -> np.ndarray:
    """Coming-from nautical degrees -> going-to math radians."""
    return np.deg2rad((270.0 - np.asarray(dir_nautical_deg, dtype=float)) % 360.0)


def theta_to_dir(theta_rad: np.ndarray) -> np.ndarray:
    """Going-to math radians -> coming-from nautical degrees."""
    return (270.0 - np.rad2deg(np.asarray(theta_rad, dtype=float))) % 360.0


def _perimeter_coord(
    x: np.ndarray, y: np.ndarray, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    """Position along the bbox perimeter, anti-clockwise from the SW corner.

    Points not exactly on the perimeter are assigned to the nearest edge.
    """
    xmin, xmax, ymin, ymax = bounds
    w, h = xmax - xmin, ymax - ymin
    x = np.clip(np.asarray(x, dtype=float), xmin, xmax)
    y = np.clip(np.asarray(y, dtype=float), ymin, ymax)
    d_edges = np.stack([y - ymin, xmax - x, ymax - y, x - xmin])  # S, E, N, W
    edge = np.argmin(d_edges, axis=0)
    p_edges = np.stack(
        [
            x - xmin,  # south, running east
            w + (y - ymin),  # east, running north
            w + h + (xmax - x),  # north, running west
            2 * w + h + (ymax - y),  # west, running south
        ]
    )
    return np.take_along_axis(p_edges, edge[None, ...], axis=0)[0]


def _circular_bracket_weights(
    p: np.ndarray, pk: np.ndarray, period: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linear interpolation weights between circular bracketing nodes.

    ``pk`` must be sorted ascending within [0, period). Returns index of the
    lower and upper bracketing node and their weights, vectorised over p.
    """
    p = np.asarray(p, dtype=float) % period
    k = pk.size
    if k == 1:
        zeros = np.zeros_like(p, dtype=int)
        return zeros, zeros, np.ones_like(p), np.zeros_like(p)
    j = np.searchsorted(pk, p)
    hi = j % k
    lo = (j - 1) % k
    gap = (pk[hi] - pk[lo]) % period
    gap = np.where(gap == 0, period, gap)
    w_hi = ((p - pk[lo]) % period) / gap
    return lo, hi, 1.0 - w_hi, w_hi


@dataclass
class TransferOperator:
    """Precomputed linear transfer from K boundary spectra to one target point."""

    T: np.ndarray  # (nf, ndir_t, K, ndir_b)
    freq: np.ndarray  # (nf,) Hz
    dir_t: np.ndarray  # (ndir_t,) coming-from nautical deg, target bins
    dir_b: np.ndarray  # (ndir_b,) coming-from nautical deg, boundary bins
    bp_x: np.ndarray  # (K,) boundary point x [m]
    bp_y: np.ndarray  # (K,) boundary point y [m]
    target_x: float
    target_y: float
    depth_target: float
    attrs: dict = field(default_factory=dict)

    @property
    def n_boundary(self) -> int:
        return self.bp_x.size

    def apply(self, efth_boundary: np.ndarray) -> np.ndarray:
        """Apply to stacked boundary spectra ``(..., K, nf, ndir_b)``.

        Returns the target spectra ``(..., nf, ndir_t)`` in the same density
        units as the input (the coefficients are density ratios).
        """
        e = np.asarray(efth_boundary, dtype=float)
        if e.shape[-3:] != (self.n_boundary, self.freq.size, self.dir_b.size):
            raise ValueError(
                f"expected trailing dims (K={self.n_boundary}, nf={self.freq.size}, "
                f"ndir_b={self.dir_b.size}), got {e.shape[-3:]}"
            )
        return np.einsum("fjkl,...kfl->...fj", self.T, e)

    # ------------------------------------------------------------------ #
    def to_dataset(self) -> xr.Dataset:
        ds = xr.Dataset(
            {
                "T": (("freq", "dir", "bp", "dir_b"), self.T),
                "bp_x": (("bp",), self.bp_x),
                "bp_y": (("bp",), self.bp_y),
            },
            coords={
                "freq": self.freq,
                "dir": self.dir_t,
                "dir_b": self.dir_b,
                "bp": np.arange(self.n_boundary),
            },
            attrs={
                "target_x": self.target_x,
                "target_y": self.target_y,
                "depth_target": self.depth_target,
                "package": f"waveray {_pkg_version}",
                **self.attrs,
            },
        )
        return ds

    def to_netcdf(self, path: str) -> None:
        self.to_dataset().to_netcdf(path)

    @classmethod
    def from_dataset(cls, ds: xr.Dataset) -> TransferOperator:
        attrs = dict(ds.attrs)
        target_x = float(attrs.pop("target_x"))
        target_y = float(attrs.pop("target_y"))
        depth_target = float(attrs.pop("depth_target"))
        return cls(
            T=ds["T"].values,
            freq=ds["freq"].values,
            dir_t=ds["dir"].values,
            dir_b=ds["dir_b"].values,
            bp_x=ds["bp_x"].values,
            bp_y=ds["bp_y"].values,
            target_x=target_x,
            target_y=target_y,
            depth_target=depth_target,
            attrs=attrs,
        )

    @classmethod
    def from_netcdf(cls, path: str) -> TransferOperator:
        with xr.open_dataset(path) as ds:
            return cls.from_dataset(ds.load())


def build_operator(
    grid: LocalGrid,
    target_xy: tuple[float, float],
    boundary_xy: np.ndarray,
    freqs: np.ndarray,
    dirs: np.ndarray,
    nsub: int = 7,
    ds: float | None = None,
    max_steps: int | None = None,
    d_min: float = 0.3,
    cf_jonswap: float | None = 0.038,
) -> TransferOperator:
    """Build a transfer operator by backward ray tracing.

    Parameters
    ----------
    grid : local bathymetry grid.
    target_xy : target point (x, y) in grid metres.
    boundary_xy : (K, 2) boundary spectra points in grid metres; they are
        projected onto the domain perimeter for alongshore interpolation.
    freqs : (nf,) frequencies [Hz] of the boundary spectra.
    dirs : (ndir,) direction bins [coming-from nautical deg] of the boundary
        spectra; also used as the target bins. Must be uniformly spaced.
    nsub : sub-rays per direction bin (averaged; smooths caustics).
    ds : ray integration step [m]; default min(grid spacing) / 3.
    max_steps : default enough to traverse ~1.5 perimeters.
    d_min : stopping depth [m]; rays shallower than this are blocked.
    cf_jonswap : JONSWAP bottom friction coefficient [m^2 s^-3] integrated
        along each ray path (0.038 = SWAN swell default); None disables
        friction (pure refraction + shoaling).
    """
    freqs = np.asarray(freqs, dtype=float)
    dirs = np.asarray(dirs, dtype=float)
    boundary_xy = np.atleast_2d(np.asarray(boundary_xy, dtype=float))
    if boundary_xy.shape[1] != 2:
        raise ValueError("boundary_xy must be (K, 2)")

    ddir = np.diff(np.sort(dirs % 360.0))
    widths = np.r_[ddir, 360.0 - ddir.sum()]
    if not np.allclose(widths, widths[0], atol=1e-6):
        raise ValueError("direction bins must be uniformly spaced")
    bin_width = float(widths[0])

    dx, dy = grid.spacing
    if ds is None:
        ds = min(dx, dy) / 3.0
    xmin, xmax, ymin, ymax = grid.bounds
    perimeter = 2.0 * ((xmax - xmin) + (ymax - ymin))
    if max_steps is None:
        max_steps = int(np.ceil(1.5 * perimeter / ds))

    tx, ty = float(target_xy[0]), float(target_xy[1])
    depth_t = float(grid.sample_depth(np.array([tx]), np.array([ty]))[0])
    if depth_t < d_min:
        raise ValueError(f"target depth {depth_t:.2f} m is below d_min={d_min} m")

    # Boundary point perimeter parameterisation, sorted.
    p_bp = _perimeter_coord(boundary_xy[:, 0], boundary_xy[:, 1], grid.bounds)
    order = np.argsort(p_bp)
    p_bp_sorted = p_bp[order]

    # Boundary-direction bins sorted for circular interpolation.
    db_order = np.argsort(dirs % 360.0)
    dirs_b_sorted = (dirs % 360.0)[db_order]

    nf, ndt, ndb, kk = freqs.size, dirs.size, dirs.size, boundary_xy.shape[0]
    t_op = np.zeros((nf, ndt, kk, ndb))
    n_rays = n_lost = n_landed = 0

    # Sub-ray direction offsets across each target bin (bin-centre sampling).
    offsets = (np.arange(nsub) + 0.5) / nsub - 0.5  # in units of bin width
    theta_t = dir_to_theta(dirs)  # (ndt,) going-to rad of bin centres
    # going-to angle decreases as coming-from degree increases; offsets are
    # symmetric so the sign does not matter.
    theta0 = (theta_t[:, None] + np.deg2rad(offsets * bin_width)[None, :]).ravel()

    for i, f in enumerate(freqs):
        omega = 2.0 * np.pi * f
        fld = SpeedField.build(grid, omega, d_min=d_min, cf_jonswap=cf_jonswap)
        ccg_t = float(ccg(np.array(omega), np.array(depth_t)))

        fan = trace_backward(fld, tx, ty, theta0, ds=ds, max_steps=max_steps, d_min=d_min)

        ok = fan.status == STATUS_EXITED
        n_rays += fan.status.size
        n_lost += int(np.sum(fan.status == STATUS_LOST))
        n_landed += int(np.sum(fan.status == STATUS_LANDED))
        if not ok.any():
            continue
        jbin = np.repeat(np.arange(ndt), nsub)[ok]
        depth_exit = grid.sample_depth(fan.x[ok], fan.y[ok])
        # The ccg ratio is the pointwise density invariant; composed with the
        # direction-bin interpolation below it yields the energy-flux
        # directional transform (the interpolation column-sum supplies the
        # dtheta_t/dtheta_b Jacobian). Do not "fix" this to a flux ratio.
        coef = ccg(omega, depth_exit) / ccg_t / nsub * np.exp(-fan.atten[ok])

        # boundary-point weights from exit perimeter position
        p_exit = _perimeter_coord(fan.x[ok], fan.y[ok], grid.bounds)
        blo, bhi, wlo, whi = _circular_bracket_weights(p_exit, p_bp_sorted, perimeter)
        klo, khi = order[blo], order[bhi]

        # boundary-direction weights from exit propagation direction
        d_exit = theta_to_dir(fan.theta[ok])
        dlo, dhi, vlo, vhi = _circular_bracket_weights(d_exit, dirs_b_sorted, 360.0)
        llo, lhi = db_order[dlo], db_order[dhi]

        for kb, wb in ((klo, wlo), (khi, whi)):
            for lb, wd in ((llo, vlo), (lhi, vhi)):
                np.add.at(t_op[i], (jbin, kb, lb), coef * wb * wd)

    return TransferOperator(
        T=t_op,
        freq=freqs,
        dir_t=dirs,
        dir_b=dirs,
        bp_x=boundary_xy[:, 0],
        bp_y=boundary_xy[:, 1],
        target_x=tx,
        target_y=ty,
        depth_target=depth_t,
        attrs={
            "nsub": nsub,
            "ds": ds,
            "d_min": d_min,
            "max_steps": max_steps,
            # 0.0 == friction disabled (physically identical to None)
            "cf_jonswap": 0.0 if cf_jonswap is None else cf_jonswap,
            "lost_fraction": n_lost / max(n_rays, 1),
            "landed_fraction": n_landed / max(n_rays, 1),
        },
    )
