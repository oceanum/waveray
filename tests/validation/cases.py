"""Shared SWAN-vs-waveray validation cases.

Each :class:`Case` owns one bathymetry, one boundary spectrum and a list of
target points, and can render itself as a stationary SWAN run *and* as a set
of waveray site models. Both models are fed the **same** boundary spectrum
(written directly to SWAN's stationary ASCII spectral format) and the same
frequency range, so most of what is left at the target is the transformation.

Two residual differences are worth knowing before reading a small
disagreement as physics:

* **Directional bins are staggered half a bin.** SWAN centres its bins at
  5, 15, ... 355 degrees for ``CIRCLE 36``; this package uses 0, 10, ... 350.
  Frequencies agree to ~5e-5 Hz.
* **Boundary coverage differs.** SWAN prescribes energy on the named side
  only and treats the others as zero-energy; in ``bbox`` mode waveray gives
  boundary energy to a ray leaving *any* edge. For the shore-normal cases
  here most of the rays that exit elsewhere sit in the near-zero tail of the
  spectrum, which is why the plane-beach agreement is still 0.3 %, but the
  setup is not a perfectly matched boundary condition.

Used by ``tests/test_validation_swan.py`` and by the validation notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from waveray import SiteModel
from waveray.bathymetry import LocalGrid
from waveray.breaking import hm0, spectral_moment

from .swan import read_table, run_swan, swan_freqs, write_bottom, write_spectrum

# A spectral grid both models share. The upper frequency is deliberately
# modest: unbalanced wind input runs away at high frequency (see the wind
# forcing guide), and 0.4 Hz spans the energy-containing range of a swell
# downscale.
FREQS = swan_freqs(0.04, 0.4, 24)
DIRS = np.arange(0.0, 360.0, 10.0)


def jonswap_spectrum(
    freqs: np.ndarray,
    dirs: np.ndarray,
    hs: float,
    tp: float,
    dpm: float,
    dspr: float = 25.0,
    gamma: float = 3.3,
) -> xr.DataArray:
    """JONSWAP frequency shape with a cos^2s directional spread.

    Returns an ``efth(freq, dir)`` DataArray in m^2 / Hz / deg, scaled so its
    Hm0 equals ``hs`` exactly under the package's quadrature.
    """
    from wavespectra.construct.direction import cartwright
    from wavespectra.construct.frequency import jonswap

    ef = jonswap(xr.DataArray(freqs, dims="freq", coords={"freq": freqs}), fp=1.0 / tp, gamma=gamma)
    gth = cartwright(xr.DataArray(dirs, dims="dir", coords={"dir": dirs}), dm=dpm, dspr=dspr)
    efth = (ef * gth).transpose("freq", "dir")
    scale = (hs / hm0(efth.values[None], freqs, dirs)[0]) ** 2
    out = (efth * scale).astype(float)
    out.name = "efth"
    return out


def write_swan_boundary(path: Path, efth: xr.DataArray, x: float, y: float) -> None:
    """Write ``efth(freq, dir)`` as a stationary SWAN ASCII boundary spectrum."""
    write_spectrum(
        path,
        efth.transpose("freq", "dir").values,
        efth["freq"].values,
        efth["dir"].values,
        x=x,
        y=y,
    )


@dataclass
class Case:
    """One bathymetry + boundary spectrum + targets, runnable in both models."""

    name: str
    x: np.ndarray  # (nx,) local metres, ascending
    y: np.ndarray  # (ny,) local metres, ascending
    depth: np.ndarray  # (ny, nx) positive down; <= 0 is land
    targets: list[tuple[float, float]]
    boundary_xy: np.ndarray  # (K, 2) waveray boundary points
    efth: xr.DataArray | None = None  # boundary spectrum, None = wind sea only
    wind: tuple[float, float] | None = None  # (U10 m/s, coming-from nautical deg)
    boundary_side: str = "W"
    swan_full_physics: bool = False  # quadruplets + whitecapping on (wind cases)
    freqs: np.ndarray = field(default_factory=lambda: FREQS)
    dirs: np.ndarray = field(default_factory=lambda: DIRS)

    # ------------------------------------------------------------------ #
    @property
    def grid(self) -> LocalGrid:
        return LocalGrid(x=self.x, y=self.y, depth=self.depth)

    def _swan_input(self) -> str:
        nx, ny = self.x.size - 1, self.y.size - 1
        dx = float(np.median(np.diff(self.x)))
        dy = float(np.median(np.diff(self.y)))
        x0, y0 = float(self.x[0]), float(self.y[0])
        xlen, ylen = float(self.x[-1] - self.x[0]), float(self.y[-1] - self.y[0])
        msc = self.freqs.size - 1
        flow, fhigh = float(self.freqs[0]), float(self.freqs[-1])
        pts = " ".join(f"{tx:.2f} {ty:.2f}" for tx, ty in self.targets)

        lines = [
            f"$ waveray validation case: {self.name}",
            "PROJ 'wray' 'V1'",
            # maxerr 2: SWAN raises a level-2 error for third-generation wind
            # without quadruplets, which is exactly the configuration waveray
            # implements; the no-wind cases never trip it.
            "SET 0. 90. 0.05 200 2 NAUTICAL",
            "MODE STATIONARY TWODIMENSIONAL",
            "COORDINATES CARTESIAN",
            f"CGRID REGULAR {x0} {y0} 0. {xlen} {ylen} {nx} {ny} "
            f"CIRCLE {self.dirs.size} {flow} {fhigh} {msc}",
            f"INPGRID BOTTOM REGULAR {x0} {y0} 0. {nx} {ny} {dx} {dy}",
            "READINP BOTTOM 1. 'bottom.bot' 3 0 FREE",
        ]
        if self.efth is not None:
            lines.append(f"BOUN SIDE {self.boundary_side} CCW CONSTANT FILE 'boun.sp2'")
        if self.wind is not None:
            lines.append(f"WIND {self.wind[0]} {self.wind[1]}")
            lines.append("GEN3 KOMEN DRAG FIT AGROW")
        else:
            lines.append("GEN3 KOMEN DRAG FIT")
        if not self.swan_full_physics:
            lines += ["OFF QUAD", "OFF WCAPPING"]
        lines += [
            "OFF BREAKING",  # waveray applies breaking as a target post-step
            "NUM STOPC 0.005 0.01 0.005 99.5 STAT 80",
            f"POINTS 'tg' {pts}",
            f"TABLE 'tg' HEAD '{self.name}.tbl' XP YP DEP HSIGN TM01 DIR DSPR",
            f"SPECOUT 'tg' SPEC2D ABS '{self.name}.sp2'",
            "COMPUTE",
            "STOP",
        ]
        return "\n".join(lines) + "\n"

    def write(self, workdir: Path) -> None:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        write_bottom(workdir / "bottom.bot", self.depth)
        if self.efth is not None:
            write_swan_boundary(workdir / "boun.sp2", self.efth, self.x[0], 0.0)
        (workdir / f"{self.name}.swn").write_text(self._swan_input())

    def run_swan(self, workdir: Path) -> dict[str, np.ndarray]:
        """Run SWAN and return the target table columns."""
        workdir = Path(workdir)
        self.write(workdir)
        run_swan(workdir, self.name)
        return read_table(
            workdir / f"{self.name}.tbl",
            ["XP", "YP", "DEP", "HSIGN", "TM01", "DIR", "DSPR"],
        )

    def run_waveray(self, **build_kwargs) -> dict[str, np.ndarray]:
        """Build one operator per target and transform the boundary spectrum.

        Returns Hs / Tm01 / Dir arrays in the same order as ``targets``,
        computed with the package's own quadrature.
        """
        grid = self.grid
        kwargs = dict(cf_jonswap=None, nsub=5)
        kwargs.update(build_kwargs)
        if self.wind is not None:
            kwargs.setdefault("wind", self.wind)
            kwargs.setdefault("agrow", True)

        k = self.boundary_xy.shape[0]
        if self.efth is not None:
            e_b = xr.concat([self.efth] * k, dim="site").transpose("site", "freq", "dir")
        else:  # wind-sea only: zero boundary energy
            e_b = xr.DataArray(
                np.zeros((k, self.freqs.size, self.dirs.size)),
                dims=("site", "freq", "dir"),
                coords={"freq": self.freqs, "dir": self.dirs},
            )

        hs, tm01, pdir = [], [], []
        for tx, ty in self.targets:
            model = SiteModel.build(
                bathy=grid,
                target=(tx, ty),
                boundary_points=self.boundary_xy,
                freqs=self.freqs,
                dirs=self.dirs,
                **kwargs,
            )
            out = model.transform(e_b, breaking=False).values
            m0 = spectral_moment(out[None], self.freqs, self.dirs, n=0)[0]
            m1 = spectral_moment(out[None], self.freqs, self.dirs, n=1)[0]
            hs.append(4.0 * np.sqrt(max(m0, 0.0)))
            tm01.append(m0 / m1 if m1 > 0 else np.nan)
            # mean direction from the first directional moment (nautical)
            th = np.deg2rad(self.dirs)
            ef = out.sum(axis=0)
            sx, sy = float((ef * np.sin(th)).sum()), float((ef * np.cos(th)).sum())
            pdir.append(np.rad2deg(np.arctan2(sx, sy)) % 360.0 if ef.sum() > 0 else np.nan)
        return {
            "HSIGN": np.array(hs),
            "TM01": np.array(tm01),
            "DIR": np.array(pdir),
        }


# ---------------------------------------------------------------------- #
# Idealised cases
# ---------------------------------------------------------------------- #
def plane_beach_case(
    name: str = "beach",
    dpm: float = 270.0,
    d_offshore: float = 30.0,
    d_shore: float = 4.0,
    length: float = 15_000.0,
    halfwidth: float = 40_000.0,
    nx: int = 76,
    ny: int = 81,
    hs: float = 2.0,
    tp: float = 10.0,
) -> Case:
    """Alongshore-uniform plane beach: pure refraction + shoaling.

    The domain is deliberately much wider than it is long so the arrival cone
    of the target is fully supplied by the offshore boundary and neither model
    sees a lateral energy deficit.
    """
    x = np.linspace(0.0, length, nx)
    y = np.linspace(-halfwidth, halfwidth, ny)
    prof = d_offshore + (d_shore - d_offshore) * x / length
    depth = np.tile(prof, (ny, 1))
    targets = [
        (float(length * (d_offshore - d) / (d_offshore - d_shore)), 0.0) for d in (20.0, 12.0, 6.0)
    ]
    return Case(
        name=name,
        x=x,
        y=y,
        depth=depth,
        targets=targets,
        boundary_xy=np.array([[0.0, -halfwidth * 0.9], [0.0, halfwidth * 0.9]]),
        efth=jonswap_spectrum(FREQS, DIRS, hs=hs, tp=tp, dpm=dpm),
    )


def island_case(
    name: str = "island",
    depth_flat: float = 25.0,
    size: float = 30_000.0,
    n: int = 121,
    island_radius: float = 3_000.0,
    hs: float = 2.0,
    tp: float = 12.0,
) -> Case:
    """Flat bottom with a circular island: tests sheltering in the lee."""
    x = np.linspace(-size / 2, size / 2, n)
    y = np.linspace(-size / 2, size / 2, n)
    xx, yy = np.meshgrid(x, y)
    depth = np.where(np.hypot(xx, yy) < island_radius, -1.0, depth_flat)
    targets = [(6_000.0, 0.0), (6_000.0, 6_000.0), (-6_000.0, 0.0)]  # lee, side, upwave
    return Case(
        name=name,
        x=x,
        y=y,
        depth=depth,
        targets=targets,
        boundary_xy=np.array([[-size / 2, -size * 0.45], [-size / 2, size * 0.45]]),
        efth=jonswap_spectrum(FREQS, DIRS, hs=hs, tp=tp, dpm=270.0, dspr=20.0),
    )


def wind_growth_case(
    name: str = "windsea",
    depth_flat: float = 30.0,
    length: float = 6_000.0,
    halfwidth: float = 30_000.0,
    nx: int = 61,
    ny: int = 61,
    u10: float = 12.0,
    swan_full_physics: bool = False,
) -> Case:
    """Flat bottom, zero boundary energy, uniform onshore wind: fetch-limited
    growth.

    SWAN runs with quadruplets and whitecapping **off**, so both models carry
    wind input and nothing else — the like-for-like test of the source term.
    SWAN raises a level-2 error for a third-generation wind without
    quadruplets (``SET`` maxerr=2 overrides it) but then converges in 9
    iterations to 100 % and reproduces byte-identically, whereas the same case
    with full physics stops at the 80-iteration cap having reached only ~92 %
    of the required 99.5 % and lands in one of two answer families 3.6x apart.
    Never compare against that.
    """
    x = np.linspace(0.0, length, nx)
    y = np.linspace(-halfwidth, halfwidth, ny)
    depth = np.full((ny, nx), depth_flat)
    targets = [(1_000.0, 0.0), (3_000.0, 0.0), (5_000.0, 0.0)]
    return Case(
        name=name,
        x=x,
        y=y,
        depth=depth,
        targets=targets,
        boundary_xy=np.array([[0.0, -halfwidth * 0.9], [0.0, halfwidth * 0.9]]),
        efth=None,
        wind=(u10, 270.0),
        swan_full_physics=swan_full_physics,
    )


def real_bathymetry_case(
    name: str = "gebco",
    datasource: str = "gebco_2025",
    bbox: tuple[float, float, float, float] = (3.90, 52.10, 4.45, 52.45),
    hs: float = 2.5,
    tp: float = 9.0,
    dpm: float = 285.0,
    max_nodes: int = 140,
) -> Case:
    """Real GEBCO bathymetry off Noordwijk aan Zee (NL), swell from the NW.

    The Datamesh grid is resampled onto a local-metre grid (SWAN and waveray
    both take it in metres), land is kept as negative depth so both models
    block on it, and the targets sit on a shore-normal transect. Requires
    ``DATAMESH_TOKEN`` and the ``datamesh`` extra.
    """
    from waveray.bathymetry import fetch_datamesh_bathymetry

    grid = fetch_datamesh_bathymetry(datasource, bbox=bbox, positive="up")

    # thin to a manageable, uniformly spaced local grid
    def _thin(coord: np.ndarray) -> np.ndarray:
        step = max(1, int(np.ceil(coord.size / max_nodes)))
        return np.arange(coord.size)[::step]

    ix, iy = _thin(grid.x), _thin(grid.y)
    x, y = grid.x[ix], grid.y[iy]
    x = np.linspace(x[0], x[-1], x.size)  # enforce exactly uniform spacing
    y = np.linspace(y[0], y[-1], y.size)
    xx, yy = np.meshgrid(x, y)
    depth = grid.sample_depth(xx.ravel(), yy.ravel()).reshape(xx.shape)
    # sample_depth fills land with 0; restore it as land for both models
    depth = np.where(depth <= 0.05, -1.0, depth)

    # shore-normal transect: deepest column in the west, stepping east
    x0 = float(x[2])
    targets = [
        (float(x0 + frac * (x[-3] - x0)), float(np.median(y))) for frac in (0.35, 0.55, 0.70)
    ]
    # keep only targets that are wet with a workable depth
    targets = [
        (tx, ty) for tx, ty in targets if grid.sample_depth(np.array([tx]), np.array([ty]))[0] > 3.0
    ]
    return Case(
        name=name,
        x=x,
        y=y,
        depth=depth,
        targets=targets,
        boundary_xy=np.array([[float(x[0]), float(y[3])], [float(x[0]), float(y[-4])]]),
        efth=jonswap_spectrum(FREQS, DIRS, hs=hs, tp=tp, dpm=dpm, dspr=28.0),
    )


def wind_on_swell_case(
    name: str = "windswell",
    u10: float = 12.0,
    swan_full_physics: bool = True,
    **kwargs,
) -> Case:
    """Plane beach with swell *and* wind over a 15 km fetch.

    With ``swan_full_physics=True`` (default) this is the realistic downscale
    configuration: SWAN dissipates as well as generates, waveray adds only the
    input term, so the comparison bounds what that omission costs.

    With ``swan_full_physics=False`` both models carry wind input alone, and
    both run away — SWAN reaches Hs 77.8 m from a 2 m swell here, waveray 35.0 m
    with the growth ceiling disabled. That is the measurement behind the
    ``max_growth`` guard: the instability belongs to the unbalanced source
    term, not to this implementation of it.
    """
    case = plane_beach_case(name=name, **kwargs)
    case.wind = (u10, 270.0)
    case.swan_full_physics = swan_full_physics
    return case
