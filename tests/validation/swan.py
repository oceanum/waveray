"""Run stationary SWAN (official delftwaves/swan docker image) and read output.

The runner writes a case directory (input file, bottom grid, optional
boundary spectra), executes ``swanrun`` inside the container as the calling
user, checks for SWAN errors, and returns the output paths. Outputs are read
back with ``wavespectra.read_swan`` (2D spectra) and a small table parser.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

SWAN_IMAGE = "delftwaves/swan:latest"


def docker_available() -> bool:
    """True when a docker daemon is reachable."""
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "ps"], check=True, capture_output=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    return True


def swan_image_available(pull: bool = True) -> bool:
    """True when the SWAN image is present (optionally pulling it)."""
    if not docker_available():
        return False
    have = (
        subprocess.run(["docker", "image", "inspect", SWAN_IMAGE], capture_output=True).returncode
        == 0
    )
    if have or not pull:
        return have
    return subprocess.run(["docker", "pull", "-q", SWAN_IMAGE]).returncode == 0


def swan_freqs(flow: float, fhigh: float, msc: int) -> np.ndarray:
    """SWAN's logarithmic frequency grid: msc+1 freqs from flow to fhigh."""
    return flow * (fhigh / flow) ** (np.arange(msc + 1) / msc)


def write_bottom(path: Path, depth: np.ndarray) -> None:
    """Write a SWAN bottom file, IDLA=3 (rows south to north), FREE format.

    ``depth`` is (ny, nx) positive-down metres with y ascending; NaN / land
    is written as -9 (SWAN treats non-positive depth as dry with LEVEL 0).
    """
    d = np.where(np.isfinite(depth), depth, -9.0)
    with open(path, "w") as f:
        for row in d:  # row 0 = southernmost = IDLA 3 start
            f.write(" ".join(f"{v:.3f}" for v in row) + "\n")


def write_spectrum(
    path: Path,
    efth: np.ndarray,
    freqs: np.ndarray,
    dirs: np.ndarray,
    x: float = 0.0,
    y: float = 0.0,
) -> None:
    """Write a *stationary* SWAN ASCII 2D spectrum file (one location).

    ``efth`` is (nfreq, ndir) in m^2/Hz/deg with nautical coming-from
    directions. wavespectra's ``to_swan`` always stamps a TIME block, which
    stationary SWAN rejects, so the file is written directly here.
    """
    efth = np.asarray(efth, dtype=float)
    if efth.shape != (freqs.size, dirs.size):
        raise ValueError(f"efth shape {efth.shape} != {(freqs.size, dirs.size)}")
    peak = float(np.max(efth))
    factor = peak / 1.0e4 if peak > 0 else 1.0e-8

    lines = [
        "SWAN   1                                Swan standard spectral file",
        "$   Written by waveray validation harness",
        "LOCATIONS                               locations in x-y-space",
        "     1",
        f"  {x:.4f}  {y:.4f}",
        "AFREQ                                   absolute frequencies in Hz",
        f"  {freqs.size}",
    ]
    lines += [f"  {f:.6f}" for f in freqs]
    lines += [
        "NDIR                                    spectral nautical directions in degr",
        f"  {dirs.size}",
    ]
    lines += [f"  {d:.4f}" for d in dirs]
    lines += [
        "QUANT",
        "     1                                  number of quantities",
        "VaDens                                  variance densities in m2/Hz/degr",
        "m2/Hz/degr",
        "   -0.9900E+02                          exception value",
        "FACTOR",
        f"  {factor:.8E}",
    ]
    for i in range(freqs.size):
        lines.append(" ".join(f"{int(round(v / factor)):8d}" for v in efth[i]))
    Path(path).write_text("\n".join(lines) + "\n")


def convergence(prt_path: Path) -> tuple[float, float, int, bool]:
    """(achieved %, required %, iterations, converged?) from a SWAN PRINT file.

    SWAN stops either when the accuracy criterion is met or at the iteration
    cap. A run that stopped at the cap has *not* solved the equations: its
    answer depends on where the iteration happened to be, and repeat runs can
    land in entirely different families. Never use one as a test oracle.
    """
    text = Path(prt_path).read_text()
    acc = re.findall(r"accuracy OK in\s+([\d.]+) %.*?\(\s*([\d.]+) % required\)", text)
    iters = re.findall(r"iteration\s+(\d+); sweep", text)
    n_iter = int(iters[-1]) if iters else 0
    if not acc:
        return 0.0, 0.0, n_iter, False
    achieved, required = float(acc[-1][0]), float(acc[-1][1])
    return achieved, required, n_iter, achieved >= required


def run_swan(
    workdir: Path, name: str, timeout: int = 600, require_convergence: bool = True
) -> Path:
    """Run ``swanrun -input <name>`` in the SWAN container; return the .prt path.

    Raises RuntimeError with the SWAN error context if the run fails, or if it
    stopped at the iteration cap without meeting its accuracy criterion
    (``require_convergence``). The latter is not pedantry: a non-converged
    stationary SWAN run is not reproducible — the same input can land in
    answer families several-fold apart — so comparing against one produces a
    test that passes or fails at random.
    """
    workdir = Path(workdir)
    uid, gid = os.getuid(), os.getgid()
    cmd = [
        "docker",
        "run",
        "--rm",
        "--user",
        f"{uid}:{gid}",
        "-v",
        f"{workdir.resolve()}:/case",
        "-w",
        "/case",
        SWAN_IMAGE,
        "swanrun",
        "-input",
        name,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    prt = workdir / f"{name}.prt"
    errfile = workdir / "Errfile"
    errors: list[str] = []
    if proc.returncode != 0:
        errors.append(f"swanrun exited {proc.returncode}")
    if errfile.exists() and errfile.read_text().strip():
        errors.append(f"Errfile:\n{errfile.read_text()}")
    prt_text = prt.read_text() if prt.exists() else ""
    if "Severe error" in prt_text or "SEVERE ERROR" in prt_text:
        sev = [ln for ln in prt_text.splitlines() if "evere" in ln]
        errors.append("severe errors in PRINT: " + " | ".join(sev[:5]))
    if not prt.exists():
        errors.append(f"no PRINT file; stdout tail: {proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    if errors:
        raise RuntimeError(f"SWAN run '{name}' failed: " + "\n".join(errors))
    if require_convergence:
        achieved, required, n_iter, ok = convergence(prt)
        if not ok:
            raise RuntimeError(
                f"SWAN run '{name}' did NOT converge: reached {achieved:.2f}% of wet grid "
                f"points within tolerance after {n_iter} iterations, {required:.2f}% required. "
                "Its answer is not reproducible and must not be used as a reference — raise "
                "the iteration cap, or fix the case (a wind sea whose peak frequency falls "
                "outside the spectral grid will not converge)."
            )
    return prt


def swan_version(prt_path: Path) -> str:
    """SWAN version string from the PRINT file header."""
    for line in Path(prt_path).read_text().splitlines()[:20]:
        if "version" in line.lower():
            return line.strip()
    return "unknown"


def read_table(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    """Parse a SWAN TABLE ... HEAD output into named 1D arrays."""
    rows = []
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        rows.append([float(v) for v in s.split()])
    arr = np.asarray(rows, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(columns):
        raise ValueError(f"table {path} has shape {arr.shape}, expected {len(columns)} columns")
    return {c: arr[:, i] for i, c in enumerate(columns)}
