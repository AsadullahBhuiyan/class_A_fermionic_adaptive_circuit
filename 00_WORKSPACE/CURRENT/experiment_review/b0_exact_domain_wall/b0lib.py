"""Numerical kernels and audited I/O for the deterministic B0 campaign.

The Hamiltonian itself always comes from the CPU ``classA_U1FGTN`` object.  The
momentum representation below is an exact block diagonalization of that canonical
real-space Hamiltonian and is checked against it in preflight.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.linalg import polar
from scipy.optimize import curve_fit, linear_sum_assignment


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]
FGTN_SRC = REPO_ROOT / "src" / "fgtn"
if str(FGTN_SRC) not in sys.path:
    sys.path.insert(0, str(FGTN_SRC))

from classA_U1FGTN import classA_U1FGTN  # noqa: E402


SIGMA_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
HOP_X = -0.5 * SIGMA_Z - 0.5j * SIGMA_X
HOP_Y = -0.5 * SIGMA_Z - 0.5j * SIGMA_Y


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_config(path: Path | None = None) -> tuple[dict[str, Any], Path, str]:
    path = PACKAGE_DIR / "campaign_config.v2.json" if path is None else Path(path)
    raw = path.read_bytes()
    config = json.loads(raw)
    if config.get("config_locked") is not True:
        raise ValueError(f"Campaign configuration is not locked: {path}")
    return config, path, hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name)


def json_safe(value: Any) -> Any:
    """Convert NumPy values and non-finite floats to strict-JSON values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(name)


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=False
        )
        return proc.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short").splitlines(),
    }


def environment_metadata() -> dict[str, Any]:
    import scipy

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "hostname": platform.node(),
        "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
        "thread_env": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def make_model(nx: int, ny: int, config: dict[str, Any]) -> classA_U1FGTN:
    # The canonical constructor prints geometry information; keep worker logs concise.
    with contextlib.redirect_stdout(io.StringIO()):
        model = classA_U1FGTN(
            Nx=int(nx),
            Ny=int(ny),
            DW=True,
            alpha_1=float(config["alpha_top"]),
            alpha_2=float(config["alpha_triv"]),
        )
    return model


def construction_flag(construction: str) -> bool:
    if construction not in {"coupled", "hard_exterior"}:
        raise ValueError(f"Unknown construction {construction!r}")
    return construction == "hard_exterior"


def canonical_hamiltonian(model: classA_U1FGTN, construction: str) -> np.ndarray:
    return np.asarray(
        model._domain_wall_hamiltonian(
            periodic=True, triv_region_local_mode=construction_flag(construction)
        ),
        dtype=np.complex128,
    )


def active_x_mask(model: classA_U1FGTN) -> np.ndarray:
    x0, x1 = sorted(int(x) for x in model.DW_loc)
    mask = np.zeros(model.Nx, dtype=bool)
    mask[x0 : x1 + 1] = True
    return mask


def momentum_hamiltonian(
    model: classA_U1FGTN, k: float, construction: str
) -> np.ndarray:
    """Exact y-momentum block of the canonical real-space Hamiltonian."""
    nx = model.Nx
    h = np.zeros((2 * nx, 2 * nx), dtype=np.complex128)
    hard = construction_flag(construction)
    mask = active_x_mask(model)
    alpha_x = np.real(np.asarray(model.alpha_profile)[:, 0])

    def sl(x: int) -> slice:
        return slice(2 * x, 2 * x + 2)

    for x in range(nx):
        if hard and not mask[x]:
            h[sl(x), sl(x)] = -np.eye(2, dtype=np.complex128)
        else:
            h[sl(x), sl(x)] = alpha_x[x] * SIGMA_Z
            h[sl(x), sl(x)] += HOP_Y * np.exp(1j * k)
            h[sl(x), sl(x)] += HOP_Y.conj().T * np.exp(-1j * k)
    for x in range(nx):
        xp = (x + 1) % nx
        if hard and not (mask[x] and mask[xp]):
            continue
        h[sl(x), sl(xp)] += HOP_X
        h[sl(xp), sl(x)] += HOP_X.conj().T
    return 0.5 * (h + h.conj().T)


def momentum_reconstruct_real(
    model: classA_U1FGTN, construction: str, phi: float = 0.0
) -> np.ndarray:
    nx, ny = model.Nx, model.Ny
    blocks = [
        momentum_hamiltonian(model, 2 * np.pi * n / ny + phi / ny, construction)
        for n in range(ny)
    ]
    nblock = 2 * nx
    out = np.empty((nblock * ny, nblock * ny), dtype=np.complex128)
    for y in range(ny):
        for yp in range(ny):
            d = y - yp
            out[
                y * nblock : (y + 1) * nblock,
                yp * nblock : (yp + 1) * nblock,
            ] = sum(
                np.exp(1j * (2 * np.pi * n / ny) * d) * block
                for n, block in enumerate(blocks)
            ) / ny
    return 0.5 * (out + out.conj().T)


def seam_twisted_hamiltonian(
    model: classA_U1FGTN, construction: str, phi: float, seam_y: int = 0
) -> np.ndarray:
    """Canonical real-space H with the full flux placed on one +y bond layer."""
    h = canonical_hamiltonian(model, construction).copy()
    nx, ny = model.Nx, model.Ny
    nblock = 2 * nx
    y0 = int(seam_y) % ny
    y1 = (y0 + 1) % ny
    s0 = slice(y0 * nblock, (y0 + 1) * nblock)
    s1 = slice(y1 * nblock, (y1 + 1) * nblock)
    h[s0, s1] *= np.exp(1j * phi)
    h[s1, s0] *= np.exp(-1j * phi)
    return 0.5 * (h + h.conj().T)


def occupied_frame(
    model: classA_U1FGTN, k: float, construction: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a declared rank-Nx occupied frame and its block spectrum/projector."""
    h = momentum_hamiltonian(model, k, construction)
    evals, evecs = np.linalg.eigh(h)
    nx = model.Nx
    if construction == "coupled":
        frame = evecs[:, :nx]
    else:
        mask = active_x_mask(model)
        active = np.array(
            [2 * x + mu for x in range(nx) if mask[x] for mu in (0, 1)], dtype=int
        )
        exterior_occ = np.array([2 * x for x in range(nx) if not mask[x]], dtype=int)
        ha = h[np.ix_(active, active)]
        _, va = np.linalg.eigh(ha)
        va = va[:, : active.size // 2]
        frame = np.zeros((2 * nx, nx), dtype=np.complex128)
        frame[np.ix_(active, np.arange(va.shape[1]))] = va
        for col, idx in enumerate(exterior_occ, start=va.shape[1]):
            frame[idx, col] = 1.0
    projector = frame @ frame.conj().T
    return frame, evals, projector


def declared_complete_eigenbasis(
    model: classA_U1FGTN, k: float, construction: str
) -> tuple[np.ndarray, np.ndarray]:
    """Complete orthonormal instantaneous basis used for overlap continuation."""
    h = momentum_hamiltonian(model, k, construction)
    if construction == "coupled":
        return np.linalg.eigh(h)
    nx = model.Nx
    mask = active_x_mask(model)
    active = np.array([2 * x + mu for x in range(nx) if mask[x] for mu in (0, 1)], dtype=int)
    exterior = np.array([2 * x + mu for x in range(nx) if not mask[x] for mu in (0, 1)], dtype=int)
    values_active, vectors_active = np.linalg.eigh(h[np.ix_(active, active)])
    values = np.concatenate([values_active, -np.ones(exterior.size)])
    vectors = np.zeros((2 * nx, 2 * nx), dtype=np.complex128)
    vectors[np.ix_(active, np.arange(active.size))] = vectors_active
    for col, idx in enumerate(exterior, start=active.size):
        vectors[idx, col] = 1.0
    return values, vectors


def occupied_blocks(
    model: classA_U1FGTN, construction: str, phi: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ny = model.Ny
    frames, spectra, projectors = [], [], []
    for n in range(ny):
        k = 2 * np.pi * n / ny + phi / ny
        frame, evals, projector = occupied_frame(model, k, construction)
        frames.append(frame)
        spectra.append(evals)
        projectors.append(projector)
    return np.asarray(frames), np.asarray(spectra), np.asarray(projectors)


def correlation_displacements(projectors: np.ndarray, phi: float = 0.0) -> np.ndarray:
    """C[d] = <c^dagger_(y') c_y> block for d=y-y'."""
    ny = projectors.shape[0]
    # ``projectors[n]`` is evaluated at k_n + phi/Ny, but the uniform-gauge
    # Bloch basis itself remains exp(i k_n y).  Including phi in this Fourier
    # phase would impose a non-periodic basis and destroy projector idempotency
    # when the displacement is reduced modulo Ny.
    ks = 2 * np.pi * np.arange(ny) / ny
    return np.asarray(
        [
            np.einsum("k,kab->ab", np.exp(1j * ks * d), projectors, optimize=True) / ny
            for d in range(ny)
        ],
        dtype=np.complex128,
    )


def full_covariance_from_displacements(cdisp: np.ndarray) -> np.ndarray:
    ny, nblock, _ = cdisp.shape
    out = np.empty((ny * nblock, ny * nblock), dtype=np.complex128)
    for y in range(ny):
        for yp in range(ny):
            out[
                y * nblock : (y + 1) * nblock,
                yp * nblock : (yp + 1) * nblock,
            ] = cdisp[(y - yp) % ny]
    return 0.5 * (out + out.conj().T)


def strip_correlation(cdisp: np.ndarray, ay: int) -> np.ndarray:
    ny, nblock, _ = cdisp.shape
    out = np.empty((ay * nblock, ay * nblock), dtype=np.complex128)
    for y in range(ay):
        for yp in range(ay):
            out[
                y * nblock : (y + 1) * nblock,
                yp * nblock : (yp + 1) * nblock,
            ] = cdisp[(y - yp) % ny]
    return 0.5 * (out + out.conj().T)


def translated_half_window_average(cfull: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Average translated half-cylinder reductions in a common relative-y basis."""
    nblock = 2 * int(nx)
    ay = int(ny) // 2
    expected = int(ny) * nblock
    cfull = np.asarray(cfull, dtype=np.complex128)
    if cfull.shape != (expected, expected):
        raise ValueError(f"Expected full correlation shape {(expected, expected)}, got {cfull.shape}")
    average = np.zeros((ay * nblock, ay * nblock), dtype=np.complex128)
    for y0 in range(int(ny)):
        indices = np.asarray(
            [
                mu + 2 * x + nblock * ((y0 + yrel) % int(ny))
                for yrel in range(ay)
                for x in range(int(nx))
                for mu in (0, 1)
            ],
            dtype=int,
        )
        average += cfull[np.ix_(indices, indices)]
    average /= float(ny)
    return 0.5 * (average + average.conj().T)


def modular_hamiltonian_from_correlation(
    correlation: np.ndarray, clip: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return h=log[(1-C)/C] and its clipped occupation eigensystem."""
    correlation = 0.5 * (
        np.asarray(correlation, dtype=np.complex128)
        + np.asarray(correlation, dtype=np.complex128).conj().T
    )
    occupations, vectors = np.linalg.eigh(correlation)
    occupations = np.clip(np.real(occupations), float(clip), 1.0 - float(clip))
    energies = np.log((1.0 - occupations) / occupations)
    hmod = (vectors * energies[None, :]) @ vectors.conj().T
    return 0.5 * (hmod + hmod.conj().T), occupations, vectors


def renyi_weight(lam: np.ndarray, q: int) -> np.ndarray:
    lam = np.clip(np.asarray(lam, dtype=float), 1e-12, 1 - 1e-12)
    if int(q) == 1:
        return -(lam * np.log(lam) + (1 - lam) * np.log(1 - lam))
    return np.log(lam**q + (1 - lam) ** q) / (1 - q)


def chord(ny: int, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.log((ny / np.pi) * np.sin(np.pi * values / ny))


def linear_fit(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    design = np.column_stack([x, np.ones_like(x)])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ beta
    residuals = y - predicted
    dof = max(1, x.size - 2)
    sigma2 = float(residuals @ residuals / dof)
    cov = sigma2 * np.linalg.pinv(design.T @ design)
    denom = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(residuals @ residuals) / denom if denom > 0 else 1.0
    return {
        "slope": float(beta[0]),
        "intercept": float(beta[1]),
        "covariance": cov,
        "residuals": residuals,
        "predicted": predicted,
        "r2": float(r2),
        "n_points": int(x.size),
    }


def cyclic_distance(x: np.ndarray | float, center: float, period: int) -> np.ndarray:
    value = np.asarray(x, dtype=float)
    return np.abs((value - center + period / 2) % period - period / 2)


def wall_columns(center: int, nx: int, width: int) -> np.ndarray:
    if width % 2 != 1:
        raise ValueError("wall_window_columns must be odd")
    half = width // 2
    return np.array([(center + dx) % nx for dx in range(-half, half + 1)], dtype=int)


@dataclass
class EntropyResult:
    payload: dict[str, np.ndarray]
    rows: list[dict[str, Any]]
    half_evals: np.ndarray
    half_evecs: np.ndarray


def entropy_observables(
    cdisp: np.ndarray, model: classA_U1FGTN, config: dict[str, Any]
) -> EntropyResult:
    nx, ny = model.Nx, model.Ny
    ay_max = ny // 2
    qs = np.asarray(config["entropy_q"], dtype=int)
    entropies = np.full((qs.size, ay_max + 1), np.nan, dtype=float)
    contours = np.full((qs.size, ay_max + 1, nx, ay_max), np.nan, dtype=float)
    spectra = np.full((ay_max + 1, 2 * nx * ay_max), np.nan, dtype=float)
    wall_values = np.full((2, qs.size, ay_max + 1), np.nan, dtype=float)
    walls = sorted(int(v) for v in model.DW_loc)
    columns = [wall_columns(w, nx, int(config["wall_window_columns"])) for w in walls]
    half_evals = np.empty(0)
    half_evecs = np.empty((0, 0), dtype=np.complex128)

    for ay in range(1, ay_max + 1):
        csub = strip_correlation(cdisp, ay)
        evals, evecs = np.linalg.eigh(csub)
        evals = np.clip(np.real(evals), 0.0, 1.0)
        spectra[ay, : evals.size] = evals
        if ay == ay_max:
            half_evals, half_evecs = evals, evecs
        probabilities = np.abs(evecs) ** 2
        for qi, q in enumerate(qs):
            weights = renyi_weight(evals, int(q))
            entropies[qi, ay] = float(np.sum(weights))
            local = (probabilities @ weights).reshape(ay, nx, 2).sum(axis=2).T
            contours[qi, ay, :, :ay] = local
            for wi, cols in enumerate(columns):
                wall_values[wi, qi, ay] = float(np.sum(local[cols, :]))

    rows: list[dict[str, Any]] = []
    fit_min = int(config["entropy_fit_ay_min"])
    fit_ays = np.arange(fit_min, ay_max + 1)
    if fit_ays.size >= 3:
        xfit = chord(ny, fit_ays)
        for qi, q in enumerate(qs):
            full = linear_fit(xfit, entropies[qi, fit_ays])
            rows.append(
                {
                    "quantity": "full_strip",
                    "wall_index": -1,
                    "q": int(q),
                    "slope": full["slope"],
                    "c_estimate": 6 * full["slope"] / (1 + 1 / q),
                    "intercept": full["intercept"],
                    "r2": full["r2"],
                    "n_points": full["n_points"],
                    "ay_min": int(fit_ays[0]),
                    "ay_max": int(fit_ays[-1]),
                }
            )
            for wi in range(2):
                fit = linear_fit(xfit, wall_values[wi, qi, fit_ays])
                rows.append(
                    {
                        "quantity": "physical_wall_contour",
                        "wall_index": wi,
                        "q": int(q),
                        "slope": fit["slope"],
                        "c_estimate": 6 * fit["slope"] / (1 + 1 / q),
                        "intercept": fit["intercept"],
                        "r2": fit["r2"],
                        "n_points": fit["n_points"],
                        "ay_min": int(fit_ays[0]),
                        "ay_max": int(fit_ays[-1]),
                    }
                )
    contour_sum_error = np.nanmax(
        np.abs(np.nansum(contours, axis=(2, 3)) - entropies)
    )
    payload = {
        "entropy_q": qs,
        "entropy_ay": np.arange(ay_max + 1),
        "entropy_values": entropies,
        "entropy_spectra": spectra,
        "entropy_contours": contours,
        "wall_contour_values": wall_values,
        "wall_positions": np.asarray(walls, dtype=int),
        "wall_columns": np.asarray(columns, dtype=int),
        "contour_sum_max_error": np.asarray(contour_sum_error),
    }
    return EntropyResult(payload, rows, half_evals, half_evecs)


def localize_low_doublet(
    h: np.ndarray, nx: int, walls: Iterable[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(h)
    chosen = np.argsort(np.abs(evals))[:2]
    basis = evecs[:, chosen]
    xcoord = np.repeat(np.arange(nx, dtype=float), 2)
    projected_x = basis.conj().T @ (xcoord[:, None] * basis)
    _, rotate = np.linalg.eigh(0.5 * (projected_x + projected_x.conj().T))
    localized = basis @ rotate
    centers = np.real(np.einsum("ij,i,ij->j", localized.conj(), xcoord, localized))
    energies = np.real(np.diag(localized.conj().T @ h @ localized))
    wall_array = np.asarray(list(walls), dtype=float)
    cost = np.stack([cyclic_distance(centers, w, nx) for w in wall_array])
    rows, cols = linear_sum_assignment(cost)
    order = np.empty(2, dtype=int)
    order[rows] = cols
    localized = localized[:, order]
    centers = centers[order]
    energies = energies[order]
    profiles = np.abs(localized.reshape(nx, 2, 2)) ** 2
    profiles = profiles.sum(axis=1).T
    return energies, centers, profiles, localized


def spectral_observables(
    model: classA_U1FGTN, construction: str, config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, float]]:
    nx = model.Nx
    walls = sorted(int(v) for v in model.DW_loc)
    nk = int(config["spectrum_k_points"])
    nlow = int(config["spectrum_low_modes"])
    # A centered odd grid must contain k=0 exactly.  An endpoint-based linspace
    # with odd ``nk`` misses the crossing and biases the fitted mass by the
    # nearest-grid level spacing, producing a spurious width-independent gap.
    if nk % 2 != 1:
        raise ValueError("spectrum_k_points must be odd so the Dirac crossing is sampled")
    ks = 2 * np.pi * np.arange(-(nk // 2), nk // 2 + 1, dtype=float) / nk
    low_e = np.empty((nk, nlow), dtype=float)
    low_v = np.empty((nk, 2 * nx, nlow), dtype=np.complex128)
    wall_e = np.empty((nk, 2), dtype=float)
    wall_centers = np.empty((nk, 2), dtype=float)
    wall_profiles = np.empty((nk, 2, nx), dtype=float)
    localized_vectors = np.empty((nk, 2 * nx, 2), dtype=np.complex128)
    exact_pair = np.empty((nk, 2), dtype=float)
    tracked_e = np.empty((nk, nlow), dtype=float)
    tracked_v = np.empty((nk, 2 * nx, nlow), dtype=np.complex128)
    tracked_assignment = np.empty((nk, nlow), dtype=int)
    tracked_overlap = np.full((nk, nlow), np.nan, dtype=float)
    for ki, k in enumerate(ks):
        h = momentum_hamiltonian(model, k, construction)
        evals, evecs = np.linalg.eigh(h)
        selected = np.argsort(np.abs(evals))[:nlow]
        selected = selected[np.argsort(evals[selected])]
        low_e[ki] = evals[selected]
        low_v[ki] = evecs[:, selected]
        if ki == 0:
            branch_indices = selected
            branch_vectors = evecs[:, branch_indices]
        else:
            overlap_abs = np.abs(tracked_v[ki - 1].conj().T @ evecs)
            rows, cols = linear_sum_assignment(-overlap_abs)
            branch_indices = np.empty(nlow, dtype=int)
            branch_indices[rows] = cols
            branch_vectors = evecs[:, branch_indices]
            diagonal_overlap = np.einsum(
                "ij,ij->j", tracked_v[ki - 1].conj(), branch_vectors
            )
            phases = np.where(
                np.abs(diagonal_overlap) > 0,
                np.exp(-1j * np.angle(diagonal_overlap)),
                1.0,
            )
            branch_vectors = branch_vectors * phases[None, :]
            tracked_overlap[ki] = np.abs(diagonal_overlap)
        tracked_e[ki] = evals[branch_indices]
        tracked_v[ki] = branch_vectors
        tracked_assignment[ki] = branch_indices
        pair = np.argsort(np.abs(evals))[:2]
        exact_pair[ki] = np.sort(evals[pair])
        energies, centers, profiles, vectors = localize_low_doublet(h, nx, walls)
        wall_e[ki] = energies
        wall_centers[ki] = centers
        wall_profiles[ki] = profiles
        localized_vectors[ki] = vectors

    edge_rows: list[dict[str, Any]] = []
    crossings, velocities = [], []
    for wi in range(2):
        own_weight = np.asarray(
            [
                wall_profiles[ki, wi, wall_columns(walls[wi], nx, int(config["wall_window_columns"]))].sum()
                for ki in range(nk)
            ]
        )
        eligible = own_weight >= float(config["wall_weight_min"])
        idx0 = int(np.flatnonzero(eligible)[np.argmin(np.abs(wall_e[eligible, wi]))]) if np.any(eligible) else int(np.argmin(np.abs(wall_e[:, wi])))
        k_guess = float(ks[idx0])
        dk = (ks - k_guess + np.pi) % (2 * np.pi) - np.pi
        use = np.abs(dk) <= float(config["edge_linear_window"])
        fit = linear_fit(dk[use], wall_e[use, wi])
        k0 = (k_guess - fit["intercept"] / fit["slope"] + np.pi) % (2 * np.pi) - np.pi
        crossings.append(k0)
        velocities.append(fit["slope"])
        edge_rows.append(
            {
                "wall_index": wi,
                "wall_x": walls[wi],
                "k0": k0,
                "velocity": fit["slope"],
                "velocity_stderr": float(np.sqrt(max(0.0, fit["covariance"][0, 0]))),
                "linear_r2": fit["r2"],
                "center_at_nearest_k": wall_centers[idx0, wi],
            }
        )

    k0_mean = float(np.angle(np.mean(np.exp(1j * np.asarray(crossings)))))
    dk = (ks - k0_mean + np.pi) % (2 * np.pi) - np.pi
    use = np.abs(dk) <= float(config["dirac_fit_window"])
    xdata = np.repeat(dk[use], 2)
    ydata = np.abs(exact_pair[use]).reshape(-1)

    def dirac_model(delta: np.ndarray, speed: float, mass: float, shift: float) -> np.ndarray:
        return np.sqrt((speed * (delta - shift)) ** 2 + mass**2)

    p0 = [float(np.mean(np.abs(velocities))), float(np.min(ydata)), 0.0]
    try:
        popt, pcov = curve_fit(
            dirac_model,
            xdata,
            ydata,
            p0=p0,
            bounds=([0.0, 0.0, -0.2], [np.inf, np.inf, 0.2]),
            maxfev=20000,
        )
    except Exception:
        popt = np.asarray(p0, dtype=float)
        pcov = np.full((3, 3), np.nan)
    residuals = ydata - dirac_model(xdata, *popt)

    xi_values = []
    localization_fit = np.full((2, nx), np.nan, dtype=float)
    localization_profiles = np.full((2, nx), np.nan, dtype=float)
    localization_k_values = np.full((2, 2), np.nan, dtype=float)
    for wi, (wall, k0) in enumerate(zip(walls, crossings)):
        offset = float(config["localization_k_offset"])
        profile_samples = []
        for oi, signed_offset in enumerate((-offset, offset)):
            kval = (k0 + signed_offset + np.pi) % (2 * np.pi) - np.pi
            localization_k_values[wi, oi] = kval
            _, _, profiles_at_k, _ = localize_low_doublet(
                momentum_hamiltonian(model, kval, construction), nx, walls
            )
            profile_samples.append(profiles_at_k[wi])
        profile = np.mean(profile_samples, axis=0)
        localization_profiles[wi] = profile
        # Fit against integer lattice distance from the declared interface.  The
        # exact k=0 state is compact at the fine-tuned alpha_top=1 point, so the
        # fit uses symmetric offsets within the declared linear edge window.
        distances = cyclic_distance(np.arange(nx), wall, nx)
        other_distance = cyclic_distance(walls[1 - wi], wall, nx)
        max_distance = max(1, int(np.floor(other_distance / 2)))
        radial_distance = np.arange(1, max_distance + 1, dtype=float)
        radial_profile = np.asarray(
            [np.sum(profile[np.isclose(distances, value)]) for value in radial_distance]
        )
        use_radial = radial_profile > max(float(np.max(radial_profile)) * 1e-14, 1e-300)
        if np.count_nonzero(use_radial) >= 2:
            fit = linear_fit(
                radial_distance[use_radial], np.log(radial_profile[use_radial])
            )
            xi = -2.0 / fit["slope"] if fit["slope"] < 0 else float("inf")
            localization_fit[wi] = np.exp(fit["intercept"] + fit["slope"] * distances)
        else:
            xi = float("nan")
        xi_values.append(xi)
        edge_rows[wi]["localization_length"] = xi

    speed_abs = float(abs(popt[0]))
    mass = float(abs(popt[1]))
    level_spacing = 2 * np.pi * speed_abs / model.Ny
    ratio = mass / level_spacing if level_spacing > 0 else float("inf")
    scalars = {
        "dirac_velocity": speed_abs,
        "hybridization_mass": mass,
        "width_ratio": ratio,
        "level_spacing": level_spacing,
        "k0": float((k0_mean + popt[2] + np.pi) % (2 * np.pi) - np.pi),
        "velocity_wall_0": float(velocities[0]),
        "velocity_wall_1": float(velocities[1]),
        "xi_wall_0": float(xi_values[0]),
        "xi_wall_1": float(xi_values[1]),
    }
    payload = {
        "spectrum_k": ks,
        "spectrum_low_energies": low_e,
        "spectrum_low_vectors": low_v,
        "spectrum_tracked_energies": tracked_e,
        "spectrum_tracked_vectors": tracked_v,
        "spectrum_branch_assignments": tracked_assignment,
        "spectrum_branch_overlaps": tracked_overlap,
        "wall_branch_energies": wall_e,
        "wall_branch_centers": wall_centers,
        "wall_branch_profiles": wall_profiles,
        "wall_localized_vectors": localized_vectors,
        "dirac_exact_pair": exact_pair,
        "dirac_fit_parameters": np.asarray(popt),
        "dirac_fit_covariance": np.asarray(pcov),
        "dirac_fit_residuals": residuals,
        "localization_profile_fits": localization_fit,
        "localization_profile_fit_inputs": localization_profiles,
        "localization_k_values": localization_k_values,
    }
    return payload, edge_rows, scalars


def information_score(y: np.ndarray, pred: np.ndarray, nparams: int) -> tuple[float, float]:
    n = int(np.size(y))
    rss = max(float(np.sum((np.asarray(y) - np.asarray(pred)) ** 2)), 1e-300)
    aic = n * np.log(rss / n) + 2 * nparams
    aicc = aic + (2 * nparams * (nparams + 1) / (n - nparams - 1)) if n > nparams + 1 else np.inf
    bic = n * np.log(rss / n) + nparams * np.log(n)
    return float(aicc), float(bic)


def correlator_observables(
    cdisp: np.ndarray, model: classA_U1FGTN, config: dict[str, Any]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    nx, ny = model.Nx, model.Ny
    rvals = np.arange(1, ny // 2 + 1)
    cg = np.empty((nx, rvals.size), dtype=float)
    for ri, r in enumerate(rvals):
        gblock = 2.0 * cdisp[(-int(r)) % ny]
        for x in range(nx):
            block = gblock[2 * x : 2 * x + 2, 2 * x : 2 * x + 2]
            cg[x, ri] = 0.5 * float(np.sum(np.abs(block) ** 2))
    walls = sorted(int(v) for v in model.DW_loc)
    windows = [wall_columns(w, nx, int(config["wall_window_columns"])) for w in walls]
    derived = np.vstack([cg[w] for w in walls] + [cg[cols].sum(axis=0) for cols in windows] + [cg.mean(axis=0)])
    labels = ["wall_0", "wall_1", "wall_window_0", "wall_window_1", "full_x_mean"]
    rows: list[dict[str, Any]] = []

    def power(x: np.ndarray, amp: float, beta: float) -> np.ndarray:
        return amp * ((ny / np.pi) * np.sin(np.pi * x / ny)) ** (-beta)

    def exponential(x: np.ndarray, amp: float, xi: float) -> np.ndarray:
        return amp * np.exp(-x / xi)

    def power_floor(x: np.ndarray, amp: float, beta: float, floor: float) -> np.ndarray:
        return power(x, amp, beta) + floor

    def exponential_floor(x: np.ndarray, amp: float, xi: float, floor: float) -> np.ndarray:
        return exponential(x, amp, xi) + floor

    models = {
        "chord_power": (power, [1e-2, 2.0], ([0, 0], [np.inf, 10])),
        "exponential": (exponential, [1e-2, 3.0], ([0, 1e-6], [np.inf, np.inf])),
        "chord_power_plus_floor": (power_floor, [1e-2, 2.0, 0.0], ([0, 0, 0], [np.inf, 10, np.inf])),
        "exponential_plus_floor": (exponential_floor, [1e-2, 3.0, 0.0], ([0, 1e-6, 0], [np.inf, np.inf, np.inf])),
    }
    for curve_name, curve in zip(labels, derived):
        for shift in config["correlator_endpoint_shifts"]:
            lo = max(1, int(config["correlator_fit_min"]) + int(shift))
            hi = min(int(config["correlator_fit_max"]), ny // 2)
            use = (rvals >= lo) & (rvals <= hi)
            x, y = rvals[use].astype(float), curve[use]
            if x.size < 4:
                continue
            for name, (func, p0, bounds) in models.items():
                try:
                    popt, pcov = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=20000)
                    pred = func(x, *popt)
                    status = "ok"
                except Exception as exc:
                    popt = np.full(len(p0), np.nan)
                    pcov = np.full((len(p0), len(p0)), np.nan)
                    pred = np.full_like(y, np.nan)
                    status = f"failed:{type(exc).__name__}"
                aicc, bic = information_score(y, pred, len(p0)) if np.all(np.isfinite(pred)) else (np.inf, np.inf)
                denom = float(np.sum((y - np.mean(y)) ** 2))
                rss = float(np.sum((y - pred) ** 2)) if np.all(np.isfinite(pred)) else np.inf
                rows.append(
                    {
                        "curve": curve_name,
                        "model": name,
                        "endpoint_shift": int(shift),
                        "r_min": lo,
                        "r_max": hi,
                        "n_points": int(x.size),
                        "parameter_0": float(popt[0]),
                        "parameter_1": float(popt[1]),
                        "parameter_2": float(popt[2]) if len(popt) == 3 else "",
                        "stderr_0": float(np.sqrt(max(0, pcov[0, 0]))) if np.isfinite(pcov[0, 0]) else "",
                        "stderr_1": float(np.sqrt(max(0, pcov[1, 1]))) if np.isfinite(pcov[1, 1]) else "",
                        "r2": 1 - rss / denom if denom > 0 and np.isfinite(rss) else "",
                        "aicc": aicc,
                        "bic": bic,
                        "status": status,
                    }
                )
    return {
        "correlator_r": rvals,
        "correlator_cg_xr": cg,
        "correlator_derived": derived,
        "correlator_derived_labels": np.asarray(labels, dtype="U32"),
    }, rows


def apply_block_function(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    vector_real: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Apply exp(-i H t) to a real-y vector using prediagonalized momentum blocks."""
    vk = np.fft.fft(vector_real, axis=0)
    coeff = np.einsum("kij,kj->ki", eigvecs.conj().transpose(0, 2, 1), vk, optimize=True)
    out = np.empty((times.size,) + vector_real.shape, dtype=np.complex128)
    for ti, time in enumerate(times):
        evolved_k = np.einsum(
            "kij,kj->ki", eigvecs, coeff * np.exp(-1j * eigvals * time), optimize=True
        )
        out[ti] = np.fft.ifft(evolved_k, axis=0)
    return out


def response_velocity(
    profile_ty: np.ndarray, times: np.ndarray, method: str = "peak"
) -> tuple[float, np.ndarray, dict[str, Any]]:
    ny = profile_ty.shape[1]
    displacement = (np.arange(ny) + ny // 2) % ny - ny // 2
    weight = np.abs(profile_ty)
    if method == "peak":
        centers = displacement[np.argmax(weight, axis=1)].astype(float)
        centers = np.unwrap(2 * np.pi * centers / ny) * ny / (2 * np.pi)
    elif method == "absolute_center_of_mass":
        denom = weight.sum(axis=1)
        centers = np.divide(
            weight @ displacement, denom, out=np.zeros_like(denom), where=denom > 0
        )
    else:
        raise ValueError(f"Unknown wavefront center method {method!r}")
    use = (times >= max(2.0, times[1] if times.size > 1 else 0.0)) & (
        times <= times[-1] * 0.75
    )
    if np.count_nonzero(use) >= 3:
        fit = linear_fit(times[use], centers[use])
    else:
        fit = {
            "slope": float("nan"),
            "intercept": float("nan"),
            "r2": float("nan"),
            "n_points": int(np.count_nonzero(use)),
            "residuals": np.empty(0),
            "predicted": np.empty(0),
            "covariance": np.full((2, 2), np.nan),
        }
    predicted_full = np.full(times.shape, np.nan, dtype=float)
    predicted_full[use] = fit["intercept"] + fit["slope"] * times[use]
    fit["predicted_full"] = predicted_full
    fit["fit_mask"] = use
    return float(fit["slope"]), centers, fit


def physical_response(
    model: classA_U1FGTN,
    construction: str,
    projectors: np.ndarray,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    nx, ny = model.Nx, model.Ny
    nblock = 2 * nx
    ks = 2 * np.pi * np.arange(ny) / ny + float(config["occupation_twist"]) / ny
    eigvals, eigvecs = [], []
    for k in ks:
        values, vectors = np.linalg.eigh(momentum_hamiltonian(model, k, construction))
        eigvals.append(values)
        eigvecs.append(vectors)
    eigvals, eigvecs = np.asarray(eigvals), np.asarray(eigvecs)
    horizon = float(config["response_horizon_factor"]) * ny
    times = np.arange(0.0, horizon + 0.5 * float(config["response_dt"]), float(config["response_dt"]))
    source_ys = np.floor(np.linspace(0, ny, int(config["response_source_count"]), endpoint=False)).astype(int)
    walls = sorted(int(v) for v in model.DW_loc)
    epsilon = float(config["response_epsilon"])
    sinc = np.sin(epsilon) / epsilon
    samples = np.empty((2, source_ys.size, times.size, nx, ny), dtype=float)
    aligned = np.empty_like(samples)
    for source_wall, source_x in enumerate(walls):
        for si, sy in enumerate(source_ys):
            chi = np.zeros((times.size, ny, nblock), dtype=float)
            for mu in (0, 1):
                source = np.zeros((ny, nblock), dtype=np.complex128)
                source[sy, 2 * source_x + mu] = 1.0
                source_k = np.fft.fft(source, axis=0)
                csource = np.fft.ifft(
                    np.einsum("kij,kj->ki", projectors, source_k, optimize=True), axis=0
                )
                evolved_c = apply_block_function(eigvals, eigvecs, csource, times)
                evolved_s = apply_block_function(eigvals, eigvecs, source, times)
                chi += -2.0 * sinc * np.imag(evolved_c * evolved_s.conj())
            cell = chi.reshape(times.size, ny, nx, 2).sum(axis=3).transpose(0, 2, 1)
            samples[source_wall, si] = cell
            aligned[source_wall, si] = np.roll(cell, -sy, axis=2)
    mean_aligned = np.mean(aligned, axis=1)
    rows: list[dict[str, Any]] = []
    centers_all = []
    predicted_all = []
    for wi, wall in enumerate(walls):
        cols = wall_columns(wall, nx, int(config["wall_window_columns"]))
        profile = mean_aligned[wi][:, cols, :].sum(axis=1)
        velocity, centers, fit = response_velocity(profile, times, method="peak")
        centers_all.append(centers)
        predicted_all.append(fit["predicted_full"])
        rows.append(
            {
                "wall_index": wi,
                "wall_x": wall,
                "physical_velocity": velocity,
                "wavefront_intercept": fit["intercept"],
                "wavefront_r2": fit["r2"],
                "wavefront_n_points": fit["n_points"],
                "wavefront_velocity_stderr": float(np.sqrt(max(0.0, fit["covariance"][0, 0])))
                if np.isfinite(fit["covariance"][0, 0])
                else float("nan"),
            }
        )
    wall_profiles = np.asarray(
        [
            mean_aligned[wi][
                :, wall_columns(wall, nx, int(config["wall_window_columns"])), :
            ].sum(axis=1)
            for wi, wall in enumerate(walls)
        ]
    )
    chi_kw = np.fft.fftshift(np.fft.fft2(wall_profiles, axes=(1, 2)), axes=(1, 2))
    omega = np.fft.fftshift(np.fft.fftfreq(times.size, d=float(config["response_dt"]))) * 2 * np.pi
    momentum = np.fft.fftshift(np.fft.fftfreq(ny)) * 2 * np.pi
    multipliers = np.asarray(config["response_epsilon_multipliers"], dtype=float)
    epsilon_scales = np.sin(epsilon * multipliers) / (epsilon * multipliers)
    reference = epsilon_scales[np.argmin(np.abs(multipliers - 1.0))]
    convergence = np.abs(epsilon_scales / reference - 1.0)
    return {
        "response_times": times,
        "response_source_y": source_ys,
        "response_chi_samples": samples,
        "response_chi_aligned_mean": mean_aligned,
        "response_wall_centers": np.asarray(centers_all),
        "response_wavefront_fit": np.asarray(predicted_all),
        "response_chi_kw": chi_kw,
        "response_momentum": momentum,
        "response_omega": omega,
        "response_epsilon_multipliers": multipliers,
        "response_epsilon_relative_errors": convergence,
        "response_central_difference_prefactor": np.asarray(sinc),
        "response_kick_convention": np.asarray("exact_plus_minus_epsilon_density_phase_central_difference"),
        "response_wavefront_center_method": np.asarray("absolute_response_peak"),
        "response_charge_drift": np.asarray(np.max(np.abs(samples.sum(axis=(-2, -1))))),
    }, rows


def _modular_source(
    nx: int,
    ay: int,
    columns: Iterable[int],
    endpoints: Iterable[int],
    *,
    normalize: bool,
) -> np.ndarray:
    source = np.zeros(2 * int(nx) * int(ay), dtype=np.complex128)
    for endpoint in endpoints:
        for x in columns:
            start = int(endpoint) * 2 * int(nx) + 2 * int(x)
            source[start : start + 2] += 1.0
    if normalize:
        norm = float(np.linalg.norm(source))
        if norm == 0:
            raise ValueError("Cannot normalize an empty modular source")
        source /= norm
    return source


def _fit_modular_drift(
    times: np.ndarray, drift: np.ndarray, window: Iterable[float]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    lo, hi = (float(value) for value in window)
    mask = (times >= lo - 1e-12) & (times <= hi + 1e-12)
    if np.count_nonzero(mask) < 3:
        raise ValueError(f"Modular fit window {(lo, hi)} contains fewer than three samples")
    fit = linear_fit(times[mask], drift[mask])
    predicted = np.full(times.shape, np.nan, dtype=float)
    residual = np.full(times.shape, np.nan, dtype=float)
    predicted[mask] = fit["intercept"] + fit["slope"] * times[mask]
    residual[mask] = drift[mask] - predicted[mask]
    return fit, predicted, residual


def modular_packet(
    cdisp: np.ndarray,
    model: classA_U1FGTN,
    construction: str,
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Legacy-informed endpoint packets and a symmetry-cancelling handedness fit."""
    nx, ny, ay = model.Nx, model.Ny, model.Ny // 2
    c_half = strip_correlation(cdisp, ay)
    _, occupations, vectors = modular_hamiltonian_from_correlation(
        c_half, float(config["modular_eigenvalue_clip"])
    )
    modular_eigs = np.log((1.0 - occupations) / occupations)
    dt = float(config["modular_dt"])
    times = dt * np.arange(int(round(float(config["modular_time_max"]) / dt)) + 1)
    walls = sorted(int(value) for value in model.DW_loc)
    endpoints = np.asarray([0, ay - 1], dtype=int)
    widths = np.asarray(config["modular_source_widths"], dtype=int)
    sources: list[np.ndarray] = []
    source_index: list[tuple[int, int, int]] = []
    for width_index, width in enumerate(widths):
        for wall_index, wall in enumerate(walls):
            columns = wall_columns(wall, nx, int(width))
            for endpoint_index, endpoint in enumerate(endpoints):
                sources.append(
                    _modular_source(nx, ay, columns, [int(endpoint)], normalize=True)
                )
                source_index.append((width_index, wall_index, endpoint_index))
    source_matrix = np.column_stack(sources)
    coefficients = vectors.conj().T @ source_matrix
    packets = np.empty(
        (widths.size, len(walls), endpoints.size, times.size, nx, ay), dtype=float
    )
    norms = np.empty((widths.size, len(walls), endpoints.size, times.size), dtype=float)
    for time_index, time in enumerate(times):
        states = vectors @ (coefficients * np.exp(-1j * modular_eigs * time)[:, None])
        for source_column, (width_index, wall_index, endpoint_index) in enumerate(source_index):
            state = states[:, source_column]
            norms[width_index, wall_index, endpoint_index, time_index] = float(
                np.vdot(state, state).real
            )
            packets[width_index, wall_index, endpoint_index, time_index] = (
                np.abs(state) ** 2
            ).reshape(ay, nx, 2).sum(axis=2).T

    retention = np.empty((widths.size, len(walls), endpoints.size, times.size), dtype=float)
    centers = np.empty_like(retention)
    y_coordinates = np.arange(ay, dtype=float)
    for width_index, width in enumerate(widths):
        for wall_index, wall in enumerate(walls):
            columns = wall_columns(wall, nx, int(width))
            for endpoint_index in range(endpoints.size):
                profile = packets[width_index, wall_index, endpoint_index][:, columns, :].sum(axis=1)
                retained = profile.sum(axis=1)
                retention[width_index, wall_index, endpoint_index] = retained
                centers[width_index, wall_index, endpoint_index] = np.divide(
                    profile @ y_coordinates,
                    retained,
                    out=np.full(times.shape, np.nan, dtype=float),
                    where=retained > 1e-14,
                )
    handedness = 0.5 * (centers[:, :, 0] + centers[:, :, 1] - float(ay - 1))

    fit_windows = [list(config["modular_primary_fit_window"])] + [
        list(window) for window in config["modular_sensitivity_fit_windows"]
    ]
    fit_predictions = np.full(
        (widths.size, len(walls), len(fit_windows), times.size), np.nan, dtype=float
    )
    fit_residuals = np.full_like(fit_predictions, np.nan)
    fit_covariances = np.full((widths.size, len(walls), len(fit_windows), 2, 2), np.nan)
    rows: list[dict[str, Any]] = []
    for width_index, width in enumerate(widths):
        for wall_index, wall in enumerate(walls):
            for fit_index, window in enumerate(fit_windows):
                fit, predicted, residual = _fit_modular_drift(
                    times, handedness[width_index, wall_index], window
                )
                fit_predictions[width_index, wall_index, fit_index] = predicted
                fit_residuals[width_index, wall_index, fit_index] = residual
                fit_covariances[width_index, wall_index, fit_index] = fit["covariance"]
                mask = np.isfinite(predicted)
                rows.append(
                    {
                        "wall_index": wall_index,
                        "wall_x": wall,
                        "source_width": int(width),
                        "fit_window_index": fit_index,
                        "fit_window_min": float(window[0]),
                        "fit_window_max": float(window[1]),
                        "is_primary": bool(
                            int(width) == int(config["modular_primary_source_width"])
                            and fit_index == 0
                        ),
                        "modular_velocity": float(fit["slope"]),
                        "fit_intercept": float(fit["intercept"]),
                        "fit_r2": float(fit["r2"]),
                        "fit_n_points": int(fit["n_points"]),
                        "velocity_stderr": float(np.sqrt(max(0.0, fit["covariance"][0, 0]))),
                        "minimum_wall_retention": float(
                            np.min(retention[width_index, wall_index, :, mask])
                        ),
                    }
                )

    primary_width_index = int(
        np.flatnonzero(widths == int(config["modular_primary_source_width"]))[0]
    )
    primary_rows = [row for row in rows if row["is_primary"]]
    primary_signs = np.sign([row["modular_velocity"] for row in primary_rows]).astype(int)
    sign_stable = []
    for wall_index in range(len(walls)):
        wall_signs = np.sign(
            [row["modular_velocity"] for row in rows if row["wall_index"] == wall_index]
        ).astype(int)
        sign_stable.append(bool(np.all(wall_signs == primary_signs[wall_index])))
    norm_drift = float(np.max(np.abs(norms - norms[..., :1])))
    primary_mask = (
        times >= float(config["modular_primary_fit_window"][0]) - 1e-12
    ) & (times <= float(config["modular_primary_fit_window"][1]) + 1e-12)
    primary_min_retention = float(
        np.min(retention[primary_width_index, :, :, primary_mask])
    )
    diagnostics = {
        "covariance_mode": "translated_half_window_average_exact_by_y_translation",
        "primary_source_width": int(config["modular_primary_source_width"]),
        "primary_velocities": [float(row["modular_velocity"]) for row in primary_rows],
        "primary_signs": primary_signs.tolist(),
        "sign_stable": sign_stable,
        "primary_minimum_wall_retention": primary_min_retention,
        "max_norm_drift": norm_drift,
        "pass": bool(
            len(primary_rows) == 2
            and primary_rows[0]["modular_velocity"] * primary_rows[1]["modular_velocity"] < 0
            and all(sign_stable)
            and primary_min_retention >= float(config["modular_primary_retention_min"])
            and norm_drift <= float(config["modular_norm_drift_max"])
        ),
    }

    payload: dict[str, np.ndarray] = {
        "modular_covariance_mode": np.asarray(diagnostics["covariance_mode"]),
        "modular_times": times,
        "modular_occupations": occupations,
        "modular_eigenvalues": modular_eigs,
        "modular_source_widths": widths,
        "modular_endpoints": endpoints,
        "modular_packets": packets,
        "modular_norms": norms,
        "modular_wall_retention": retention,
        "modular_endpoint_centers": centers,
        "modular_handedness": handedness,
        "modular_fit_windows": np.asarray(fit_windows, dtype=float),
        "modular_fit_predictions": fit_predictions,
        "modular_fit_residuals": fit_residuals,
        "modular_fit_covariances": fit_covariances,
        "modular_primary_width_index": np.asarray(primary_width_index),
        "modular_primary_minimum_wall_retention": np.asarray(primary_min_retention),
        "modular_max_norm_drift": np.asarray(norm_drift),
    }

    if (
        construction == "coupled"
        and nx == int(config["legacy_modular_nx"])
        and ny == int(config["legacy_modular_ny"])
    ):
        legacy_dt = float(config["legacy_modular_dt"])
        legacy_times = legacy_dt * np.arange(
            int(round(float(config["legacy_modular_time_max"]) / legacy_dt)) + 1
        )
        aggregate_sources = []
        for width in widths:
            columns = np.concatenate(
                [wall_columns(wall, nx, int(width)) for wall in walls]
            )
            aggregate_sources.append(
                _modular_source(nx, ay, columns, endpoints, normalize=False)
            )
        aggregate_matrix = np.column_stack(aggregate_sources)
        aggregate_coefficients = vectors.conj().T @ aggregate_matrix
        aggregate_packets = np.empty(
            (widths.size, legacy_times.size, nx, ay), dtype=float
        )
        aggregate_norms = np.empty((widths.size, legacy_times.size), dtype=float)
        for time_index, time in enumerate(legacy_times):
            states = vectors @ (
                aggregate_coefficients * np.exp(-1j * modular_eigs * time)[:, None]
            )
            for width_index in range(widths.size):
                state = states[:, width_index]
                aggregate_norms[width_index, time_index] = float(np.vdot(state, state).real)
                aggregate_packets[width_index, time_index] = (
                    np.abs(state) ** 2
                ).reshape(ay, nx, 2).sum(axis=2).T
        payload.update(
            {
                "legacy_aggregate_times": legacy_times,
                "legacy_aggregate_source_widths": widths,
                "legacy_aggregate_packets": aggregate_packets,
                "legacy_aggregate_norms": aggregate_norms,
            }
        )
    return payload, rows, diagnostics


def twist_observables(
    model: classA_U1FGTN,
    construction: str,
    config: dict[str, Any],
    points: int | None = None,
    sign: int = 1,
    save_links: bool = True,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    nx, ny = model.Nx, model.Ny
    points = int(config["twist_points"] if points is None else points)
    base = float(config["occupation_twist"])
    phis = np.linspace(0.0, sign * 2 * np.pi, points)
    frames_all = np.empty((points, ny, 2 * nx, nx), dtype=np.complex128)
    spectra = np.empty((points, ny, 2 * nx), dtype=float)
    wall_energy = np.empty((points, ny, 2), dtype=float)
    wall_weight = np.empty((points, ny, 2, 2), dtype=float)
    branch_assignments = np.full((points, ny, nx), -1, dtype=int)
    walls = sorted(int(v) for v in model.DW_loc)
    for pi, phi in enumerate(phis):
        for n in range(ny):
            k = 2 * np.pi * n / ny + (base + phi) / ny
            if pi == 0:
                frame, _, _ = occupied_frame(model, k, construction)
                _, candidates = declared_complete_eigenbasis(model, k, construction)
                rows, cols = linear_sum_assignment(-np.abs(frame.conj().T @ candidates))
                seed_order = np.empty(nx, dtype=int)
                seed_order[rows] = cols
                branch_assignments[pi, n] = seed_order
            else:
                _, candidates = declared_complete_eigenbasis(model, k, construction)
                previous = frames_all[pi - 1, n]
                overlap_abs = np.abs(previous.conj().T @ candidates)
                rows, cols = linear_sum_assignment(-overlap_abs)
                order = np.empty(nx, dtype=int)
                order[rows] = cols
                branch_assignments[pi, n] = order
                selected = candidates[:, order]
                overlap = previous.conj().T @ selected
                unitary, _ = polar(overlap)
                frame = selected @ unitary.conj().T
            values = np.linalg.eigvalsh(momentum_hamiltonian(model, k, construction))
            frames_all[pi, n] = frame
            spectra[pi, n] = values
            energies, _, profiles, _ = localize_low_doublet(
                momentum_hamiltonian(model, k, construction), nx, walls
            )
            wall_energy[pi, n] = energies
            for target in range(2):
                for source_wall, wall in enumerate(walls):
                    cols = wall_columns(wall, nx, int(config["wall_window_columns"]))
                    wall_weight[pi, n, target, source_wall] = profiles[target, cols].sum()
    singular = np.empty((points - 1, ny, nx), dtype=float)
    determinant_phase = np.empty((points - 1, ny), dtype=float)
    overlap_matrices = np.empty((points - 1, ny, nx, nx), dtype=np.complex128)
    polar_links = np.empty((points - 1, ny, nx, nx), dtype=np.complex128) if save_links else np.empty((0, 0, 0, 0), dtype=np.complex128)
    for pi in range(points - 1):
        for n in range(ny):
            overlap = frames_all[pi, n].conj().T @ frames_all[pi + 1, n]
            overlap_matrices[pi, n] = overlap
            unitary, _ = polar(overlap)
            svals = np.linalg.svd(overlap, compute_uv=False)
            singular[pi, n] = svals
            determinant_phase[pi, n] = np.angle(np.linalg.det(unitary))
            if save_links:
                polar_links[pi, n] = unitary
    flow_rows = []
    for wi in range(2):
        crossings = []
        for pi in range(points - 1):
            before, after = wall_energy[pi, :, wi], wall_energy[pi + 1, :, wi]
            weight_before = wall_weight[pi, :, wi, wi]
            weight_after = wall_weight[pi + 1, :, wi, wi]
            mask = (
                ((before * after < 0) | (np.abs(before) < 1e-13) | (np.abs(after) < 1e-13))
                & (np.maximum(np.abs(before), np.abs(after)) <= float(config["twist_crossing_energy_max"]))
                & (np.minimum(weight_before, weight_after) >= float(config["wall_weight_min"]))
            )
            for b, a in zip(before[mask], after[mask]):
                if a != b:
                    crossings.append(int(np.sign(a - b)))
        signed = int(np.sum(crossings))
        flow_rows.append(
            {
                "wall_index": wi,
                "wall_x": walls[wi],
                "twist_sign": int(sign),
                "twist_points": points,
                "signed_flow": signed,
                "absolute_flow": int(np.sum(np.abs(crossings))),
                "crossing_events": len(crossings),
                "min_overlap_singular_value": float(np.min(singular)),
            }
        )
    payload = {
        "twist_phi": phis,
        "twist_energies": spectra,
        "twist_uniform_gauge_energies": spectra,
        "twist_seam_gauge_energies": spectra.copy(),
        "twist_seam_energy_method": np.asarray(
            "unitary_gauge_transform_from_uniform_family; independent seam checks saved in preflight and accepted-size validation"
        ),
        "twist_wall_branch_energies": wall_energy,
        "twist_wall_weights": wall_weight,
        "twist_overlap_singular_values": singular,
        "twist_overlap_matrices": overlap_matrices,
        "twist_polar_links": polar_links,
        "twist_polar_determinant_phase": determinant_phase,
        "twist_frame_rank": np.asarray(nx),
        "twist_branch_assignments": branch_assignments,
        "twist_seam_positions": np.asarray(config["twist_seam_positions"], dtype=int),
        "twist_uniform_gauge_shift": (base + phis) / ny,
    }
    return payload, flow_rows


def preflight_checks(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    nx, ny = int(config["preflight_nx"]), int(config["preflight_ny"])
    tol = float(config["numerical_tolerance"])
    rows: dict[str, Any] = {"nx": nx, "ny": ny, "tolerance": tol, "checks": {}}
    arrays: dict[str, np.ndarray] = {}
    for construction in config["constructions"]:
        model = make_model(nx, ny, config)
        canonical = canonical_hamiltonian(model, construction)
        reconstructed = momentum_reconstruct_real(model, construction)
        h_error = float(np.max(np.abs(canonical - reconstructed)))
        hermiticity = float(np.max(np.abs(canonical - canonical.conj().T)))
        _, _, blocks = occupied_blocks(model, construction, float(config["occupation_twist"]))
        cdisp = correlation_displacements(blocks, float(config["occupation_twist"]))
        cfull = full_covariance_from_displacements(cdisp)
        projector_error = float(np.max(np.abs(cfull @ cfull - cfull)))
        trace_error = float(abs(np.trace(cfull).real - nx * ny))
        fixed_half = strip_correlation(cdisp, ny // 2)
        translated_half = translated_half_window_average(cfull, nx, ny)
        translated_half_error = float(np.max(np.abs(fixed_half - translated_half)))
        hmod, modular_occ, modular_vec = modular_hamiltonian_from_correlation(
            translated_half, float(config["modular_eigenvalue_clip"])
        )
        legacy_g = 2.0 * modular_occ - 1.0
        legacy_energy = -2.0 * np.arctanh(legacy_g)
        legacy_hmod = (modular_vec * legacy_energy[None, :]) @ modular_vec.conj().T
        modular_convention_error = float(np.max(np.abs(hmod - legacy_hmod)))
        entropy = entropy_observables(cdisp, model, config)
        contour_error = float(entropy.payload["contour_sum_max_error"])
        phi = 0.731
        seam0 = np.linalg.eigvalsh(seam_twisted_hamiltonian(model, construction, phi, 0))
        seam1 = np.linalg.eigvalsh(seam_twisted_hamiltonian(model, construction, phi, 1))
        uniform = np.linalg.eigvalsh(momentum_reconstruct_real(model, construction, phi))
        gauge_error = float(max(np.max(np.abs(seam0 - uniform)), np.max(np.abs(seam1 - uniform))))
        closure = float(
            np.max(
                np.abs(
                    np.linalg.eigvalsh(momentum_reconstruct_real(model, construction, 0.0))
                    - np.linalg.eigvalsh(momentum_reconstruct_real(model, construction, 2 * np.pi))
                )
            )
        )
        checks = {
            "momentum_real_max_error": h_error,
            "hermiticity_max_error": hermiticity,
            "projector_idempotency_max_error": projector_error,
            "half_filling_trace_error": trace_error,
            "translated_half_window_error": translated_half_error,
            "modular_convention_error": modular_convention_error,
            "entropy_contour_sum_error": contour_error,
            "seam_uniform_spectrum_error": gauge_error,
            "twist_closure_spectrum_error": closure,
        }
        checks["pass"] = bool(
            all(
                value
                < (
                    float(config["modular_convention_tolerance"])
                    if key == "modular_convention_error"
                    else tol
                )
                for key, value in checks.items()
                if key != "pass"
            )
        )
        rows["checks"][construction] = checks
        arrays[f"{construction}_canonical_hamiltonian"] = canonical
        arrays[f"{construction}_reconstructed_hamiltonian"] = reconstructed
        arrays[f"{construction}_projector"] = cfull
        arrays[f"{construction}_translated_half_correlation"] = translated_half
        arrays[f"{construction}_modular_hamiltonian"] = hmod
        arrays[f"{construction}_seam_spectrum"] = seam0
        arrays[f"{construction}_uniform_spectrum"] = uniform
    rows["pass"] = bool(all(item["pass"] for item in rows["checks"].values()))
    return rows, arrays


def geometry_key(construction: str, nx: int, ny: int) -> str:
    return f"{construction}__Nx{int(nx):03d}__Ny{int(ny):03d}"


def run_geometry(
    construction: str,
    nx: int,
    ny: int,
    config: dict[str, Any],
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    model = make_model(nx, ny, config)
    started = utc_now()
    spectral_payload, edge_rows, edge_scalars = spectral_observables(model, construction, config)
    frames, block_spectra, projectors = occupied_blocks(
        model, construction, float(config["occupation_twist"])
    )
    cdisp = correlation_displacements(projectors, float(config["occupation_twist"]))
    entropy = entropy_observables(cdisp, model, config)
    corr_payload, corr_rows = correlator_observables(cdisp, model, config)
    response_payload, response_rows = physical_response(model, construction, projectors, config)
    modular_payload, modular_rows, modular_diagnostics = modular_packet(
        cdisp, model, construction, config
    )
    for wi, row in enumerate(response_rows):
        spectral_velocity = float(edge_rows[wi]["velocity"])
        row["spectral_velocity"] = spectral_velocity
        row["absolute_velocity_ratio_to_spectral"] = abs(float(row["physical_velocity"])) / max(
            abs(spectral_velocity), 1e-300
        )
    for row in modular_rows:
        wi = int(row["wall_index"])
        row["spectral_velocity"] = float(edge_rows[wi]["velocity"])
        row["physical_velocity"] = float(response_rows[wi]["physical_velocity"])
    twist_payload, flow_rows = twist_observables(model, construction, config)
    h = canonical_hamiltonian(model, construction)
    hermiticity = float(np.max(np.abs(h - h.conj().T)))
    idempotency = float(np.max(np.abs(projectors @ projectors - projectors)))
    occupation_count = int(round(sum(np.trace(p).real for p in projectors)))
    summary = {
        "schema_version": 1,
        "key": geometry_key(construction, nx, ny),
        "construction": construction,
        "nx": int(nx),
        "ny": int(ny),
        "started_utc": started,
        "completed_utc": utc_now(),
        "dtype": "complex128",
        "canonical_source": "classA_U1FGTN._domain_wall_hamiltonian",
        "canonical_source_sha256": sha256_file(FGTN_SRC / "classA_U1FGTN.py"),
        "wall_positions": sorted(int(v) for v in model.DW_loc),
        "mass_profile": np.real(model.alpha_profile[:, 0]).tolist(),
        "occupation_count": occupation_count,
        "target_occupation_count": int(nx * ny),
        "hermiticity_max_error": hermiticity,
        "projector_idempotency_max_error": idempotency,
        "edge": edge_scalars,
        "edge_rows": edge_rows,
        "entropy_rows": entropy.rows,
        "correlator_rows": corr_rows,
        "response_rows": response_rows,
        "modular_rows": modular_rows,
        "modular_diagnostics": modular_diagnostics,
        "flow_rows": flow_rows,
        "contour_sum_max_error": float(entropy.payload["contour_sum_max_error"]),
        "response_charge_drift": float(response_payload["response_charge_drift"]),
        "response_epsilon_max_relative_error": float(
            np.max(response_payload["response_epsilon_relative_errors"])
        ),
        "pass_fixed_rank": occupation_count == nx * ny,
        "pass_numerics": bool(
            max(hermiticity, idempotency, float(entropy.payload["contour_sum_max_error"]))
            < float(config["numerical_tolerance"])
        ),
    }
    arrays: dict[str, Any] = {
        "schema_version": np.asarray(1),
        "construction": np.asarray(construction),
        "nx": np.asarray(nx),
        "ny": np.asarray(ny),
        "mass_profile": np.real(model.alpha_profile),
        "wall_positions": np.asarray(summary["wall_positions"]),
        "occupation_count": np.asarray(occupation_count),
        "block_spectra": block_spectra,
        "occupied_frames": frames,
        "occupied_projectors": projectors,
        "correlation_displacements": cdisp,
        **spectral_payload,
        **entropy.payload,
        **corr_payload,
        **response_payload,
        **modular_payload,
        **twist_payload,
    }
    atomic_npz(output_path, **arrays)
    summary["npz_sha256"] = sha256_file(output_path)
    atomic_json(summary_path, summary)
    return summary


def run_twist_validation(
    model: classA_U1FGTN,
    construction: str,
    config: dict[str, Any],
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    payload: dict[str, np.ndarray] = {"schema_version": np.asarray(1)}
    rows: list[dict[str, Any]] = []
    for points in config["twist_validation_points"]:
        for sign in config["twist_signs"]:
            item, item_rows = twist_observables(
                model, construction, config, points=int(points), sign=int(sign), save_links=True
            )
            prefix = f"p{int(points)}_s{'plus' if int(sign) > 0 else 'minus'}__"
            payload.update({prefix + key: value for key, value in item.items()})
            rows.extend(item_rows)
    phi = 0.913
    seam_spectra = []
    for seam in config["twist_seam_positions"]:
        seam_spectra.append(np.linalg.eigvalsh(seam_twisted_hamiltonian(model, construction, phi, int(seam))))
    uniform = np.linalg.eigvalsh(momentum_reconstruct_real(model, construction, phi))
    payload["seam_validation_phi"] = np.asarray(phi)
    payload["seam_validation_spectra"] = np.asarray(seam_spectra)
    payload["uniform_validation_spectrum"] = uniform
    gauge_error = float(np.max(np.abs(np.asarray(seam_spectra) - uniform[None, :])))
    atomic_npz(output_path, **payload)
    summary = {
        "schema_version": 1,
        "construction": construction,
        "nx": model.Nx,
        "ny": model.Ny,
        "rows": rows,
        "seam_uniform_spectrum_max_error": gauge_error,
        "pass_gauge": gauge_error < float(config["numerical_tolerance"]),
        "npz_sha256": sha256_file(output_path),
        "completed_utc": utc_now(),
    }
    atomic_json(summary_path, summary)
    return summary
