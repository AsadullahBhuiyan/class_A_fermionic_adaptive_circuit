"""Spectral-only raster-y channel gaps. No covariance/trajectory evolution."""
import os
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np
import scipy
from scipy.linalg import eig
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
REPO = PROJECT.parents[2]
sys.path.insert(0, str(PROJECT))
from matched_model import build_model

SIZES = (20, 24, 28, 32, 36, 40, 44, 50)
ORDER = ("Ap", "Am", "Bp", "Bm")
UNIT_TOL = 1e-10


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def config(alpha, ny):
    return dict(revision="hard_wall_raster_y_spectral_gap_v1", Nx=20, Ny=ny,
                alpha_1=alpha, alpha_2=30, nshell=1, walls=[5, 15],
                DW=True, dw_truncation=True, all_slabs_active=True,
                trial_orbitals="X", twist_y=0, boundary_conditions="periodic_x_y",
                dtype="complex128", perfect_correction=True, dephasing=True,
                sequence="raster_y", channel_order=list(ORDER),
                explicit_dynamics=False, decay_rate_units="inverse_cycle",
                canonical_source="classA_U1FGTN.construct_OW_projectors",
                represented_channel="classA_U1FGTN.run_markov_channel")


def sources():
    paths = [Path(__file__), PROJECT / "matched_model.py",
             REPO / "src/fgtn/classA_U1FGTN.py", REPO / "src/fgtn/occupied_frame.py"]
    return {str(p.relative_to(REPO)): sha(p) for p in paths}


def make_model(cfg):
    return build_model({"model": dict(Nx=cfg["Nx"], Ny=cfg["Ny"],
        domain_wall=True, wall_locations=cfg["walls"], alpha_run_in=cfg["alpha_1"],
        alpha_run_out=30, nshell=1, trial_orbitals="X", dw_truncation=True)})


def raster_word(model):
    coords = list(model._sequence_helper("raster_y")["iter_fn"]())
    expected = [(x, y) for x in range(model.Nx) for y in range(model.Ny)]
    if coords != expected:
        raise ValueError("Canonical raster-y order differs from the declared contract")
    return coords


def construct_product(model, progress=True):
    """Left-multiply the identity by each Q using the canonical compact payload."""
    n = 2 * model.Nx * model.Ny
    result = np.eye(n, dtype=np.complex128)
    max_norm_error = 0.0
    mode_hash = hashlib.sha256()
    for x, y in tqdm(raster_word(model), desc="construct Q product", unit="site", disable=not progress):
        payload = model._get_ow_local_support_data(x, y)
        indices = payload["idx"]
        mode_hash.update(np.asarray([x, y], dtype=np.int64).tobytes())
        mode_hash.update(indices.tobytes())
        for name in ORDER:
            vector = payload[name]
            error = abs(np.vdot(vector, vector).real - 1)
            max_norm_error = max(max_norm_error, float(error))
            if error > 1e-12:
                raise ValueError(f"Unnormalized OW mode {x,y,name}: {error}")
            mode_hash.update(vector.tobytes())
            rows = result[indices, :]
            result[indices, :] = rows - np.outer(vector, vector.conj() @ rows)
    return result, {"maximum_mode_norm_error": max_norm_error,
                    "ordered_mode_sha256": mode_hash.hexdigest()}


def independent_action(model, vectors):
    """Independent full-length WF-vector action; no compact-support helper."""
    out = np.array(vectors, dtype=np.complex128, copy=True)
    for x in range(model.Nx):
        for y in range(model.Ny):
            for name in ORDER:
                w = np.asarray(getattr(model, "WF_" + name)[:, x, y])
                w = w / np.linalg.norm(w)
                out -= np.outer(w, w.conj() @ out)
    return out


def classify_radius(radius):
    if not np.isfinite(radius) or radius < 0 or radius > 1 + UNIT_TOL:
        raise ValueError(f"Noncontractive or invalid spectral radius: {radius}")
    if abs(radius - 1) <= UNIT_TOL:
        return "unresolved_unit_modulus"
    if radius == 0:
        return "zero_radius_finite_step_decay"
    return "resolved_positive"


def verified_complete(folder, cfg, source_hashes):
    try:
        receipt = json.loads((folder / "completion.json").read_text())
        if (receipt["config"] != cfg or receipt["sources"] != source_hashes
                or receipt["status"] != "complete" or receipt["result_filename"] != "spectrum.npz"):
            return False
        path = folder / "spectrum.npz"
        if path.stat().st_size != receipt["result_bytes"] or sha(path) != receipt["result_sha256"]:
            return False
        with np.load(path, allow_pickle=False) as data:
            return bool(json.loads(str(data["config_json"])) == cfg
                    and data["eigenvalues"].shape == (2*cfg["Nx"]*cfg["Ny"],)
                    and np.isfinite(data["eigenvalues"]).all())
    except (OSError, KeyError, ValueError, EOFError):
        return False


def run_case(folder, cfg, source_hashes):
    folder.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    print("[configuration]", json.dumps(cfg), flush=True)
    model = make_model(cfg)
    constructed = time.monotonic()
    matrix, checks = construct_product(model)
    product_done = time.monotonic()
    rng = np.random.default_rng(2026092901 + cfg["alpha_1"]*1000 + cfg["Ny"])
    probes = rng.normal(size=(matrix.shape[0], 3)) + 1j*rng.normal(size=(matrix.shape[0], 3))
    probes /= np.linalg.norm(probes, axis=0)
    direct = independent_action(model, probes)
    action_error = float(np.linalg.norm(matrix @ probes - direct))
    if action_error > 1e-11:
        raise ValueError(f"Independent product check failed: {action_error}")
    print(f"[eigensolver start] full complex spectrum: dimension={len(matrix)}", flush=True)
    eigen_start = time.monotonic()
    values, left, right = eig(matrix, left=True, right=True, check_finite=True)
    eigen_seconds = time.monotonic() - eigen_start
    index = int(np.argmax(abs(values)))
    radius = float(abs(values[index]))
    vector = right[:, index] / np.linalg.norm(right[:, index])
    left_vector = left[:, index] / np.linalg.norm(left[:, index])
    residual = float(np.linalg.norm(matrix @ vector - values[index]*vector))
    independent_residual = float(np.linalg.norm(independent_action(model, vector[:, None])[:, 0]
                                                 - values[index]*vector))
    left_residual = float(np.linalg.norm(matrix.conj().T @ left_vector
                                        - values[index].conjugate()*left_vector))
    if max(residual, independent_residual, left_residual) > 1e-10:
        raise ValueError("Dominant eigenpair residual failed")
    status = classify_radius(radius)
    # Preserve raw rates near unity, including tiny negative values; never clip.
    raw_gap = float(-2*np.log(radius)) if radius > 0 else None
    overlap = float(abs(np.vdot(left_vector, vector)))
    details = dict(**checks, independent_product_action_error=action_error,
        dominant_residual=residual, independent_dominant_residual=independent_residual,
        dominant_left_residual=left_residual, dominant_left_right_overlap=overlap,
        spectral_radius=radius, covariance_gap_raw=raw_gap, gap_status=status,
        covariance_multiplier_gap=float(1-radius**2), unit_modulus_tolerance=UNIT_TOL,
        ow_build_seconds=constructed-start, product_seconds=product_done-constructed,
        eigensolver_seconds=eigen_seconds, elapsed_seconds=time.monotonic()-start,
        numpy_version=np.__version__, scipy_version=scipy.__version__,
        cpu_affinity=sorted(os.sched_getaffinity(0)), completed_utc=datetime.now(timezone.utc).isoformat())
    print(f"[eigensolver done] rho={radius:.12g}, gap={raw_gap}, {status}; {eigen_seconds:.1f}s", flush=True)
    temporary = folder / "spectrum.tmp.npz"
    np.savez_compressed(temporary, eigenvalues=values, dominant_eigenvector=vector,
        dominant_left_eigenvector=left_vector, dominant_eigenvalue=values[index],
        spectral_radius=radius, covariance_gap_raw=np.inf if raw_gap is None else raw_gap,
        covariance_multiplier_gap=1-radius**2,
        site_word=np.asarray([x+model.Nx*y for x,y in raster_word(model)], dtype=np.int64),
        config_json=np.asarray(json.dumps(cfg, sort_keys=True)), diagnostics_json=np.asarray(json.dumps(details)))
    with np.load(temporary, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["eigenvalues"], values)
    temporary.replace(folder / "spectrum.npz")
    path = folder / "spectrum.npz"
    receipt = dict(status="complete", config=cfg, sources=source_hashes, diagnostics=details,
                   result_filename=path.name, result_bytes=path.stat().st_size, result_sha256=sha(path))
    atomic_json(folder / "completion.json", receipt)
    assert verified_complete(folder, cfg, source_hashes)
    print(f"[complete] {folder.name}", flush=True)


def worker(args):
    os.sched_setaffinity(0, {args.cpu})
    os.nice(10)
    source_hashes = sources()
    count = 0
    failures = []
    for ny in tqdm(SIZES, desc=f"alpha={args.alpha} cases", unit="case"):
        folder = args.root / f"alpha{args.alpha}_Ny{ny:03d}"
        cfg = config(args.alpha, ny)
        if verified_complete(folder, cfg, source_hashes):
            print(f"[skip verified] {folder.name}", flush=True)
            count += 1
            continue
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "case", "--root", str(args.root),
                   "--alpha", str(args.alpha), "--ny", str(ny), "--cpu", str(args.cpu)]
        with (args.root / f"{folder.name}.log").open("a", buffering=1) as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode == 0 and verified_complete(folder, cfg, source_hashes):
            count += 1
        else:
            failures.append(folder.name)
        print(f"[inventory] complete={count}, failed={len(failures)}, pending={8-count-len(failures)}", flush=True)
    atomic_json(args.root / f"worker_alpha{args.alpha}.json", dict(complete=count, failures=failures))
    return int(bool(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "worker", "case", "report"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--cpus", default="0,7")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--alpha", type=int, choices=(1,3))
    parser.add_argument("--ny", type=int, choices=SIZES)
    args = parser.parse_args()
    if args.mode == "launch":
        cpus = [int(x) for x in args.cpus.split(",")]
        if len(set(cpus)) != 2 or len(cpus) != 2 or not set(cpus) <= os.sched_getaffinity(0):
            parser.error("Provide two distinct available CPU IDs")
        if args.root is None:
            args.root = HERE / "results" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.root = args.root.resolve()
        args.root.mkdir(parents=True, exist_ok=True)
        jobs = []
        for alpha, cpu in zip((1,3), cpus):
            session = f"spectralgap_a{alpha}_{args.root.name}"
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--root", str(args.root),
                       "--alpha", str(alpha), "--cpu", str(cpu)]
            shell = "set -o pipefail; " + shlex.join(command) + " 2>&1 | tee -a " + shlex.quote(str(args.root/f"worker_alpha{alpha}.log"))
            subprocess.run(["tmux", "new-session", "-d", "-s", session, "-c", str(HERE), "bash", "-lc", shell], check=True)
            jobs.append(dict(alpha=alpha, cpu=cpu, session=session, command=command))
        atomic_json(args.root / "launch.json", dict(jobs=jobs, sources=sources(), explicit_dynamics=False))
        print(json.dumps(dict(root=str(args.root), jobs=jobs), indent=2))
        return 0
    if args.root is None:
        parser.error("--root is required")
    if args.mode == "report":
        rows = [dict(alpha=a, Ny=ny, complete=verified_complete(args.root/f"alpha{a}_Ny{ny:03d}", config(a,ny), sources()))
                for a in (1,3) for ny in SIZES]
        print(json.dumps(dict(complete=sum(r["complete"] for r in rows), total=16, cases=rows), indent=2))
        return 0
    if args.cpu is None or args.alpha is None:
        parser.error("--cpu and --alpha are required")
    if args.mode == "worker":
        return worker(args)
    if args.ny is None:
        parser.error("--ny is required")
    os.sched_setaffinity(0, {args.cpu})
    folder = args.root / f"alpha{args.alpha}_Ny{args.ny:03d}"
    try:
        run_case(folder, config(args.alpha,args.ny), sources())
    except Exception:
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / "failure.json", dict(traceback=traceback.format_exc()))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
