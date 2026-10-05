"""Verified, compact endpoint loading and trajectory-first correlator fits."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import analyze_finite_size_power_law as prior

ROOT = Path(__file__).resolve().parents[4]
PROJECT = Path(__file__).resolve().parent
BUNDLE = ROOT / "00_WORKSPACE/CURRENT/final_production_new_designs/14_hard_wall_xresolved_correlator_scaling"
REVISION = "hard_wall_xresolved_nx20_ny40-50-60_a1-1_nsh1_s100_2ny_raster_endpoint_frame_halfcov_occupations_v2_30gib_batched"
LARGE_ROOT = BUNDLE / "gpu_data" / REVISION
SIZES = (24, 28, 32, 40, 50, 60)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path, role):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": sha256(path), "role": role}


@dataclass
class Endpoint:
    ny: int
    cohort: str
    ids: np.ndarray
    xavg: np.ndarray
    xresolved: np.ndarray | None
    provenance: list


def preparation_identity():
    """Read the checksum-bound configuration without importing a GPU runner."""
    manifest = json.loads((LARGE_ROOT / "DOWNLOAD_MANIFEST.json").read_text())
    hashes = manifest["validation"]["source_hashes"]
    for name, expected in hashes.items():
        if sha256(BUNDLE / name) != expected:
            raise ValueError(f"Cannot resolve preparation: archived source differs: {name}")
    tree = ast.parse((BUNDLE / "run_campaign.py").read_text())
    namespace = {}
    function = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("EXPECTED_") or name == "GPU_MEMORY_HARD_LIMIT_GIB":
                namespace[name] = ast.literal_eval(node.value)
        elif isinstance(node, ast.FunctionDef) and node.name == "expected_config":
            function = node
    if function is None:
        raise ValueError("No checksum-bound configuration function")
    # Execute only the pure configuration-returning function, never runner code.
    function.returns = None
    exec(compile(ast.Module(body=[function], type_ignores=[]), "config_metadata", "exec"), namespace)
    config = namespace["expected_config"]()
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    if digest != manifest["configuration_sha256"]:
        raise ValueError("Reconstructed configuration does not match the archived identity")
    for key, expected in {"DW": True, "domain_wall_interval": [5, 15], "init_mode": "default",
                          "filling_frac": 0.5, "trial_orbitals": "X", "postselect": False,
                          "state_representation": "physical_frame"}.items():
        if config["protocol"][key] != expected:
            raise ValueError(f"Preparation mismatch: {key}")
    return {"configuration_sha256": digest, "source_hashes": hashes, "config": config,
            "evidence": [file_record(LARGE_ROOT / "DOWNLOAD_MANIFEST.json", "import_manifest")]
            + [file_record(BUNDLE / name, "preparation_source") for name in hashes]}


def verify_pair(path, completion):
    if completion.get("status") != "complete" or completion.get("result_filename") != path.name:
        raise ValueError(f"Incomplete or incompatible result pair: {path.name}")
    if path.stat().st_size != completion.get("result_bytes") or sha256(path) != completion.get("result_sha256"):
        raise ValueError(f"Result checksum/size mismatch: {path.name}")


def extract_endpoint(data, ny, samples, every_cycle):
    """Access only compact members: never frame/covariance/spectrum members."""
    expected = {"Nx": 20, "Ny": ny, "alpha_1": 1., "alpha_2": 30., "nshell": 1,
                "dtype": "complex128", "sequence": "raster_y", "perfect_correction": True,
                "dw_truncation": True, "meas_slab_only": True,
                "canonical_entry_point": "classA_U1FGTN_gpu.run_markov_circuit"}
    for key, value in expected.items():
        if np.asarray(data[key]).item() != value:
            raise ValueError(f"Ny={ny}: incompatible {key}")
    if every_cycle and (np.asarray(data["init_mode"]).item() != "default"
                        or np.asarray(data["construction"]).item() != "hard"):
        raise ValueError("Smaller ensemble preparation mismatch")
    cycles = np.asarray(data["cycles"])
    np.testing.assert_array_equal(cycles, np.arange(2 * ny + 1) if every_cycle else [2 * ny])
    np.testing.assert_array_equal(data["ry_values"], np.arange(ny // 2 + 1))
    np.testing.assert_array_equal(data["x_values"], np.arange(20))
    np.testing.assert_array_equal(data["dw_location" if every_cycle else "wall_locations"], [5, 15])
    ids = np.asarray(data["global_sample_indices"], dtype=int)
    x = np.asarray(data["x_resolved_square_correlator"], dtype=float)
    avg = np.asarray(data["xavg_square_correlator_vs_ry"], dtype=float)
    if x.shape != (samples, len(cycles), 20, ny // 2 + 1) or avg.shape != (samples, len(cycles), ny // 2 + 1):
        raise ValueError("Unexpected correlator shape")
    np.testing.assert_allclose(x.mean(axis=2), avg, rtol=2e-14, atol=1e-15)
    if ids.shape != (samples,) or np.unique(ids).size != samples:
        raise ValueError("Invalid sample indices")
    return ids.copy(), avg[:, -1].copy(), x[:, -1].copy()


def assemble(ny, cohort, pieces, provenance):
    ids = np.concatenate([p[0] for p in pieces])
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.arange(100)):
        raise ValueError(f"Ny={ny}: expected exactly 100 unique sample indices 0..99")
    return Endpoint(ny, cohort, ids[order], np.concatenate([p[1] for p in pieces])[order],
                    np.concatenate([p[2] for p in pieces])[order], provenance)


def load_endpoint(ny, identity=None):
    small = ny in (24, 28, 32)
    directory = (prior.NEW_ROOT / f"Ny{ny:03}/alpha1_1/nshell_1" if small
                 else LARGE_ROOT / f"results/Ny{ny:03}")
    paths = sorted(directory.glob("*.npz"))
    if len(paths) != (4 if small else 20) or len(list(directory.glob("*.complete.json"))) != len(paths):
        raise ValueError(f"Ny={ny}: missing/extra result pairs")
    pieces, provenance = [], []
    for path in tqdm(paths, desc=f"verify Ny={ny}", unit="shard", leave=False):
        receipt_path = path.with_suffix(".complete.json")
        receipt = json.loads(receipt_path.read_text())
        verify_pair(path, receipt)
        for key, value in {"Nx": 20, "Ny": ny, "cycles": 2 * ny,
                           "canonical_entry_point": "classA_U1FGTN_gpu.run_markov_circuit"}.items():
            if receipt.get(key) != value:
                raise ValueError(f"Completion contract mismatch: {key}")
        if not small:
            if identity is None or any(receipt.get(k) != identity[k] for k in ("source_hashes", "configuration_sha256")):
                raise ValueError("Unresolved preparation/source identity")
            if receipt.get("sampling_revision") != REVISION:
                raise ValueError("Sampling revision mismatch")
        with np.load(path, allow_pickle=False) as data:
            if not small:
                if str(data["configuration_sha256"].item()) != identity["configuration_sha256"] or str(data["sampling_revision"].item()) != REVISION:
                    raise ValueError("NPZ campaign identity mismatch")
                np.testing.assert_array_equal(data["global_sample_indices"], receipt["global_sample_indices"])
            piece = extract_endpoint(data, ny, 25 if small else 5, small)
        pieces.append(piece)
        provenance.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                           "sha256": receipt["result_sha256"], "role": "verified_endpoint"})
        provenance.append(file_record(receipt_path, "completion"))
    return assemble(ny, "all_cycle" if small else "large_endpoint", pieces, provenance)


def observables(endpoint):
    x = endpoint.xresolved
    result = {"xavg": endpoint.xavg}
    if x is not None:
        result.update(left=x[:, 5], right=x[:, 15], walls=(x[:, 5] + x[:, 15]) / 2,
                      pair_left=(x[:, 5] + x[:, 6]) / 2, pair_right=(x[:, 14] + x[:, 15]) / 2,
                      pairs=(x[:, 5] + x[:, 6] + x[:, 14] + x[:, 15]) / 4)
    return result


def fit(curve, ny, lo, hi, cutoff=0., coordinate="chord"):
    r = np.arange(len(curve))
    window = (r >= max(1, lo)) & (r <= min(ny // 2, hi))
    keep = window & np.isfinite(curve) & (curve > cutoff)
    n = int(keep.sum())
    result = {"beta": None, "log_amplitude": None, "r_squared": None, "n_points": n,
              "excluded_points": int(window.sum()) - n, "window_points": int(window.sum()),
              "valid": False, "reason": "fewer_than_four_valid_points"}
    if n < 4:
        return result
    x = np.log(prior.chord(ny, r[keep]) if coordinate == "chord" else r[keep])
    y = np.log(np.asarray(curve)[keep])
    slope, intercept = np.polyfit(x, y, 1)
    residual = np.sum((y - (slope * x + intercept))**2)
    variance = np.sum((y - y.mean())**2)
    return dict(result, beta=float(-slope), log_amplitude=float(intercept),
                r_squared=float(1 - residual / variance) if variance > 0 else None,
                valid=True, reason="ok")


def windows(ny):
    result = {"primary": (2, ny // 4), "fixed_short": (2, 8), "full_half": (2, ny // 2),
              "legacy_tail": (5, ny // 2), "outer_tail": ((ny + 3) // 4, ny // 2)}
    for lo in (2, 3, 4, 6):
        for denominator in (4, 3, 2):
            result[f"grid_r{lo}_N{denominator}"] = (lo, ny // denominator)
    return result


def distribution(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    if not len(values):
        return {"n": 0, "mean": None, "median": None, "sd": None, "q25": None, "q75": None,
                "q16": None, "q84": None, "q025": None, "q975": None}
    q = np.quantile(values, [.025, .16, .25, .5, .75, .84, .975])
    return dict(n=len(values), mean=float(values.mean()), sd=float(values.std(ddof=1)) if len(values)>1 else None,
                **dict(zip(("q025", "q16", "q25", "median", "q75", "q84", "q975"), map(float, q))))


def typical(endpoint):
    betas = [fit(c, endpoint.ny, 2, 8, coordinate="raw")["beta"] for c in endpoint.xavg]
    if any(b is None for b in betas):
        raise ValueError("Representative selection has invalid historical-window fits")
    median = float(np.median(betas))
    position = int(np.argmin(np.abs(np.asarray(betas) - median)))
    return {"position": position, "sample_id": int(endpoint.ids[position]),
            "selected_beta": betas[position], "median_beta": median,
            "selection": "nearest median raw-r exponent on 2..8; tie broken by sorted sample ID"}
