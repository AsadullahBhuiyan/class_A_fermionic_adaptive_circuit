from __future__ import annotations

import concurrent.futures
import json
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
BUNDLE = REPO / "00_WORKSPACE" / "CURRENT" / "matched_markov_lindblad_campaign"
if str(BUNDLE) not in sys.path:
    sys.path.insert(0, str(BUNDLE))

from campaign_schema import case_requires_response, load_config
from markov_adapter import run_markov_channel_case
from matched_model import build_model
from observables import enrich_terminal_arrays, translation_residual
from run_campaign import (
    AdapterResult,
    _parse_cpu_range,
    _smoke_cases,
    _initialize_worker,
    _run_one_worker,
    _source_hashes,
    _validate_adapter_result,
    _validate_complete_case,
)
import run_campaign as campaign_runner


@pytest.fixture(scope="module")
def config() -> dict:
    payload, _ = load_config(BUNDLE / "campaign_config.v1.json")
    return payload


@pytest.fixture(scope="module")
def markov_case(config: dict) -> dict:
    return next(
        case
        for case in _smoke_cases(config)
        if case["dynamics"]["family"] == "markov_channel"
        and case["model"]["dw_truncation"]
    )


@pytest.fixture(scope="module")
def markov_result(config: dict, markov_case: dict):
    return run_markov_channel_case(markov_case, config)


def test_smoke_matrix_covers_discrete_and_continuous_arms(config: dict) -> None:
    cases = _smoke_cases(config)
    signatures = {
        (
            case["dynamics"]["family"],
            case["dynamics"]["dephasing"],
            case["model"]["dw_truncation"],
        )
        for case in cases
    }
    assert len(cases) == 6
    assert signatures == {
        ("markov_channel", True, True),
        ("markov_channel", True, False),
        ("lindblad", False, True),
        ("lindblad", True, True),
        ("lindblad", False, False),
        ("lindblad", True, False),
    }
    assert {case["dynamics"]["sample_seeds"][0] for case in cases} == {20260814}
    assert all(case["dynamics"]["cycles"] == 2 * case["model"]["Ny"] for case in cases)


def test_markov_adapter_sparse_contract_and_random_words(
    config: dict, markov_case: dict, markov_result
) -> None:
    wrapped = AdapterResult(
        arrays=dict(markov_result.arrays), metadata=dict(markov_result.metadata)
    )
    _validate_adapter_result(markov_case, wrapped, config)
    arrays = markov_result.arrays
    assert arrays["translation_residual"].shape == (1, 13)
    assert arrays["spectral_checkpoint_cycle"].tolist() == [0, 3, 6, 9, 12]
    assert arrays["half_occupation_gap"].shape == (1, 5)
    assert arrays["G_final"].shape == (1, 48, 48)
    assert arrays["schedule_site_ids"].shape == (1, 12, 24)
    expected_sites = np.arange(24)
    for word in arrays["schedule_site_ids"][0]:
        assert np.array_equal(np.sort(word), expected_sites)
    assert markov_result.metadata["channel_order"] == ["Ap", "Am", "Bp", "Bm"]
    assert markov_result.metadata["late_cycle_window_inclusive"] == [7, 12]
    assert markov_result.metadata["response_enabled"] is True
    assert case_requires_response(markov_case, config)


def test_markov_schedule_is_reproducible_from_declared_seed(
    markov_case: dict, markov_result
) -> None:
    nx, ny = markov_case["model"]["Nx"], markov_case["model"]["Ny"]
    coords = [(x, y) for x in range(nx) for y in range(ny)]
    rng = np.random.default_rng(markov_case["dynamics"]["sample_seeds"][0])
    expected = []
    for _ in range(markov_case["dynamics"]["cycles"]):
        word = list(coords)
        rng.shuffle(word)
        expected.append([x + nx * y for x, y in word])
    assert np.array_equal(
        markov_result.arrays["schedule_site_ids"][0], np.asarray(expected)
    )


def test_streamed_late_average_matches_full_history(
    config: dict, markov_case: dict, markov_result
) -> None:
    model = build_model(markov_case)
    raw = model.run_markov_channel(
        G_history=True,
        progress=False,
        cycles=markov_case["dynamics"]["cycles"],
        init_mode="maxmix",
        save=False,
        n_a=float(config["model"]["n_a_metadata"]),
        sequence="random",
        decoh=True,
        perfect_correction=True,
        schedule_seed=markov_case["dynamics"]["sample_seeds"][0],
    )["G_hist"]
    identity = np.eye(raw.shape[-1])
    expected = 0.5 * (np.mean(raw[7:13], axis=0) + identity)
    assert np.allclose(markov_result.arrays["G_late_cycle_average"][0], expected)


def test_terminal_enrichment_uses_an_exact_y_twirl(
    markov_case: dict, markov_result
) -> None:
    enriched = enrich_terminal_arrays(markov_case, markov_result.arrays)
    twirled = enriched["G_late_cycle_average_twirl"][0]
    assert translation_residual(twirled, 4, 6) < 1e-14
    assert enriched["twirled_ky_natural_occupations"].shape == (1, 6, 8)
    assert np.all(enriched["twirled_ky_natural_occupations"] >= -1e-10)
    assert np.all(enriched["twirled_ky_natural_occupations"] <= 1.0 + 1e-10)


def test_atomic_case_output_is_immutable_and_hash_validated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: dict,
    markov_case: dict,
    markov_result,
) -> None:
    wrapped = AdapterResult(
        arrays=dict(markov_result.arrays), metadata=dict(markov_result.metadata)
    )
    monkeypatch.setattr(
        campaign_runner, "run_markov_channel_case", lambda case, cfg: wrapped
    )
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    receipt = campaign_runner._run_one(markov_case, config, cases_dir)
    reconstructed = _validate_complete_case(
        markov_case, cases_dir / markov_case["case_id"], config, receipt
    )
    assert reconstructed == receipt
    with np.load(cases_dir / markov_case["case_id"] / "observables.npz") as archive:
        assert translation_residual(
            archive["G_late_cycle_average_twirl"][0], 4, 6
        ) < 1e-14
    with pytest.raises(FileExistsError):
        campaign_runner._run_one(markov_case, config, cases_dir)


def test_spawned_runner_pins_one_worker_and_writes_a_receipt(
    tmp_path: Path, config: dict, markov_case: dict
) -> None:
    cpu = min(os.sched_getaffinity(0))
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    context = multiprocessing.get_context("spawn")
    cpu_queue = context.Queue()
    cpu_queue.put(cpu)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=1,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(cpu_queue, 1),
    ) as executor:
        receipt = executor.submit(
            _run_one_worker, markov_case, config, cases_dir
        ).result(timeout=60)
    cpu_queue.close()
    assert receipt["status"] == "complete"
    metadata = json.loads(
        (cases_dir / markov_case["case_id"] / "metadata.json").read_text()
    )
    assert metadata["execution"]["cpu"] == cpu
    assert metadata["execution"]["affinity"] == [cpu]
    assert metadata["execution"]["blas_threads"] == 1


def test_source_hash_manifest_covers_every_numerical_layer(config: dict) -> None:
    hashes = _source_hashes(BUNDLE / "campaign_config.v1.json")
    assert {
        "runner",
        "config",
        "campaign_schema",
        "dynamics_adapters",
        "markov_adapter",
        "lindblad_adapter",
        "lindblad_response",
        "matched_model",
        "observables",
        "analysis",
        "canonical_cpu_class",
        "canonical_mean_lindblad",
        "lindblad_reference",
    } == set(hashes)
    assert all(value is None or len(value) == 64 for value in hashes.values())


def test_response_scope_excludes_schedule_control(config: dict, markov_case: dict) -> None:
    control = dict(markov_case)
    control["campaign_roles"] = ["channel_schedule_seed_control"]
    assert not case_requires_response(control, config)


@pytest.mark.parametrize(
    ("specification", "expected"),
    [("0-2,5,7-8", [0, 1, 2, 5, 7, 8]), ("4,4,3", [4, 3])],
)
def test_cpu_range_parser(specification: str, expected: list[int]) -> None:
    assert _parse_cpu_range(specification) == expected


@pytest.mark.parametrize("specification", ["", "3-1", "-1", "1-2-3"])
def test_cpu_range_parser_rejects_malformed_input(specification: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        _parse_cpu_range(specification)
