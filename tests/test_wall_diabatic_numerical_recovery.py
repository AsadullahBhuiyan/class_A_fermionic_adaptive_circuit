from __future__ import annotations

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = REPO_ROOT / "00_WORKSPACE/CURRENT/frozen_record_flux_charge_pilot"
if str(PILOT_ROOT) not in sys.path:
    sys.path.insert(0, str(PILOT_ROOT))

recovery = importlib.import_module("run_wall_diabatic_numerical_recovery_v2")


def _configured() -> tuple[dict, dict]:
    config, parent = recovery.load_recovery_config(recovery.DEFAULT_CONFIG)
    recovery._install_recovery_numerics(config["numerical_recovery"])
    return config, parent


def test_recovery_is_exactly_the_five_failed_v1_tasks() -> None:
    config, parent = _configured()
    rows = recovery._selected_tasks(parent, config)
    assert len(rows) == 5
    assert len({row["task_id"] for row in rows}) == 5
    assert {row["wall"] for row in rows} == {"hard"}
    assert all(row["recovery_revision"] == config["campaign_id"] for row in rows)
    assert recovery.DEFAULT_OUTPUT.name.endswith("numerical_recovery_v2")


def test_eigensolver_retries_when_numpy_driver_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configured()
    matrix = np.asarray([[0.25, 0.1j], [-0.1j, -0.5]], dtype=np.complex128)
    original_twisted = recovery.base.twisted_parent
    monkeypatch.setattr(recovery.base, "twisted_parent", lambda *args, **kwargs: matrix.copy())
    monkeypatch.setattr(
        recovery.np.linalg, "eigh",
        lambda *args, **kwargs: (_ for _ in ()).throw(np.linalg.LinAlgError("forced")),
    )
    values, vectors = recovery._recovery_parent_eigensystem(
        matrix, np.zeros((2, 2)), 0.3, 2, np.arange(2), "uniform"
    )
    np.testing.assert_allclose(matrix @ vectors, vectors * values[None, :], atol=1e-13)
    monkeypatch.setattr(recovery.base, "twisted_parent", original_twisted)


def test_eigensolver_retries_nonfinite_numpy_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configured()
    matrix = np.asarray([[1.0, 0.2], [0.2, -1.0]], dtype=np.complex128)
    monkeypatch.setattr(recovery.base, "twisted_parent", lambda *args, **kwargs: matrix.copy())
    monkeypatch.setattr(
        recovery.np.linalg, "eigh",
        lambda *args, **kwargs: (
            np.asarray([np.nan, 1.0]), np.eye(2, dtype=np.complex128)
        ),
    )
    values, vectors = recovery._recovery_parent_eigensystem(
        matrix, np.zeros((2, 2)), -0.1, 2, np.arange(2), "uniform"
    )
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(vectors))


def test_polar_stabilization_preserves_subspace_projector() -> None:
    _configured()
    rng = np.random.default_rng(20260913)
    raw = rng.normal(size=(18, 6)) + 1j * rng.normal(size=(18, 6))
    frame, _ = np.linalg.qr(raw)
    mixing = np.eye(6, dtype=np.complex128)
    mixing[0, 1] = 2e-6
    drifted = frame @ mixing
    stabilized, residual = recovery._polar_stabilize(drifted)
    reference = drifted @ np.linalg.inv(drifted.conj().T @ drifted) @ drifted.conj().T
    np.testing.assert_allclose(
        stabilized @ stabilized.conj().T, reference, rtol=0.0, atol=2e-13
    )
    assert residual <= 5e-13
    np.testing.assert_allclose(
        stabilized.conj().T @ stabilized, np.eye(6), rtol=0.0, atol=5e-13
    )
