from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from fgtn.classA_U1FGTN import classA_U1FGTN


def _model(**overrides):
    kwargs = {
        "Nx": 4,
        "Ny": 4,
        "DW": True,
        "nshell": 1,
        "alpha_1": 1,
        "alpha_2": 30,
        "trial_orbitals": "X",
        "dw_truncation": True,
        "dw_interval": (1, 2),
    }
    kwargs.update(overrides)
    model = classA_U1FGTN(**kwargs)
    model.construct_OW_projectors(
        nshell=model.nshell,
        DW=model.DW,
        trial_orbitals=model.trial_orbitals,
        dw_truncation=model.dw_truncation,
    )
    return model


def test_explicit_dw_interval_preserves_default_when_omitted():
    # Nx=20 distinguishes the GPU-production Nx//4 rule from the superseded
    # CPU-only Nx//3 default.
    default = classA_U1FGTN(Nx=20, Ny=3, DW=True, nshell=0)
    assert default.DW_loc == [5, 15]
    assert default.dw_interval is None
    assert default.DW_slab_half_width_rule == "max(1, Nx // 4)"
    assert default.DW_slab_half_width == 5
    assert default.DW_slab_width_sites == 11

    explicit = classA_U1FGTN(
        Nx=8,
        Ny=3,
        DW=True,
        nshell=0,
        alpha_1=1,
        alpha_2=30,
        dw_interval=(3, 5),
    )
    assert explicit.DW_loc == [3, 5]
    assert explicit.dw_interval == (3, 5)
    assert explicit.DW_slab_half_width_rule == "explicit_inclusive_interval"
    np.testing.assert_allclose(explicit.alpha_profile[3:6], 1.0)
    np.testing.assert_allclose(explicit.alpha_profile[:3], 30.0)
    np.testing.assert_allclose(explicit.alpha_profile[6:], 30.0)


@pytest.mark.parametrize("nx", [8, 12, 16, 20, 24])
def test_default_domain_wall_interval_matches_gpu_engine(nx):
    gpu_module = pytest.importorskip("fgtn.classA_U1FGTN_gpu")
    cpu = classA_U1FGTN(Nx=nx, Ny=2, DW=True, nshell=0)
    gpu = gpu_module.classA_U1FGTN_gpu(
        Nx=nx,
        Ny=2,
        DW=True,
        nshell=0,
        device="cpu",
        dtype="complex128",
        backend="local",
    )

    assert cpu.DW_loc == gpu.DW_loc
    assert cpu.DW_slab_width_sites == gpu.DW_loc[1] - gpu.DW_loc[0] + 1


@pytest.mark.parametrize(
    "interval",
    [(-1, 2), (2, 8), (4, 3), (1,), (1, 2, 3), (1.5, 3), "1,3"],
)
def test_explicit_dw_interval_validation(interval):
    with pytest.raises(ValueError):
        classA_U1FGTN(Nx=8, Ny=3, DW=True, nshell=0, dw_interval=interval)

    if interval == (1,):
        with pytest.raises(ValueError, match="requires DW=True"):
            classA_U1FGTN(Nx=8, Ny=3, DW=False, nshell=0, dw_interval=(1, 3))


def test_perfect_channel_uses_canonical_within_cell_order_and_local_support():
    model = _model()
    result = model.run_markov_channel(
        G_history=False,
        progress=False,
        cycles=1,
        init_mode="maxmix",
        save=False,
        sequence="raster_y",
        decoh=True,
        perfect_correction=True,
    )

    expected = np.zeros_like(result["G_final"])
    identity = np.eye(expected.shape[0], dtype=np.complex128)
    channel_word = (
        ("WF_Ap", -1.0),
        ("WF_Am", 1.0),
        ("WF_Bp", -1.0),
        ("WF_Bm", 1.0),
    )
    for x in range(model.Nx):
        for y in range(model.Ny):
            for projector_name, target_covariance in channel_word:
                chi = getattr(model, projector_name)[:, x, y]
                projector = np.outer(chi, chi.conj())
                complement = identity - projector
                expected = (
                    complement @ expected @ complement
                    + target_covariance * projector
                )

    np.testing.assert_allclose(result["G_final"], expected, atol=1e-13, rtol=1e-13)
    assert result["run_config"]["channel_order"] == ["Ap", "Am", "Bp", "Bm"]
    occupations = np.linalg.eigvalsh(
        0.5 * (result["G_final"] + identity)
    )
    assert occupations.min() >= -1e-12
    assert occupations.max() <= 1.0 + 1e-12


def test_random_schedule_seed_and_selected_cycle_observer_are_reproducible():
    def run(seed):
        observed = []
        model = _model(Nx=3, Ny=3, DW=False, dw_truncation=False, dw_interval=None)

        def observer(**payload):
            observed.append(
                (payload["cycle"], payload["ordered_site_ids"].copy())
            )

        result = model.run_markov_channel(
            G_history=False,
            progress=False,
            cycles=3,
            init_mode="maxmix",
            save=False,
            sequence="random",
            decoh=True,
            perfect_correction=True,
            schedule_seed=seed,
            cycle_observer=observer,
            cycle_observer_cycles=(0, 2),
        )
        return result, observed

    first, first_observed = run(20260814)
    second, second_observed = run(20260814)
    different, different_observed = run(20260815)

    np.testing.assert_allclose(
        first["G_final"], second["G_final"], atol=2e-15, rtol=2e-15
    )
    assert [cycle for cycle, _ in first_observed] == [0, 2]
    assert [cycle for cycle, _ in second_observed] == [0, 2]
    for left, right in zip(first_observed, second_observed):
        np.testing.assert_array_equal(left[1], right[1])
    assert first_observed[0][1].size == 0
    assert not np.array_equal(first_observed[1][1], different_observed[1][1])
    assert not np.array_equal(first["G_final"], different["G_final"])


@pytest.mark.parametrize("seed", [-1, 1.25, True])
def test_markov_channel_schedule_seed_validation(seed):
    with pytest.raises(ValueError, match="schedule_seed"):
        _model().run_markov_channel(
            G_history=False,
            progress=False,
            cycles=0,
            init_mode="maxmix",
            save=False,
            schedule_seed=seed,
        )


def test_markov_channel_observer_selection_validation():
    model = _model()
    with pytest.raises(ValueError, match="requires cycle_observer"):
        model.run_markov_channel(
            G_history=False,
            progress=False,
            cycles=1,
            init_mode="maxmix",
            save=False,
            cycle_observer_cycles=(0, 1),
        )
    with pytest.raises(ValueError, match="0..1"):
        model.run_markov_channel(
            G_history=False,
            progress=False,
            cycles=1,
            init_mode="maxmix",
            save=False,
            cycle_observer=lambda **_: None,
            cycle_observer_cycles=(2,),
        )


def test_markov_channel_save_key_and_metadata_distinguish_protocol(tmp_path, monkeypatch):
    module = importlib.import_module("fgtn.classA_U1FGTN")
    monkeypatch.setattr(module.time, "sleep", lambda _: None)

    paths = []
    cases = (
        dict(
            alpha_1=1,
            alpha_2=30,
            trial_orbitals="X",
            dw_truncation=True,
            schedule_seed=11,
            sequence="raster_y",
        ),
        dict(
            alpha_1=2,
            alpha_2=20,
            trial_orbitals="Y",
            dw_truncation=False,
            schedule_seed=12,
            sequence="random",
        ),
    )
    for case in cases:
        model = _model(
            alpha_1=case["alpha_1"],
            alpha_2=case["alpha_2"],
            trial_orbitals=case["trial_orbitals"],
            dw_truncation=case["dw_truncation"],
        )
        monkeypatch.setattr(model, "_g_history_outdir", lambda: str(tmp_path))
        monkeypatch.setattr(model, "_g_history_outdir_rel", lambda: str(tmp_path))
        result = model.run_markov_channel(
            G_history=False,
            progress=False,
            cycles=0,
            init_mode="maxmix",
            save=True,
            n_a=0.5,
            sequence=case["sequence"],
            decoh=True,
            perfect_correction=True,
            schedule_seed=case["schedule_seed"],
        )
        paths.append(Path(result["save_path"]))

    assert paths[0] != paths[1]
    assert all(path.exists() for path in paths)
    assert "_dwtrunc1_" in paths[0].name
    assert "_a11_" in paths[0].name
    assert "_trialX_" in paths[0].name
    assert "_seq-raster_y_seed11_order-Ap-Am-Bp-Bm_" in paths[0].name
    assert "_dwtrunc0_" in paths[1].name
    assert "_a12_" in paths[1].name
    assert "_trialY_" in paths[1].name
    assert "_seq-random_seed12_order-Ap-Am-Bp-Bm_" in paths[1].name

    with np.load(paths[0], allow_pickle=False) as archive:
        config = json.loads(archive["run_config"].item())
    assert config["DW_loc"] == [1, 2]
    assert config["dw_interval"] == [1, 2]
    assert config["dw_truncation"] is True
    assert config["alpha_top"] == 1.0
    assert config["alpha_triv"] == 30.0
    assert config["trial_orbitals"] == "X"
    assert config["sequence"] == "raster_y"
    assert config["schedule_seed"] == 11
    assert config["channel_order"] == ["Ap", "Am", "Bp", "Bm"]
