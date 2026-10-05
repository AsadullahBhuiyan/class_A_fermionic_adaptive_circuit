import numpy as np
import pytest


torch = pytest.importorskip("torch")

from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu


def _model():
    return classA_U1FGTN_gpu(
        Nx=1,
        Ny=2,
        DW=False,
        nshell=None,
        device="cpu",
        dtype="complex128",
        backend="dense",
    )


def _run(observer=None):
    torch.manual_seed(1729)
    return _model().run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        samples=2,
        save=False,
        return_data=True,
        sequence="random",
        perfect_correction=True,
        batch_size=2,
        record_observer=observer,
    )


def test_record_observer_is_diagnostic_only_and_emits_normalized_events():
    baseline = _run()
    callbacks = []
    observed = _run(lambda **payload: callbacks.append(payload))

    np.testing.assert_allclose(observed["G_final"], baseline["G_final"], atol=0.0, rtol=0.0)
    assert callbacks

    seen = set()
    for payload in callbacks:
        assert payload["channel_labels"] == ("Ap", "Am", "Bp", "Bm")
        probabilities = payload["occupation_probability"]
        outcomes = payload["outcome_occupied"]
        targets = payload["target_occupied"]
        realized = payload["realized_probability"]
        log_probability = payload["conditional_log_probability"]
        transfer = payload["transfer"]

        assert probabilities.shape == outcomes.shape == targets.shape
        assert realized.shape == log_probability.shape == transfer.shape
        assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
        assert torch.all((realized >= 0.0) & (realized <= 1.0))
        torch.testing.assert_close(log_probability.exp(), realized)
        torch.testing.assert_close(
            transfer,
            targets.to(torch.int8) - outcomes.to(torch.int8),
        )

        for sample_index, update_index in zip(
            payload["sample_indices"].tolist(),
            [payload["update_index"]] * payload["batch_count"],
        ):
            key = (payload["cycle"], sample_index, update_index)
            assert key not in seen
            seen.add(key)

    assert seen == {
        (cycle, sample, update)
        for cycle in (1, 2)
        for sample in (0, 1)
        for update in (0, 1)
    }


def test_record_observer_rejects_forced_postselection():
    with pytest.raises(ValueError, match="record_observer"):
        _model().run_markov_circuit(
            G_history=False,
            progress=False,
            cycles=1,
            samples=1,
            save=False,
            return_data=False,
            sequence="random",
            perfect_correction=True,
            postselect_probability=0.5,
            batch_size=1,
            record_observer=lambda **_: None,
        )


def test_zero_onsite_noise_observer_is_diagnostic_only():
    torch.manual_seed(991)
    baseline = _model().run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        samples=2,
        save=False,
        return_data=True,
        sequence="random",
        perfect_correction=True,
        batch_size=2,
    )
    rows = []
    torch.manual_seed(991)
    observed = _model().run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=2,
        samples=2,
        save=False,
        return_data=True,
        sequence="random",
        perfect_correction=True,
        batch_size=2,
        onsite_phase_noise_sigma=0.0,
        noise_observer=lambda **payload: rows.append(payload),
    )
    np.testing.assert_allclose(observed["G_final"], baseline["G_final"], atol=0.0, rtol=0.0)
    assert len(rows) == 2
    for row in rows:
        assert torch.count_nonzero(row["theta"]) == 0
        torch.testing.assert_close(row["phase"], torch.ones_like(row["phase"]))


def test_onsite_phase_noise_preserves_covariance_spectrum():
    model = _model()
    torch.manual_seed(18)
    G = model._prepare_initial_batch(batch_size=2, init_mode="default")
    before = torch.linalg.eigvalsh(G)
    noisy, theta, phase = model._apply_onsite_phase_noise(G, sigma=0.3)
    after = torch.linalg.eigvalsh(noisy)
    torch.testing.assert_close(after, before, atol=1e-11, rtol=1e-11)
    assert torch.all(theta >= 0.0)
    assert torch.all(theta < 0.3)
    torch.testing.assert_close(torch.abs(phase), torch.ones_like(theta))


def test_frozen_record_replay_reproduces_final_covariance_and_log_probabilities():
    samples, cycles, sites = 2, 2, 2
    schedule = np.full((samples, cycles, sites), -1, dtype=np.int64)
    outcomes = np.zeros((samples, cycles, sites, 4), dtype=np.bool_)
    original_logs = np.full((samples, cycles, sites, 4), np.nan, dtype=np.float64)

    def capture(**payload):
        sample_indices = payload["sample_indices"].detach().cpu().numpy()
        site_ids = payload["site_ids"].detach().cpu().numpy()
        cycle_index = payload["cycle"] - 1
        update_index = payload["update_index"]
        schedule[sample_indices, cycle_index, update_index] = site_ids
        outcomes[sample_indices, cycle_index, update_index] = (
            payload["outcome_occupied"].detach().cpu().numpy()
        )
        original_logs[sample_indices, cycle_index, update_index] = (
            payload["conditional_log_probability"].detach().cpu().numpy()
        )

    torch.manual_seed(431)
    original = _model().run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=cycles,
        samples=samples,
        save=False,
        return_data=True,
        sequence="random",
        perfect_correction=True,
        batch_size=samples,
        record_observer=capture,
    )
    assert np.all(schedule >= 0)
    replay_logs = np.full_like(original_logs, np.nan)

    def capture_replay(**payload):
        sample_indices = payload["sample_indices"].detach().cpu().numpy()
        replay_logs[
            sample_indices, payload["cycle"] - 1, payload["update_index"]
        ] = payload["conditional_log_probability"].detach().cpu().numpy()

    torch.manual_seed(431)
    replay = _model().run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=cycles,
        samples=samples,
        save=False,
        return_data=True,
        sequence="random",
        perfect_correction=True,
        batch_size=samples,
        frozen_schedule=schedule,
        frozen_outcomes=outcomes,
        record_observer=capture_replay,
    )
    np.testing.assert_allclose(replay["G_final"], original["G_final"], atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(replay_logs, original_logs, atol=1e-11, rtol=1e-11)


def test_seam_gauge_controller_twist_closes_at_two_pi():
    model = _model()
    reference = model.WF_Ap_sites.clone()
    metadata = model.set_controller_twist(2.0 * np.pi, gauge="seam", seam_y=1)
    assert metadata["gauge"] == "seam"
    torch.testing.assert_close(model.WF_Ap_sites, reference, atol=2e-12, rtol=2e-12)
    model.set_controller_twist(0.0, gauge="seam", seam_y=0)
    torch.testing.assert_close(model.WF_Ap_sites, reference, atol=4e-12, rtol=4e-12)


def test_public_mean_replacement_matches_explicit_rank_one_formula():
    model = _model()
    torch.manual_seed(77)
    initial = model._prepare_initial_batch(batch_size=1, init_mode="maxmix")
    coords = model._sequence_helper("raster_y", meas_slab_only=False)["coords_for_len"]
    sites = [int(x + model.Nx * y) for x, y in coords]
    schedule = np.asarray([[sites]], dtype=np.int64)
    observed = {}

    def capture(**payload):
        observed[int(payload["cycle"])] = payload["G"].detach().clone()

    model.run_markov_circuit(
        cycles=1,
        samples=1,
        batch_size=1,
        init_mode="maxmix",
        G_init=initial,
        sequence="raster_y",
        meas_slab_only=False,
        frozen_schedule=schedule,
        mean_replacement=True,
        return_data=False,
        cycle_observer=capture,
    )
    expected = initial.clone()
    for site in sites:
        chi_ap, chi_bp, chi_am, chi_bm = model._site_spinors(site)
        for chi, eta in ((chi_ap, -1.0), (chi_am, 1.0), (chi_bp, -1.0), (chi_bm, 1.0)):
            chi = chi / torch.linalg.vector_norm(chi)
            projector = chi[:, None] * chi.conj()[None, :]
            q = torch.eye(model.Nlayer, dtype=model.dtype) - projector
            expected = q[None] @ expected @ q[None] + eta * projector[None]
    expected = 0.5 * (expected + expected.conj().transpose(-2, -1))
    torch.testing.assert_close(observed[1], expected, atol=2e-11, rtol=2e-11)
