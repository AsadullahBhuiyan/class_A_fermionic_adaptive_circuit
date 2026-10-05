from __future__ import annotations

import numpy as np
import pytest

from src.fgtn.classA_U1FGTN import classA_U1FGTN


torch = pytest.importorskip("torch")

from src.fgtn.classA_U1FGTN_gpu import classA_U1FGTN_gpu


NX = 4
NY = 2


def _pure_centered_covariance(seed: int, *, force_orbital_zero_empty: bool = False):
    dimension = 2 * NX * NY
    rank = dimension // 2
    generator = np.random.default_rng(seed)
    rows = dimension - int(force_orbital_zero_empty)
    trial = generator.normal(size=(rows, rank)) + 1j * generator.normal(
        size=(rows, rank)
    )
    frame, _ = np.linalg.qr(trial, mode="reduced")
    if force_orbital_zero_empty:
        padded = np.zeros((dimension, rank), dtype=np.complex128)
        padded[1:] = frame
        frame = padded
    return 2.0 * (frame @ frame.conj().T) - np.eye(dimension)


def _model(engine: str, *, dw_truncation: bool = True):
    common = dict(
        Nx=NX,
        Ny=NY,
        DW=True,
        nshell=1,
        alpha_1=1,
        alpha_2=30,
        dw_truncation=dw_truncation,
    )
    if engine == "cpu":
        return classA_U1FGTN(**common)
    return classA_U1FGTN_gpu(
        **common,
        device="cpu",
        dtype="complex128",
        backend="dense",
    )


def _active_and_exterior(model):
    active = np.asarray(model.active_top_layer_indices(True), dtype=np.int64)
    dimension = model.Ntot // 2
    exterior = np.setdiff1d(np.arange(dimension, dtype=np.int64), active)
    return active, exterior


def _capture_run(
    engine: str,
    *,
    initial: np.ndarray,
    state_representation: str = "covariance",
    meas_slab_only: bool = True,
    postselect: bool = False,
    postselect_probability: float = 0.0,
    dw_truncation: bool = True,
):
    model = _model(engine, dw_truncation=dw_truncation)
    observed = {}

    def capture(*, cycle, G, **_):
        value = G.detach().cpu().numpy() if torch.is_tensor(G) else np.asarray(G)
        if value.ndim == 3:
            value = value[0]
        observed[int(cycle)] = np.array(value, dtype=np.complex128, copy=True)

    common = dict(
        cycles=1,
        samples=1,
        G_init=initial,
        sequence="raster_y",
        perfect_correction=True,
        postselect=postselect,
        postselect_probability=postselect_probability,
        meas_slab_only=meas_slab_only,
        state_representation=state_representation,
        G_history=False,
        save=False,
        progress=False,
        cycle_observer=capture,
    )
    if engine == "cpu":
        result = model.run_markov_circuit(random_seed=771, **common)
    else:
        torch.manual_seed(771)
        result = model.run_markov_circuit(
            batch_size=1,
            return_data=False,
            **common,
        )
    return model, observed, result


def _assert_exterior_product_state(G, active, exterior, expected_diagonal=None):
    assert np.max(np.abs(G[np.ix_(exterior, active)])) < 2e-10
    exterior_block = G[np.ix_(exterior, exterior)]
    diagonal = np.real(np.diag(exterior_block))
    np.testing.assert_allclose(
        exterior_block,
        np.diag(diagonal),
        atol=2e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(np.abs(diagonal), 1.0, atol=2e-10, rtol=0.0)
    if expected_diagonal is not None:
        np.testing.assert_allclose(
            diagonal,
            expected_diagonal,
            atol=2e-10,
            rtol=0.0,
        )


@pytest.mark.parametrize("engine", ["cpu", "gpu"])
def test_born_conditioned_exterior_is_prepared_before_cycle_zero_and_stays_fixed(
    engine,
):
    initial = _pure_centered_covariance(1043, force_orbital_zero_empty=True)
    model, observed, result = _capture_run(
        engine,
        initial=initial,
        postselect_probability=0.5,
    )
    active, exterior = _active_and_exterior(model)
    assert np.linalg.norm(initial[np.ix_(exterior, active)]) > 0.1
    assert sorted(observed) == [0, 1]
    _assert_exterior_product_state(observed[0], active, exterior)
    assert observed[0][0, 0].real == pytest.approx(-1.0, abs=2e-10)
    np.testing.assert_allclose(
        observed[1][np.ix_(exterior, exterior)],
        observed[0][np.ix_(exterior, exterior)],
        atol=2e-10,
        rtol=0.0,
    )
    assert np.max(np.abs(observed[1][np.ix_(exterior, active)])) < 2e-10
    np.testing.assert_allclose(
        observed[0] @ observed[0],
        np.eye(observed[0].shape[0]),
        atol=3e-10,
        rtol=0.0,
    )
    assert result["exterior_preparation_mode"] == "born_conditioned"
    assert result["exterior_preparation_basis"] == "canonical_unit_cell_orbital"
    assert result["exterior_preparation_orbital_count"] == exterior.size


@pytest.mark.parametrize("engine", ["cpu", "gpu"])
@pytest.mark.parametrize("state_representation", ["covariance", "physical_frame"])
def test_forced_postselection_prepares_every_exterior_orbital_occupied(
    engine, state_representation
):
    initial = _pure_centered_covariance(2207)
    model, observed, result = _capture_run(
        engine,
        initial=initial,
        state_representation=state_representation,
        postselect=True,
    )
    active, exterior = _active_and_exterior(model)
    assert sorted(observed) == [0, 1]
    _assert_exterior_product_state(
        observed[0], active, exterior, expected_diagonal=np.ones(exterior.size)
    )
    _assert_exterior_product_state(
        observed[1], active, exterior, expected_diagonal=np.ones(exterior.size)
    )
    assert result["exterior_preparation_mode"] == "forced_occupied"
    assert result["exterior_preparation"] == "forced_occupied_onsite_before_cycle_0"


@pytest.mark.parametrize("engine", ["cpu", "gpu"])
def test_meas_slab_only_does_not_prepare_exterior_without_dw_truncation(engine):
    initial = _pure_centered_covariance(3301)
    _, observed, result = _capture_run(
        engine,
        initial=initial,
        meas_slab_only=True,
        dw_truncation=False,
    )
    np.testing.assert_allclose(observed[0], initial, atol=2e-10, rtol=0.0)
    assert result["exterior_preparation"] is None
    assert result["exterior_preparation_mode"] is None
    assert result["exterior_preparation_basis"] is None
    assert result["exterior_preparation_orbital_count"] == 0


def _random_frame(seed: int) -> np.ndarray:
    dimension = 2 * NX * NY
    generator = np.random.default_rng(seed)
    trial = generator.normal(size=(dimension, dimension // 2)) + 1j * generator.normal(
        size=(dimension, dimension // 2)
    )
    frame, _ = np.linalg.qr(trial, mode="reduced")
    return np.asarray(frame, dtype=np.complex128)


def _gpu_frame_run_kwargs(frame: np.ndarray) -> dict:
    return {
        "G_history": False,
        "progress": False,
        "cycles": 1,
        "samples": 1,
        "frame_init": frame,
        "state_representation": "physical_frame",
        "return_native_state": True,
        "save": False,
        "sequence": "raster_y",
        "perfect_correction": True,
        "meas_slab_only": True,
        "batch_size": 1,
    }


def test_gpu_omitted_prepared_flag_matches_explicit_false():
    frame = _random_frame(4417)
    torch.manual_seed(991)
    omitted = _model("gpu").run_markov_circuit(**_gpu_frame_run_kwargs(frame))
    torch.manual_seed(991)
    explicit = _model("gpu").run_markov_circuit(
        frame_init_prepared=False,
        **_gpu_frame_run_kwargs(frame),
    )

    omitted_frame = omitted["native_final"]["frame"]
    explicit_frame = explicit["native_final"]["frame"]
    omitted_projector = omitted_frame[0] @ omitted_frame[0].conj().T
    explicit_projector = explicit_frame[0] @ explicit_frame[0].conj().T
    np.testing.assert_allclose(omitted_projector, explicit_projector, atol=1e-13, rtol=0.0)
    np.testing.assert_array_equal(
        omitted["native_final"]["ranks"], explicit["native_final"]["ranks"]
    )
    assert omitted["frame_init_prepared"] is False
    assert explicit["frame_init_prepared"] is False
    assert omitted["exterior_preparation_performed"] is True
    assert explicit["exterior_preparation_performed"] is True
    assert omitted["exterior_preparation"] == "born_conditioned_onsite_before_cycle_0"
    assert explicit["exterior_preparation"] == "born_conditioned_onsite_before_cycle_0"


def test_gpu_prepared_frame_skips_hard_wall_exterior(monkeypatch):
    model = _model("gpu")
    initial = model._prepare_initial_frame_batch(
        1,
        init_mode="default",
        frame_init=_random_frame(5521),
    )
    torch.manual_seed(773)
    prepared = model._prepare_exterior_product_frame_batched(
        initial,
        mode="born_conditioned",
    ).snapshot(cpu=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("hard-wall exterior preparation was repeated")

    monkeypatch.setattr(model, "_prepare_exterior_product_frame_batched", forbidden)
    result = model.run_markov_circuit(
        frame_init=prepared["frame"],
        frame_ranks=prepared["ranks"],
        frame_init_prepared=True,
        **{
            key: value
            for key, value in _gpu_frame_run_kwargs(prepared["frame"]).items()
            if key != "frame_init"
        },
    )

    assert result["frame_init_prepared"] is True
    assert result["exterior_preparation"] == "skipped_prepared_frame"
    assert result["exterior_preparation_mode"] == "already_prepared"
    assert result["exterior_preparation_performed"] is False


def test_gpu_prepared_flag_requires_frame_init():
    with pytest.raises(ValueError, match="requires frame_init"):
        _model("gpu").run_markov_circuit(frame_init_prepared=True)


def _segmented_gpu_model() -> classA_U1FGTN_gpu:
    return classA_U1FGTN_gpu(
        Nx=4,
        Ny=4,
        DW=True,
        nshell=1,
        filling_frac=0.5,
        alpha_1=1.0,
        alpha_2=30.0,
        trial_orbitals="X",
        dw_truncation=True,
        device="cpu",
        dtype="complex128",
        backend="local",
    )


def _segmented_gpu_run(
    model: classA_U1FGTN_gpu,
    *,
    cycles: int,
    frame: np.ndarray | None = None,
    ranks: np.ndarray | None = None,
    prepared: bool = False,
) -> dict:
    return model.run_markov_circuit(
        G_history=False,
        progress=False,
        cycles=cycles,
        postselect=False,
        postselect_probability=0.0,
        perfect_correction=True,
        samples=2,
        init_mode="default",
        frame_init=frame,
        frame_ranks=ranks,
        frame_init_prepared=prepared,
        save=False,
        n_a=0.5,
        sequence="raster_y",
        meas_slab_only=True,
        batch_size=2,
        return_data=True,
        state_representation="physical_frame",
        return_native_state=True,
        require_no_covariance_materialization=True,
        frame_reorthonormalize_interval=1,
    )


def _assert_numpy_rng_states_equal(left: tuple, right: tuple) -> None:
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_gpu_hard_wall_segmented_prepared_frame_is_bitwise_exact():
    seed = 67891
    np.random.seed(seed)
    torch.manual_seed(seed)
    uninterrupted = _segmented_gpu_run(_segmented_gpu_model(), cycles=8)
    uninterrupted_numpy_rng = np.random.get_state()
    uninterrupted_torch_rng = torch.get_rng_state().clone()

    np.random.seed(seed)
    torch.manual_seed(seed)
    segmented_model = _segmented_gpu_model()
    first = _segmented_gpu_run(segmented_model, cycles=5)
    checkpoint_numpy_rng = np.random.get_state()
    checkpoint_torch_rng = torch.get_rng_state().clone()

    np.random.seed(1)
    torch.manual_seed(2)
    np.random.set_state(checkpoint_numpy_rng)
    torch.set_rng_state(checkpoint_torch_rng)
    resumed = _segmented_gpu_run(
        segmented_model,
        cycles=3,
        frame=first["native_final"]["frame"],
        ranks=first["native_final"]["ranks"],
        prepared=True,
    )
    resumed_numpy_rng = np.random.get_state()
    resumed_torch_rng = torch.get_rng_state().clone()

    for key in ("frame", "ranks"):
        np.testing.assert_array_equal(
            resumed["native_final"][key], uninterrupted["native_final"][key]
        )
    _assert_numpy_rng_states_equal(resumed_numpy_rng, uninterrupted_numpy_rng)
    torch.testing.assert_close(
        resumed_torch_rng, uninterrupted_torch_rng, rtol=0, atol=0
    )
    assert first["frame_init_prepared"] is False
    assert first["exterior_preparation_performed"] is True
    assert resumed["frame_init_prepared"] is True
    assert resumed["exterior_preparation"] == "skipped_prepared_frame"
    assert resumed["exterior_preparation_performed"] is False
