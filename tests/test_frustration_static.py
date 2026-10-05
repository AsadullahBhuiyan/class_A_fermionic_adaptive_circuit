from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fgtn.classA_U1FGTN import classA_U1FGTN
from fgtn.diagnostics import compute_completion_from_constraints, compute_static_completion


def _basis(dimension: int, index: int) -> np.ndarray:
    vector = np.zeros((dimension,), dtype=np.complex128)
    vector[index] = 1.0
    return vector


def test_compatible_controller_frames_have_exact_completion():
    vectors = np.column_stack((_basis(4, 0), _basis(4, 2)))
    result = compute_completion_from_constraints(
        vectors,
        np.asarray([1, 0]),
        target_rank=2,
    )

    assert result.filled_rank == 1
    assert result.empty_rank == 1
    np.testing.assert_allclose(result.principal_cosines, [0.0], atol=1e-14)
    assert result.rank_bounds_satisfied
    assert result.exact_completion_exists
    assert result.completion_constraint_error < 1e-12
    assert result.f_star < 1e-12

    completion = result.completion_projector
    assert completion is not None
    np.testing.assert_allclose(completion, completion.conj().T, atol=1e-12)
    np.testing.assert_allclose(completion @ completion, completion, atol=1e-12)
    np.testing.assert_allclose(np.linalg.eigvalsh(completion), [0.0, 0.0, 1.0, 1.0], atol=1e-12)


def test_rank_obstruction_has_unit_minimum_frustration():
    vectors = np.column_stack((_basis(4, 0), _basis(4, 1), _basis(4, 2)))
    result = compute_completion_from_constraints(
        vectors,
        np.ones((3,), dtype=np.int8),
        target_rank=2,
    )

    assert result.filled_rank == 3
    assert not result.rank_bounds_satisfied
    assert not result.exact_completion_exists
    assert result.completion_projector is None
    np.testing.assert_allclose(result.f_star, 1.0, atol=1e-12)
    np.testing.assert_allclose(result.f_star_formula, result.f_star, atol=1e-12)


def test_nonorthogonal_frames_reproduce_analytic_principal_angle_and_fstar():
    overlap_vector = (_basis(2, 0) + _basis(2, 1)) / np.sqrt(2.0)
    vectors = np.column_stack((_basis(2, 0), overlap_vector))
    result = compute_completion_from_constraints(
        vectors,
        np.asarray([1, 0]),
        target_rank=1,
    )

    expected_overlap = 1.0 / np.sqrt(2.0)
    expected_fstar = 1.0 - expected_overlap
    np.testing.assert_allclose(result.principal_cosines, [expected_overlap], atol=1e-12)
    np.testing.assert_allclose(result.sigma_max, expected_overlap, atol=1e-12)
    np.testing.assert_allclose(result.overlap_phi, 0.5, atol=1e-12)
    assert result.rank_bounds_satisfied
    assert not result.exact_completion_exists
    np.testing.assert_allclose(result.f_star, expected_fstar, atol=1e-12)
    np.testing.assert_allclose(result.f_star_formula, expected_fstar, atol=1e-12)


def _dw_model(alpha_2: float) -> classA_U1FGTN:
    model = classA_U1FGTN(
        4,
        6,
        DW=True,
        nshell=1,
        alpha_1=1,
        alpha_2=alpha_2,
        trial_orbitals="X",
        dw_truncation=True,
    )
    model.construct_OW_projectors(
        nshell=1,
        DW=True,
        trial_orbitals="X",
        dw_truncation=True,
    )
    return model


def test_dw_truncated_active_frames_and_static_metrics_ignore_exterior_alpha_2():
    alpha_one = _dw_model(1)
    alpha_thirty = _dw_model(30)
    active_one = alpha_one.active_top_layer_indices(meas_slab_only=True)
    active_thirty = alpha_thirty.active_top_layer_indices(meas_slab_only=True)
    np.testing.assert_array_equal(active_one, active_thirty)

    for channel in ("Ap", "Am", "Bp", "Bm"):
        frame_one = np.asarray(getattr(alpha_one, f"WF_{channel}"))[active_one]
        frame_thirty = np.asarray(getattr(alpha_thirty, f"WF_{channel}"))[active_thirty]
        np.testing.assert_allclose(frame_one, frame_thirty, rtol=0.0, atol=1e-13)

    result_one = compute_static_completion(alpha_one)
    result_thirty = compute_static_completion(alpha_thirty)
    assert result_one.payload().keys() == result_thirty.payload().keys()
    for key in result_one.payload():
        np.testing.assert_allclose(
            result_one.payload()[key],
            result_thirty.payload()[key],
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
            err_msg=key,
        )
