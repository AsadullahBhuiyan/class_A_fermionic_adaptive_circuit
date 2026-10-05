import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from extract_endpoint_singular_modes import decompose, x_profiles


class SingularModeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(731)
        self.matrix = rng.normal(size=(5, 8)) + 1j * rng.normal(size=(5, 8))

    def test_reconstruction_orientation_and_phases(self):
        with threadpool_limits(limits=1):
            d = decompose(self.matrix, 3., 12, np.arange(5), nx=2)
        u, s, vh = (d[k] for k in ("left_vectors_initial_active", "singular_values_normalized",
                                   "right_vectors_endpoint_dagger"))
        np.testing.assert_allclose((u * s) @ vh, self.matrix, atol=1e-12)
        pivots = d["left_vector_phase_pivot_rows"]
        pivot = u[pivots, np.arange(5)]
        np.testing.assert_allclose(pivot.imag, 0, atol=1e-14)
        self.assertTrue(np.all(pivot.real > 0))
        # The saved covariance action takes initial LEFT modes to endpoint RIGHT modes.
        i, j = 0, 1
        h0 = np.outer(u[:, i], u[:, j].conj())
        v = vh.conj().T
        expected = s[i] * s[j] * np.outer(v[:, i], v[:, j].conj())
        np.testing.assert_allclose(self.matrix.conj().T @ h0 @ self.matrix, expected, atol=1e-12)

    def test_logscale_keeps_large_scale_finite(self):
        with threadpool_limits(limits=1):
            d = decompose(self.matrix, 1000., 80, np.arange(5))
        np.testing.assert_allclose(d["raw_log_singular_values"],
                                   np.log(d["singular_values_normalized"]) + 1000.)
        self.assertTrue(np.isfinite(d["resolved_one_leg_rates_per_cycle"]).all())

    def test_null_mask_does_not_claim_exact_physical_zeros(self):
        matrix = np.zeros((3, 5), complex)
        matrix[0, 0] = 1
        matrix[1, 1] = .01
        with threadpool_limits(limits=1):
            d = decompose(matrix, 0., 10, np.arange(3))
        self.assertEqual(int(d["numerical_rank"]), 2)
        self.assertTrue(np.isnan(d["resolved_log_singular_values"][-1]))
        np.testing.assert_array_equal(d["one_leg_nearest_zero_mode_indices"], [0, 1])

    def test_spatial_index_order(self):
        vectors = np.zeros((8, 2), complex)
        vectors[2, 0] = 1  # x=1,y=0,orbital=0
        vectors[5, 1] = 1  # x=0,y=1,orbital=1
        np.testing.assert_array_equal(x_profiles(vectors, np.arange(8), 2), [[0, 1], [1, 0]])


if __name__ == "__main__":
    unittest.main()
