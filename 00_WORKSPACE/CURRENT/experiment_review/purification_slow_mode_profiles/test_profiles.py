import unittest
import numpy as np
from analyze_profiles import extract, finite_mask, profiles


class ProfileTests(unittest.TestCase):
    def test_cap_policy(self):
        np.testing.assert_array_equal(finite_mask([0, 1e-10, 1e-8, .5, 1-1e-8, 1]), [False, False, True, True, True, False])

    def test_known_modes_and_rates(self):
        nu = np.array([0., .2, .5, .8, 1.])
        d = extract(np.diag(2*nu-1), np.arange(5), 10)
        self.assertEqual(int(d['finite_count']), 3)
        np.testing.assert_allclose(d['finite_signed_rates'], [0, np.log(4)/20, -np.log(4)/20], atol=1e-15)
        np.testing.assert_allclose(d['finite_mode_x_profiles'].sum(axis=1), 1)

    def test_degenerate_subspace_flag(self):
        d = extract(np.diag([-.6, -.6, 1.]), np.arange(3), 10)
        self.assertFalse(d['finite_mode_individually_separated'].any())
        self.assertTrue(bool(d['finite_subspace_separated_from_caps']))

    def test_subspace_profile_rotation_invariant(self):
        v = np.eye(4, dtype=complex)[:, :2]
        rotation = np.array([[1, 1j], [1j, 1]])/np.sqrt(2)
        np.testing.assert_allclose(profiles(v, np.arange(4)).mean(axis=0), profiles(v@rotation, np.arange(4)).mean(axis=0))

    def test_empty_endpoint_does_not_invent_modes(self):
        d = extract(np.diag([-1., -1., 1., 1.]), np.arange(4), 10)
        self.assertEqual(int(d['finite_count']), 0)
        self.assertEqual(d['finite_eigenvectors'].shape, (4, 0))
        self.assertTrue(np.isnan(d['finite_subspace_mean_x_profile']).all())


if __name__ == '__main__': unittest.main()
