import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plot_endpoint_gaps import summary, trajectory_gaps, verify_pair


class GapEstimatorTests(unittest.TestCase):
    def test_absolute_value_before_averaging(self):
        gamma, signed = trajectory_gaps(np.array([[.1, -.3], [-.1, -.4]]))
        np.testing.assert_allclose(gamma, [.1, .1])
        self.assertEqual(signed.mean(), 0.)
        mean, sem = summary(gamma)
        self.assertAlmostEqual(mean, .1)
        self.assertEqual(sem, 0.)

    def test_not_old_factor_two_gap(self):
        rates = np.array([[-.2, -.3], [.1, -.2]])
        gamma, _ = trajectory_gaps(rates)
        np.testing.assert_allclose(gamma, [.2, .1])
        np.testing.assert_allclose(gamma, .5 * np.abs(-2 * rates).min(axis=1))

    def test_sem_and_true_zero(self):
        gamma, _ = trajectory_gaps(np.array([[0., -.1], [-.2, -.3]]))
        np.testing.assert_allclose(summary(gamma), [.1, .1])

    def test_invalid_saved_rates_rejected(self):
        for rates in (np.array([[np.nan, .2]]), np.array([[.3, .1]]), np.ones(4)):
            with self.assertRaises(ValueError):
                trajectory_gaps(rates)

    def test_receipt_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.npz"
            result.write_bytes(b"abcd")
            receipt = root / "result.complete.json"
            receipt.write_text(json.dumps({"result_filename": result.name, "result_bytes": 4,
                                           "result_sha256": hashlib.sha256(b"abcd").hexdigest()}))
            self.assertEqual(verify_pair(receipt)[1], result)
            result.write_bytes(b"abce")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_pair(receipt)
            result.write_bytes(b"abc")
            with self.assertRaisesRegex(ValueError, "byte-count"):
                verify_pair(receipt)


if __name__ == "__main__":
    unittest.main()
