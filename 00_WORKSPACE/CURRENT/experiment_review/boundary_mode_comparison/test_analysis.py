import unittest
import numpy as np
from analyze import extract, density, nearest_mask


class AnalysisTests(unittest.TestCase):
    def test_known_schmidt_spectrum_and_frame_gauge(self):
        # Nx=20, Ny=2; two occupied modes entangle the two halves.
        f = np.zeros((80, 2), complex)
        f[10,0], f[50,0] = np.sqrt(.3), np.sqrt(.7)
        f[30,1], f[70,1] = np.sqrt(.8), np.sqrt(.2)
        r = extract(f, 2)
        np.testing.assert_allclose(r['occupations'][-2:], [.3,.8], atol=1e-14)
        np.testing.assert_allclose(np.sort(r['selected_energies']), np.sort(np.log([.7/.3,.2/.8])))
        q = np.array([[1,1j],[1j,1]],complex)/np.sqrt(2)
        s = extract(f@q, 2)
        np.testing.assert_allclose(r['occupations'], s['occupations'], atol=1e-14)
        np.testing.assert_allclose(r['selected_xy'], s['selected_xy'], atol=1e-14)
        np.testing.assert_allclose(r['all_mode_wall_weights'][r['selected_indices']], 1)

    def test_density_indexing(self):
        v = np.zeros((80,1),complex); v[2*20+2*15+1] = 1
        p=density(v,np.arange(80),2)
        self.assertEqual(p[0,1,15],1)
        self.assertEqual(p.sum(),1)

    def test_include_spectral_ties(self):
        np.testing.assert_array_equal(nearest_mask(np.array([.1,-.1,.2]),1),[True,True,False])

    def test_reject_nonorthonormal_frame(self):
        with self.assertRaises(ValueError): extract(np.ones((80,1),complex),2)


if __name__ == '__main__': unittest.main()
