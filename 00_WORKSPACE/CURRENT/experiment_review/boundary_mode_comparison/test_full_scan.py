import unittest
import numpy as np
from full_scan import resolve


class FullScanTests(unittest.TestCase):
    def test_wall_away_vs_cut(self):
        # Half-y=6 rows. First mode at wall x5,y3 (away); second at x10,y0 (cut).
        f=np.zeros((480,2),complex)
        f[40*3+10,0]=np.sqrt(.2);f[40*9+10,0]=np.sqrt(.8)
        f[20,1]=np.sqrt(.7);f[40*6+20,1]=np.sqrt(.3)
        r=resolve(f,12)
        np.testing.assert_allclose(r['wall_weight'],[1,0])
        np.testing.assert_allclose(r['cut_weight'],[0,1])
        np.testing.assert_allclose(r['wall_away_from_cut_weight'],[1,0])
        np.testing.assert_array_equal(r['robust_candidate'],[True,False])
        np.testing.assert_allclose(r['modular_energies'],np.log([4,3/7]))

    def test_degenerate_vectors_not_individually_claimed(self):
        f=np.zeros((480,2),complex)
        f[130,0]=np.sqrt(.2);f[370,0]=np.sqrt(.8)
        f[150,1]=np.sqrt(.2);f[390,1]=np.sqrt(.8)
        r=resolve(f,12)
        self.assertFalse(r['individually_separated'].any())
        self.assertFalse(r['robust_candidate'].any())

    def test_all_modes_beyond_first_sixteen(self):
        f=np.zeros((480,20),complex)
        nu=np.linspace(.01,.99,20)
        for i,n in enumerate(nu):f[i,i]=np.sqrt(n);f[i+240,i]=np.sqrt(1-n)
        r=resolve(f,12)
        self.assertEqual(r['eigenvectors'].shape,(240,20))
        np.testing.assert_allclose(r['occupations'][r['finite_indices']],nu)


if __name__=='__main__':unittest.main()
