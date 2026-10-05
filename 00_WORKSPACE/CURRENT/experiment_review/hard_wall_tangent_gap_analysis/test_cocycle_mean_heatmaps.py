import unittest
import numpy as np
from plot_cocycle_mean_heatmaps import mean_density, near_zero


class HeatmapTests(unittest.TestCase):
    def test_coordinates_and_subspace_gauge(self):
        indices=np.array([2*5+40*3,2*15+40*7+1])
        v=np.eye(2,dtype=complex)
        p=mean_density(v,indices)
        self.assertEqual(p[3,5],.5);self.assertEqual(p[7,15],.5)
        q=np.array([[1,1j],[1j,1]])/np.sqrt(2)
        np.testing.assert_allclose(p,mean_density(v@q,indices),atol=1e-14)

    def test_ties_and_null_exclusion(self):
        rates=np.array([.1,-.1,.2,-.3,.3,np.nan])
        np.testing.assert_array_equal(near_zero(rates,np.isfinite(rates)),np.arange(5))

    def test_equal_sample_not_mode_weight(self):
        p=mean_density(np.eye(2),np.array([10,11]))
        q=mean_density(np.ones((1,1)),np.array([30]))
        result=(p+q)/2
        self.assertEqual(result[0,5],.5);self.assertEqual(result[0,15],.5)


if __name__=='__main__':unittest.main()
