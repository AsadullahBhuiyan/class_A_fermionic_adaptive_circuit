"""Small deterministic algebra tests; no monitored dynamics or data writes."""
import unittest
import numpy as np
from reproduce import adj, load_native


class DerivationTests(unittest.TestCase):
    def test_two_families_reconstruct_target(self):
        m, removed = load_native()
        self.assertEqual(len(removed), 2)
        k = 2*np.pi*np.arange(31)/31
        kx, ky = np.meshgrid(k, k, indexing='ij')
        p = m.band_projector(kx, ky, 1., -1)
        frame = np.zeros_like(p)
        for trial in m.TRIAL_SPINORS:
            w = np.sqrt(2)*(p@trial)
            self.assertAlmostEqual(float(np.mean(np.sum(abs(w)**2, axis=-1))), 1.)
            frame += w[..., :, None]*w.conj()[..., None, :]/2
        np.testing.assert_allclose(frame, p, atol=1e-14)

    def test_separate_ket_bra_transform_and_translation_sum(self):
        rng = np.random.default_rng(2026100601)
        cells, orbitals = 7, 2
        w = rng.normal(size=(cells, orbitals))+1j*rng.normal(size=(cells, orbitals))
        w /= np.linalg.norm(w)
        translated = np.stack([np.roll(w, r, axis=0).ravel() for r in range(cells)], axis=1)
        frame = translated@adj(translated)
        q = 2*np.pi*np.arange(cells)/cells
        fourier = np.exp(-1j*q[:, None]*np.arange(cells))/np.sqrt(cells)
        transform = np.kron(fourier, np.eye(orbitals))
        actual = transform@frame@adj(transform)
        wf = np.sqrt(cells)*(fourier@w)
        expected = np.zeros_like(actual)
        for k in range(cells):
            expected[2*k:2*k+2, 2*k:2*k+2] = np.outer(wf[k], wf[k].conj())
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_general_unequal_normalization_frame(self):
        f = np.array([[.4, .2+.1j], [.2-.1j, .6]])
        trials = np.array([[1, 1], [1, -1]])/np.sqrt(2)
        for operator, norms in ((f, [.3, .6]), (np.eye(2)-f, [.4, .7])):
            d = sum(np.outer(t, t)/z for t, z in zip(trials, norms))
            direct = sum(np.outer(operator@t, (operator@t).conj())/z
                         for t, z in zip(trials, norms))
            np.testing.assert_allclose(direct, operator@d@adj(operator), atol=1e-14)

    def test_projector_homotopy(self):
        p = np.diag([1., 0.])
        v = np.array([.1+.2j, np.sqrt(.95)])
        r = np.outer(v, v.conj())
        for s in (0., .2, .7, 1.):
            b = p+(1-s)*(np.eye(2)-p)
            rs = b@r@b
            denominator = np.trace(rs).real
            self.assertGreater(denominator, 0.)
            self.assertAlmostEqual(denominator, .05+(1-s)**2*.95)
            rs /= denominator
            np.testing.assert_allclose(rs@rs, rs, atol=1e-14)
            if s == 0:
                np.testing.assert_allclose(rs, r, atol=1e-14)
            if s == 1:
                np.testing.assert_allclose(rs, p, atol=1e-14)

    def test_rank_one_addition_removal_and_mismatch(self):
        rng = np.random.default_rng(2026100602)
        u, _ = np.linalg.qr(rng.normal(size=(8, 3))+1j*rng.normal(size=(8, 3)))
        gamma = u@adj(u)
        v = rng.normal(size=8)+1j*rng.normal(size=8)
        for sign, component in ((1, (np.eye(8)-gamma)@v), (-1, gamma@v)):
            phi = component/np.linalg.norm(component)
            updated = gamma+sign*np.outer(phi, phi.conj())
            np.testing.assert_allclose(updated@updated, updated, atol=1e-14)
            self.assertAlmostEqual(np.trace(updated).real, 3+sign)
            self.assertAlmostEqual(np.linalg.norm(updated-gamma, 2), 1.)
        ur, _ = np.linalg.qr(rng.normal(size=(8, 4))+1j*rng.normal(size=(8, 4)))
        r = ur@adj(ur)
        particles = np.trace((np.eye(8)-r)@gamma).real
        holes = np.trace(r@(np.eye(8)-gamma)).real
        self.assertAlmostEqual(particles-holes, np.trace(gamma-r).real)
        self.assertAlmostEqual(particles+holes, np.trace((gamma-r)@(gamma-r)).real)

    def test_nonholomorphic_boundary_homotopy(self):
        angle = np.linspace(0, 2*np.pi, 721)
        z = np.exp(1j*angle)
        f = z
        g = z+.2*z.conj()+.1
        self.assertLess(np.max(abs(g-f)), np.min(abs(f)))
        for s in (0., .25, .75, 1.):
            h = (1-s)*f+s*g
            self.assertGreater(np.min(abs(h)), .69)
            winding = np.angle(h[1:]*h[:-1].conj()).sum()/(2*np.pi)
            self.assertAlmostEqual(winding, 1.)


if __name__ == '__main__':
    unittest.main()
