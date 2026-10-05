"""Exact three-mode Fock-space checks; no circuit simulation or GPU required."""
import numpy as np
from scipy.linalg import expm


def test_wick_and_gaussian_probe_identities():
    rng = np.random.default_rng(240927)
    n = 3
    c = []
    for j in range(n):
        op = np.zeros((2**n, 2**n), complex)
        for ket in range(2**n):
            if (ket >> j) & 1:
                op[ket ^ (1 << j), ket] = (-1)**((ket & ((1 << j)-1)).bit_count())
        c.append(op)
    number = [op.conj().T @ op for op in c]

    def quadratic(h):
        return sum(h[i,j]*c[i].conj().T@c[j] for i in range(n) for j in range(n))

    def correlation(rho):
        return np.array([[np.trace(rho@c[i].conj().T@c[j]) for j in range(n)] for i in range(n)])

    h = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    h = (h+h.conj().T)/2
    rho = expm(-quadratic(h)); rho /= np.trace(rho)
    G = correlation(rho)
    k = rng.normal(size=(n,n)) + 1j*rng.normal(size=(n,n))
    k = (k+k.conj().T)/2
    U = expm(-.7j*quadratic(k)); X = expm(-.7j*k)
    Z = expm(1j*np.pi*number[0])
    eps = 1e-5
    for mixture in (False, True):
        def channel(state):
            evolved = U@state@U.conj().T
            return .37*evolved + .63*Z@evolved@Z.conj().T if mixture else evolved
        for j in range(n):
            P = np.zeros((n,n)); P[j,j] = 1
            initial_mean = G[j,j]
            filtered, rotated = {}, {}
            for sign in (-1,1):
                e = sign*eps
                E = expm(e*number[j]/2)
                r = E@rho@E; r /= np.trace(r)
                filtered[sign] = channel(r)
                F = expm(e*P/2); u = np.expm1(e)
                expected = F@(G-u/(1+u*initial_mean)*(G@P@G))@F
                np.testing.assert_allclose(correlation(r), expected, atol=2e-14)
                phase = expm(-1j*e*number[j])
                rp = phase@rho@phase.conj().T
                np.testing.assert_allclose(correlation(rp),expm(1j*e*P)@G@expm(-1j*e*P),atol=2e-14)
                rotated[sign] = channel(rp)
            for i in range(n):
                exact = np.trace(number[i]@channel(number[j]@rho)) - np.trace(number[i]@channel(rho))*initial_mean
                real = np.trace(number[i]@(filtered[1]-filtered[-1]))/(2*eps)
                imag = np.trace(number[i]@(rotated[1]-rotated[-1]))/(4*eps)
                np.testing.assert_allclose(real+1j*imag,exact,atol=2e-10,rtol=1e-8)
                if not mixture:
                    wick = (X.conj()@G)[i,j]*(X@(np.eye(n)-G.T))[i,j]
                    np.testing.assert_allclose(wick,exact,atol=2e-14)
    # Gaussian trajectories do not justify Wick-factorizing averaged contractions.
    p = .3
    rho1 = np.diag([1-p,p]); n1 = np.diag([0,1]); parity = np.diag([1,-1])
    dephase = lambda a: (a+parity@a@parity)/2
    exact = np.trace(n1@dephase(n1@rho1))-p*p
    np.testing.assert_allclose(exact,p*(1-p))
    assert exact != 0  # Averaged one-body propagator is zero, but density memory persists.
