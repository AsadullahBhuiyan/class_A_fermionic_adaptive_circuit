"""Exact, small-Fock-space checks for the gain/loss boundary note.

These are algebra checks, not runs of the adaptive circuit or production engine.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm


def fermions(count):
    lower = np.array([[0, 1], [0, 0]], dtype=complex)
    parity = np.diag([1, -1])
    result = []
    for index in range(count):
        op = np.ones((1, 1), dtype=complex)
        for site in range(count):
            op = np.kron(
                op, parity if site < index else lower if site == index else np.eye(2)
            )
        result.append(op)
    return result


def mode(operators, column):
    return sum(x.conjugate() * c for x, c in zip(column, operators))


def random_column(rng, count):
    x = rng.normal(size=count) + 1j * rng.normal(size=count)
    return x / np.linalg.norm(x)


def majoranas(operators):
    return [g for c in operators for g in (c + c.conj().T, -1j * (c - c.conj().T))]


def transfer(operator, gamma):
    inverse = np.linalg.inv(operator)
    dimension = operator.shape[0]
    return np.array([
        [np.trace(gb @ operator @ ga @ inverse) / dimension for gb in gamma]
        for ga in gamma
    ])


def correlation(state, operators):
    """C_ij = <psi_j^dagger psi_i>; BPJ G = C.T."""
    state = state / np.linalg.norm(state)
    return np.array([
        [np.vdot(state, cj.conj().T @ ci @ state) for cj in operators]
        for ci in operators
    ])


def anomalous(state, operators):
    state = state / np.linalg.norm(state)
    return np.array([
        [np.vdot(state, ci @ cj @ state) for cj in operators]
        for ci in operators
    ])


def slater(frame, operators):
    state = np.zeros(operators[0].shape[0], dtype=complex)
    state[0] = 1
    for column in frame.T[::-1]:
        state = mode(operators, column).conj().T @ state
    return state


def even_operator(matrix, operators):
    return expm(-sum(
        matrix[i, j] * ci.conj().T @ cj
        for i, ci in enumerate(operators) for j, cj in enumerate(operators)
    ))


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("epsilon", [1.0, 0.3, 0.03])
def test_polar_reflection_and_singular_values(count, epsilon):
    rng = np.random.default_rng(941 + count)
    ops = fermions(count)
    c = mode(ops, random_column(rng, count))
    identity = np.eye(2**count)
    n = c.conj().T @ c
    flip = c + c.conj().T
    strength = -0.5 * np.log(epsilon)
    gamma = majoranas(ops)
    for sign in (1, -1):
        jump = (c.conj().T + epsilon * c if sign == 1
                else c + epsilon * c.conj().T)
        v = jump / np.sqrt(epsilon)
        assert_allclose(jump @ jump, epsilon * identity, atol=2e-14)
        assert_allclose(v @ v, identity, atol=2e-13)
        assert_allclose(v, flip @ expm(-sign * strength * (2*n-identity)), atol=3e-13)
        z = np.array([np.trace(g @ v)/len(v) for g in gamma])
        assert_allclose(z @ z, 1, atol=3e-13)
        reflected = -np.eye(2*count) + 2*np.outer(z, z)/(z @ z)
        t = transfer(v, gamma)
        assert_allclose(t, reflected, atol=3e-12)
        assert_allclose(t.T @ t, np.eye(2*count), atol=3e-11)
        assert_allclose(np.linalg.det(t), -1, atol=3e-12)
        assert_allclose(np.sort(np.abs(np.linalg.eigvals(t))), 1, atol=3e-12)
        assert_allclose(
            np.linalg.svd(t, compute_uv=False),
            [1/epsilon] + [1.]*(2*count-2) + [epsilon], atol=3e-12,
        )


@pytest.mark.parametrize("epsilon", [1.0, 0.2, 0.01, 0.0])
@pytest.mark.parametrize("feedback", ["fill", "deplete", "both"])
def test_physical_instruments_and_parity(epsilon, feedback):
    rng = np.random.default_rng(513)
    ops = fermions(3)
    c = mode(ops, random_column(rng, 3))
    identity = np.eye(8)
    n = c.conj().T @ c
    flip = c + c.conj().T
    m0 = (identity-n+epsilon*n)/np.sqrt(1+epsilon**2)
    m1 = (n+epsilon*(identity-n))/np.sqrt(1+epsilon**2)
    branches = (flip@m0 if feedback in ("fill", "both") else m0,
                flip@m1 if feedback in ("deplete", "both") else m1)
    assert_allclose(sum(k.conj().T@k for k in branches), identity, atol=2e-14)
    assert_allclose(branches[0].conj().T@branches[0], m0@m0, atol=2e-14)
    assert_allclose(branches[1].conj().T@branches[1], m1@m1, atol=2e-14)
    parity = np.diag([(-1)**i.bit_count() for i in range(8)])
    x = rng.normal(size=(8, 8)) + 1j*rng.normal(size=(8, 8))
    rho = x@x.conj().T
    rho = (rho+parity@rho@parity)/2
    rho /= np.trace(rho)
    for k in branches:
        out = k@rho@k.conj().T
        assert_allclose(parity@out@parity, out, atol=2e-14)
        if epsilon:
            assert np.min(np.linalg.svd(k, compute_uv=False)) > 0
    if epsilon == 0:
        expected = {"fill": (c.conj().T, n),
                    "deplete": (identity-n, c),
                    "both": (c.conj().T, c)}[feedback]
        for actual, target in zip(branches, expected):
            assert_allclose(actual, target, atol=2e-14)


def test_stm_product_order_with_nonunitary_even_and_odd_factors():
    rng = np.random.default_rng(982)
    ops = fermions(3)
    gamma = majoranas(ops)
    c = mode(ops, random_column(rng, 3))
    odd = c.conj().T + .4*c
    matrix = .13*(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
    even = even_operator(matrix, ops)
    t_odd, t_even = transfer(odd, gamma), transfer(even, gamma)
    assert_allclose(transfer(even@odd, gamma), t_odd@t_even, atol=2e-13)
    assert np.linalg.norm(t_odd@t_even-t_even@t_odd) > .1
    for i, c_i in enumerate(ops):
        assert_allclose(even@c_i@np.linalg.inv(even),
                        sum(expm(matrix)[i, j]*ops[j] for j in range(3)), atol=2e-13)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_exact_projector_updates_and_even_gram_norm(rank):
    rng = np.random.default_rng(207 + rank)
    count = 4
    ops = fermions(count)
    frame, _ = np.linalg.qr(rng.normal(size=(count, rank))+1j*rng.normal(size=(count, rank)))
    state = slater(frame, ops)
    C = frame@frame.conj().T
    assert_allclose(correlation(state, ops), C, atol=2e-14)
    chi = random_column(rng, count)
    c = mode(ops, chi)
    for gain in (False, True):
        vector = (np.eye(count)-C)@chi if gain else C@chi
        prob = np.vdot(vector, vector).real
        output = (c.conj().T if gain else c)@state
        expected = C+(1 if gain else -1)*np.outer(vector, vector.conj())/prob
        assert_allclose(np.vdot(output, output).real, prob, atol=2e-14)
        assert_allclose(correlation(output, ops), expected, atol=2e-14)
        assert_allclose(expected@expected, expected, atol=2e-14)
        assert_allclose(expected@chi, chi if gain else np.zeros(count), atol=2e-14)
    matrix = .15*(rng.normal(size=(count, count))+1j*rng.normal(size=(count, count)))
    A = expm(-matrix)
    image = A@frame
    gram = image.conj().T@image
    output = even_operator(matrix, ops)@state
    assert_allclose(np.vdot(output, output), np.linalg.det(gram), atol=2e-13)
    assert_allclose(correlation(output, ops), image@np.linalg.inv(gram)@image.conj().T,
                    atol=2e-13)


def test_chronological_nonorthogonal_word_matches_fock_state_and_probability():
    rng = np.random.default_rng(892)
    ops = fermions(3)
    frame = random_column(rng, 3)[:, None]
    state = slater(frame, ops)
    C = frame@frame.conj().T
    accumulated = 1.
    for gain in (True, False, True, False):
        chi = random_column(rng, 3)
        c = mode(ops, chi)
        v = (np.eye(3)-C)@chi if gain else C@chi
        prob = np.vdot(v, v).real
        assert prob > 1e-7
        accumulated *= prob
        C = C+(1 if gain else -1)*np.outer(v, v.conj())/prob
        state = (c.conj().T if gain else c)@state
        assert_allclose(correlation(state, ops), C, atol=2e-13)
        assert_allclose(np.vdot(state, state), accumulated, atol=2e-13)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_projector_outcomes_from_gain_loss_factorization(rank):
    rng = np.random.default_rng(317 + rank)
    ops = fermions(4)
    frame, _ = np.linalg.qr(rng.normal(size=(4, rank))+1j*rng.normal(size=(4, rank)))
    state = slater(frame, ops)
    C = frame@frame.conj().T
    chi = random_column(rng, 4)
    c = mode(ops, chi)
    n = c.conj().T@c
    P = np.outer(chi, chi.conj())
    p1 = np.vdot(chi, C@chi).real
    occupied = C-np.outer(C@chi, (C@chi).conj())/p1+P
    empty_vector = (np.eye(4)-C)@chi
    empty = C+np.outer(empty_vector, empty_vector.conj())/(1-p1)-P
    assert_allclose(correlation(n@state, ops), occupied, atol=2e-13)
    assert_allclose(correlation((np.eye(16)-n)@state, ops), empty, atol=2e-13)
    assert_allclose(np.vdot(n@state, n@state), p1, atol=2e-14)
    assert_allclose(np.vdot((np.eye(16)-n)@state, (np.eye(16)-n)@state), 1-p1, atol=2e-14)


def choi_setup():
    rng = np.random.default_rng(409)
    ops = fermions(4)
    ref = np.zeros(16, dtype=complex)
    ref[0] = 1
    for index in reversed(range(2)):
        ref = (ops[index].conj().T+ops[index+2].conj().T)@ref/np.sqrt(2)
    physical = ops[2:]
    gain = mode(physical, random_column(rng, 2))
    loss = mode(physical, random_column(rng, 2))
    matrix = .1*(rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2)))
    even = even_operator(matrix, physical)
    exact = loss@even@gain.conj().T
    return ops, ref, gain, loss, even, exact


@pytest.mark.parametrize("ratio", [.2, 1., 3.])
def test_normal_quadratic_but_anomalous_linear_choi_limit(ratio):
    ops, ref, gain, loss, even, exact = choi_setup()
    base = exact@ref
    assert np.linalg.norm(base) > .1
    C0 = correlation(base, ops)
    normal_errors, anomalous_errors = [], []
    for epsilon in (.002, .001):
        word = (loss+ratio*epsilon*loss.conj().T)@even@(gain.conj().T+epsilon*gain)
        state = word@ref
        normal_errors.append(np.linalg.norm(correlation(state, ops)-C0))
        anomalous_errors.append(np.linalg.norm(anomalous(state, ops)))
        assert_allclose(np.vdot(state, state), np.trace(word.conj().T@word)/16,
                        atol=2e-14)
    assert 3.98 < normal_errors[0]/normal_errors[1] < 4.02
    assert 1.99 < anomalous_errors[0]/anomalous_errors[1] < 2.01


@pytest.mark.parametrize("powers", [(1, 1), (1, 2), (2, 1)])
def test_unequal_regulator_paths_same_nonzero_endpoint(powers):
    ops, ref, gain, loss, even, exact = choi_setup()
    epsilon = .001
    e1, e2 = [epsilon**p for p in powers]
    word = (loss+e2*loss.conj().T)@even@(gain.conj().T+e1*gain)
    assert np.linalg.norm(correlation(word@ref, ops)-correlation(exact@ref, ops)) < 1e-5


def test_zero_word_path_dependence_and_alternating_lemma_counterexample():
    c = fermions(1)[0]
    cd = c.conj().T
    n = cd@c
    identity = np.eye(2)
    e1, e2 = .02, .03
    assert_allclose((cd+e2*c)@(cd+e1*c), e1*n+e2*(identity-n))
    assert_allclose(cd@cd, np.zeros((2, 2)))
    # Nonzero alternating GLGL disproves an unconditional "only if blocked".
    S = c@cd@c@cd
    assert_allclose(S, identity-n)
    assert_allclose(c@S, np.zeros((2, 2)))
    # Normalizing a zero exact word gives path-dependent probabilities.
    def normalized_weights(ratio):
        W = ratio*n+(identity-n)
        return W@W.conj().T/np.trace(W@W.conj().T)
    assert np.linalg.norm(normalized_weights(1)-normalized_weights(2)) > .1


def test_regulator_singular_values_are_not_long_time_growth_rates():
    c = fermions(1)[0]
    epsilon = .01
    v = (c.conj().T+epsilon*c)/np.sqrt(epsilon)
    t = transfer(v, majoranas([c]))
    assert np.max(np.linalg.svd(t, compute_uv=False)) > 99
    assert_allclose(np.linalg.matrix_power(t, 100), np.eye(2), atol=1e-10)
    assert_allclose((np.sqrt(epsilon)*v)@(np.sqrt(epsilon)*v),
                    epsilon*np.eye(2), atol=1e-14)


@pytest.mark.parametrize("seed", [3, 13, 23])
def test_maxmix_many_body_spectrum_from_occupations(seed):
    rng = np.random.default_rng(seed)
    count = 3
    ops = fermions(count)
    gain = mode(ops, random_column(rng, count)).conj().T
    loss = mode(ops, random_column(rng, count))
    matrix = .2*(rng.normal(size=(count, count))+1j*rng.normal(size=(count, count)))
    word = loss@even_operator(matrix, ops)@gain
    positive = word@word.conj().T
    Z = np.trace(positive).real
    rho = positive/Z
    C = np.array([[np.trace(rho@cj.conj().T@ci) for cj in ops] for ci in ops])
    nu = np.linalg.eigvalsh(C)
    weights = [np.prod([nu[a] if bits & (1 << a) else 1-nu[a]
                       for a in range(count)]) for bits in range(2**count)]
    assert_allclose(np.sort(Z*np.array(weights)),
                    np.sort(np.linalg.svd(word, compute_uv=False)**2), atol=2e-13)
