'''Small-matrix checks reused unchanged from channel_gap_discussion/build_note.py.

The source SHA-256 is recorded in validation.json. No campaign is simulated.
'''
import itertools
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment

def superop(kraus):
    return sum(np.kron(k.conj(), k) for k in kraus)


def dissipator(k):
    ident = np.eye(k.shape[0])
    n = k.conj().T @ k
    return np.kron(k.conj(), k) - (np.kron(ident, n) + np.kron(n.T, ident)) / 2


def spectrum_error(actual, expected):
    distances = abs(np.asarray(actual)[:, None] - np.asarray(expected)[None, :])
    ii, jj = linear_sum_assignment(distances)
    return float(np.max(distances[ii, jj]))


def check_math():
    rng = np.random.default_rng(20261005)
    n, d = 3, 8
    annihilators = []
    for i in range(n):
        c = np.zeros((d, d), complex)
        for state in range(d):
            if state & (1 << i):
                c[state ^ (1 << i), state] = (-1) ** ((state & ((1 << i) - 1)).bit_count())
        annihilators.append(c)
    ident = np.eye(d)
    s_total, a_total = np.eye(d * d), np.eye(n)
    raw = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = raw @ raw.conj().T
    rho /= np.trace(rho)

    def correlation(state):
        return np.array([[np.trace(state @ ci.conj().T @ cj)
                          for cj in annihilators] for ci in annihilators])

    errors = {}
    for j, target in enumerate([0, 1, 0, 1, 1, 0, 1]):
        v = rng.normal(size=n) + 1j * rng.normal(size=n)
        v /= np.linalg.norm(v)
        c = sum(x * op for x, op in zip(v, annihilators))
        number = c.conj().T @ c
        jump = c.conj().T if target else c
        kraus = [jump, number if target else ident - number]
        sj = superop(kraus)
        errors[f'reset_Id_plus_L_{j}'] = float(np.max(abs(
            sj - np.eye(d * d) - dissipator(jump) - dissipator(number))))
        p = np.outer(v, v.conj())
        q = np.eye(n) - p
        actual = sum(k @ rho @ k.conj().T for k in kraus)
        errors[f'covariance_reset_{j}'] = float(np.max(abs(
            correlation(actual) - q @ correlation(rho) @ q - target * p)))
        rho = actual
        s_total = sj @ s_total
        a_total = q @ a_total
    aeig = np.linalg.eigvals(a_total)
    radius = max(abs(aeig))
    neutral = [i + d * j for j in range(d) for i in range(d)
               if i.bit_count() == j.bit_count()]
    even = [i + d * j for j in range(d) for i in range(d)
            if (i.bit_count() - j.bit_count()) % 2 == 0]
    expected = []
    for p in range(n + 1):
        products = [np.prod(aeig[list(indices)]) for indices in itertools.combinations(range(n), p)]
        expected.extend(x * y.conjugate() for x in products for y in products)
    errors['complete_neutral_hierarchy_spectrum'] = spectrum_error(
        np.linalg.eigvals(s_total[np.ix_(neutral, neutral)]), expected)
    for label, indices, expected_r in [('neutral', neutral, radius**2), ('even', even, radius**2),
                                       ('unrestricted', list(range(d*d)), radius)]:
        eig = np.linalg.eigvals(s_total[np.ix_(indices, indices)])
        stationary = int(np.argmin(abs(eig - 1)))
        errors[label + '_stationary_eigenvalue'] = float(abs(eig[stationary] - 1))
        eig = np.delete(eig, stationary)
        errors[label + '_spectral_radius'] = float(abs(max(abs(eig)) - expected_r))
        if label == 'even':
            for p in [0.03, 0.2, 1.0]:
                errors[f'poisson_gap_p{p}'] = float(abs(1 - max(abs(1 - p + p*eig))
                                                       - p*(1 - radius**2)))
            errors['poisson_generator_gap'] = float(abs(min(1-eig.real) - (1-radius**2)))
    # Global maximally mixed replacement: every traceless multiplier is 1-p.
    t = np.outer((ident/d).reshape(-1, order='F'), ident.reshape(-1, order='F'))
    p = 0.37
    errors['global_depolarizing_spectrum'] = spectrum_error(
        np.linalg.eigvals((1-p)*np.eye(d*d)+p*t), [1]+[1-p]*(d*d-1))
    errors['global_poisson_semigroup'] = float(np.max(abs(
        expm(0.7*(t-np.eye(d*d))) - t - np.exp(-0.7)*(np.eye(d*d)-t))))
    # Local qubit depolarization, checked on the complete two-qubit operator space.
    paulis = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
    local_l = sum(dissipator(np.kron(pauli, np.eye(2))) / 4
                  + dissipator(np.kron(np.eye(2), pauli)) / 4 for pauli in paulis)
    errors['local_depolarizing_spectrum'] = spectrum_error(
        np.linalg.eigvals(local_l), [0]+[-1]*6+[-2]*9)
    c = np.array([[0, 1], [0, 0]])
    errors['balanced_fermionic_spectrum'] = spectrum_error(
        np.linalg.eigvals(dissipator(c) + dissipator(c.T)), [0, -1, -1, -2])
    assert max(errors.values()) < 1e-11, errors
    return dict(test_count=len(errors), maximum_error=max(errors.values()), errors=errors)


def check_pedagogy():
    """Check the worked examples added in the one-column pedagogical revision."""
    errors = {}
    c = np.array([[0, 1], [0, 0]], dtype=complex)
    number = c.conj().T @ c
    rho = np.array([[0.6, 0.15+0.07j], [0.15-0.07j, 0.4]])
    for target in [0, 1]:
        jump = c.conj().T if target else c
        proj = number if target else np.eye(2)-number
        actual = jump@rho@jump.conj().T + proj@rho@proj
        errors[f'one_mode_reset_target{target}'] = float(
            np.max(abs(actual-np.diag([1-target, target]))))
    for theta in [0.0, 0.2, 0.7, np.pi/2]:
        ct, st = np.cos(theta), np.sin(theta)
        v1, v2 = np.array([1., 0.]), np.array([ct, st])
        q1, q2 = np.eye(2)-np.outer(v1, v1), np.eye(2)-np.outer(v2, v2)
        a = q2@q1
        name = f'two_mode_theta{theta:.6f}'
        errors[name+'_product'] = float(np.max(abs(a-np.array([[0, -ct*st], [0, ct**2]]))))
        errors[name+'_spectrum'] = spectrum_error(np.linalg.eigvals(a), [0, ct**2])
        errors[name+'_covariance_spectrum'] = spectrum_error(
            np.linalg.eigvals(np.kron(a.conj(), a)), [0, 0, 0, ct**4])
        errors[name+'_singular_value'] = float(abs(np.linalg.svd(a, compute_uv=False)[0]-abs(ct)))
    a = np.array([[0.3, 0.2j], [0, 0.4]])
    b = np.diag([0.1, 0.2])
    g0 = np.array([[0.4, 0.1j], [-0.1j, 0.6]])
    evolved = g0.copy()
    for _ in range(4):
        evolved = a@evolved@a.conj().T+b
    a4 = np.linalg.matrix_power(a, 4)
    closed = a4@g0@a4.conj().T + sum(
        np.linalg.matrix_power(a, k)@b@np.linalg.matrix_power(a, k).conj().T
        for k in range(4))
    errors['finite_cycle_affine_solution'] = float(np.max(abs(evolved-closed)))
    for steady in [0., 0.5, 0.99]:
        q, initial, occupation = 0.37, 0.2, 0.2
        for _ in range(7):
            occupation = q*occupation+(1-q)*steady
        errors[f'scalar_source_stationary{steady}'] = float(
            abs(occupation-steady-q**7*(initial-steady)))
    ny = 7
    k = 2*np.pi*np.arange(ny)/ny
    kernel = sum(np.exp(1j*r*(k[:, None]-k[None, :])) for r in range(ny))/ny
    errors['translation_twirl_momentum_selection'] = float(np.max(abs(kernel-np.eye(ny))))
    assert max(errors.values()) < 1e-11, errors
    return dict(test_count=len(errors), maximum_error=max(errors.values()), errors=errors)


def check_additions():
    """Periodicity, fixed spaces, Poissonization, and left/right conventions."""
    errors = {}
    c = np.array([[0, 1], [0, 0]], dtype=complex)
    flip = superop([c, c.conj().T])
    eye = np.eye(4)
    state = (np.eye(2) / 2).reshape(-1, order='F')
    z = np.diag([1, -1]).reshape(-1, order='F')
    errors['period_two_spectrum'] = spectrum_error(np.linalg.eigvals(flip), [1, -1, 0, 0])
    errors['period_two_stationary_state'] = float(np.max(abs(flip @ state - state)))
    errors['period_two_stationary_dimension'] = float(abs(4 - np.linalg.matrix_rank(flip-eye) - 1))
    errors['period_two_alternating_population'] = float(np.max(abs(flip @ z + z)))
    multipliers = np.linalg.eigvals(flip)
    nonstationary = np.delete(multipliers, np.argmin(abs(multipliers-1)))
    errors['period_two_absolute_gap_zero'] = float(abs(1-max(abs(nonstationary))))
    gen = 0.7 * (flip-eye)
    errors['period_two_poissonized_spectrum'] = spectrum_error(
        np.linalg.eigvals(gen), [0, -1.4, -0.7, -0.7])
    errors['period_two_poissonized_stationary_dimension'] = float(
        abs(4 - np.linalg.matrix_rank(gen) - 1))
    errors['period_two_poissonized_balanced_gain_loss'] = float(np.max(abs(
        gen - 0.7*(dissipator(c)+dissipator(c.conj().T)))))
    errors['period_two_finite_time_spectrum'] = spectrum_error(
        np.linalg.eigvals(expm(gen*0.4)), [1, np.exp(-0.56), np.exp(-0.28), np.exp(-0.28)])
    for p in [0.2, 0.5, 0.9, 1.0]:
        lazy = (1-p)*eye + p*flip
        errors[f'period_two_lazy_spectrum_{p}'] = spectrum_error(
            np.linalg.eigvals(lazy), [1, 1-2*p, 1-p, 1-p])
        errors[f'period_two_lazy_fixed_space_{p}'] = float(
            abs(4 - np.linalg.matrix_rank(lazy-eye) - 1))
    # Two stationary states produce a trace-zero fixed direction; do not discard it.
    dephase = superop([np.diag([1, 0]), np.diag([0, 1])])
    errors['multiple_stationary_states_dimension'] = float(
        abs(4-np.linalg.matrix_rank(dephase-eye)-2))
    errors['multiple_stationary_states_trace_zero_direction'] = float(np.max(abs(dephase@z-z)))

    # Independent three-mode reset product: explicit dominant observable and odd similarity.
    n, d = 3, 8
    cs = []
    for i in range(n):
        op = np.zeros((d, d), complex)
        for state_index in range(d):
            if state_index & (1 << i):
                op[state_index ^ (1 << i), state_index] = (-1)**(
                    (state_index & ((1 << i)-1)).bit_count())
        cs.append(op)
    rng = np.random.default_rng(20261006)
    s, a, b = np.eye(d*d), np.eye(n), np.zeros((n, n), complex)
    parity = np.diag([(-1)**i.bit_count() for i in range(d)])
    similarity = np.kron(np.eye(d), parity)
    for target in [1, 0, 1, 0, 1, 0, 1]:
        v = rng.normal(size=n)+1j*rng.normal(size=n)
        v /= np.linalg.norm(v)
        chi = sum(x*op for x, op in zip(v, cs))
        number = chi.conj().T @ chi
        jump = chi.conj().T if target else chi
        proj = number if target else np.eye(d)-number
        sj = superop([jump, proj])
        predicted = -superop([jump]).conj().T + superop([proj]).conj().T
        errors[f'odd_similarity_target{target}'] = max(
            errors.get(f'odd_similarity_target{target}', 0),
            float(np.max(abs(similarity @ sj.conj().T @ similarity-predicted))))
        q = np.eye(n)-np.outer(v, v.conj())
        a, b, s = q@a, q@b@q + target*np.outer(v, v.conj()), sj@s
    gss = np.linalg.solve(np.eye(n*n)-np.kron(a.conj(), a),
                         b.reshape(-1, order='F')).reshape((n, n), order='F')
    vals, vecs = np.linalg.eig(a.conj().T)
    which = np.argmax(abs(vals))
    u, multiplier = vecs[:, which], abs(vals[which])**2
    observable = sum(u[i].conjugate()*u[j]*(cs[i].conj().T@cs[j]-gss[i, j]*np.eye(d))
                     for i in range(n) for j in range(n))
    ov = observable.reshape(-1, order='F')
    errors['dominant_left_observable'] = float(np.max(abs(s.conj().T@ov-multiplier*ov)))
    av, aw = np.linalg.eig(a)
    w = aw[:, np.argmax(abs(av))]
    ww = np.outer(w, w.conj())
    errors['dominant_right_covariance'] = float(np.max(abs(a@ww@a.conj().T-multiplier*ww)))
    errors['covariance_pairwise_spectrum'] = spectrum_error(
        np.linalg.eigvals(np.kron(a.conj(), a)), (av[:, None]*av.conj()[None, :]).ravel())
    assert max(errors.values()) < 1e-11, errors
    return dict(test_count=len(errors), maximum_error=max(errors.values()), errors=errors)
