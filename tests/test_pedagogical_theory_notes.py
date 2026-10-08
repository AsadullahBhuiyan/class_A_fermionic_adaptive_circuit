"""Small algebra checks for the two theory notes; no simulation/data inputs."""

from pathlib import Path
import re

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm


ROOT = Path(__file__).resolve().parents[1]
METHODS = ROOT / "00_WORKSPACE/CURRENT/Paper Methods"


def fermions(count):
    annihilator = np.array([[0, 1], [0, 0]], dtype=complex)
    parity = np.diag([1, -1])
    result = []
    for index in range(count):
        operator = np.ones((1, 1), dtype=complex)
        for site in range(count):
            factor = parity if site < index else (
                annihilator if site == index else np.eye(2)
            )
            operator = np.kron(operator, factor)
        result.append(operator)
    return result


def operator_rank(operator, left_dimension, right_dimension):
    reshaped = operator.reshape(
        left_dimension, right_dimension, left_dimension, right_dimension
    ).transpose(0, 2, 1, 3)
    return np.linalg.matrix_rank(
        reshaped.reshape(left_dimension**2, right_dimension**2), tol=1e-11
    )


def test_parity_expansion_kraus_completeness_and_operator_ranks():
    rng = np.random.default_rng(20261006)
    coefficients = rng.normal(size=4) + 1j * rng.normal(size=4)
    coefficients /= np.linalg.norm(coefficients)
    modes = fermions(4)
    chi = sum(value * mode for value, mode in zip(coefficients, modes))
    local = fermions(2)
    left = coefficients[0] * local[0] + coefficients[1] * local[1]
    right = coefficients[2] * local[0] + coefficients[3] * local[1]
    parity = np.diag([1, -1, -1, 1])
    identity = np.eye(16)
    np.testing.assert_allclose(
        chi, np.kron(left, np.eye(4)) + np.kron(parity, right), atol=1e-14
    )
    number = chi.conj().T @ chi
    expansion = (
        np.kron(left.conj().T @ left, np.eye(4))
        + np.kron(left.conj().T @ parity, right)
        + np.kron(parity @ left, right.conj().T)
        + np.kron(np.eye(4), right.conj().T @ right)
    )
    np.testing.assert_allclose(number, expansion, atol=1e-14)
    assert [operator_rank(op, 4, 4) for op in
            (chi, chi.conj().T, number, identity - number)] == [2, 2, 4, 4]
    projectors = (identity - number, number)
    for target, branches in enumerate(
        ((projectors[0], chi), (chi.conj().T, projectors[1]))
    ):
        completeness = np.zeros_like(chi)
        for outcome, branch in enumerate(branches):
            np.testing.assert_allclose(
                branch.conj().T @ branch, projectors[outcome], atol=1e-14
            )
            np.testing.assert_allclose(
                projectors[target] @ branch, branch, atol=1e-14
            )
            completeness += branch.conj().T @ branch
            # One crossing operation on a product state obeys rank multiplication.
            state = np.kron(rng.normal(size=4), rng.normal(size=4))
            output = branch @ state
            assert np.linalg.matrix_rank(output.reshape(4, 4), tol=1e-11) <= (
                operator_rank(branch, 4, 4)
            )
        np.testing.assert_allclose(completeness, identity, atol=1e-14)


def test_fswap_dilation_including_spectator_parity():
    c0, c1, ancillary = fermions(3)
    chi = (c0 + 1j * c1) / np.sqrt(2)
    number = chi.conj().T @ chi
    anc_number = ancillary.conj().T @ ancillary
    swap = np.eye(8) - number - anc_number
    swap += chi.conj().T @ ancillary + ancillary.conj().T @ chi
    np.testing.assert_allclose(swap.conj().T @ swap, np.eye(8), atol=1e-14)
    p0, p1 = fermions(2)
    physical_chi = (p0 + 1j * p1) / np.sqrt(2)
    physical_number = physical_chi.conj().T @ physical_chi
    physical_parity = np.diag([1, -1, -1, 1])
    expected = (
        (np.eye(4) - physical_number, physical_chi),
        (physical_chi.conj().T, physical_number),
    )
    for target in (0, 1):
        for outcome in (0, 1):
            projection = number if outcome else np.eye(8) - number
            joint_branch = (
                projection if outcome == target else swap @ projection
            )
            outgoing = target if outcome == target else outcome
            block = joint_branch[outgoing::2, target::2]
            other = joint_branch[1 - outgoing::2, target::2]
            np.testing.assert_allclose(other, 0, atol=1e-14)
            # A fixed ordinary tensor ordering can differ by an input-parity
            # phase from the graded Kraus operator; physical branches agree.
            for parity in (-1, 1):
                sector = (np.eye(4) + parity * physical_parity) / 2
                actual = block @ sector
                wanted = expected[target][outcome] @ sector
                assert min(np.linalg.norm(actual - wanted),
                           np.linalg.norm(actual + wanted)) < 1e-13


def test_modular_transpose_and_gaussian_fock_space_entropy():
    rng = np.random.default_rng(173)
    raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    h = (raw + raw.conj().T) / 3
    modes = fermions(3)
    many_body_h = sum(
        h[i, j] * modes[i].conj().T @ modes[j]
        for i in range(3) for j in range(3)
    )
    rho = expm(-many_body_h)
    rho /= np.trace(rho)
    correlation = np.array([
        [np.trace(rho @ modes[i].conj().T @ modes[j]) for j in range(3)]
        for i in range(3)
    ])
    np.testing.assert_allclose(
        correlation, np.linalg.inv(np.eye(3) + expm(h.T)), atol=1e-13
    )
    occupations, unitary = np.linalg.eigh(correlation)
    energies = np.log((1 - occupations) / occupations)
    reconstructed = (unitary * energies) @ unitary.conj().T
    np.testing.assert_allclose(reconstructed.T, h, atol=1e-13)
    # The test must genuinely distinguish transpose from no transpose.
    assert np.linalg.norm(reconstructed - h) > 0.1
    natural = [
        sum(unitary[i, a] * modes[i] for i in range(3)) for a in range(3)
    ]
    natural_G = np.array([
        [np.trace(rho @ fa.conj().T @ fb) for fb in natural] for fa in natural
    ])
    np.testing.assert_allclose(natural_G, np.diag(occupations), atol=1e-13)
    probabilities = np.linalg.eigvalsh(rho)
    exact_s1 = -np.sum(probabilities * np.log(probabilities))
    spectral_s1 = -np.sum(
        occupations * np.log(occupations)
        + (1 - occupations) * np.log(1 - occupations)
    )
    assert spectral_s1 == pytest.approx(exact_s1, abs=1e-13)
    for q in (0.5, 2, 3):
        spectral = np.sum(np.log(
            occupations**q + (1 - occupations)**q
        )) / (1 - q)
        direct = np.log(np.sum(probabilities**q)) / (1 - q)
        assert spectral == pytest.approx(direct, abs=1e-13)
    q = 1 + 1e-6
    near_one = np.sum(np.log(
        occupations**q + (1 - occupations)**q
    )) / (1 - q)
    assert near_one == pytest.approx(spectral_s1, abs=1e-6)


@pytest.mark.parametrize("q", [0.5, 1, 2, 3])
def test_modular_kernel_integrals_and_branch_factors(q):
    def entropy_kernel(energy):
        if q == 1:
            return np.logaddexp(0, -energy) + energy / (1 + np.exp(energy))
        return (
            np.logaddexp(0, -q * energy)
            - q * np.logaddexp(0, -energy)
        ) / (1 - q)

    entropy_integral = 2 * quad(entropy_kernel, 0, 200, epsabs=1e-12)[0]
    variance_integral = 2 * quad(
        lambda x: 1 / (4 * np.cosh(x / 2)**2), 0, 200, epsabs=1e-12
    )[0]
    assert variance_integral == pytest.approx(1, abs=1e-12)
    assert entropy_integral == pytest.approx(
        np.pi**2 * (1 + 1/q) / 6, abs=1e-11
    )
    for branches in (1, 2):
        a_s = branches * (1 + 1/q) / 12
        a_f = branches / (2 * np.pi**2)
        assert a_s / a_f == pytest.approx(entropy_integral, abs=1e-11)
    assert (2 * (1 + 1/q) / 12) * 6 / (1 + 1/q) == pytest.approx(1)


def test_current_integration_and_pure_mixed_cross_blocks():
    length, interval, level = 31.0, 8.3, 1.7
    step = 0.002
    variance = lambda x: level / (2*np.pi**2) * np.log(
        length / np.pi * np.sin(np.pi*x/length)
    )
    second_derivative = (
        variance(interval + step) - 2*variance(interval)
        + variance(interval - step)
    ) / step**2
    correlation = -level / (4*np.pi**2) * (
        np.pi/length / np.sin(np.pi*interval/length)
    )**2
    assert second_derivative == pytest.approx(2*correlation, abs=2e-10)
    assert variance(interval) == pytest.approx(variance(length-interval))
    rng = np.random.default_rng(20)
    raw = rng.normal(size=(6, 6)) + 1j*rng.normal(size=(6, 6))
    unitary, _ = np.linalg.qr(raw)
    for occupations in (np.array([0, 0, 0, 1, 1, 1]),
                        np.linspace(0.1, 0.9, 6)):
        G = (unitary * occupations) @ unitary.conj().T
        GA = G[:3, :3]
        V = GA @ (np.eye(3) - GA)
        cross = G[:3, 3:] @ G[3:, :3]
        np.testing.assert_allclose(
            V, cross + (G-G@G)[:3, :3], atol=1e-13
        )
        if np.all((occupations == 0) | (occupations == 1)):
            np.testing.assert_allclose(V, cross, atol=1e-13)


@pytest.mark.parametrize("stem", [
    "adaptive_entangling_capacity", "kac_moody_renyi_validation"
])
def test_theory_notes_are_self_contained_onecolumn_documents(stem):
    directory = METHODS / stem
    source = (directory / f"{stem}.tex").read_text()
    assert source.startswith(
        r"\documentclass[aps,prb,onecolumn,nofootinbib,superscriptaddress]{revtex4-2}"
    )
    assert r"\tableofcontents" in source
    assert r"\newcommand{\ket}[1]{\lvert #1\rangle}" in source
    assert r"\ketket" not in source
    assert r"\braketket" not in source
    assert r"\rangle\!\rangle" not in source
    assert r"\langle\!\langle" not in source
    assert not re.search(r"\\(?:input|include|includegraphics)\b", source)
    assert not re.search(r"\\mathbbm?\{?(?:1|I)\}?", source)
    assert "geometry}" not in source
    labels = re.findall(r"\\label\{([^}]+)\}", source)
    assert len(labels) == len(set(labels))
    references = re.findall(r"\\(?:eqref|ref)\{([^}]+)\}", source)
    assert set(references) <= set(labels)
    bib = (directory / "references.bib").read_text()
    bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = {
        key.strip()
        for citation in re.findall(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}", source)
        for key in citation.split(",")
    }
    assert cited <= bibkeys
