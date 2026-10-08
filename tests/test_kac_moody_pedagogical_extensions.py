"""Independent algebra checks for current modes and interface-defect spectra.

These are small quadratures and finite sums, not adaptive-circuit simulations.
The defect convention is that of Eisler--Peschel (2010), Eqs. (17)--(32),
and Peschel--Eisler (2012), Secs. II and IV: x and omega are *half* modular
energies. The physical signed modular energies of the XX model are +/-2 omega.
"""

from pathlib import Path
import re

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit, spence, xlogy


def _omega(x, amplitude):
    return np.arccosh(np.cosh(x) / amplitude)


def _entropy_kernel(energy, order):
    energy = np.abs(energy)
    if order == 1:
        return np.logaddexp(0, -energy) + energy * expit(-energy)
    if np.isinf(order):
        return np.logaddexp(0, -energy)
    return (
        np.logaddexp(0, -order * energy)
        - order * np.logaddexp(0, -energy)
    ) / (1 - order)


def _j_integral(amplitude, order):
    if amplitude == 0:
        return 0.0
    return quad(
        lambda x: _entropy_kernel(2 * _omega(x, amplitude), order),
        0,
        60,
        epsabs=2e-12,
        epsrel=2e-12,
    )[0]


def _von_neumann_dilog(amplitude):
    if amplitude == 0:
        return 0.0
    # scipy.special.spence(z) = Li_2(1-z); xlogy handles s=1 safely.
    weighted_logs = xlogy(1 + amplitude, 1 + amplitude) + xlogy(
        1 - amplitude, 1 - amplitude
    )
    return -0.5 * (
        weighted_logs * np.log(amplitude)
        + (1 + amplitude) * spence(1 + amplitude)
        + (1 - amplitude) * spence(1 - amplitude)
    )


def _j_closed(amplitude, order):
    if order == 0.5:
        angle = np.arcsin(amplitude)
        return angle * (np.pi - angle) / 2
    if order == 1:
        return _von_neumann_dilog(amplitude)
    if order == 2:
        return np.arcsin(amplitude / np.sqrt(2)) ** 2
    if order == 3:
        return np.arcsin(np.sqrt(3) * amplitude / 2) ** 2 / 2
    if np.isinf(order):
        return spence(1 - amplitude**2) / 4
    raise ValueError("No closed form provided for this order")


@pytest.mark.parametrize("amplitude", [0.07, 0.35, 0.8, 1.0])
def test_defect_half_energy_mapping_and_susceptibility(amplitude):
    reference = np.array([0.0, 0.03, 0.4, 1.3, 5.0])
    defect = _omega(reference, amplitude)
    np.testing.assert_allclose(
        np.cosh(defect), np.cosh(reference) / amplitude, rtol=2e-15
    )
    assert defect[0] == pytest.approx(np.arccosh(1 / amplitude))
    assert np.all(np.diff(defect) > 0)

    reference_occupations = expit(-2 * reference)
    defect_occupations = expit(-2 * defect)
    reference_v = reference_occupations * (1 - reference_occupations)
    defect_v = defect_occupations * (1 - defect_occupations)
    np.testing.assert_allclose(
        defect_v, amplitude**2 * reference_v, rtol=3e-15
    )
    np.testing.assert_allclose(
        defect_v, 1 / (4 * np.cosh(defect) ** 2), rtol=3e-15
    )


def test_bond_amplitude_transmission_and_strong_weak_duality():
    bonds = np.array([0.03, 0.2, 0.7, 1.0, 1.6, 9.0])
    amplitudes = 2 * bonds / (1 + bonds**2)
    np.testing.assert_allclose(amplitudes, np.sin(2 * np.arctan(bonds)))
    np.testing.assert_allclose(
        amplitudes, 2 / bonds / (1 + (1 / bonds) ** 2)
    )
    phase = 2 * np.arctan(bonds) - np.pi / 2
    np.testing.assert_allclose(np.cos(phase) ** 2, amplitudes**2)
    assert np.all((amplitudes > 0) & (amplitudes <= 1))


@pytest.mark.parametrize("amplitude", [0.12, 0.6, 1.0])
def test_signed_density_jacobian_and_integrated_mode_count(amplitude):
    def inverse_energy(energy):
        return 2 * np.arccosh(amplitude * np.cosh(energy / 2))

    def jacobian(energy):
        return amplitude * np.sinh(abs(energy) / 2) / np.sqrt(
            amplitude**2 * np.cosh(energy / 2) ** 2 - 1
        )

    # Stay away from the square-root edge; the inverse-map count tests the
    # Jacobian without asking a quadrature to resolve an endpoint singularity.
    lower, upper = 0.17, 2.7
    energy_lower, energy_upper = 2 * _omega(
        np.array([lower, upper]), amplitude
    )
    physical_energy = 2 * _omega(0.83, amplitude)
    step = 2e-5
    derivative = (
        inverse_energy(physical_energy + step)
        - inverse_energy(physical_energy - step)
    ) / (2 * step)
    assert derivative == pytest.approx(jacobian(physical_energy), rel=2e-9)
    assert jacobian(-physical_energy) == jacobian(physical_energy)
    positive_count = quad(jacobian, energy_lower, energy_upper, epsabs=1e-12)[0]
    assert positive_count == pytest.approx(2 * (upper - lower), abs=1e-11)
    signed_count = 2 * positive_count
    assert signed_count == pytest.approx(4 * (upper - lower), abs=2e-11)
    if amplitude == 1:
        assert jacobian(physical_energy) == pytest.approx(1)
    else:
        assert energy_lower > 2 * np.arccosh(1 / amplitude) > 0
        assert jacobian(physical_energy) > 1


@pytest.mark.parametrize("amplitude", [0.07, 0.4, 0.85, 1.0])
@pytest.mark.parametrize("order", [0.5, 1, 2, 3, np.inf])
def test_defect_renyi_quadratures_match_closed_forms(amplitude, order):
    assert _j_integral(amplitude, order) == pytest.approx(
        _j_closed(amplitude, order), abs=2e-12, rel=2e-11
    )


@pytest.mark.parametrize("order", [0.5, 1, 2, 3, np.inf])
def test_transparent_and_disconnected_defect_limits(order):
    expected = np.pi**2 * (1 + 1 / order) / 24
    assert _j_closed(1, order) == pytest.approx(expected, abs=1e-13)
    assert _j_closed(0, order) == pytest.approx(0, abs=1e-15)
    # These factors refer to the same one-cut open-chain geometry: the XX
    # entropy has twice the mode content of TI, not an extra pair of cuts.
    ti_slope = expected / np.pi**2
    xx_slope = 2 * expected / np.pi**2
    assert ti_slope == pytest.approx((1 + 1 / order) / 24)
    assert xx_slope == pytest.approx((1 + 1 / order) / 12)


@pytest.mark.parametrize("amplitude", [0.2, 0.6, 0.9])
def test_von_neumann_limit_and_dilog_second_derivative(amplitude):
    q_step = 2e-4
    near_one = (
        _j_integral(amplitude, 1 - q_step)
        + _j_integral(amplitude, 1 + q_step)
    ) / 2
    assert near_one == pytest.approx(_j_integral(amplitude, 1), abs=2e-8)
    s_step = 2e-4
    second_derivative = (
        _von_neumann_dilog(amplitude + s_step)
        - 2 * _von_neumann_dilog(amplitude)
        + _von_neumann_dilog(amplitude - s_step)
    ) / s_step**2
    assert second_derivative == pytest.approx(
        -np.log(amplitude) / (1 - amplitude**2), abs=3e-7
    )


def test_small_transmission_order_dependence():
    amplitude = 0.001
    leading_s1 = amplitude**2 * (0.75 - 0.5 * np.log(amplitude))
    assert _j_integral(amplitude, 1) == pytest.approx(leading_s1, rel=8e-7)
    for order in (2, 3, np.inf):
        coefficient = 0.25 if np.isinf(order) else order / (4 * (order - 1))
        assert _j_closed(amplitude, order) / amplitude**2 == pytest.approx(
            coefficient, rel=1e-6
        )
    assert _j_closed(amplitude, 0.5) / amplitude == pytest.approx(
        np.pi / 2, rel=7e-4
    )


@pytest.mark.parametrize("amplitude", [0.1, 0.5, 1.0])
def test_xx_one_cut_variance_factor_and_nonconstant_entropy_ratio(amplitude):
    variance_integral = quad(
        lambda x: 1 / (4 * np.cosh(_omega(x, amplitude)) ** 2),
        0,
        60,
        epsabs=1e-13,
    )[0]
    assert variance_integral == pytest.approx(amplitude**2 / 4, abs=1e-13)
    variance_slope = 2 * variance_integral / np.pi**2
    assert variance_slope == pytest.approx(amplitude**2 / (2 * np.pi**2))
    entropy_slope = 2 * _j_integral(amplitude, 1) / np.pi**2
    ratio = entropy_slope / variance_slope
    assert ratio == pytest.approx(4 * _von_neumann_dilog(amplitude) / amplitude**2)
    if amplitude == 1:
        assert ratio == pytest.approx(np.pi**2 / 3)
    else:
        assert ratio > np.pi**2 / 3


def _regulated_variance(interval, length, level, radius):
    angle = 2 * np.pi * interval / length
    numerator = (1 - radius) ** 2 + 4 * radius * np.sin(angle / 2) ** 2
    return level / (4 * np.pi**2) * np.log(numerator / (1 - radius) ** 2)


@pytest.mark.parametrize("radius", [0.35, 0.8, 0.97])
def test_regulated_current_mode_sum_contact_and_integrated_variance(radius):
    length, interval, level = 17.0, 5.3, 1.7
    modes = np.arange(1, 2501, dtype=float)
    angle = 2 * np.pi * interval / length
    factors = radius**modes
    current_sum = level / length**2 * np.sum(modes * factors * np.cos(modes * angle))
    complex_radius = radius * np.exp(1j * angle)
    current_closed = level / length**2 * np.real(
        complex_radius / (1 - complex_radius) ** 2
    )
    assert current_sum == pytest.approx(current_closed, abs=2e-14)
    variance_sum = level / (2 * np.pi**2) * np.sum(
        factors * (1 - np.cos(modes * angle)) / modes
    )
    variance_closed = _regulated_variance(interval, length, level, radius)
    assert variance_sum == pytest.approx(variance_closed, abs=2e-14)
    assert variance_closed >= 0
    step = 0.002
    curvature = (
        _regulated_variance(interval + step, length, level, radius)
        - 2 * variance_closed
        + _regulated_variance(interval - step, length, level, radius)
    ) / step**2
    assert curvature == pytest.approx(2 * current_closed, abs=3e-10)
    assert _regulated_variance(0, length, level, radius) == 0
    assert _regulated_variance(length, length, level, radius) == pytest.approx(0)
    assert _regulated_variance(length - interval, length, level, radius) == (
        pytest.approx(variance_closed)
    )
    # The finite regulator keeps a positive contact peak, while its integral
    # over the entire circle vanishes in a fixed-total-charge sector.
    contact = level / length**2 * np.sum(modes * factors)
    assert contact > 0
    assert contact == pytest.approx(level * radius / (length**2 * (1 - radius) ** 2))
    integrated_current = quad(
        lambda y: level / length**2 * np.real(
            radius * np.exp(2j * np.pi * y / length)
            / (1 - radius * np.exp(2j * np.pi * y / length)) ** 2
        ),
        0,
        length,
        epsabs=2e-12,
    )[0]
    assert integrated_current == pytest.approx(0, abs=3e-12)


def test_regulator_continuum_limit_and_charge_zero_mode():
    length, interval, level, cutoff = 31.0, 8.7, 1.0, 1e-4
    radius = np.exp(-2 * np.pi * cutoff / length)
    regulated = _regulated_variance(interval, length, level, radius)
    continuum = level / (2 * np.pi**2) * np.log(
        length / (np.pi * cutoff) * np.sin(np.pi * interval / length)
    )
    assert regulated == pytest.approx(continuum, abs=1e-10)

    # A mixture of oscillator vacua carrying different global charges has
    # exactly this extra zero-mode contribution; it is not removed by a UV
    # regulator or by the current commutator.
    charges = np.array([-1.0, 0.0, 2.0])
    weights = np.array([0.2, 0.5, 0.3])
    mean = weights @ charges
    total_charge_variance = weights @ (charges - mean) ** 2

    def mixed_variance(size):
        oscillator = _regulated_variance(size, length, level, radius)
        return oscillator + (size / length) ** 2 * total_charge_variance

    assert mixed_variance(length) == pytest.approx(total_charge_variance)
    assert mixed_variance(0) == 0
    fraction = interval / length
    conditional_means = fraction * charges
    classical_variance = weights @ (conditional_means - weights @ conditional_means) ** 2
    assert classical_variance == pytest.approx(fraction**2 * total_charge_variance)
    assert mixed_variance(interval) - mixed_variance(length - interval) == (
        pytest.approx((2 * fraction - 1) * total_charge_variance)
    )


@pytest.mark.parametrize("velocity,forward,backward", [
    (1.0, 0.0, 0.0), (1.2, 0.7, 0.3), (0.8, 0.2, -0.9),
])
def test_density_dictionary_luttinger_stiffnesses_and_legendre_transform(
    velocity, forward, backward
):
    # Columns are (d_y Phi, Pi=d_y varphi), rows are (+,-). The user's
    # two physical chiral densities both have the same positive prefactor.
    chiral_map = 0.5 * np.array([[1.0, -1.0], [1.0, 1.0]])
    density_map = chiral_map / np.sqrt(np.pi)
    density_couplings = np.array([[forward, backward], [backward, forward]])
    quadratic = velocity * np.eye(2) / 2
    quadratic += density_map.T @ density_couplings @ density_map
    momentum_stiffness = velocity + (forward - backward) / np.pi
    spatial_stiffness = velocity + (forward + backward) / np.pi
    np.testing.assert_allclose(
        quadratic,
        np.diag([spatial_stiffness, momentum_stiffness]) / 2,
        atol=2e-16,
    )
    assert min(momentum_stiffness, spatial_stiffness) > 0
    sound_velocity = np.sqrt(momentum_stiffness * spatial_stiffness)
    parameter = np.sqrt(momentum_stiffness / spatial_stiffness)
    assert sound_velocity * parameter == pytest.approx(momentum_stiffness)
    assert sound_velocity / parameter == pytest.approx(spatial_stiffness)

    for gradient, momentum, time_derivative in ((0.3, -0.7, 1.1), (-0.8, 0.5, -0.2)):
        densities = density_map @ np.array([gradient, momentum])
        assert 2 * np.prod(densities) == pytest.approx(
            (gradient**2 - momentum**2) / (2 * np.pi)
        )
        assert densities @ densities == pytest.approx(
            (gradient**2 + momentum**2) / (2 * np.pi)
        )
        phase_space_lagrangian = momentum * time_derivative - (
            momentum_stiffness * momentum**2 + spatial_stiffness * gradient**2
        ) / 2
        effective_lagrangian = (
            time_derivative**2 / sound_velocity - sound_velocity * gradient**2
        ) / (2 * parameter)
        square = momentum_stiffness / 2 * (
            momentum - time_derivative / momentum_stiffness
        ) ** 2
        assert phase_space_lagrangian == pytest.approx(effective_lagrangian - square)


@pytest.mark.parametrize("sound_velocity,parameter", [(0.7, 0.4), (1.3, 1.0), (2.1, 2.5)])
def test_luttinger_rescaled_eigenfields_propagate_in_opposite_directions(
    sound_velocity, parameter
):
    # Hamilton's equations for (Phi,varphi), after discarding a uniform
    # integration mode, are d_t fields = propagation @ d_y fields.
    propagation = np.array([
        [0.0, sound_velocity * parameter],
        [sound_velocity / parameter, 0.0],
    ])
    eigenfields = 0.5 * np.array([
        [1 / np.sqrt(parameter), -np.sqrt(parameter)],
        [1 / np.sqrt(parameter), np.sqrt(parameter)],
    ])
    np.testing.assert_allclose(
        eigenfields @ propagation,
        np.diag([-sound_velocity, sound_velocity]) @ eigenfields,
        atol=5e-16,
    )
    # Rescaling the field and its canonical momentum preserves their bracket.
    canonical_form = np.array([[0.0, 1.0], [-1.0, 0.0]])
    canonical_rescaling = np.diag([1 / np.sqrt(parameter), np.sqrt(parameter)])
    np.testing.assert_allclose(
        canonical_rescaling @ canonical_form @ canonical_rescaling.T,
        canonical_form,
        atol=2e-16,
    )
    physical_density = np.array([1.0, 0.0]) / np.sqrt(np.pi)
    np.testing.assert_allclose(
        np.sqrt(parameter / np.pi) * np.array([1.0, 1.0]) @ eigenfields,
        physical_density,
        atol=2e-16,
    )
    physical_current = np.array([0.0, -sound_velocity * parameter]) / np.sqrt(np.pi)
    np.testing.assert_allclose(
        physical_density @ propagation + physical_current, 0, atol=2e-15
    )


@pytest.mark.parametrize("parameter", [0.35, 1.0, 2.7])
def test_scalar_covariance_gives_luttinger_charge_variance(parameter):
    # Frequency integration of the inverse Euclidean quadratic kernel fixes
    # the equal-time covariance K/(2|p|), independently of the velocity.
    momentum = 0.83
    for velocity in (0.6, 1.9):
        covariance = quad(
            lambda frequency: parameter
            / (frequency**2 / velocity + velocity * momentum**2) / np.pi,
            0,
            np.inf,
            epsabs=1e-12,
        )[0]
        assert covariance == pytest.approx(parameter / (2 * momentum), abs=1e-12)

    length, interval, radius = 29.0, 7.2, 0.94
    modes = np.arange(1, 1501, dtype=float)
    weights = parameter / (2 * np.pi) * radius**modes / modes
    coincident = np.sum(weights)
    separated = np.sum(weights * np.cos(2 * np.pi * modes * interval / length))
    covariance_matrix = np.array([[coincident, separated], [separated, coincident]])
    charge_difference = np.array([-1.0, 1.0]) / np.sqrt(np.pi)
    variance = charge_difference @ covariance_matrix @ charge_difference
    assert variance == pytest.approx(
        2 * _regulated_variance(interval, length, parameter, radius), abs=3e-14
    )
    assert np.min(np.linalg.eigvalsh(covariance_matrix)) > 0
    # The two propagating physical currents each have k=K, whereas the
    # entropy retains c_R=c_L=1. This checks the sum, not a chiral anomaly.
    logarithmic_variance_coefficient = 2 * parameter / (2 * np.pi**2)
    assert logarithmic_variance_coefficient == pytest.approx(parameter / np.pi**2)


@pytest.mark.parametrize("stiffness", [1.0, 3.0, 5.0])
def test_first_order_chiral_action_symplectic_bracket_and_right_moving_eom(stiffness):
    # A small odd periodic grid avoids a Nyquist mode. Restricting the
    # symplectic inverse to the nonzero-mode subspace matches the continuum
    # caveat that the first-order action leaves a uniform mode unspecified.
    count, length, velocity = 9, 13.0, 1.4
    spacing = length / count
    momenta = 2 * np.pi * np.fft.fftfreq(count, d=spacing)
    derivative = np.fft.ifft(
        1j * momenta[:, None] * np.fft.fft(np.eye(count), axis=0), axis=0
    ).real
    np.testing.assert_allclose(derivative.T, -derivative, atol=2e-16)
    one_form = -stiffness * spacing * derivative
    symplectic = one_form.T - one_form
    poisson = np.linalg.pinv(symplectic)
    nonzero_modes = np.eye(count) - np.ones((count, count)) / count
    np.testing.assert_allclose(
        derivative @ poisson,
        nonzero_modes / (2 * stiffness * spacing),
        atol=5e-16,
    )
    hessian = 2 * velocity * stiffness * spacing * derivative.T @ derivative
    np.testing.assert_allclose(poisson @ hessian, -velocity * derivative, atol=2e-14)
    # rho=Phi'/sqrt(pi), so [rho,rho]=i{rho,rho} has the negative derivative
    # Schwinger term and physical level 1/M in this orientation convention.
    density_bracket = derivative @ poisson @ derivative.T / np.pi
    np.testing.assert_allclose(
        density_bracket,
        -derivative / (2 * np.pi * stiffness * spacing),
        atol=3e-16,
    )


@pytest.mark.parametrize("stiffness", [1, 3, 5])
def test_chiral_electron_vertex_weight_charge_and_fermionic_exchange(stiffness):
    # [Q,Phi]=i/(2 M sqrt(pi)) follows by integrating the derivative of the
    # line commutator i sgn(y-y')/(4M). The exponent includes its own i.
    charge_field_bracket = 1j / (2 * stiffness * np.sqrt(np.pi))
    electron_exponent = stiffness * np.sqrt(4 * np.pi)
    charge = 1j * electron_exponent * charge_field_bracket
    assert charge == pytest.approx(-1)
    # A chiral contraction is -log(z)/(4 pi M). Opposite vertex exponents
    # therefore give a two-point exponent 2h=alpha^2/(4 pi M).
    weight = electron_exponent**2 / (8 * np.pi * stiffness)
    assert weight == pytest.approx(stiffness / 2)
    current_level = 1 / stiffness
    assert weight == pytest.approx(abs(charge)**2 / (2 * current_level))
    exchange = np.exp(-1j * electron_exponent**2 / (4 * stiffness))
    assert exchange == pytest.approx(-1, abs=1e-14)

    minimal_exponent = np.sqrt(4 * np.pi)
    assert 1j * minimal_exponent * charge_field_bracket == pytest.approx(-1 / stiffness)
    assert minimal_exponent**2 / (8 * np.pi * stiffness) == pytest.approx(1 / (2 * stiffness))
    # The supplied constant cross-chiral bracket supplies the relative
    # fermion sign by itself; extra Klein factors must not double this sign.
    cross_exponent_bracket = (1j * np.sqrt(4 * np.pi)) * (
        -1j * np.sqrt(4 * np.pi)
    ) * (1j / 4)
    assert np.exp(cross_exponent_bracket) == pytest.approx(-1, abs=1e-14)


def test_continuum_sections_present_and_source_has_no_control_characters():
    root = Path(__file__).resolve().parents[1]
    source_path = root / (
        "00_WORKSPACE/CURRENT/Paper Methods/kac_moody_renyi_validation/"
        "kac_moody_renyi_validation.tex"
    )
    source = source_path.read_text()
    controls = [
        (position, ord(character))
        for position, character in enumerate(source)
        if (ord(character) < 32 and character not in "\n\r\t") or ord(character) == 127
    ]
    assert not controls, f"Unexpected control characters (offset, codepoint): {controls}"
    labels = set(re.findall(r"\\label\{([^}]+)\}", source))
    required = {
        "sec:continuum-models", "eq:dirac-hamiltonian", "eq:dirac-action",
        "eq:dirac-euclidean", "eq:chiral-fermion", "eq:user-boson-fields",
        "eq:user-boson-commutators", "eq:user-bosonization",
        "eq:free-boson-hamiltonian", "eq:free-boson-action",
        "eq:free-boson-euclidean", "eq:user-charge-current",
        "eq:user-boson-propagators", "eq:density-interactions",
        "eq:ll-hamiltonian", "eq:ll-parameters", "eq:ll-actions",
        "eq:ll-variance", "eq:chiral-boson-action",
        "eq:chiral-boson-euclidean", "eq:chiral-boson-bracket",
    }
    assert required <= labels, f"Missing continuum equations: {sorted(required - labels)}"
