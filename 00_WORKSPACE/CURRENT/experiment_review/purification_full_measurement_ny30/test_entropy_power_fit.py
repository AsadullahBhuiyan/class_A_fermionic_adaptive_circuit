import numpy as np

from make_figure import entropy_power_fit


def test_exact_mean_first_power_law_and_temporal_covariance():
    times = np.maximum(np.arange(61), 1) / 30
    amplitudes = np.linspace(.02, .04, 100)
    samples = 30 * amplitudes[:, None] * times[None, :]**(-1.25)
    fit = entropy_power_fit(samples)
    np.testing.assert_allclose(fit['amplitude'], amplitudes.mean(), atol=1e-14)
    np.testing.assert_allclose(fit['exponent'], 1.25, atol=1e-14)
    np.testing.assert_allclose(fit['amplitude_sem'], amplitudes.std(ddof=1)/10, atol=1e-14)
    assert fit['exponent_sem'] < 1e-14  # common amplitude noise cancels in slope
    np.testing.assert_allclose(fit['log_space_r2'], 1, atol=1e-14)
    samples[:, 0] = np.nan  # cycle zero must never enter the logarithmic fit
    assert entropy_power_fit(samples)['exponent'] == fit['exponent']


def test_sampling_sem_matches_finite_difference_influence():
    rng = np.random.default_rng(173)
    samples = np.exp(rng.normal(size=(100, 61)))
    fit = entropy_power_fit(samples, 5)
    mean = samples.mean(axis=0)
    x = np.log(np.arange(5, 61)/30)
    def exponent(y):
        return -np.polyfit(x, np.log(y[5:]/30), 1)[0]
    eps = 1e-5
    influence = np.array([(exponent(mean+eps*(row-mean))-
                           exponent(mean-eps*(row-mean)))/(2*eps)
                          for row in samples])
    np.testing.assert_allclose(fit['exponent_sem'], influence.std(ddof=1)/10, rtol=1e-7)


def test_requested_window_includes_15_through_60_only():
    times = np.maximum(np.arange(61), 1) / 30
    samples = np.tile(30*.03*times**(-1.27), (100, 1))
    samples[:, :15] = np.nan
    fit = entropy_power_fit(samples, 15, 60)
    assert fit['point_count'] == 46
    assert fit['normalized_start'] == .5 and fit['normalized_end'] == 2
    np.testing.assert_allclose(fit['exponent'], 1.27, atol=1e-14)
