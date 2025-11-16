import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Tuple


def load_final_time_samples(npz_path: Path) -> np.ndarray:
    """
    Load cached Markov circuit histories and return the final-time G samples.
    Parameters
    ----------
    npz_path : Path
        Path to the cached `.npz` file containing `G_hist`.
    Returns
    -------
    np.ndarray
        Array of shape (n_samples, Nlayer, Nlayer) with complex-valued G matrices.
    """
    with np.load(npz_path) as data:
        G_hist = data["G_hist"]
    return G_hist[:, -1]


def frobenius_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix, ord="fro"))


def occupation_trace(matrix: np.ndarray) -> float:
    return float(np.trace(matrix).real)


def spectral_radius(matrix: np.ndarray) -> float:
    eigvals = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(eigvals)))


def frobenius_standard_error(samples: np.ndarray) -> float:
    """
    Compute the standard error of the Frobenius norm across raw samples.
    """
    count = samples.shape[0]
    if count <= 1:
        return float("nan")
    sample_norms = np.sqrt(np.sum(np.abs(samples) ** 2, axis=(1, 2)))
    return float(sample_norms.std(ddof=1) / np.sqrt(count))


def mean_estimator(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=0)


def median_of_means_estimator(samples: np.ndarray, n_groups: int = None) -> np.ndarray:
    """
    Elementwise median of means estimator.
    """
    count = samples.shape[0]
    if n_groups is None:
        n_groups = max(1, int(np.sqrt(count)))
    n_groups = max(1, min(n_groups, count))
    groups = [grp for grp in np.array_split(samples, n_groups, axis=0) if grp.size > 0]
    group_means = np.stack([grp.mean(axis=0) for grp in groups], axis=0)
    return np.median(group_means, axis=0)


def analyze_estimator(
    samples: np.ndarray,
    sample_counts: Iterable[int],
    estimator_fn: Callable[[np.ndarray], np.ndarray],
) -> Mapping[str, List[float]]:
    """
    Apply an estimator to increasing sample counts and collect diagnostics.
    Returns a dictionary of metric -> list of values aligned to `sample_counts`.
    """
    metrics = {
        "frobenius_norm": [],
        "frobenius_se": [],
        "successive_diff": [],
        "occupation_trace": [],
        "spectral_radius": [],
    }
    previous_estimate = None

    for count in sample_counts:
        subset = samples[:count]
        estimate = estimator_fn(subset)

        metrics["frobenius_norm"].append(frobenius_norm(estimate))
        metrics["frobenius_se"].append(frobenius_standard_error(subset))
        metrics["occupation_trace"].append(occupation_trace(estimate))
        metrics["spectral_radius"].append(spectral_radius(estimate))

        if previous_estimate is None:
            metrics["successive_diff"].append(float("nan"))
        else:
            metrics["successive_diff"].append(frobenius_norm(estimate - previous_estimate))
        previous_estimate = estimate

    return metrics


def collect_diagnostics(
    final_samples: np.ndarray,
    sample_counts: Iterable[int],
    *,
    include_mom: bool = True,
    mom_groups: int = None,
) -> Dict[str, Mapping[str, List[float]]]:
    """
    Compute diagnostics for the selected estimators.
    """
    sample_counts = list(sample_counts)
    estimators: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "Mean": mean_estimator,
    }
    if include_mom:
        estimators["Median-of-Means"] = lambda data: median_of_means_estimator(
            data, n_groups=mom_groups
        )
    diagnostics: Dict[str, Mapping[str, List[float]]] = {}
    for name, fn in estimators.items():
        diagnostics[name] = analyze_estimator(final_samples, sample_counts, fn)
    return diagnostics


def plot_diagnostics(
    sample_counts: Iterable[int],
    diagnostics: Mapping[str, Mapping[str, List[float]]],
    output_path: Path = None,
) -> plt.Figure:
    """
    Plot five diagnostics for the provided estimators.
    """
    sample_counts = list(sample_counts)
    metric_defs: List[Tuple[str, str, str]] = [
        ("frobenius_norm", "Frobenius Norm of Estimate", "||⟨G⟩||₍fro₎"),
        ("frobenius_se", "Std Error of Frobenius Norm", "SE[||G||₍fro₎]"),
        ("successive_diff", "Successive Estimate Difference", "||Δ⟨G⟩||₍fro₎"),
        ("occupation_trace", "Occupation Trace", "Tr(⟨G⟩)"),
        ("spectral_radius", "Spectral Radius", "ρ(⟨G⟩)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    ax_list = axes.flatten()
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, (metric_key, title, ylabel) in enumerate(metric_defs):
        ax = ax_list[idx]
        for color, (est_name, metrics) in zip(colors, diagnostics.items()):
            values = np.array(metrics[metric_key], dtype=float)
            if metric_key == "successive_diff":
                non_positive = ~np.isfinite(values) | (values <= 0)
                values[non_positive] = np.nan
                ax.set_yscale("log")
            ax.plot(sample_counts, values, marker="o", label=est_name, color=color)
        ax.set_title(title)
        ax.set_xlabel("Samples")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.legend()

    ax_list[-1].axis("off")
    fig.suptitle("Convergence Diagnostics: Mean vs Median-of-Means", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)

    return fig


# Example usage (copy-paste friendly for notebooks):
#
# data_path = Path(
#     "cache/G_history_samples/N16x16_C20_S100_nshNone_DW1_init-default_n_a0.5_markov_circuit.npz"
# )
# final_samples = load_final_time_samples(data_path)
# sample_counts = [1, 10, 25, 50, 75, 100]
# diagnostics = collect_diagnostics(final_samples, sample_counts, mom_groups=5)
# fig = plot_diagnostics(
#     sample_counts,
#     diagnostics,
#     output_path=Path("frobenius_convergence_outputs/G_convergence_diagnostics.png"),
# )
# plt.show()
