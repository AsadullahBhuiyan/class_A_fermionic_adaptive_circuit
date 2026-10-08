#!/usr/bin/env python3
"""Capture native A1 plot artists from saved diagnostics, without fitting.

This one-time provenance step evaluates the native analytical form-factor cuts.
The delivery renderer only reads the resulting compact numerical cache.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

OUT = Path(__file__).resolve().parents[1]
DATA = OUT / "data" / "ow_truncation"


def digest(path):
    return {"bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    native_path = root / "technical_report/analyze_ow_truncation.py"
    source_data = native_path.parent / "data"
    names = ["ow_truncation_diagnostics.csv", "ow_truncation_fit.json",
             "flattened_pauli_gap_vs_w.csv", "flattened_pauli_gap_analysis.json"]
    original_paths = [native_path, *[source_data / name for name in names],
                      root / "technical_report/figures/ow_truncation_summary.pdf",
                      root / "technical_report/figures/ow_truncation_summary.png",
                      OUT.parent / "ow_truncation_summary.pdf"]
    originals = {str(path.relative_to(root)): digest(path) for path in original_paths}
    spec = importlib.util.spec_from_file_location("native_ow_truncation", native_path)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    native.configure_plotting()
    figures = []
    native.save_figure = lambda figure, stem: figures.append(figure)
    # These routines must never be reached in this extraction.
    def forbidden(*args, **kwargs):
        raise RuntimeError("A1 typography extraction must not optimize or refit")
    for name in ("critical_fit", "locate_overlap_zero", "locate_flattened_critical_alpha",
                 "brentq", "curve_fit", "minimize_scalar"):
        setattr(native, name, forbidden)
    rows = pd.read_csv(source_data / names[0])
    fits = json.loads((source_data / names[1]).read_text())
    fit = fits["critical_fit"]
    gap_rows = pd.read_csv(source_data / names[2])
    gap_fits = json.loads((source_data / names[3]).read_text())
    coefficient_minus, _ = native.projector_fourier_coefficients(native.ALPHA)
    native.make_summary_figure(
        coefficient_minus, rows,
        np.array([fit[key] for key in ("alpha_infinity", "A", "b", "p")]),
        gap_rows, gap_fits,
    )
    assert len(figures) == 1
    figure = figures[0]
    arrays, axes = {}, []
    for axis_index, axis in enumerate(figure.axes):
        metadata = dict(xlim=list(axis.get_xlim()), ylim=list(axis.get_ylim()),
                        xscale=axis.get_xscale(), yscale=axis.get_yscale(),
                        xticks=axis.get_xticks().tolist(), yticks=axis.get_yticks().tolist(),
                        xlabel=axis.get_xlabel(), ylabel=axis.get_ylabel(), lines=[])
        for line_index, line in enumerate(axis.lines):
            prefix = f"axis{axis_index}_line{line_index}"
            arrays[prefix + "_x"] = np.array(line.get_xdata())
            arrays[prefix + "_y"] = np.array(line.get_ydata())
            transform = "data"
            if line.get_transform() == axis.get_xaxis_transform():
                transform = "xaxis"
            elif line.get_transform() == axis.get_yaxis_transform():
                transform = "yaxis"
            metadata["lines"].append(dict(
                prefix=prefix, label=line.get_label(), transform=transform,
                color=line.get_color(), marker=line.get_marker(),
                linestyle=line.get_linestyle(), dash_pattern=line._unscaled_dash_pattern,
                linewidth=line.get_linewidth(), markersize=line.get_markersize(),
                markerfacecolor=line.get_markerfacecolor(),
                markeredgecolor=line.get_markeredgecolor(),
                markeredgewidth=line.get_markeredgewidth(), markevery=line.get_markevery(),
                clip_on=line.get_clip_on(), alpha=line.get_alpha(), zorder=line.get_zorder(),
            ))
        axes.append(metadata)
    assert [len(axis["lines"]) for axis in axes] == [12, 2, 2, 2, 3]
    with tempfile.TemporaryDirectory(prefix="native_ow_compare_") as directory:
        rendered_path = Path(directory) / "native.png"
        figure.savefig(rendered_path, dpi=300, bbox_inches="tight")
        old = np.asarray(Image.open(original_paths[-2]))
        new = np.asarray(Image.open(rendered_path))
        assert old.shape == new.shape
        delta = np.abs(old.astype(int) - new.astype(int))
        changed_pixels = int(np.any(delta != 0, axis=-1).sum())
        assert delta.max() <= 1 and changed_pixels <= 25
        raster_validation = dict(shape=list(old.shape), changed_pixels=changed_pixels,
                                 total_pixels=int(np.prod(old.shape[:2])),
                                 maximum_channel_difference=int(delta.max()),
                                 mean_absolute_channel_difference=float(delta.mean()))
    plt.close(figure)
    DATA.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy2(source_data / name, DATA / name)
    np.savez_compressed(DATA / "plot_data.npz", **arrays)
    (DATA / "native_artists.json").write_text(json.dumps(axes, indent=2) + "\n")
    for relative, expected in originals.items():
        assert digest(root / relative) == expected, relative
    provenance = dict(
        source_renderer="technical_report/analyze_ow_truncation.py:make_summary_figure",
        extraction="Saved diagnostics/fit coefficients plus native analytical Fourier cuts; no optimization, refitting, or circuit evolution",
        original_sources=originals, native_raster_comparison=raster_validation,
        preserved_line_artists=21, saved_critical_fit=fit,
        saved_tail_fit=gap_fits["tail_fits"]["X"],
        bundle_inputs={name: digest(DATA / name)
                       for name in [*names, "plot_data.npz", "native_artists.json"]},
        originals_unchanged=True,
    )
    (DATA / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(raster_validation, indent=2))


if __name__ == "__main__":
    main()
