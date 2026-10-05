#!/usr/bin/env python3
import inspect
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)

from fgtn.classA_U1FGTN import classA_U1FGTN


def main():
    sig = inspect.signature(classA_U1FGTN.run_markov_channel)
    assert "p" not in sig.parameters
    assert "dw_exclude" not in sig.parameters
    assert "decoh" in sig.parameters
    assert "perfect_correction" in sig.parameters

    kwargs = dict(
        Nx=4,
        Ny=6,
        DW=True,
        nshell=None,
        alpha_1=1,
        alpha_2=30,
    )
    run_kwargs = dict(
        G_history=False,
        progress=False,
        cycles=1,
        init_mode="maxmix",
        save=False,
        n_a=0.5,
        sequence="raster_y",
        perfect_correction=False,
    )

    g_decoh = classA_U1FGTN(**kwargs).run_markov_channel(**run_kwargs, decoh=True)["G_final"]
    g_nodecoh = classA_U1FGTN(**kwargs).run_markov_channel(**run_kwargs, decoh=False)["G_final"]
    diff = float(np.max(np.abs(g_decoh - g_nodecoh)))
    assert diff > 1e-10, f"decoh branches unexpectedly agree: max diff {diff:.3e}"

    g_pc = classA_U1FGTN(**kwargs).run_markov_channel(
        **{**run_kwargs, "perfect_correction": True},
        decoh=True,
    )["G_final"]
    pc_diff = float(np.max(np.abs(g_pc - g_decoh)))
    assert pc_diff > 1e-10, f"perfect_correction did not change the channel: max diff {pc_diff:.3e}"

    save_base = dict(
        G_history=False,
        progress=False,
        cycles=0,
        init_mode="maxmix",
        save=True,
        n_a=0.5,
        sequence="raster_y",
        save_suffix="_smoke_decoh_api",
    )
    path_a = classA_U1FGTN(**kwargs).run_markov_channel(
        **save_base,
        decoh=True,
        perfect_correction=False,
    )["save_path"]
    path_b = classA_U1FGTN(**kwargs).run_markov_channel(
        **save_base,
        decoh=False,
        perfect_correction=False,
    )["save_path"]
    path_c = classA_U1FGTN(**kwargs).run_markov_channel(
        **save_base,
        decoh=True,
        perfect_correction=True,
    )["save_path"]
    assert path_a and path_b and path_c
    assert len({path_a, path_b, path_c}) == 3
    assert "_decoh1_pc0_" in path_a
    assert "_decoh0_pc0_" in path_b
    assert "_decoh1_pc1_" in path_c

    for path, expected in [
        (path_a, {"decoh": True, "perfect_correction": False}),
        (path_b, {"decoh": False, "perfect_correction": False}),
        (path_c, {"decoh": True, "perfect_correction": True}),
    ]:
        with np.load(path, allow_pickle=False) as data:
            config = data["run_config"].item()
        for key, value in expected.items():
            assert f'"{key}": {str(value).lower()}' in config
        Path(path).unlink(missing_ok=True)

    print("smoke_markov_channel_decoh: ok")


if __name__ == "__main__":
    main()
