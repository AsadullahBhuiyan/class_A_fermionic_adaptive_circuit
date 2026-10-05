# Equilibrium wall-crossing explorer

Open `equilibrium_wall_crossing_explorer.ipynb` in Jupyter and run it from top
to bottom.  The notebook uses the canonical CPU OW constructor, not monitored
dynamics, and writes no production data.

The default `20 x 24`, `n_shell=1`, soft-wall calculation takes about 30
seconds on the local workstation.  Two `ipywidgets` sliders then expose:

- the finite-size avoided crossing and the exchange of left/right wall
  character by the energy-ordered modes;
- the difference between instantaneous lowest-energy refilling and
  previous-overlap spectral continuation through a complete flux quantum.

Edit `CPU_RANGE` in the first code cell before imports.  Edit `CONFIG` in the
single configuration cell to switch between soft/hard walls or CCW/CW paths.
The last cell prints the raw translation-block, overlap, charge-conservation,
and endpoint diagnostics used by the displayed plots.

`wall_crossing_explorer.py` contains the numerical construction and
`build_notebook.py` regenerates the notebook artifact.
