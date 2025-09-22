import numpy as np
import importlib
import classA_U1FGTN
importlib.reload(classA_U1FGTN)
from matplotlib import pyplot as plt
from classA_U1FGTN import classA_U1FGTN
from IPython.display import Image, display
import os

# Local chern marker

# Small-ish grid and a few cycles so it runs quickly in a notebook.
Nx, Ny   = 21, 21
cycles   = 6         # increase if you want a longer animation

# Build the model
m = classA_U1FGTN(Nx, Ny, DW=True, nshell=5)

# Evolve (keep history -> frames for the GIF)
m.run_adaptive_circuit(cycles=cycles, G_history=True, progress=True)

# Make animation (GIF @ 1 fps) and save final static frame (PDF)
gif_path, final_path, C_last, G_last = m.chern_marker_dynamics()

print("Saved files:")
print("  GIF :", gif_path)
print("  PDF :", final_path)

# Show the GIF inline
if os.path.exists(gif_path):
    display(Image(filename=gif_path))

# Also show the final Chern-marker array inline (same data as the saved PDF)
plt.figure(figsize=(3.6, 4.2))
plt.imshow(C_last, cmap='RdBu_r', vmin=-1.0, vmax=1.0, origin='upper', aspect='equal')
plt.xlabel("y"); plt.ylabel("x")
plt.title(r"Final $\tanh\mathcal{C}(\mathbf{r})$")
plt.colorbar(orientation='horizontal', ticks=[-1,0,1], pad=0.08)
plt.show()