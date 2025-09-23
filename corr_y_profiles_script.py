import numpy as np
import importlib
import classA_U1FGTN
importlib.reload(classA_U1FGTN)
from classA_U1FGTN import classA_U1FGTN
from matplotlib import pyplot as plt
import os
import time

def format_interval(seconds):
    """Convert seconds to H:MM:SS (or D:HH:MM:SS if >1 day)."""
    seconds = int(round(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{days}d {hours:02}:{minutes:02}:{seconds:02}" if days else f"{hours:02}:{minutes:02}:{seconds:02}"

# ----------------------------
# Correlation y-profile test
# ----------------------------

# Small-ish grid so it runs in a notebook reasonably fast; bump cycles/samples as needed.
Nx, Ny   = 30, 30
cycles   = 10          # circuit sweeps used by helper inside plotter (m.cycles)
samples  = 10          # number of samples when doing resolved/averaged
ry_max   = None       # defaults to Ny//2 if None
DW       = True       # include a domain wall so profiles are interesting
nshell   = 5          # Wannier truncation (None for no truncation)

# Choose what to compute:
# - If both are False -> single trajectory C_G
# - If only resolved  -> \bar{C}_G
# - If only averaged  -> C_{\bar{G}}
# - If both True      -> two-panel figure with both (computed efficiently)
trajectory_resolved = True
trajectory_averaged = True

t0 = time.time()

# Build the model
m = classA_U1FGTN(Nx, Ny, cycles = cycles, samples = samples, DW=DW, nshell=nshell)

# Let the method pick smart x-positions from alpha; you can also pass a list like [(x0,"label"), ...]
#x_positions = range(8)
#time.sleep(10)
# Run the plotter (this will run the circuit internally if needed)
pdf_path = m.plot_corr_y_profiles_v2(backend="loky")              # let it auto-name the PDF


elapsed = time.time() - t0
print("Saved file:")
print("  PDF :", os.path.abspath(pdf_path))
print(f"Total time elapsed: {format_interval(elapsed)}")

# Optional: open the PDF automatically in some environments (commented out by default)
# import webbrowser; webbrowser.open(f"file://{os.path.abspath(pdf_path)}")

# If you want to also quickly visualize one curve inline (e.g., single trajectory), you can:
# (Leave commented; the PDF already contains the full figure.)
# plt.figure(figsize=(6,4))
# ... (your quick-view code)
# plt.show()