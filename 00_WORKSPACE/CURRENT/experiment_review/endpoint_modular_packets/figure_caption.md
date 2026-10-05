Modular packet evolution from hard-wall 20 x 32 endpoint states. The monitored
states were prepared from random pure initial states (with hard-wall exterior
product preparation) for 64 raster-y cycles using perfect correction,
alpha2=30, nshell=1, and complex128. Each alpha1 ensemble contains S=100
independent trajectories. For every trajectory and all 32 translated origins,
restrict the covariance to the full x width and 16 consecutive y rows, then
construct h_A=-2 atanh(G_A) with eigenvalues clipped to
[-1+1e-10,1-1e-10]. Independently propagate charge-2, two-orbital packets
initialized at (x,y-y0)=(5,8) and (15,8) under exp(-i h_A t_mod).

(a) Evolved densities averaged over cut origins within trajectories and then
over trajectories, for alpha1=1. Purple, orange, green, and pink denote modular
times 0, 0.5, 1, and 1.5. Marker area is proportional to the square root of
mean local density, with the same normalization in (a) and (c). The plotting
threshold of 1e-4 does not affect numerical averages. Dotted lines indicate
wall columns x=5 and 15.

(b) Alpha1=1 signed y center-of-mass displacement within the radius-2 x window
around each source wall, normalized separately for each realization before
averaging. The orange solid and blue dashed curves denote sources (5,8) and
(15,8). Bands are +/- one SEM across the 100 trajectory-level origin means.
The saved time grid is 0..32 in steps of 0.01; panel (b) displays only 0..4.
There is no smoothing or sign
reorientation. Full-subsystem COM and retained wall-window charge are saved
as diagnostics. No velocity or other time-window fit is used.

(c) The same averaged-density construction as (a), for alpha1=3 as the control.
All panels average observables after sample/cut-resolved evolution; none uses
an ensemble-averaged Hamiltonian. Modular time is distinct from monitored
circuit time. The initial position is the midpoint row of the retained region,
not either of the opposite corners used in the earlier figure.
