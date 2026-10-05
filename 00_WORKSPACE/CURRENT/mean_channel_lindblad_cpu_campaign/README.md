# Mean-channel and Lindblad CPU campaign

This local deterministic campaign evaluates a finite completely positive Gaussian bath
channel and its continuous-time gain/loss generator without Born trajectories or random
schedule sampling. It is the production replacement for the retired Lindblad Colab bundle
and the former schedule-paired M1 descendants.

The hybrid production grid uses `Nx=20`.  Its static eight-point mass scan runs only at
`Ny=64`.  The `(alpha_1,alpha_2)=(1,30)` response endpoint runs at `Ny=32,48,64`,
`nshell=1,2,None`, and both domain-wall truncation settings.  Domain-wall truncation is on
for the canonical arm and off only as an endpoint robustness control.  This gives 39
unique primary wall cases.  There is no `Ny=24` production arm.

The response is the exact central difference of local `+/- epsilon` density-phase
unitaries applied to the stationary mean covariance.  It uses `epsilon=1e-3`, runs through
`Ny/2`, and saves only wall-projected density profiles, signed directionality, positive-lobe
velocities, retention, linearity checks, and finite-channel convergence.  The rank-four
response is propagated through momentum blocks without constructing a full response
covariance.  It is a directional density response, not quantized charge transport.

Only `nshell=None` cases may emit momentum-resolved arrays. Finite-shell cases use the
same translation symmetry internally for speed but archive only integrated occupation
histograms, real-space wall profiles, gaps, scalar time series, relaxation rates, and
channel-to-continuous errors. No case saves a covariance matrix or covariance history.

Run a smoke matrix:

```bash
python run_campaign.py --smoke --workers 3 --blas-threads 1 --cpu-list 0-11
```

Run or resume production.  Keep one BLAS thread per process; on the audited 56-core host,
28 workers are the efficient default and 46 use every physical core left free by the
current ten-core pilot:

```bash
python run_campaign.py --resume --workers 28 --blas-threads 1 --cpu-list 8,11-55
```

Analyze a completed run:

```bash
python analyze_campaign.py --run-root results/mean_channel_lindblad_cpu_v2_hybrid_response_production_<id>
```

Every completed case has a JSON metadata record and compact NPZ selected-observable
product. Response-enabled cases remain compact and do not save kicked covariances. The run
manifest records configuration and solver hashes, CPU allocation, timings,
and checksums. The historical bath-density dephasing closure is retained only as a labelled
small-system control; it is not silently mixed into the primary gain/loss grid.

`source_manifest.json` pins the immutable source/configuration files and the working-note
and atlas PDFs used to define and present this campaign. Regenerate it whenever any listed
artifact changes; production run manifests additionally pin the exact configuration and
solver hashes used by each output directory.
