# v3 implementation matrix

Ordinary stochastic cases use 10 trajectories in shards `0` and `1`, representing global
sample indices `0..4` and `5..9`.  H3 keeps its explicit one-record exception.  Frozen P1
keeps its original 25-trajectory contract. The redesigned P1 is an independent revision
with 25 trajectories in five shards and is excluded from the v4 totals below.

| Campaign | Bundle | Geometry/matrix | Cases | Shards |
|---|---|---|---:|---:|
| redesigned P1 | `01_p1_chern_dynamics` | square `L=16,24,32,64`; `n_shell=1,2,None` | 12 | 60 |
| P1 | `01_p1_existing_completion` | frozen original square shell sweep | frozen | up to 240 |
| fixed baseline | `01_bulk_width_gate` | 4 constructions, `Nx=20`, five `Ny` | 20 | 40 |
| S1/T1/B2 parents | `02_pure_wall_master` | 4 constructions, five `Ny` | 20 | 40 |
| R1 | `02_pure_wall_master` | 2 records, five `Ny` | 10 | 20 |
| H3 | `03_chirality_replay` | 2 one-record `20x40` descendants | 2 | 2 jobs |
| P2 | `04_maxmix_master` | 2 arms, five `Ny` | 10 | 20 |
| S2 | `05_scans_and_controls` | dense alpha at `Ny=40`; selected alpha at other four sizes; 3 constructions | 72 | 144 |
| M2 | `05_scans_and_controls` | dense correction at `Ny=40`; selected correction at other four sizes | 20 | 40 |
| M3 bulk | `05_scans_and_controls` | dense noise at `20x20`; selected noise at `20x30,40,50,60` | 45 | 90 |
| M3 wall | `05_scans_and_controls` | 2 protocols, five `Ny`, 3–5 gated noise values | 30–50 | 60–100 |
| B1 | `06_b1_controller_frame` | 2 protocols at `20x40` | 2 | 4 |

The exact ordinary-stochastic total is 458 shards for a three-value M3 wall bracket and
498 shards for a five-value bracket.  The S2/M2/M3 subtotal is 334–374 shards; this is the
arithmetically consistent subtotal behind the 458–498 overall range.
