# L=24 soft-domain-wall reference-anisotropy campaign

This bundle is the matched soft-wall twin of
`00_WORKSPACE/CURRENT/reference_ancilla_anisotropy_cpu_pilot`.

The physics, sampling, seeds, convergence gates, temporal grids, reference
observable, and canonical CPU dynamics entry point are unchanged. The two
intentional protocol differences are:

- `dw_truncation=False`: overcomplete Wannier modes may cross the domain-wall
  interfaces.
- `meas_slab_only=False`: adaptive measurements act on the full lattice rather
  than only the topological slab.

The mass domain wall remains enabled with `alpha_1=1` in the topological slab
and `alpha_2=30` in the trivial exterior. References are inserted at the two
interface columns recorded by `model.DW_loc`.

Hard-wall checkpoints and throughput benchmarks are not reused because their
checkpoint signatures and computational workload do not match this protocol.
This campaign generates fresh paired checkpoints using the same locked random
seeds, enabling a direct hard-wall/soft-wall comparison.

Launch with:

```bash
python 00_WORKSPACE/CURRENT/reference_ancilla_anisotropy_cpu_softwall/launch_tmux.py
```
