# Source-subtracted frozen-record charge response

This analysis reuses the completed `16x20`, 40-cycle frozen-record pilot. No
trajectory or covariance evolution was rerun. The saved ordered measurement
word determines the direct regional correction source

```text
A_a(phi) = sum_e [target_e - outcome_e] tr[R_a P_e(phi)].
```

The reported conditional redistribution is

```text
q_a(phi) = N_a^final(phi) - N_a^final(0) - [A_a(phi) - A_a(0)],
q_x(phi) = [q_R(phi) - q_L(phi)] / 2.
```

## Result

| Construction | `q_x` near half twist | maximum `|q_x|` | maximum direct-source contribution |
|---|---:|---:|---:|
| Soft, increasing twist | -0.49936045 | 0.62806858 | 2.93e-7 |
| Soft, decreasing twist | -0.49936042 | 0.62806862 | 2.93e-7 |
| Hard, increasing twist | +0.37775821 | 0.43334964 | 1.96e-7 |
| Hard, decreasing twist | +0.37775785 | 0.43334967 | 1.96e-7 |

The largest direct-source correction is less than `4.7e-7` of the largest raw
wall response. Source normalization closes to `4.6e-14`, and the corrected
left-plus-right balance closes to `1.2e-13`. Therefore the large intermediate-
twist imbalance is not caused by a twist-dependent relocation of the explicit
feedback source.

The nearest nonzero pair is strongly direction-odd. At
`phi = +/-0.392699`, the soft-wall response is `+0.062303/-0.062351`, giving
`2*pi*dq_x/dphi = 0.99723`; the hard-wall response is
`+0.062428/-0.062446`, giving `0.99899`. The hard principal branch remains an
almost linear sawtooth through `|phi| = 2.75`. This is clear evidence in these
two frozen records for a handed, spectral-flow-like conditional response.

The endpoint relative to exact zero twist is approximately `1.6e-8`; this is
the expected residual from evaluating the final grid point at `2*pi +/- 1e-7`.
Using each direction's own `+/-1e-7` origin gives exact large-gauge closure.
The result establishes a source-subtracted conditional endpoint redistribution.
Because every twist is an independently initialized static replay, it does not
establish a time-integrated current or a quantized pump. Because there is only
one record per wall construction and no matched trivial or reversed-Chern arm,
the chirality evidence is diagnostic rather than an ensemble-level claim.
