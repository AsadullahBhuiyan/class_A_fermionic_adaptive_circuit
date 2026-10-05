# Fresh-record twist-torus validation

This campaign used a 16×16 physical torus, a 16×16 boundary-twist mesh, and one independently sampled occupied-frame trajectory per twist point.

No full measurement record was replayed. The `fresh_full_record` surface refreshed both the site order and outcomes, while `fresh_outcomes` held only the site schedule fixed.

## Result

- Exact target: `defined_C1`, C = 1.0
- fresh schedule + outcomes: `undefined_rank_mismatch`, C = None, ranks [253, 254, 255, 256, 257, 258], valid-link fraction 0.375000
  Real-space topology gate: False (maximum final error 0.0886365).
- fixed schedule + fresh outcomes: `undefined_rank_mismatch`, C = None, ranks [253, 254, 255, 256, 257, 258, 259], valid-link fraction 0.388672
  Real-space topology gate: False (maximum final error 0.0527637).

Preliminary support for stitching fresh records: **False**.

An undefined result is not rounded or repaired. Rank-changing neighboring Slater determinants live in different particle-number sectors, and singular overlaps do not define a stable Berry link.

The 256 twist points form one surface, not 256 independent statistical replicates.
