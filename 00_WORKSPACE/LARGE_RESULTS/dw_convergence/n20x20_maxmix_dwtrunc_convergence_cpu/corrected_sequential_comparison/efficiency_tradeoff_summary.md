# Efficiency Tradeoff Summary

Corrected sequential-site CPU runs only. Times are measured from completed run manifests.

## Mean Cycle Cost

| condition | intrinsic sec / trajectory-cycle | wall sec / ensemble-cycle | total runtime hr |
|---|---:|---:|---:|
| dwtrunc0 | 138.939 | 1389.39 | 15.4377 |
| dwtrunc1 | 98.8915 | 988.915 | 10.9879 |

## Direct Entropy-Loss Efficiency

| condition   |   S0_bits |   S_final_bits |   entropy_loss_bits |   entropy_loss_bits_per_cycle |   wall_sec_per_cycle |   intrinsic_sec_per_trajectory_cycle |   bits_lost_per_wall_sec |   wall_sec_per_bit_lost |   bits_lost_per_intrinsic_sec |   intrinsic_sec_per_bit_lost |   cycle_time_speedup_false_over_true |   entropy_loss_per_cycle_true_over_false |   bits_per_wall_sec_true_over_false |   wall_sec_per_bit_true_over_false |
|:------------|----------:|---------------:|--------------------:|------------------------------:|---------------------:|-------------------------------------:|-------------------------:|------------------------:|------------------------------:|-----------------------------:|-------------------------------------:|-----------------------------------------:|------------------------------------:|-----------------------------------:|
| dwtrunc0    |       800 |      0.0165708 |             799.983 |                       19.9996 |             1389.39  |                             138.939  |                0.0143945 |                 69.471  |                      0.143945 |                      6.9471  |                              1.40496 |                                 0.549602 |                            0.772171 |                            1.29505 |
| dwtrunc1    |       440 |      0.327814  |             439.672 |                       10.9918 |              988.915 |                              98.8915 |                0.011115  |                 89.9684 |                      0.11115  |                      8.99684 |                              1.40496 |                                 0.549602 |                            0.772171 |                            1.29505 |

Direct-rate interpretation: `dwtrunc1` is cheaper per cycle, but its entropy loss per cycle is smaller. The `bits_lost_per_wall_sec` column is the net entropy-throughput measure.

## Threshold Crossings

| metric                     |   threshold |   dwtrunc0_cycle |   dwtrunc1_cycle |   dwtrunc0_wall_hr |   dwtrunc1_wall_hr |   dwtrunc0_intrinsic_sec |   dwtrunc1_intrinsic_sec |   speedup_false_over_true_wall | winner   | dwtrunc0_method      | dwtrunc1_method        |
|:---------------------------|------------:|-----------------:|-----------------:|-------------------:|-------------------:|-------------------------:|-------------------------:|-------------------------------:|:---------|:---------------------|:-----------------------|
| absolute_entropy_bits      |     100     |          1.12536 |          1.08404 |           0.434324 |           0.297784 |                  156.357 |                 107.202  |                       1.45852  | dwtrunc1 | linear_interpolation | linear_interpolation   |
| absolute_entropy_bits      |      50     |          1.73852 |          1.74205 |           0.670968 |           0.47854  |                  241.549 |                 172.274  |                       1.40212  | dwtrunc1 | linear_interpolation | linear_interpolation   |
| absolute_entropy_bits      |      20     |          2.5321  |          2.63961 |           0.977242 |           0.725099 |                  351.807 |                 261.036  |                       1.34774  | dwtrunc1 | linear_interpolation | linear_interpolation   |
| absolute_entropy_bits      |      10     |          3.42012 |          3.87939 |           1.31997  |           1.06566  |                  475.188 |                 383.639  |                       1.23863  | dwtrunc1 | linear_interpolation | linear_interpolation   |
| absolute_entropy_bits      |       5     |          4.76717 |          6.18858 |           1.83985  |           1.7      |                  662.347 |                 611.998  |                       1.08227  | dwtrunc1 | linear_interpolation | linear_interpolation   |
| absolute_entropy_bits      |       1     |         14.3664  |         25.2299  |           5.54458  |           6.93061  |                 1996.05  |                2495.02   |                       0.800013 | dwtrunc0 | linear_interpolation | linear_interpolation   |
| absolute_entropy_bits      |       0.5   |         21.0926  |         31.876   |           8.14051  |           8.75629  |                 2930.58  |                3152.26   |                       0.929676 | dwtrunc0 | linear_interpolation | linear_interpolation   |
| relative_entropy           |       0.1   |          1.37063 |          1.82101 |           0.528982 |           0.50023  |                  190.433 |                 180.083  |                       1.05748  | dwtrunc1 | linear_interpolation | linear_interpolation   |
| relative_entropy           |       0.05  |          1.86115 |          2.51661 |           0.718297 |           0.691308 |                  258.587 |                 248.871  |                       1.03904  | dwtrunc1 | linear_interpolation | linear_interpolation   |
| relative_entropy           |       0.01  |          3.77479 |          6.82661 |           1.45685  |           1.87526  |                  524.465 |                 675.094  |                       0.776877 | dwtrunc0 | linear_interpolation | linear_interpolation   |
| relative_entropy           |       0.005 |          5.39709 |         11.9581  |           2.08296  |           3.28487  |                  749.866 |                1182.55   |                       0.634107 | dwtrunc0 | linear_interpolation | linear_interpolation   |
| relative_entropy           |       0.001 |         17.0676  |         32.9546  |           6.58711  |           9.0526   |                 2371.36  |                3258.94   |                       0.727649 | dwtrunc0 | linear_interpolation | linear_interpolation   |
| frobenius_successive_delta |      20     |          1.30971 |          1       |           0.505471 |           0.274699 |                  181.97  |                  98.8915 |                       1.84009  | dwtrunc1 | linear_interpolation | initial_or_first_point |
| frobenius_successive_delta |      10     |          1.76558 |          1.61752 |           0.681411 |           0.444331 |                  245.308 |                 159.959  |                       1.53357  | dwtrunc1 | linear_interpolation | linear_interpolation   |
| frobenius_successive_delta |       8     |          1.85675 |          1.75546 |           0.716599 |           0.482223 |                  257.976 |                 173.6    |                       1.48603  | dwtrunc1 | linear_interpolation | linear_interpolation   |
| frobenius_successive_delta |       7     |          1.90234 |          1.82443 |           0.734193 |           0.501169 |                  264.309 |                 180.421  |                       1.46496  | dwtrunc1 | linear_interpolation | linear_interpolation   |

## Sustained Entropy Threshold Crossings

| metric                |   threshold |   dwtrunc0_sustained_cycle |   dwtrunc1_sustained_cycle |   dwtrunc0_sustained_wall_hr |   dwtrunc1_sustained_wall_hr |   speedup_false_over_true_wall | winner   | dwtrunc0_method             | dwtrunc1_method             |
|:----------------------|------------:|---------------------------:|---------------------------:|-----------------------------:|-----------------------------:|-------------------------------:|:---------|:----------------------------|:----------------------------|
| absolute_entropy_bits |     100     |                          2 |                          2 |                     0.771884 |                     0.549397 |                       1.40496  | dwtrunc1 | sustained_discrete_crossing | sustained_discrete_crossing |
| absolute_entropy_bits |      50     |                          2 |                          2 |                     0.771884 |                     0.549397 |                       1.40496  | dwtrunc1 | sustained_discrete_crossing | sustained_discrete_crossing |
| absolute_entropy_bits |      20     |                          3 |                          3 |                     1.15783  |                     0.824096 |                       1.40496  | dwtrunc1 | sustained_discrete_crossing | sustained_discrete_crossing |
| absolute_entropy_bits |      10     |                          4 |                          4 |                     1.54377  |                     1.09879  |                       1.40496  | dwtrunc1 | sustained_discrete_crossing | sustained_discrete_crossing |
| absolute_entropy_bits |       5     |                          5 |                          7 |                     1.92971  |                     1.92289  |                       1.00355  | dwtrunc1 | sustained_discrete_crossing | sustained_discrete_crossing |
| absolute_entropy_bits |       1     |                         15 |                         26 |                     5.78913  |                     7.14217  |                       0.810556 | dwtrunc0 | sustained_discrete_crossing | sustained_discrete_crossing |
| absolute_entropy_bits |       0.5   |                         22 |                         32 |                     8.49072  |                     8.79036  |                       0.965913 | dwtrunc0 | sustained_discrete_crossing | sustained_discrete_crossing |
| relative_entropy      |       0.1   |                          2 |                          2 |                     0.771884 |                     0.549397 |                       1.40496  | dwtrunc1 | sustained_discrete_crossing | sustained_discrete_crossing |
| relative_entropy      |       0.05  |                          2 |                          3 |                     0.771884 |                     0.824096 |                       0.936643 | dwtrunc0 | sustained_discrete_crossing | sustained_discrete_crossing |
| relative_entropy      |       0.01  |                          4 |                          7 |                     1.54377  |                     1.92289  |                       0.802837 | dwtrunc0 | sustained_discrete_crossing | sustained_discrete_crossing |
| relative_entropy      |       0.005 |                          6 |                         12 |                     2.31565  |                     3.29638  |                       0.702482 | dwtrunc0 | sustained_discrete_crossing | sustained_discrete_crossing |
| relative_entropy      |       0.001 |                         18 |                         33 |                     6.94695  |                     9.06506  |                       0.766344 | dwtrunc0 | sustained_discrete_crossing | sustained_discrete_crossing |

Interpretation: speedup is `dwtrunc0_wall_hr / dwtrunc1_wall_hr`; values above 1 mean `dw_truncation=True` reaches that threshold faster in wall-clock ensemble time.
Threshold times are first crossings of the ensemble-mean curve, linearly interpolated between saved cycles. A later increase above the same threshold does not change the reported first-crossing time.