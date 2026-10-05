# How to read the refined charge-pump plots

1. continued_pump_forward_reverse: both twist directions are sign-aligned. Agreement with the dashed diagonal means one charge is transferred over one flux quantum.
2. continued_left_right_charge_balance: the left wall must lose exactly what the right wall gains.
3. pump_mesh_convergence_and_control: continued circles should approach 1; instantaneous crosses should approach 0 as the twist mesh is refined.
4. final_wall_localized_charge_profile: the transferred density must form opposite peaks at the two implemented walls, not in the bulk.

This is quasistatic occupied-state continuation of the exact flattened Hamiltonian. It is not a finite-time circuit evolution and does not compute an instantaneous current.
