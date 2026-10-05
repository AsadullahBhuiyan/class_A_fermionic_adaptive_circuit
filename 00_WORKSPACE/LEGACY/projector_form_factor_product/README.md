# Projected-Wannier Form-Factor Product Notebook

Quick exploratory notebook:

```bash
jupyter nbconvert --to notebook --execute \
  projector_form_factor_product/projector_form_factor_product_random_sequences.ipynb \
  --output projector_form_factor_product_random_sequences.executed.ipynb \
  --output-dir projector_form_factor_product \
  --ExecutePreprocessor.timeout=300
```

The notebook builds a finite real-space two-band class-A/Qi-Wu-Zhang
Hamiltonian using the repository convention
`n_z(k)=alpha-cos(kx)-cos(ky)`.  It constructs normalized projected
overcomplete Wannier orbitals from the two `sigma_x` trial spinors, samples 10
random orderings, and compares:

- the literal rank-one projector chain `P_{a_M} ... P_{a_1}`;
- the regularized product `prod_a (I + lambda P_a)`;
- lower-band, upper-band, and multiplied both-band products `A_+ A_-`;
- uniform `alpha=1`, uniform `alpha=3`, and a `1|3` domain-wall profile.

Mathematically, the bare chain is

```tex
P_{a_M}\cdots P_{a_1}
=
\left[\prod_{j=1}^{M-1}<w_{a_{j+1}}|w_{a_j}>\right]
|w_{a_M}><w_{a_1}|,
```

so it has only one nonzero singular value.  The full log-singular-value
spectrum comes from the regularized product.  For exact complementary
Hermitian bands, upper and lower projectors obey `P_a^+ P_b^- = 0`; therefore
the multiplied both-band regularized product `A_+ A_-` has the union of the
active upper/lower spectra up to numerical error and commutes with `A_- A_+`
up to numerical precision.

Current executed outputs live in `outputs/`.
