### Numerical outcome

- **Hard wall:** cycle-40 source slopes are wall0_y0 = 1.514 [1.391, 1.637], wall0_y9 = 0.0112 [0.0003294, 0.02462], wall1_y0 = -0.01643 [-0.02515, -0.01031], wall1_y9 = -1.417 [-1.565, -1.258]. Mean static transverse leakage is 0.001 [0.000, 0.001], and the primary clipped-mode fraction is 0.644.
- **Soft wall:** cycle-40 source slopes are wall0_y0 = 0.3841 [0.3469, 0.4314], wall0_y9 = 0.01277 [0.006544, 0.01954], wall1_y0 = -0.01342 [-0.02228, -0.0046], wall1_y9 = -1.286 [-1.516, -1.017]. Mean static transverse leakage is 0.001 [0.000, 0.001], and the primary clipped-mode fraction is 0.528.

- The spread of the handed slope across the six late checkpoints is 0.1356 for the hard wall and 0.1844 for the soft wall. This measures late-cycle drift descriptively; checkpoints from one trajectory were not treated as independent samples.
- Combining the two visibly active, oppositely oriented sources as $H=[v(\mathrm{wall0\_y0})-v(\mathrm{wall1\_y9})]/2$ gives $H=1.466$ [1.35, 1.575] for the hard wall and $H=0.8352$ [0.7149, 0.9432] for the soft wall. Their unpaired hard-minus-soft difference is 0.6304 [0.4735, 0.7953]. The active sources have opposite signs, while the complementary endpoints remain much weaker over the prespecified early-time window; this is endpoint-selective handed response, not four independent confirmations.
- Across occupation clips $10^{-8}$, $10^{-10}$, and $10^{-12}$, the cycle-40 handed slope spans 0.001926 (hard) and 0.0001266 (soft). The clipping fractions printed above should be considered whenever judging that sensitivity.
- The cycle-40 static susceptibility is strongly local: the source-averaged integrated value is 0.1352 for the hard wall and 0.1259 for the soft wall, while the mean transverse leakage is reported above at roughly the $10^{-3}$ level. In this pilot the static response is therefore a local compressibility diagnostic, not a useful chirality estimator.
- The maximum retarded total-charge residual is 1.038e-15. Analytic/finite-difference relative errors are 1.668e-09 for the retarded response and 1.434e-09 for the static susceptibility.

### Interpretation limits

The endpoint- and wall-resolved signs in Fig. 3 are the relevant orientation test; a source whose bootstrap interval crosses zero is not evidence for a definite propagation direction. Differences between hard and soft static leakage quantify the response of these saved states to the two interface constructions, but they do not establish universality or a quantized coefficient. These histories contain only ten independent trajectories and use a maximally mixed initializer. They therefore validate the estimator and expose qualitative hard/soft behavior, but they do **not** replace a matched random-pure production campaign or establish a manuscript gate.


### Crude Fourier-dispersion result

- **hard wall0_y0 (k<0):** $v_{\rm F}=0.279$ [0.270, 0.290] across 9 trajectories; ensemble $R^2=0.999$; the damping scan gives 0.266--0.318.
- **hard wall1_y9 (k>0):** $v_{\rm F}=0.282$ [0.274, 0.291] across 10 trajectories; ensemble $R^2=0.999$; the damping scan gives 0.267--0.309.
- **soft wall0_y0 (k<0):** unresolved; the selected peaks remain on the lower frequency-search boundary.
- **soft wall1_y9 (k>0):** $v_{\rm F}=0.284$ [0.277, 0.294] across 9 trajectories; ensemble $R^2=0.999$; the damping scan gives 0.271--0.321.

The hard-wall response contains mirror-related low-frequency ridges in opposite momentum sectors on its two physical walls. Their slopes agree within trajectory fluctuations. The soft-wall response resolves essentially the same slope on `wall1_y9`, but its `wall0_y0` low-frequency ridge cannot be separated from near-zero-frequency weight with this window. Thus the crude system supports an approximately linear $\omega\sim |k|$ branch wherever the ridge is resolved, with opposite physical momentum sectors for the two wall orientations. It does **not** establish a clean two-wall soft-interface dispersion.

Only four nonzero momenta enter each fit. The fitted intercept moves substantially with the temporal damping, and the slope changes over the reported damping range. The heat maps also contain strong finite-aperture replicas. These results are therefore a qualitative Fourier consistency check, not a controlled conformal-dispersion or velocity measurement.

