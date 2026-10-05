#!/usr/bin/env python3
"""Unnormalized panel b with linear cycles and logarithmic entropy."""
from plot_spatial_purification_loglog import ROOT, main

if __name__ == '__main__':
    main(cutoff=2, figsize=(7.05,4.8), fontsize=12, normalized=False, xscale='linear',
         output=ROOT/'analysis_outputs/spatial_purification_panel_b_2ny_large_semilog_v1')
