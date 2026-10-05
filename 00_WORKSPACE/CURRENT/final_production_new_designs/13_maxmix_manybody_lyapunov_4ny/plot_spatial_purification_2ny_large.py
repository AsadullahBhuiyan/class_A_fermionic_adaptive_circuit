#!/usr/bin/env python3
"""Enlarged panel b, same Ny=20,30,40 curves and SEM through 2Ny."""
from plot_spatial_purification_loglog import ROOT, main

if __name__ == '__main__':
    main(cutoff=2, figsize=(7.05,4.8), fontsize=12,
         output=ROOT/'analysis_outputs/spatial_purification_panel_b_2ny_large_v1')
