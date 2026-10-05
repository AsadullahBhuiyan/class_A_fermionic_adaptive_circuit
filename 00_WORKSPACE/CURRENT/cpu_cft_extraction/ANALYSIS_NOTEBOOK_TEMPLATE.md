# Analysis Notebook Default Structure

Use this structure for local analysis notebooks in this folder unless a task
requires a different format.

## 1. Title and Summary

Start with a markdown title and a short description of the question answered by
the notebook. State whether the notebook loads existing data, launches a run, or
only postprocesses saved artifacts.

## 2. Basic Theory and Estimators

Before plotting, define the quantities being measured. For the CFT extraction
workflow this should include, as relevant,

```tex
Y_s(L,t)=-\log p_{\xi_s(0:t)}, \qquad
f_0(L,t)=\frac{1}{\alpha L t}\overline{Y_s(L,t)},
```

```tex
f_0(L)=f_\infty-\frac{\pi c_{\rm eff}}{6L^2}+O(L^{-4}),
```

and

```tex
\Delta_r=\epsilon_1+\cdots+\epsilon_r,\qquad
x_r(L,t)=\frac{L\Delta_r(L,t)}{2\pi\alpha}.
```

State explicitly which objects come from Born probabilities, tangent-cocycle
spectra, Choi rapidities, endpoint/null bookkeeping, or finite-size fits.

## 3. Runtime Setup

For local notebooks, the first executable cell must provide CPU allocation
controls. Use a variable such as `CPU_RANGE = ""` so a range like `0-7` can be
set before execution. Keep dependency imports and plotting defaults close to the
top of the notebook.

## 4. Data Loading and Run Parameters

Load `manifest.json`, scalar CSV files, and spectrum archives from the selected
campaign directory. Render a markdown summary containing the actual run
parameters, sample counts, geometry, postselection status, fitting convention,
and data path. Do not rely only on hardcoded text.

## 5. Analysis Sections

Separate each analysis stage with a markdown header and a short description of
what the plot/table tests. For this workflow, the default sections are:

- `Leading Free Energy and c_eff`
- `One-Body and Fock-Lifted Transfer Gaps`
- `Size-Resolved Scaling-Dimension Estimates`
- `Endpoint and Null-Sector Bookkeeping`
- optional `Rank Dependence`
- optional `Time-Convergence Diagnostics`

## 6. One Figure Per Cell

Each output figure must have its own editable plotting cell. Put the complete
figure creation code immediately above the displayed output so the plot can be
edited in place without chasing helper cells.

## 7. Final Diagnostics

End with raw fit dictionaries, scalar rows, censor/null counts, path summaries,
and any convergence flags. If a fit is not meaningful, say why in markdown and
allow the printed diagnostics to show the missing input, such as a single-size
campaign.
