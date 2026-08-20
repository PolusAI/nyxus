# Audit: `test_2d_gldm_ibsi.h` goldens vs the IBSI digital phantom

Companion to `gldm_2d_pyradiomics_vetting_report.md`, which carries the family's exact-tier
measurement. This one records what the IBSI file's 14 goldens are, where each comes from, and what
the published precision buys.

## Method

The goldens are transcribed from the IBSI reference manual (IBSI Documentation, Release 0.0.1dev,
Dec 13 2021, dataset "dig phantom", aggregation "2D, averaged") and are not produced by a tool, so
there is no generator for this file. Verifying them is a two-step check:

1. **Are they the right published values?** They are the manual's **NGLDM** table — IBSI defines no
   GLDM. The mapping and the measurement establishing that the two are the same statistic are in
   `gldm_2d_pyradiomics_vetting_report.md`; the same numbers are pinned under their IBSI names in
   `test_2d_ngldm_ibsi.h`, so the transcription can be diffed against a second copy in this
   repository rather than re-typed from the PDF.
2. **What residual do they leave?** Computed against the full-precision values from the PyRadiomics
   run, which agree with Nyxus to 9.8e-16.

## Result table

`rel` is the published 3-significant-figure value against the full-precision one.

| feature | IBSI published | full precision | rel | IBSI page |
|---|---|---|---|---|
| GLDM_SDE | 0.158 | 0.15807024738501638 | 4.4e-4 | 120 |
| GLDM_LDE | 19.2 | 19.173821809425526 | 1.4e-3 | 121 |
| GLDM_LGLE | 0.702 | 0.7017531915300232 | 3.5e-4 | 121 |
| GLDM_HGLE | 7.49 | 7.486949604403165 | 4.1e-4 | 122 |
| GLDM_SDLGLE | 0.0473 | 0.047290498640367454 | 2.0e-4 | 122 |
| GLDM_SDHGLE | 3.06 | 3.064914180133554 | 1.6e-3 | 123 |
| GLDM_LDLGLE | 17.6 | 17.59968920804189 | 1.8e-5 | 123 |
| GLDM_LDHGLE | 49.5 | 49.477721878224976 | 4.5e-4 | 124 |
| GLDM_GLN | 10.2 | 10.24637942896457 | **4.5e-3** | 124 |
| GLDM_DN | 3.96 | 3.9646456828345373 | 1.2e-3 | 125 |
| GLDM_DNN | 0.212 | 0.21177218060411693 | 1.1e-3 | 125 |
| GLDM_GLV | 2.7 | 2.7037332451477982 | 1.4e-3 | 127 |
| GLDM_DV | 2.73 | 2.729504577399913 | 1.8e-4 | 127 |
| GLDM_DE | 2.71 | 2.714292423281547 | 1.6e-3 | 128 |

Worst residual **0.45%** on `GLDM_GLN` (10.2 published against 10.2464 computed), which is what sets
the file's `rel=1e-2` tolerance. Every residual here is rounding in the published value, not
disagreement.

## What the two files are for

This file fixes the **definition** — it is the only assertion in the family tied to a published
consensus rather than to a tool run. `test_2d_gldm_pyradiomics.h` fixes the **digits**, at
`rel=1e-9`. Keeping both is deliberate: dropping the IBSI file would leave the family vetted only
against one implementation, and dropping the PyRadiomics file would leave a band 7 orders of
magnitude looser than the measured agreement.

That looseness is not hypothetical. `GLDM_DE` was 7.9e-4 away from its reference for as long as this
file has existed, and 7.9e-4 is invisible at `rel=1e-2`. See the PyRadiomics report.

## What changed in the file

- The table is `const` and read through `.at()` after a `count()` guard. It was a non-`const`
  `ref_vals_map` read with `operator[]`, which default-inserts 0.0 for a missing key and then
  compares against it with a tolerance of `0/frac = 0` — a green test on a golden that does not
  exist.
- The settings block and the four-slice averaging move to `test_2d_gldm_common.h`, shared with the
  PyRadiomics, regression and mechanics files. The copy here loaded slice 2's mask with
  `sizeof(ibsi_phantom_z2_intensity)` and slice 3's with `sizeof(ibsi_phantom_z3_intensity)` while
  slice 1 used its own mask's size; the arrays are all 20 entries so the counts happened to agree.
- The header comment now states the NGLDM mapping and where the identity is measured, instead of
  presenting the values as a GLDM table IBSI publishes.
- `../src/nyx/features/pixel.h` is gone: nothing in the file named a symbol from it. `NyxusPixel`
  comes from `test_data.h`.
