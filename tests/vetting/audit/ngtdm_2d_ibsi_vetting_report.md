# 2D NGTDM vs the IBSI published consensus — vetting report

`test_2d_ngtdm_ibsi.h` pins all five 2D NGTDM features against the IBSI reference manual's
digital-phantom consensus table. This report checks that the pinned numbers are what the manual
publishes, and that Nyxus reproduces them.

| | |
|---|---|
| Reference | IBSI Documentation, Release 0.0.1dev (Dec 13, 2021), <https://ibsi.readthedocs.io/en/latest/03_Image_features.html> |
| Dataset | digital phantom, aggregation method "2D, averaged" |
| Recipe | `ngtdm.ibsi_phantom_2d` |
| Nyxus config | `IBSI=true`, `GREYDEPTH=128`, each of the four phantom slices featurised on its own, values averaged |
| Test | `test_2d_ngtdm_ibsi.h` |
| Tolerance | `rel=1e-2` |

## The table is genuine

All five entries carry **three significant figures**, which is the precision the reference manual
publishes at. That is the check worth running on any `_ibsi` table: a sixteen-digit literal under an
`_ibsi` name is somebody's own run pasted in, not a transcription of a published consensus value.
Nothing in this table is longer than three figures.

## Result

Nyxus' four-slice mean against the published value. Two independent implementations — mirp 2.6.0 and
PyRadiomics 3.0.1 — were also run on the same fixture at the same config as a check that the
published table is being read correctly; they agree with each other to 1.6e-16 and with every entry
below to within the rounding:

| feature | IBSI published | Nyxus (4-slice mean) | rel |
|---|---|---|---|
| NGTDM_COARSENESS | 0.121 | 0.12051055470374192 | 4.1e-3 |
| NGTDM_CONTRAST | 0.925 | 0.9252630132885581 | 2.8e-4 |
| NGTDM_BUSYNESS | 2.99 | 2.9887939543849873 | 4.0e-4 |
| NGTDM_COMPLEXITY | 10.4 | 10.400131856837582 | 1.3e-5 |
| NGTDM_STRENGTH | 2.88 | 2.8763659173789415 | 1.3e-3 |

Worst residual **4.1e-3**, on `NGTDM_COARSENESS` — 0.121 published against 0.1205106 computed, which
is the rounding to three significant figures and nothing more. `rel=1e-2` is the band that
three-significant-figure references support; it is set by the reference's precision, not by the
disagreement.

`NGTDM_COARSENESS` is the tightest of the five against its published value precisely because it is
the smallest: at 0.121, three significant figures leaves only ~4e-3 of room. That is a property of
the published precision, not a weakness in the feature.

## Why this file stays alongside the mirp one

The two are complementary, not redundant:

- **IBSI** pins the published consensus, which fixes the *definition* of each feature — but it is
  quoted to three significant figures, so it can only assert at `rel=1e-2`.
- **mirp** pins the full-precision digits at SPEC §7's exact tier -- an absolute `1e-9` band --
  per slice as well as averaged.

Dropping the IBSI file would leave the family vetted only against tool implementations, with nothing
tying it to the published standard. Dropping the mirp file would leave a band seven orders of
magnitude looser than the measured agreement — and, on this family, would also leave the per-slice
values unasserted, which is what the negative control in `ngtdm_2d_mirp_vetting_report.md` shows a
mean cannot cover.

## Reproduction

The published values are read from the reference URL above. The independent check of them is the
mirp and PyRadiomics runs: see `ngtdm_2d_golden_regen.md`.
