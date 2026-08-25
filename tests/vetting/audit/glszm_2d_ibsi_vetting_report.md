# 2D GLSZM vs the IBSI published consensus — vetting report

`test_2d_glszm_ibsi.h` pins all sixteen 2D GLSZM features against the IBSI reference manual's
digital-phantom consensus table. This report checks that the pinned numbers are what the manual
publishes, and that Nyxus reproduces them.

| | |
|---|---|
| Reference | IBSI Documentation, Release 0.0.1dev (Dec 13, 2021), <https://ibsi.readthedocs.io/en/latest/03_Image_features.html> |
| Dataset | digital phantom, aggregation method "2D, averaged" |
| Recipe | `glszm.ibsi_phantom_2d` |
| Nyxus config | `IBSI=true`, `GREYDEPTH=128`, each of the four phantom slices featurised on its own, values averaged |
| Test | `test_2d_glszm_ibsi.h` |
| Tolerance | `rel=1e-2` |

## The table is genuine

All sixteen entries carry **three significant figures**, which is the precision the reference manual
publishes at. That is the check worth running on any `_ibsi` table: a sixteen-digit literal under an
`_ibsi` name is somebody's own run pasted in, not a transcription of a published consensus value.
Nothing in this table is longer than three figures.

## Result

Nyxus' four-slice mean against the published value. Two independent implementations — mirp 2.6.0 and
PyRadiomics 3.0.1 — were also run on the same fixture at the same config as a check that the
published table is being read correctly; they agree with each other to 7.0e-16 and with every entry
below to within the rounding:

| feature | IBSI published | Nyxus (4-slice mean) | rel |
|---|---|---|---|
| GLSZM_SAE | 0.363 | 0.36330794123204835 | 8.5e-4 |
| GLSZM_LAE | 43.9 | 43.86666666666667 | 7.6e-4 |
| GLSZM_LGLZE | 0.371 | 0.3711970899470899 | 5.3e-4 |
| GLSZM_HGLZE | 16.4 | 16.44047619047619 | 2.5e-3 |
| GLSZM_SALGLE | 0.0259 | 0.025854788674729148 | 1.7e-3 |
| GLSZM_SAHGLE | 10.3 | 10.277990480914587 | 2.1e-3 |
| GLSZM_LALGLE | 40.4 | 40.398082010582016 | 4.7e-5 |
| GLSZM_LAHGLE | 113 | 112.52142857142857 | 4.2e-3 |
| GLSZM_GLN | 1.41 | 1.4142857142857144 | 3.0e-3 |
| GLSZM_GLNN | 0.323 | 0.3229931972789115 | 2.1e-5 |
| GLSZM_SZN | 1.49 | 1.4857142857142858 | 2.9e-3 |
| GLSZM_SZNN | 0.333 | 0.3331972789115646 | 5.9e-4 |
| GLSZM_ZP | 0.24 | 0.24038957688338494 | 1.6e-3 |
| GLSZM_GLV | 3.97 | 3.9694784580498865 | 1.3e-4 |
| GLSZM_ZV | 21 | 20.997052154195014 | 1.4e-4 |
| GLSZM_ZE | 1.93 | 1.9280961666788374 | 9.9e-4 |

Worst residual **4.2e-3**, on `GLSZM_LAHGLE` — 113 published against 112.52142857 computed, which is
the rounding to three significant figures and nothing more. `rel=1e-2` is the band that
three-significant-figure references support; it is set by the reference's precision, not by the
disagreement.

`GLSZM_ZE`'s 9.9e-4 is partly the same rounding and partly the `fast_log10` approximation measured
in `glszm_2d_mirp_vetting_report.md` (2.0e-3 on the mean, against mirp and PyRadiomics alike). At
three significant figures the published value cannot separate the two, which is exactly why the
family also needs the mirp file: 1.93 is a value the approximation still rounds to.

## Why this file stays alongside the mirp one

The two are complementary, not redundant:

- **IBSI** pins the published consensus, which fixes the *definition* of each feature — but it is
  quoted to three significant figures, so it can only assert at `rel=1e-2`.
- **mirp** pins the full-precision digits at SPEC §7's exact tier -- an absolute `1e-9` band -- per
  slice as well as averaged.

Dropping the IBSI file would leave the family vetted only against one implementation, with nothing
tying it to the published standard. Dropping the mirp file would leave a band seven orders of
magnitude looser than the measured agreement — which is precisely the gap `GLSZM_ZE`'s 2.5e-3 sat
inside, unnoticed, for as long as ±1% was the only check.

Both oracles here are free and open, so keeping both costs nothing operationally.

## Reproduction

The published values are read from the reference URL above. The independent check of them is the
mirp run: see `glszm_2d_golden_regen.md`.
