# Audit: `test_2d_glrlm_ibsi.h` goldens vs the IBSI digital phantom

**Verdict: all 16 hardcoded goldens reproduce to their own 3-significant-figure rounding**, and two
independent tools recompute them from the phantom pixels checked into this repo - so the "IBSI"
label is the published consensus, not Nyxus output relabelled.

## Method

The reference is a published table rather than a tool run, so the risk is circularity: a golden
could have been copied from Nyxus' own output. To rule that out, the phantom stored in
`tests/test_data.h` (`ibsi_phantom_z1..z4_intensity` / `_mask`) was fed to **MIRP 2.6.0** and
**PyRadiomics v3.0.1** at the IBSI 2D-averaged configuration (see the two tool reports for the exact
settings), and their full-precision output compared against both the 3-s.f. goldens and Nyxus.

## Result table

| feature | published golden (3 s.f.) | MIRP fresh | PyRadiomics fresh | Nyxus |
|---|---|---|---|---|
| `GLRLM_GLN` | 5.2 | 5.197062405 | 5.197062405 | 5.197062405 |
| `GLRLM_GLNN` | 0.46 | 0.4597293367 | 0.4597293367 | 0.4597293367 |
| `GLRLM_GLV` | 3.35 | 3.353028391 | 3.353028391 | 3.353028391 |
| `GLRLM_HGLRE` | 9.82 | 9.824274454 | 9.824274454 | 9.824274454 |
| `GLRLM_LGLRE` | 0.604 | 0.6043579077 | 0.6043579077 | 0.6043579077 |
| `GLRLM_LRE` | 3.78 | 3.778384223 | 3.778384223 | 3.778384223 |
| `GLRLM_LRHGLE` | 17.4 | 17.38702711 | 17.38702711 | 17.38702711 |
| `GLRLM_LRLGLE` | 3.14 | 3.144482992 | 3.144482992 | 3.144482992 |
| `GLRLM_RE` | 2.17 | 2.169550798 | 2.169550798 | 2.167200381 |
| `GLRLM_RLN` | 6.12 | 6.122863748 | 6.122863748 | 6.122863748 |
| `GLRLM_RLNN` | 0.492 | 0.4917413809 | 0.4917413809 | 0.4917413809 |
| `GLRLM_RP` | 0.627 | 0.6270994582 | 0.6270994582 | 0.6270994582 |
| `GLRLM_RV` | 0.761 | 0.7614750609 | 0.7614750609 | 0.7614750609 |
| `GLRLM_SRE` | 0.641 | 0.6406243545 | 0.6406243545 | 0.6406243545 |
| `GLRLM_SRHGLE` | 8.57 | 8.573136765 | 8.573136765 | 8.573136765 |
| `GLRLM_SRLGLE` | 0.294 | 0.2939658468 | 0.2939658468 | 0.2939658468 |

Every golden matches both tools at its own rounding, and Nyxus matches them to double precision on
15 of 16 - run entropy being the exception at 1.1e-3, the `fast_log10` + `EPSILON` accuracy choice
documented in the tool reports. The file's 1% band covers that; it is loose for the other fifteen,
but the goldens carry only 3 significant figures, so a tighter band would assert precision the
reference does not have. The tight comparison is the job of the two tool tests, which pin the same
quantities at full precision.

## What changed in the file

- The two helpers each carried their own copy of the 4-slice loop; both now go through
  `calc_2d_glrlm_phantom_feature` in `test_2d_glrlm_common.h`, which is also what the PyRadiomics
  and MIRP tests use, so all three oracles are computed by one code path.
- The `_AVE` helper's slice-length array read `sizeof(ibsi_phantom_z1_mask)` for the first slice and
  `sizeof(..._intensity)` for the other three. Same length today, so it was not a live defect; the
  shared helper reads the intensity array for all four.
- 10 `_AVE` features had no assertion (`SRE`, `LRE`, `GLN`, `GLNN`, `RLN`, `RLNN`, `RP`, `GLV`,
  `RV`, `RE`); the file asserted 6 of the 16. All 16 are asserted now.
- The golden lookup went through `operator[]`, which default-inserts a missing key as 0 and hands
  `agrees_gt` a tolerance of 0 - an assertion that can only pass on an exact 0. It now goes through
  `find()` plus an assert.
