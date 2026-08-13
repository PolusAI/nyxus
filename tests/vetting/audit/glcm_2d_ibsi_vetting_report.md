# Audit: `test_2d_glcm_ibsi.h` goldens vs the IBSI digital phantom

**Verdict: all 28 hardcoded goldens reproduce to their own 3-significant-figure rounding.** No
drift, no oracle mismatch, and - the point of this check - no circularity: the goldens are the
published IBSI consensus values, and two independent tools reproduce them from the phantom pixels
that are checked into this repo.

## Method

The reference here is a published table, not a tool run, so the risk is the opposite of the usual
one: a golden could have been copied from Nyxus' own output and labelled "IBSI". To rule that out,
the phantom stored in `tests/test_data.h` (`ibsi_phantom_z1..z4_intensity` / `_mask`) was fed to
**two independent implementations** at the IBSI 2D-averaged configuration, and their full-precision
output compared against both the 3-s.f. goldens and Nyxus.

- **MIRP 2.6.0** - `by_slice=True`, `base_discretisation_method="none"`, `glcm_distance=1`,
  `glcm_spatial_method="2d_average"`, all 4 slices in one volume.
- **PyRadiomics v3.0.1** - `binWidth=1`, `symmetricalGLCM=True`, `distances=[1]`, `force2D=True`,
  `weightingNorm=None`, per slice, averaged over the 4 slices.
- **Nyxus** - the settings `assert_glcm_feature_ibsi` uses (`ibsi=true`, `GLCM_GREYDEPTH=0`,
  `GLCM_OFFSET=1`, angles 0/45/90/135), printed at 10 significant digits.

Masked voxels per slice: 20 / 19 / 17 / 18; in-mask grey levels {1,3,4,6} on z1/z3/z4 and
{1,3,4,6} on z2.

## Result table

| feature | published golden (3 s.f.) | MIRP fresh | PyRadiomics fresh | Nyxus |
|---|---|---|---|---|
| `GLCM_ACOR` | 5.09 | 5.094374029 | 5.094374029 | 5.094374029 |
| `GLCM_ASM` | 0.368 | 0.3675285624 | 0.3675285624 | 0.3675285624 |
| `GLCM_CLUPROM` | 79.1 | 79.11263068 | 79.11263068 | 79.11263068 |
| `GLCM_CLUSHADE` | 7 | 6.997816145 | 6.997816145 | 6.997816145 |
| `GLCM_CLUTEND` | 5.47 | 5.472932478 | 5.472932478 | 5.472932478 |
| `GLCM_CONTRAST` | 5.28 | 5.277851142 | 5.277851142 | 5.277851142 |
| `GLCM_CORRELATION` | -0.0121 | -0.01210696121 | -0.01210696121 | -0.01210696121 |
| `GLCM_DIFAVE` | 1.42 | 1.42246729 | 1.42246729 | 1.42246729 |
| `GLCM_DIFENTRO` | 1.4 | 1.396147113 | 1.396147113 | 1.393553011 |
| `GLCM_DIFVAR` | 2.9 | 2.90159075 | 2.90159075 | 2.90159075 |
| `GLCM_DIS` | 1.42 | 1.42246729 | 1.42246729 | 1.42246729 |
| `GLCM_ENTROPY` | 2.05 | 2.049664288 | 2.049664288 | 2.047605754 |
| `GLCM_HOM2` | 0.619 | 0.6187370709 | 0.6187370709 | 0.6187370709 |
| `GLCM_ID` | 0.678 | 0.6779485416 | 0.6779485416 | 0.6779485416 |
| `GLCM_IDM` | 0.619 | 0.6187370709 | 0.6187370709 | 0.6187370709 |
| `GLCM_IDMN` | 0.899 | 0.8992192901 | 0.8992192901 | 0.8992192901 |
| `GLCM_IDN` | 0.851 | 0.8513990718 | 0.8513990718 | 0.8513990718 |
| `GLCM_INFOMEAS1` | -0.155 | -0.1551195162 | -0.1551195162 | -0.1557629868 |
| `GLCM_INFOMEAS2` | 0.487 | 0.4874565677 | 0.4874565677 | 0.4883048989 |
| `GLCM_IV` | 0.0567 | 0.05669828975 | 0.05669828975 | 0.05669828975 |
| `GLCM_JAVE` | 2.14 | 2.142418606 | 2.142418606 | 2.142418606 |
| `GLCM_JE` | 2.05 | 2.049664288 | 2.049664288 | 2.047605754 |
| `GLCM_JMAX` | 0.519 | 0.5187996899 | 0.5187996899 | 0.5187996899 |
| `GLCM_JVAR` | 2.69 | 2.687695905 | 2.687695905 | 2.687695905 |
| `GLCM_SUMAVERAGE` | 4.28 | 4.284837211 | 4.284837211 | 4.284837211 |
| `GLCM_SUMENTROPY` | 1.6 | 1.603188041 | 1.603188041 | 1.601240106 |
| `GLCM_SUMVARIANCE` | 5.47 | 5.472932478 | 5.472932478 | 5.472932478 |
| `GLCM_VARIANCE` | 2.69 | 2.687695905 | 2.687695905 | 2.687695905 |

Every golden matches both tools at its own rounding. Nyxus matches the tools to <=1e-10 (the residual
is the 10-digit print precision of the Nyxus column) except on the five log-based features, where it
is 1.0e-3 to 4.1e-3 low - the `fast_log10` + `EPSILON` accuracy choice documented in the PyRadiomics
and MIRP reports. The file's 1% band covers that; it is loose for the other 23 but the goldens
themselves carry only 3 significant figures, so a tighter band would be asserting precision the
reference does not have.

## Notes

- `GLCM_HOM2` and `GLCM_ENTROPY` are Nyxus names for IBSI `IDM` (WF0Z) and `JE` (TU9B); the fresh
  runs confirm the equality on this fixture (0.6187370709 and 2.049664288 respectively).
- `GLCM_VARIANCE` has no IBSI feature of its own. It is the joint variance UR99 computed about the
  grey-level marginal mean rather than the level index, and on this phantom it lands on the UR99
  consensus (2.69) and on `GLCM_JVAR` to 12 digits.
- The redundant `test_2d_glcm_inversed_difference_moment_ibsi` (a second, identical assertion of
  `GLCM_IDM` alongside `test_2d_glcm_idm_ibsi`) was removed; 7 `_AVE` twins whose base feature had
  an IBSI golden but no `_AVE` assertion were added.
