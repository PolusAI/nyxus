# Config Recipes

A **config recipe** is the exact Nyxus setting bundle that makes a feature directly comparable to a
chosen reference tool (SPEC 5). Oracle tests reference a recipe by id; this file defines each once.

## glcm.ibsi_identity
- `ibsi=True`, grey levels = distinct levels (identity binning), symmetric matrix, all directions.
- Oracle: `ibsi` reference tables / `mirp` / `pyradiomics`. Used by: `test_2d_glcm_ibsi.h`,
  `test_2d_glcm_pyradiomics.h`, `test_2d_glcm_mirp.h`.
- PyRadiomics reaches this configuration with `binWidth=1` on an integer image (identity binning),
  `symmetricalGLCM=True`, `distances=[1]`, `force2D=True`, `weightingNorm=None`; MIRP with
  `base_discretisation_method="none"`, `glcm_distance=1`, `by_slice=True`. Both report one value per
  feature over the angle set, which is the Nyxus `*_AVE` aggregation.
- Two fixtures are pinned at this recipe and both were run against both tools: the IBSI digital
  phantom (`test_2d_glcm_ibsi.h`, in-mask levels {1,3,4,6}) and a dense 8x8 phantom where every
  level 1..8 and every level pair occurs (`test_2d_glcm_common.h`).

## glcm.pyradiomics_symmetric
- `ibsi=False`, fixed bin count, symmetric GLCM averaged over directions, distance 1.
- Oracle: `pyradiomics` (symmetricalGLCM=True, binCount matched). Used by: `test_2d_glcm_pyradiomics.h`.
- Note: Nyxus 3D GLCM ground-truth was asymmetric/1-offset/100-level; the 3D recipe must force
  symmetric + 13-direction to match pyradiomics (see MIGRATION.md 5.8).

## firstorder.pyradiomics_default
- Non-discretised intensity statistics. Oracle: `pyradiomics` firstorder. Used by: `test_2d_firstorder_pyradiomics.h`.

## firstorder.preserve_hu
- `--preserve-hu` (CT/Hounsfield mode): the loader applies a slope-1 offset (`value - floor(HU_min)`)
  instead of min-max rescaling, so first-order intensity features are reported in true Hounsfield units.
  Not a new feature — a config mode; adds absolute-HU corroboration to the existing MIN/MAX/MEAN/
  INTEGRATED_INTENSITY features. Oracle: `pydicom` (RescaleSlope/Intercept → HU on a real CT slice).
  Used by: `test_2d_hu_ct_small_pydicom.py`.

## ih.ibsi_fbn
- Fixed-bin-number discretised intensity histogram (IBSI IH family 3.4) on the IBSI digital
  phantom, GREYDEPTH = <IH_PHANTOM_NBINS>, IBSI=true, all four slices as one ROI. Oracles: `ibsi`
  (published consensus), `mirp` (fixed_bin_number, n_bins matching, `by_slice=False`), `analytic`.
  Used by: `test_2d_intensity_histogram_{ibsi,mirp,analytic}.h`, which share one fixture in
  `test_2d_intensity_histogram_common.h`.
- `IH_*_IDX` are the statistics over 1-based bin indices and vet against MIRP to double precision;
  12 of them also carry an IBSI consensus value. `IH_ROBUST_MEAN_IDX` is the exception - no tool and
  no IBSI feature reports it - so it is `analytic`.
- `IH_*_VAL` are the same statistics carried into the intensity domain, and **five conventions do
  that**; there is no single transform from `_IDX`. Measured in
  `audit/intensity_histogram_2d_analytic_vetting_report.md`:

  | convention | map | features |
  |---|---|---|
  | bin centre, location | `VAL = lo + (IDX-0.5)·binWidth` | MEAN, MEDIAN, MODE, ROBUST_MEAN |
  | bin centre, scale | `VAL = binWidth·IDX` | MAD, MEDAD, RMAD |
  | bin centre, squared scale | `VAL = binWidth²·IDX` | VARIANCE |
  | domain-invariant | `VAL = IDX` | SKEWNESS, EXCESS_KURTOSIS, ENTROPY, UNIFORMITY |
  | no map | — | P10, P90, IQR, QCoD, MIN, MAX, RANGE, COV |

  The last row is three separate things: MIN/MAX/RANGE are raw voxel values; P10/P90/IQR/QCoD are
  the grouped-data percentile `L + binWidth·(n·p - F)/f` where the `_IDX` half is the discrete
  grey-level percentile; and COV is a ratio of two differently-scaled quantities. Only the first two
  rows may be anchored to an `_IDX` value, which is what `test_2d_intensity_histogram_ibsi.h` does
  for the three deviation measures.
- **The four percentile `_VAL` features are `regression`, not `vetted`.** No reference
  implementation reproduces the grouped-data percentile - numpy's nine methods and Octave's nine
  all answer the sample percentile, which is what the `_IDX` half reports - so `IH_P10_VAL`,
  `IH_P90_VAL`, `IH_INTERQUANTILE_RANGE_VAL` and `IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL` are
  pinned as drift guards in `test_2d_intensity_histogram_regression.h` and claim no oracle. Every
  other `_VAL` feature is pinned against the closed forms in
  `oracles/gen_intensity_histogram_analytic.py`.

## moments.skimage_regionprops
- scikit-image `regionprops` moments. Caveats: skimage transposes row/col indices (skimage m[i,j] =
  Nyxus m_{j,i}); weighted moments center on the intensity-weighted centroid; Hu returned raw (not log),
  2D only; normalized moments NaN for order < 2. Used by: `test_2d_moments_skimage.h`.

## radial.cellprofiler_8bin
- CellProfiler `MeasureObjectIntensityDistribution`, 8 radial bins/slices. Oracle: `cellprofiler`.
