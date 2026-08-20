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

## radial.shape2d_native
- The 8x8 `shape2d_morphology_{mask,intensity}` fixture (`test_data.h`) at `make_shape2d_settings()`,
  one 26-pixel concave ROI with an interior hole. `RadialDistributionFeature` reads the ROI contour,
  so `ContourFeature` runs first; nothing else in the shape set feeds it. Oracles: none - the family
  is regression-only. Used by: `test_2d_radial_{regression,invariant,mechanics}.h`.
- The three features are 8-entry vectors and every entry is pinned; the whole table is decided by the
  ROI, the centre pixel and the normalising radius, which `test_2d_radial_mechanics.h` pins.

## radial.cellprofiler_8bin
- CellProfiler `MeasureObjectIntensityDistribution` on the same fixture: `center_choice="These
  objects"`, `bin_count=8`, `wants_scaled=True`, Zernikes off. Oracle: `cellprofiler`.
- **Run 2026-08-20 and it does not vet the family.** Nyxus computes a different quantity under each
  of the three `RadialDistribution_*` names, and 21 of the 24 (feature x bin) values disagree by more
  than 1%. The recipe is kept because it is the configuration that was tried and the one to re-try
  after the source divergences are resolved; the six of them are in
  `tests/vetting/audit/radial_2d_cellprofiler_vetting_report.md` and the run is
  `tests/vetting/oracles/gen_radial_cellprofiler.py`.
- The fixture's distance-to-edge maximum is attained by 8 of its 26 pixels, and CellProfiler's centre
  is that maximum, so CellProfiler's own answer moves with the label image's padding. A tie-free ROI
  is a precondition for this recipe ever vetting anything.

## morphology.shape2d_native
- The 8x8 `shape2d_morphology_{mask,intensity}` fixture (`test_data.h`) at
  `make_shape2d_settings()`: `PIXELSIZEUM=2.0`, `XYRES=1.0`, `GREYDEPTH=128`, `IBSI=false`, single
  ROI, no anisotropy. Oracles: `matlab` (Octave `regionprops`), `skimage` (`regionprops`), `imea`
  (the two DIN ISO 9276-6 macro transforms), `cellprofiler`, `analytic`. Used by:
  `test_2d_morphology_{matlab,skimage,imea,cellprofiler,analytic}.h`.
- Coordinate frames differ per property and the conversion is part of the recipe: MATLAB centroids
  are 1-based pixel centres (subtract 1); its `BoundingBox` corner is at min-0.5 in 1-based coords
  (so the 0-based min index is `BoundingBox(1) - 0.5`); skimage `orientation` is measured from the
  row axis (Nyxus is `90 - degrees(...)`).
- MATLAB and Nyxus both apply the +1/12 pixel finite-size correction to the second central moments,
  so `MAJOR_AXIS_LENGTH` / `MINOR_AXIS_LENGTH` / `ECCENTRICITY` vet against `matlab` and *not*
  against `skimage`, which omits it and differs ~1.4%.
- `PERIMETER` is **not** comparable on this fixture (Nyxus chain-code walk 26.935 vs skimage 12.657
  on a 26-pixel object with a hole); it is vetted at `morphology.perimeter_circles` instead.

## morphology.perimeter_circles
- The `roiDataForPerimeterTest` fixture (`test_data.h`), a single 14309-pixel object, `PIXELSIZEUM=100`,
  `PIXELDISTANCE=5`, `IBSI=false`. Oracle: `skimage` `measure.perimeter` (the 4-neighbourhood
  boundary walk that `regionprops.perimeter` uses). Used by: `test_2d_morphology_skimage.h`.
- On an object this size the chain-code walk and the 4-neighbourhood walk are the same algorithm and
  agree to 3.8e-15. Note `nnz(bwperim(...))` counts perimeter *pixels* (846 here) and MATLAB's
  `regionprops('Perimeter')` returns 952.848 - neither is this quantity.

## morphology.caliper_ellipse
- The filled ellipse a=20, b=10 built by `calculate_ellipse_caliper_values()`
  (`test_2d_morphology_common.h`), at `make_shape2d_settings()` (`test_main_nyxus.h`). Oracle: `imea`
  `shape_measurements_2d(mask, spatial_resolution_xy=1.0, dalpha=10)`. Used by:
  `test_2d_morphology_imea.h`.
- `dalpha=10` matches Nyxus' own caliper sweep (`rot_angle_increment = 10` degrees, `caliper.h`), so
  both sample the same angles. Worst residual across the 17 pinned goldens is 4.99%; the assertions
  use `reltol=0.06` for the definitional hull-vs-raster gap.
- The `_MODE` statistics are **excluded**: imea's own mode ranges over 19..24 as `dalpha` goes 5 to
  30, further than the Nyxus-imea gap, so no tolerance distinguishes agreement from sampling noise.
  They stay `regression`. The caliper statistics on the coarse 8x8 raster are excluded for the same
  class of reason - there imea's own values differ from Nyxus by 3.9-79.3%.

## morphology.fractal_blob512
- The 512x512 mask `tests/data/fractal_blob512_seg.ome.tif`, loaded by
  `calculate_fractal_blob512_feature_values()` (`test_2d_morphology_common.h`) as a single ROI.
  Oracle: `fraclac` (ImageJ FracLac box counting). Used by: `test_2d_morphology_fraclac.h`,
  `tests/python/test_2d_morphology_fraclac.py`.
- Generators: `oracles/fraclac/shiftgrid_boxcount.ijm` (the headless ImageJ macro) and
  `oracles/fraclac/ref_boxcount.py`. `FRACT_DIM_BOXCOUNT` is same-method (1% tolerance);
  `FRACT_DIM_PERIMETER` is cross-method — Nyxus uses a divider walk against FracLac's edge box
  count — so it carries a stated ~3% band, per SPEC 7's "known method divergence" row.

## ngldm.ibsi_phantom_2d
- The four IBSI digital-phantom slices (`ibsi_phantom_z1..z4`, `test_data.h`), each featurised on
  its own with `IBSI=true` and `GREYDEPTH=128`, and the per-feature values averaged over the four.
  Oracles: `ibsi` (published consensus) and `mirp`. Used by: `test_2d_ngldm_ibsi.h`,
  `test_2d_ngldm_mirp.h`.
- mirp reaches this configuration with `by_slice=True`, `base_discretisation_method="none"` (the
  phantom is already discrete 1..6), `ngldm_distance=1` and `ngldm_difference_level=0` — the
  alpha=0, d=1 coarseness the IBSI NGLDM definition uses. Generator:
  `oracles/gen_ngldm_mirp.py`.
- The two oracles are complementary rather than redundant. The IBSI consensus values are quoted to
  three significant figures, so that file asserts at `rel=1e-2` (measured worst case 0.45%, on
  `NGLDM_GLNU`: 10.2 published against 10.2464 computed). mirp reproduces Nyxus to **2.9e-16**, so
  the mirp file asserts at `rel=1e-9`: IBSI fixes the definition, mirp fixes the digits.
- `NGLDM_GLM` and `NGLDM_DCM` are outside this recipe: they are Nyxus mean-style rows with no IBSI
  NGLDM counterpart and no mirp column, so they stay `regression` in `test_2d_ngldm_regression.h`.

## glcm3d.pyradiomics_bincount20
- The compat phantom (`compat_int/compat_int_mri.nii` + `compat_seg/compat_seg_liver.nii`) at
  `GREYDEPTH=100`, `IBSI=false`, `GLCM_GREYDEPTH=-20` (negative activates radiomics binCount-based
  binning, i.e. binCount 20), `GLCM_OFFSET=1`, `GLCM_SPARSEINTENS=true`. Oracle: `pyradiomics`
  (`binCount: 20`, `interpolator: sitkBSpline`, `weightingNorm:` empty, `imageType: Original`).
  Used by: `test_3d_glcm_pyradiomics.h`.
- PyRadiomics reports **one value per feature over its whole direction set**, which is the Nyxus
  `*_AVE` aggregation over the 13 3D angles — not a per-angle value. Both the per-angle base feature
  (via `calc_ave`) and the stored `*_AVE` feature are compared against that same golden.
- Six `*_AVE` features are vetted through an identity rather than a golden of their own, because
  PyRadiomics does not report them under their own name: `DIS≡DIFAVE`, `ENERGY≡ASM`,
  `ENTROPY≡JE`, `HOM1≡ID`, `SUMVARIANCE≡CLUTEND`, `VARIANCE≡JVAR`. The identities are asserted at
  1e-6, and the twin carries the oracle claim.

## glcm3d.regression_ut_phantom
- The segmented phantom (`phantoms/ut_inten.nii` + `phantoms/ut_mask57.nii`, label 57) at
  `GREYDEPTH=100`, `IBSI=false`, `GLCM_GREYDEPTH=-100` (binCount 100), `GLCM_OFFSET=1`,
  `GLCM_SPARSEINTENS=true`, averaged over the 13 3D angles. **No oracle** — these are Nyxus' own
  values, pinned as drift guards in `test_3d_glcm_regression.h` and regenerated by
  `test_3d_glcm_dump_regression()`.
- A different binCount from `glcm3d.pyradiomics_bincount20`, so the two benchmarks are not
  comparable to each other by construction.
