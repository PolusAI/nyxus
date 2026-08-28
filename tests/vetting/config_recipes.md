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

## firstorder3d.pyradiomics_bincount20
- `bench_compat_liver_3d`, label 1, with PyRadiomics 3.0.1 `binCount=20`, no resampling and no
  weighting. Nyxus uses `GREYDEPTH=-20`; the negative value selects the corresponding radiomics
  bin-count mode.
- PyRadiomics computes first-order features from the original intensities. Only `Entropy` and
  `Uniformity` use the discretized histogram, so the bin count affects those two features.
- Used by: `test_3d_firstorder_pyradiomics.h`. Generator:
  `oracles/gen_firstorder3d_pyradiomics.py`.
- The measured bands are `rel=1e-9` for 12 same-definition statistics, `rel=1e-2` for the four
  percentile-derived values, and `rel=1e-3` for the population-versus-sample variance comparison.
  See `audit/firstorder_3d_pyradiomics_vetting_report.md`.

## firstorder3d.matlab_native
- `bench_ut57_3d`, label 57, with default 3D first-order settings. The fixture is put in the same
  integer domain as Nyxus' default float-NIfTI loader: shift the negative volume minimum to zero,
  then truncate nonnegative values to integers.
- MATLAB R2026a uses the named built-ins directly; derived statistics apply only their defining
  normalization to those results. `prctile(..., Method="midpoint")` and `iqr`
  use raw samples; Nyxus' percentile family uses its fixed 100-bin CDF, so that group uses `rel=1e-2`.
  The other same-definition comparisons use the SPEC `rel=1e-3` tier.
- Used by: `test_3d_firstorder_matlab.h`. Generator:
  `oracles/gen_firstorder3d_matlab.m`.

## firstorder3d.regression_ut_phantom
- `bench_ut57_3d`, label 57, with the same default settings. Snapshot-only coverage for first-order
  features with no equivalent native MATLAB function; establishes no oracle vetting.
- Used by: `test_3d_firstorder_regression.h`.

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

## gabor.cpp_static_defaults
- The `(frequency, angle)` set compiled into `GaborFeature::f0_theta_pairs` (`gabor.cpp`):
  f0 = {0, pi/4, pi/2, 3pi/4}, theta = {4, 16, 32, 64} **radians**. Every consumer of that vector
  reads `pair.first` as the frequency and `pair.second` as the angle, so this is what a run that
  sets no Gabor options computes — the gtest fixture, and a CLI run without
  `--gaborfreqs`/`--gabortheta`.
- Shared filter settings: kersize n=16, gamma=0.1, sig2lam=0.8, baseline f0LP=0.1 at theta=pi/2,
  GRAYthr=0.025. Benchmark: `bench_dsb2018_2d`. Oracle: `skimage` **at kernel level**
  (`gabor_kernel`, cropped to the 16x16 grid and L1-normalized) plus `analytic` for the f0=0
  member and for the count-ratio score, which skimage has no native equivalent of — see "what the
  oracle covers" below. Used by: `test_2d_gabor_skimage.cc`
  (`test_2d_gabor_cpp_static_defaults_skimage`).
- **The 16x16 crop and the zero-padded `full` convolution are part of the recipe, not implementation
  detail.** `gamma=0.1` makes `sigma_y` ten times `sigma_x`, so the analytic kernel runs to 369x113
  at the lowest frequency and only 7.9-47.6% of its L1 mass falls inside the window (100% from
  f0 >= 16). Scoring the same feature off skimage's own filtering (`skimage.filters.gabor`,
  untruncated support, zero-padded border) moves values by up to 1.005 — the whole range of the
  feature. Measured in `audit/gabor_2d_skimage_vetting_report.md` §4.1; generator part D2 re-runs it
  and part D1 reprints the mass table.
- The f0=0 member is degenerate: `gabor_kernel` cannot express frequency 0, and at f0=0 the
  envelope and the carrier are both identically 1, so that filter is the flat window in closed
  form. It is the only kernel in either recipe not taken from skimage.

## gabor.python_raw_defaults
- The `(frequency, angle)` set `GaborOptions::parse_input` builds from the default lists
  `gabor_freqs = [4, 16, 32, 64]`, `gabor_thetas = [0, 45, 90, 135]` degrees: f0 = {4, 16, 32, 64},
  theta = {0, pi/4, pi/2, 3pi/4} radians. This is what the Python API runs (it always passes its
  defaults through the parser) and what a CLI run that passes both flags runs.
- Same filter settings, benchmark and oracle as `gabor.cpp_static_defaults`. Used by:
  `test_2d_gabor_skimage.cc` (`test_2d_gabor_python_raw_defaults_skimage`).
- **`raw` names the units the recipe actually runs at, because the documented ones are not
  implemented.** `parse_input` converts the angle list from degrees to radians and passes the
  frequency list through untouched, so `4` reaches `GaborFeature::Gabor` as f0 = 4 and sets
  `lambda = 2*pi/4`. Both `src/nyx/python/nyxus/nyxus.py` (`gabor_freqs`: "comma-separated
  denominators of `\pi`") and `src/nyx/environment.cpp`'s CLI help describe the same list as
  denominators of pi, and nothing in the tree divides by pi — and the CLI help states a third
  default again (`1,2,4,8,16,32,64`). The recipe is therefore named for the numbers that reach the
  filter. Resolving which units are intended is a source change with public value impact
  (`audit/gabor_2d_skimage_vetting_report.md` §8 item 3); until it is resolved, no test here may be
  named for the documented contract.
- **The two recipes are not variants of one configuration — they are two different frequency and
  angle sets, and they produce values up to 0.84 apart on the same ROI.** The compiled-in default
  stores its pairs in the opposite order from the one the parser builds and every consumer reads;
  `audit/gabor_2d_skimage_vetting_report.md` measures both and records the defect.

## What `oracle = skimage` covers for the two gabor recipes
Both recipes carry `oracle=skimage` in `oracle_coverage.csv` because the registry takes one token
per row (SPEC §4), but the claim is narrower than "scikit-image computes this feature", and every
artifact presenting the token repeats the scope — the same rule SPEC §4 sets for `matlab`:
- **kernel** — skimage's, at the seven non-degenerate (f0, theta) points, cross-checked against the
  hand-derived closed form at 1.7e-18 .. 1.1e-16;
- **the f0=0 kernel** — analytic, since `gabor_kernel` cannot express frequency 0;
- **the score** — Nyxus' own WND-CHARM count ratio, reproduced in the generator. scikit-image has no
  equivalent, so nothing here is a second implementation of the whole feature.
A full-feature oracle for this family would be `wndcharm` (SPEC §4 names it as the highest-value
oracle for the Nyxus-original features); it is not built in this tree, and the gap is recorded in
`not_covered.md`.

## radial.shape2d_native
- The 8x8 `shape2d_morphology_{mask,intensity}` fixture (`test_data.h`) at `make_shape2d_settings()`,
  one 26-pixel concave ROI with an interior hole. `RadialDistributionFeature` reads the ROI contour,
  so `ContourFeature` runs first; nothing else in the shape set feeds it. Oracles: none - the family
  is regression-only. Benchmark: `bench_shape8_concave_holed`. Used by:
  `test_2d_radial_{regression,invariant,mechanics}.h`.
- The three features are 8-entry vectors and every entry is pinned; the whole table is decided by the
  ROI, the centre pixel and the normalising radius, which `test_2d_radial_mechanics.h` pins. That
  file is known-defect characterization and is credited to no feature in the registry - the values it
  pins are ones a correct fix must change (`audit/radial_2d_cellprofiler_vetting_report.md` §6).
  Config matrix: `matrix/radial.md`.

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

## zernike.shape2d_native
- The 8x8 `shape2d_morphology_{mask,intensity}` fixture (`test_data.h`) at `make_shape2d_settings()`.
  `ZernikeFeature` reads the ROI's image matrix and nothing any other feature produces, so the loader
  is the whole prerequisite. Oracle: `analytic`. Benchmark: `bench_shape8_concave_holed`. Used by:
  `test_2d_zernike_{analytic,regression,invariant,mechanics}.h`.
- The geometry is the recipe. `mb_zernike2D` builds its unit disk from the ROI's **bounding box**:
  radius = `min(bbox width, bbox height)` in pixels, centre = the ROI's **intensity centroid**,
  weights = `I / sum(I)`, and the moment carries a `(n+1)/pi` factor. On this ROI that is a 6x7 box,
  radius 6, centroid (3.8416, 4.4389) in 1-based pixel coordinates, and every one of the 42
  bounding-box pixels falls inside the disk -- which is what makes `A(0,0)` exactly `1/pi`.
  `test_2d_zernike_mechanics.h` pins all of it.
- 30 magnitudes, one per `(n, m)` with `n <= 9`, `m >= 0`, `n - m` even, emitted n-ascending then
  m-ascending.
- **Not comparable to CellProfiler `MeasureObjectIntensityDistribution`'s Zernikes**, which centre the
  disk on the object's minimum enclosing circle and normalise by pixel count rather than by total
  intensity. Both were run; the divergence is a convention gap, recorded in
  `tests/vetting/audit/zernike_2d_analytic_vetting_report.md`, not a disagreement about the moments.

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

## gldm.ibsi_phantom_2d
- The four IBSI digital-phantom slices (`ibsi_phantom_z1..z4`, `test_data.h`), each featurised on
  its own with `IBSI=true` and `GREYDEPTH=128`, and the per-feature values averaged over the four.
  Oracles: `pyradiomics` (which defines GLDM) and `ibsi` (published consensus). Used by:
  `test_2d_gldm_pyradiomics.h`, `test_2d_gldm_ibsi.h`.
- PyRadiomics reaches this configuration with `binWidth=1` (identity binning on the integer
  phantom, so neither tool discretises), `gldm_a=0`, `distances=[1]` and `force2D` — the alpha=0,
  d=1 coarseness Nyxus computes in IBSI mode. Generator: `oracles/gen_gldm_pyradiomics.py`.
- IBSI publishes these quantities under the **NGLDM** name. A GLDM dependence count is `1 +` the
  number of 8-neighbours sharing the centre's grey level, which is IBSI's `j = k + 1` at alpha=0,
  d=1, so the families line up one for one (`GLDM_SDE` against low-dependence emphasis, `GLDM_LDE`
  against high-dependence emphasis, and so on). The measurement establishing the identity —
  Nyxus GLDM bit-identical to the mirp-vetted `ngldm_2d_mirp_ref_vals` on 13 of 14 — is in
  `audit/gldm_2d_pyradiomics_vetting_report.md`.
- The two oracles are complementary rather than redundant. The IBSI consensus values are quoted to
  three significant figures, so that file asserts at `rel=1e-2` (measured worst case 0.45%, on
  `GLDM_GLN`: 10.2 published against 10.2464 computed). PyRadiomics reproduces Nyxus to **1.6e-16**
  on 13 of the 14, so the PyRadiomics file asserts at `rel=1e-9`: IBSI fixes the definition,
  PyRadiomics fixes the digits. `GLDM_DE` is the exception at `rel=2.5e-3` — `calc_DE()` takes its
  logarithm through the shared float `fast_log10()` approximation, worth a measured 1.3e-3.
- **Outside this recipe:** the production default (`IBSI=false`, MATLAB grey binning at
  `coarse_gray_depth`). There Nyxus re-bins the ROI while PyRadiomics bins at `binWidth=1`, so the
  two build their dependence matrices over different level assignments and disagree by up to 108%
  (`GLDM_SDLGLE`). That config is drift-pinned only, in `test_2d_gldm_regression.h` and
  `tests/python/test_2d_gldm_mechanics.py`.

## gldzm.ibsi_phantom_2d
- The four IBSI digital-phantom slices (`ibsi_phantom_z1..z4`, `test_data.h`), each featurised on
  its own with `IBSI=true` and `GREYDEPTH=128`. Each slice is its own ROI with its own scalar; their
  mean is what the IBSI "2D, averaged" aggregation publishes. Oracles: `ibsi` (published consensus)
  and `mirp`. Used by: `test_2d_gldzm_ibsi.h`, `test_2d_gldzm_mirp.h`.
- mirp reaches this configuration with `by_slice=True` and `base_discretisation_method="none"` (the
  phantom is already discrete 1..6). GLDZM needs no distance or coarseness parameter — the zone
  distance is a property of the ROI mask, not a setting. Generator: `oracles/gen_gldzm_mirp.py`.
- **Both the per-slice values and their mean are pinned.** A mean is blind to errors in two slices
  that cancel, and on this family that is exactly what happened: the zone-connectivity defect moved
  slice 1 and left slices 2–4 exact.
- The two oracles are complementary rather than redundant. The IBSI consensus values are quoted to
  three significant figures, so that file asserts at `rel=1e-2` (measured worst case 0.35%, on
  `GLDZM_LDE`: 1.21 published against 1.2142857 computed). mirp reproduces Nyxus to a worst
  **absolute** residual of 1.3e-15 (worst relative 7.0e-16), so the mirp file asserts at SPEC §7's
  exact tier, an absolute 1e-9 band: IBSI fixes the definition, mirp fixes the digits.
- `GLDZM_GLM` and `GLDZM_ZDM` are outside this recipe: they are Nyxus mean-style rows with no IBSI
  GLDZM counterpart and no mirp column, so they stay `regression` in `test_2d_gldzm_regression.h`.
  `GLDZM_SDLGLE` and `GLDZM_ZDV` are in the recipe but covered by mirp only — the IBSI file never
  held a published value for either.

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

## glszm.ibsi_phantom_2d
- The four IBSI digital-phantom slices (`ibsi_phantom_z1..z4`, `test_data.h`, benchmark
  `ibsi_digital_phantom`), each featurised on its own with `IBSI=true` and `GREYDEPTH=128`, and the
  per-feature values averaged over the four. Oracles: `ibsi` (published consensus) and `mirp`.
  Used by: `test_2d_glszm_ibsi.h`, `test_2d_glszm_mirp.h`. Config matrix: `matrix/glszm.md`.
- mirp reaches this configuration with `by_slice=True` and `base_discretisation_method="none"` (the
  phantom is already discrete 1..6). Generator: `oracles/gen_glszm_mirp.py`.
- Both the four-slice mean and each slice on its own are asserted. The mean is the quantity IBSI
  publishes, but it cannot vet the four values behind it: two slice errors that cancel leave it
  unmoved and a defect confined to one slice reaches it quartered.
- The two oracles are complementary rather than redundant. The IBSI consensus values are quoted to
  three significant figures, so that file asserts at `rel=1e-2` (measured worst case 0.42%, on
  `GLSZM_LAHGLE`: 113 published against 112.5214 computed). mirp agrees with Nyxus to within
  **2.0e-16** relative on fifteen of the sixteen features, a worst **absolute** residual of 7.1e-15,
  so the mirp file asserts those at SPEC §7's exact tier -- an absolute 1e-9 band: IBSI fixes the
  definition, mirp fixes the digits.
- `GLSZM_ZE` is the sixteenth and asserts at `rel=4e-3`. Nyxus computes zone entropy through
  `Nyxus::fast_log10`, a float-precision quadratic approximation of the logarithm, where mirp uses a
  double `log2`; the measured residual is 2.5e-3 per slice and 2.0e-3 on the mean. The band states
  that approximation and nothing else.
- **Outside this recipe:** Nyxus' default mode (`IBSI=false`) weights the grey-level-dependent
  features by raw intensity instead of by the grey-level index, which no reference reproduces --
  `GLSZM_HGLZE` is 16.44 here and 1497.57 there on the same fixture. Those values are drift-pinned
  only, in `test_2d_glszm_regression.h`, and no oracle row claims them.

## neighbor.scene2d_radius1
- Benchmark `bench_scene7_5roi_enclosed`; config matrix `matrix/neighbor.md`. Assertions at SPEC 7's
  exact tier, an absolute 1e-9 band.
- The `neighborhood2d_scene_labels` fixture (`test_data.h`): five labelled ROIs in one scene, built
  into a `roiData` map with basic-morphology and contour features computed first, then
  `NeighborsFeature::manual_reduce`. `PIXELDISTANCE=1`, `PIXELSIZEUM=1`, `XYRES=1`, `IBSI=false`.
  Oracles: `cellprofiler` and `analytic`. Used by: `test_2d_neighbor_cellprofiler.h`,
  `test_2d_neighbor_analytic.h`.
- **Per ROI, never aggregated.** Each of the five ROIs carries its own value for every feature and
  every one is asserted; there is no scene-level mean to hide behind.
- CellProfiler reaches this configuration with `MeasureObjectNeighbors`, `distance_method=Adjacent`,
  `neighbors_are_objects=True`, on the same label image padded by 3 px so no ROI touches the border
  (CP treats border objects specially). It reproduces Nyxus **exactly** -- residual 0 -- on
  `NUM_NEIGHBORS` and `CLOSEST_NEIGHBOR1_DIST`. Generator: `oracles/gen_neighbor_cellprofiler.py`.
- The `analytic` oracle recomputes the six centroid closed forms in numpy: direction angle
  `degrees(atan2(dy, dx))` mapped into `[0, 360)`, closest/second-closest by centroid distance with
  ties keeping ascending-label push order, `CLOSEST_NEIGHBOR2_*` = 0 when fewer than two neighbours
  lie within the radius, SAMPLE (n-1) standard deviation, and mode as the most frequent
  `round(angle)` bucket with the lowest bucket winning a tie. Agreement 1.4e-14 absolute
  (1.2e-16 relative). Generator: `oracles/gen_neighbor_analytic.py`.
- **Pixel distance is the family's only knob**, and only 1 is measured. `PIXELSIZEUM` and `XYRES` are
  set because the shared fixture's morphology and contour passes want them; the neighbour quantities
  are computed from raw pixel coordinates and do not read them. See `matrix/neighbor.md`.
- **Not circular:** the closed forms are evaluated on the neighbour *graph*, and that graph is what
  CellProfiler vets independently. The analytic oracle supplies the arithmetic, CP supplies the
  graph.
- **Two features CP cannot vet, for definitional reasons rather than disagreement.** CP's
  `AngleBetweenNeighbors` is the angle SUBTENDED at an object by two neighbours, not Nyxus' absolute
  direction angle; CP's `SecondClosestDistance` ranges over ANY object, whereas Nyxus reports the
  second-closest neighbour *within the search radius*. Different quantities, so they go to the
  analytic oracle.
- **Outside this recipe:** `PERCENT_TOUCHING`. Nyxus counts distinct contour pixels 8-adjacent to a
  neighbour over contour length; CP counts outline pixels overlapping a `disk(distance+0.5)`-dilated
  neighbour over perimeter. No CP distance method reproduces it -- measured divergence on 3 of the 5
  ROIs, up to 33.3 percentage points. Drift-pinned in `test_2d_neighbor_regression.h`, with its
  construction bounds in `test_2d_neighbor_invariant.h`.

## ngtdm.ibsi_phantom_2d
- The four IBSI digital-phantom slices (`ibsi_phantom_z1..z4`, `test_data.h`), each featurised on
  its own with `IBSI=true` and `GREYDEPTH=128`, and the per-feature values averaged over the four.
  Oracles: `ibsi` (published consensus) and `mirp`, with a `pyradiomics` run corroborating. Used by:
  `test_2d_ngtdm_ibsi.h`, `test_2d_ngtdm_mirp.h`.
- mirp reaches this configuration with `by_slice=True` and `base_discretisation_method="none"` (the
  phantom is already discrete 1..6). PyRadiomics reaches it with `binWidth=1` -- identity binning on
  an integer image -- plus `force2D=True`. Generator: `oracles/gen_ngtdm_mirp.py`. Config matrix:
  `matrix/ngtdm.md`.
- Both the four-slice mean and each slice on its own are asserted. The mean is the quantity IBSI
  publishes, but it cannot vet the four values behind it: two slice errors that cancel leave it
  unmoved and a defect confined to one slice reaches it quartered.
- **No distance parameter, on any side.** NGTDM's neighbourhood is the d=1 8-neighbourhood the IBSI
  definition fixes, and `ngtdm.cpp` never reads `PIXELDISTANCE`. A pixel distance set in an NGTDM
  test looks meaningful and changes nothing, so this recipe does not set one.
- All three references agree: mirp and PyRadiomics to **1.6e-16** of each other, Nyxus to
  **3.6e-15 absolute / 3.2e-16 relative** of both, and the published consensus to within its own
  3-significant-figure rounding (worst 0.41%, on `NGTDM_COARSENESS`). The mirp file therefore asserts
  at SPEC §7's exact tier -- an absolute `1e-9` band -- and the IBSI file at `rel=1e-2`: IBSI fixes
  the definition, mirp fixes the digits.
- **`NGTDMFeature::n_levels` is a static and is not part of this recipe.** In IBSI mode `ngtdm.cpp`
  forces the grey-binning info to 0, so the static is ignored entirely. That immunity is what
  `test_2d_ngtdm_mechanics.h` checks; being a mechanics test it establishes no vetting and no
  registry row cites it.

## ngtdm.default_fbn100
- The same four IBSI digital-phantom slices, through `NGTDMFeature` in Nyxus' **default** mode:
  `IBSI=false`, `GREYDEPTH=128` and `NGTDMFeature::n_levels=100`. The static wins over `GREYDEPTH`
  when the two differ (`ngtdm.cpp:44-49`), so the fixed bin number this recipe is named for is 100,
  assigned by MATLAB binning because the count is positive (`texture_feature.h:101-103`). **No
  oracle** -- pinned Nyxus output as drift guards in `test_2d_ngtdm_regression.h` at `rel=1e-3`.
  Config matrix: `matrix/ngtdm.md`.
- A different config point from `ngtdm.ibsi_phantom_2d` on the same fixture, not a second assertion
  on it. Default mode bins the intensities to a fixed grey count instead of using the phantom's own
  levels, and nothing reproduces that: `NGTDM_CONTRAST` is 0.925 under the IBSI recipe and 3169.93
  here. It is the mode a caller gets without asking for IBSI compliance, which is why it is pinned at
  all.
- The bin count is part of the recipe and is passed explicitly rather than assigned to the static and
  left: at the default `n_levels=0` the same fixture gives `NGTDM_CONTRAST` 6634.50, more than twice
  these pins.


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

## morphology3d.mirp_ibsi
- The segmented phantom (`phantoms/ut_inten.nii` + `phantoms/ut_mask57.nii`, label 57) at its native
  1×1×1 spacing, through `D3_SurfaceFeature` with `IBSI=true`, `GREYDEPTH=128`, `PIXELSIZEUM=100`.
  Oracle: `mirp` 2.6.0 (`by_slice=False`, `base_feature_families="morphology"`,
  `base_discretisation_method="none"`). Used by: `test_3d_morphology_mirp.h`.
- Morphology is computed from the mask geometry, so no grey-level binning applies on either side —
  `GREYDEPTH` is set only because the shared fixture sets it, and MIRP is told `none` explicitly.
- Covers the five PCA axis features (`morph_pca_*`), `3VOXEL_VOLUME` (`morph_vol_approx`), and the
  convex-hull quantity used by `3VOLUME_CONVEXHULL` and the current `3MESH_VOLUME` alias
  (`morph_volume / morph_vol_dens_conv_hull`). `morph_area_mesh` is not comparable to `3AREA`:
  MIRP integrates a marching-cubes mesh while Nyxus counts exposed voxel faces, so `3AREA` and its
  five derived features stay on `morphology3d.regression_ut_phantom`.

## morphology3d.matlab_regionprops3
- The same segmented phantom and Nyxus settings as `morphology3d.mirp_ibsi`. MATLAB Image Processing
  Toolbox reads `ut_mask57.nii`, selects label 57, and calls
  `regionprops3(mask == 57, 'Volume', 'ConvexVolume')` at native 1×1×1 spacing.
- `Volume` verifies `3VOXEL_VOLUME`; `ConvexVolume` verifies `3VOLUME_CONVEXHULL` and the current
  `3MESH_VOLUME` alias. Pinned with MATLAB R2026a.
- Used by: `test_3d_morphology_matlab.h`. Generator:
  `oracles/gen_morphology3d_matlab.m`. Bands: the existing MIRP bands shared by both oracle files —
  0.1% (`rel=1e-3`) for voxel volume and 5% for both hull aliases. The measured voxel-volume
  residual is 2.338e-04%, so both rows agree within their declared tolerance.

## morphology3d.covmatrix_numpy
- Ten fixed voxel coordinates (`morphology_3d_covmatrix_cloud`, `test_3d_morphology_mechanics.h`),
  no image and no feature: the sample covariance matrix and its eigenvalues, i.e. the arithmetic the
  PCA shape features are built on. Reference: `numpy` 2.4.6 (`cov(..., ddof=1)`, `linalg.eigvalsh`) —
  **not an oracle**: `numpy` is not a SPEC §4 oracle token and a mechanics assertion establishes no
  vetting. Used by: `test_3d_morphology_mechanics.h`.
  Generator: `oracles/gen_morphology3d_covmatrix_numpy.py`.
- `Nyxus::calc_covariance` normalises by n-1, which is what both numpy `ddof=1` and MATLAB `cov`
  compute, so the two sides are the same quantity; measured agreement is exact and the pins are
  asserted at rel=1e-9.
- These goldens were MATLAB `cov`/`eig` output quoted to five significant figures. They establish no
  vetting for any registry row (mechanics, SPEC 2); the features that consume them are vetted under
  `morphology3d.mirp_ibsi`.

## morphology3d.regression_ut_phantom
- The same phantom and settings, **no oracle** — pinned Nyxus output as drift guards in
  `test_3d_morphology_regression.h`.
- Carries `3AREA` and everything derived from it. The reason is a convention difference, not a
  numerical one: 59992 exposed voxel faces against MIRP's 46739 mesh area, ~28%.
## glrlm3d.pyradiomics_bincount20
- The compat phantom (`compat_int/compat_int_mri.nii` + `compat_seg/compat_seg_liver.nii`, label 1)
  at `GREYDEPTH=100`, `IBSI=false`, `GLRLM_GREYDEPTH=-20` (negative activates radiomics
  binCount-based binning, i.e. binCount 20). Oracle: `pyradiomics` (`binCount: 20`,
  `interpolator: sitkBSpline`, `weightingNorm:` empty, `imageType: Original`). Used by:
  `test_3d_glrlm_pyradiomics.h`, `tests/python/test_nyxus.py::test_3d_glrlm_compatibility`.
- PyRadiomics reports **one value per feature over its whole direction set**, which is the Nyxus
  `*_AVE` aggregation over the 13 3D angles — not a per-angle value. Both the per-angle base feature
  (via `calc_ave`) and the stored `*_AVE` feature are compared against that same golden.
- All 16 features of the family have a PyRadiomics counterpart, so none is vetted by identity.
  Nyxus reproduces 15 of them to double precision; `3GLRLM_RE`, the family's only sum over
  logarithms, lands 3.9e-4 away through `fast_log10` and is held to `rel=5e-3`.

## glrlm3d.regression_ut_phantom
- The segmented phantom (`phantoms/ut_inten.nii` + `phantoms/ut_mask57.nii`, label 57) at
  `GREYDEPTH=100`, `IBSI=false`, `GLRLM_GREYDEPTH=-20` (binCount 20), averaged over the 13 3D
  angles. **No oracle** — these are Nyxus' own values, pinned as drift guards in
  `test_3d_glrlm_regression.h` and regenerated by `test_3d_glrlm_dump_regression()`.
- Same grey binning as `glrlm3d.pyradiomics_bincount20`, on a different fixture. Positive
  `GLRLM_GREYDEPTH` values are deliberately avoided here: `3GLRLM_RP` leaves its [0,1] bound at
  those settings (`audit/glrlm_3d_pyradiomics_vetting_report.md`).

## glrlm3d.regression_ut_phantom_grey64
- The same segmented phantom and Nyxus settings as `glrlm3d.regression_ut_phantom` except
  `GLRLM_GREYDEPTH=+64` — positive, so `bin_intensities_3d` (`texture_feature.h`) bins through
  `matlab_grey_binning`'s fixed 64-level count rather than a bin count derived from the ROI's own
  min/max. **No oracle** — Nyxus' own values, pinned as drift guards in
  `test_3d_glrlm_regression.h` and regenerated by `test_3d_glrlm_dump_regression()`.
- Covers all 32 features: the 13 angled values of each base feature and the mean its `*_AVE` twin
  stores. Pinning only the means would be a guard nothing could trip from one direction, since two
  per-angle errors of opposite sign leave the mean where it was.
- It is the one benchmark in the family at a positive grey depth, which is where `3GLRLM_RP` is known
  to leave its [0,1] bound — and here it does: **4 of the 13 angles read 1.0231** while the average
  stays at 0.940. The pins record that, so the eventual fix shows up as a diff rather than as a
  silent change. The defect stays filed in `audit/glrlm_3d_pyradiomics_vetting_report.md`.
- Carries the aggregates the family's retired coverage sweep held (to `rel<=3.9e-16`, and `4.1e-10`
  for `RE`, the family's only `fast_log10` path), so it is not comparable to
  `glrlm3d.regression_ut_phantom` — the two bin the same voxels differently by construction.

## ngldm3d.regression_ut_phantom
- The segmented phantom (`phantoms/ut_inten.nii` + `phantoms/ut_mask57.nii`, label 57) at
  `GREYDEPTH=64`, `IBSI=false`. **No oracle** — pinned Nyxus output as drift guards in
  `test_3d_ngldm_regression.h`, regenerated by `test_3d_ngldm_dump_regression()`.
- Every 3D NGLDM feature sits here, which is unique among the 3D families. Not for want of a
  comparable tool: `ngldm3d.mirp_fbn64` below is config-matched and MIRP disagrees on 16 of the 17
  features it computes, for reasons that are Nyxus-side defects rather than convention differences.
  The pins are therefore a change detector for the eventual fix, not an endorsement — see
  `audit/ngldm_3d_mirp_vetting_report.md`.

## ngldm3d.mirp_fbn64
- The same phantom and binning through MIRP 2.6.0: `by_slice=False`,
  `base_discretisation_method="fixed_bin_number"`, `base_discretisation_n_bins=64`, distance 1,
  difference level (alpha) 0 — the IBSI NGLDM coarseness, and the same 64 grey levels the Nyxus side
  uses, so the two are directly comparable. Generator:
  `oracles/gen_ngldm3d_mirp.py`.
- **Referenced but not asserted against.** No registry row is vetted at this recipe today; it exists
  so the divergence is reproducible and so the promotion can be re-run against a fixed
  implementation. `3NGLDM_GLM` and `3NGLDM_DCM` have no MIRP counterpart at all.

## gldzm3d.regression_ut_phantom
- The segmented phantom (`phantoms/ut_inten.nii` + `phantoms/ut_mask57.nii`, label 57) at
  `GREYDEPTH=64`, `IBSI=false`. **No oracle** - pinned Nyxus output as drift guards in
  `test_3d_gldzm_regression.h`, at full `%.17g` precision and `rel=1e-9`.
- The pins are a change detector, not an endorsement: at this recipe Nyxus disagrees with MIRP on
  every one of the 16 features MIRP computes. See `gldzm3d.mirp_fbn64` below.

## gldzm3d.mirp_fbn64
- The same phantom and binning through MIRP 2.6.0: `by_slice=False`,
  `base_discretisation_method="fixed_bin_number"`, `base_discretisation_n_bins=64` - the same 64
  grey levels the Nyxus side uses, so the two are directly comparable. Generator
  `oracles/gen_gldzm3d_mirp.py`.
- **Referenced but not asserted against.** No registry row is vetted at this recipe: Nyxus'
  disagreement reaches 67x, and the generator shows why it is a defect rather than a convention -
  an independent implementation with 26-connected zones and a city-block distance transform
  reproduces MIRP to rel=3.2e-16 on the same fixture. `3GLDZM_GLM` and `3GLDZM_ZDM` have no MIRP
  counterpart at all.

## ngtdm3d.pyradiomics_binwidth1
- The 4x4x3 NGTDM phantom (`compat_int/compat_int_ngtdm_3d.nii` +
  `compat_seg/compat_seg_ngtdm_3d.nii`, label 57) at `GREYDEPTH=100`, `IBSI=false`,
  `NGTDM_GREYDEPTH=0` (no binning, so the raw levels survive) and `NGTDM_RADIUS=1`. Oracle:
  `pyradiomics` (`binWidth: 1`, `distances: [1]`, no resampling, `imageType: Original`). Generator:
  `oracles/gen_ngtdm3d_pyradiomics.py`. Used by `test_3d_ngtdm_pyradiomics.h` and
  `tests/python/test_nyxus.py::test_3d_ngtdm_compatibility`.
- **The two grey-level scales coincide here rather than being made to.** The phantom's intensities
  are the discrete values 0..5. PyRadiomics' `binWidth=1` maps a value to
  `floor(x/1) - floor(min/1) + 1`, i.e. 1..6; Nyxus does not bin at `NGTDM_GREYDEPTH=0` but shifts
  every level by one because the minimum is zero. Both sides therefore index the same six levels,
  which is what makes a `rel=1e-9` band honest on a cross-tool comparison.
- **PyRadiomics' public extractor cannot load this phantom.** Its mask is label 57 in all 48 voxels
  with no background, and `imageoperations.getMask()` raises "No labels found in this mask" whenever
  the mask has a single unique value. The generator constructs `RadiomicsNGTDM` directly, which is
  the same feature code, and cross-checks it against an independent numpy NGTDM.
- The five features are contractions of one `(i, n_i, p_i, s_i)` table, so the recipe also backs the
  per-level assertion in `test_3d_ngtdm_matrix_pyradiomics`, not only the five scalars.

## ngtdm3d.pyradiomics_binwidth1_r2
- `ngtdm3d.pyradiomics_binwidth1` with `NGTDM_RADIUS=2` — the same phantom, the same binning, the
  neighbourhood widened from 3×3×3 to 5×5×5. Oracle: `pyradiomics` at `distances: [1, 2]`, same
  generator. Used by the five `*_r2_pyradiomics` assertions and `test_3d_ngtdm_matrix_r2_pyradiomics`
  in `test_3d_ngtdm_pyradiomics.h`.
- **`distances` is a list of shells, not a radius.** `distances=[2]` is the 98 offsets at Chebyshev
  distance exactly 2 and excludes the 26 at distance 1; Nyxus scans the solid cube `-2..2`. So the
  match is `[1, 2]`, and `distances_semantics_check()` in the generator measures both readings
  against an independent numpy neighbourhood on every run rather than resting on the documentation.
- **What it is for.** The radius is the one axis of this family a default run gets wrong when it is
  left unset, so a second config point is what separates a family that honours `NGTDM_RADIUS` from
  one that ignores it: coarseness rises by half, busyness and complexity each fall by about a third.
  It is also what lets `test_3d_ngtdm_default_radius_mechanics` pin the default at exactly 1 instead
  of bounding it below.
- The radius is the only setting that moves between this recipe and
  `ngtdm3d.pyradiomics_binwidth1`. The levels and their `n_i` are unchanged — every voxel of this
  phantom has a neighbour at either radius — so the whole difference lands in `s_i`.

## ngtdm3d.regression_ut_phantom
- The segmented phantom (`phantoms/ut_inten.nii` + `phantoms/ut_mask57.nii`, label 57) at
  `GREYDEPTH=64`, `IBSI=false`, `NGTDM_GREYDEPTH=64`, `NGTDM_RADIUS=1`. **No oracle** — pinned Nyxus
  output as drift guards in `test_3d_ngtdm_regression.h`, regenerated by
  `test_3d_ngtdm_dump_regression()`.
- `NGTDM_GREYDEPTH=64` is MATLAB-style binning, which makes bin 1 the background level: a voxel
  binned there is not a matrix row of its own but still counts towards its neighbours' neighbourhood
  means. That is a different quantity from `ngtdm3d.pyradiomics_binwidth1`, where nothing is
  background, which is why this fixture carries no oracle claim.
- Both recipes state `NGTDM_RADIUS` explicitly rather than leaning on the default. At 0 the
  neighbourhood is empty and every feature is NaN, so a recipe that leaves the radius to whatever a
  settings vector happens to hold is not a recipe — which is what the unwired version of
  `test_3d_ngtdm_regression.h` did, and why its five assertions would have compared against NaN on
  their first run.

## glszm3d.pyradiomics_bincount20
- The compat phantom (`compat_int/compat_int_mri.nii` + `compat_seg/compat_seg_liver.nii`, label 1)
  at `GREYDEPTH=100`, `IBSI=false` and `GLSZM_GREYDEPTH=-20` (negative selects PyRadiomics-style
  binning, magnitude = bin count). Oracle: `pyradiomics` (`binCount: 20`, `interpolator:
  sitkBSpline`, no resampling, `weightingNorm:` empty, `imageType: Original`). Generator:
  `oracles/gen_glszm3d_pyradiomics.py`. Used by `test_3d_glszm_pyradiomics.h` and
  `tests/python/test_nyxus.py::test_3d_glszm_compatibility`.
- **`GREYDEPTH` is inert for this family and is stated only for consistency.** `D3_GLSZM_feature`
  reads `GLSZM_GREYDEPTH` and `IBSI`, nothing else, so the Python API's default `GREYDEPTH` reaches
  the same values the C++ assertions get at 100. Both are recorded rather than one, because a reader
  comparing this recipe against `glcm3d.*` would otherwise wonder which of the two decides the
  levels.
- **The two binning schemes coincide by construction, not by luck.** PyRadiomics' `binCount` lays
  `binCount` uniform edges over the ROI's own min..max and digitises to 1..N;
  `to_grayscale_radiomix` computes `(x - min) / ((max - min) / binCount) + 1` and clamps the top bin,
  which is the same partition. Both sides then gather zones over the full 26-voxel neighbourhood.
  That is what makes `rel=1e-9` honest on a cross-tool comparison: fifteen of the sixteen features
  agree to the last bit.
- **`3GLSZM_ZE` alone asserts at `rel=1e-3`.** It is the family's only sum over logarithms, and
  `calc_sums_of_P()` takes it through `fast_log10(...)/LOG10_2` where PyRadiomics uses `numpy.log2`.
  The measured residual is 1.9e-4 (6.4252043186232015 against 6.426417026786065). That is the same
  documented fast path 2D GLCM and 2D GLDM band for, not a defect of this family.
- The sixteen features are contractions of one `P(i, j)` size-zone matrix, so the recipe also backs
  the cell-by-cell assertion in `test_3d_glszm_matrix_pyradiomics`, not only the sixteen scalars.
  **PyRadiomics' `P_glszm` is not indexed by zone size.** `_calculateCoefficients()` deletes every
  column no zone occupies and keeps the surviving sizes in `jvector`; on this phantom that turns 634
  columns into 46 whose sizes run 1..16, 18..29, 31, 32, 34, 36, 44, ..., 634. Nyxus keeps the matrix
  dense at its full width, so a comparison that reads the size off the column index relabels almost
  every large zone. The generator's cross-table check is what catches that.
- **The matrix assertion reads the production table, not a copy of it.** `test_3d_glszm_matrix_pyradiomics`
  takes `P`, `I` and the four dimensions off the `D3_GLSZM_feature` the sixteen scalar assertions
  ran, so a defect in the row mapping, in `Ng`/`Ns`, in the allocation or in the fill loop is visible
  to it. It rebuilt the table beside `calculate()` until this pass, which left exactly those four
  places unasserted.
- **Two questions this recipe cannot answer**, because its fixture cannot: after `binCount` binning
  its grey levels are contiguous `1..20`, which makes the `IBSI=true` row index and the one beside it
  the same number, and its matrix has 186 populated cells, which nobody checks by hand. They belong
  to `glszm3d.pyradiomics_ibsi_gapped` and to the connectivity fixture `bench_cube4x4x3_zcross`.

## glszm3d.pyradiomics_ibsi_gapped
- `bench_cube3_gapped_levels` — a 3x3x3 literal in `test_3d_glszm_common.h` carrying grey levels 1, 3
  and 5 — at `IBSI=true` and `GLSZM_GREYDEPTH=64`, which `calculate()` overwrites with 0. Oracle:
  `pyradiomics` (`binWidth: 1`, no resampling, `weightingNorm:` empty, `force2D: False`), constructed
  on the array directly. Generator: `oracles/gen_glszm3d_pyradiomics.py`. Used by
  `test_3d_glszm_pyradiomics.h` and `test_3d_glszm_mechanics.h`.
- **This is the only recipe that reaches the `IBSI=true` branch**, and the only fixture on which that
  branch is distinguishable. `calculate()` indexes a zone's row as `zone.first - 1` under IBSI and as
  the position of that level in `I` otherwise, and reports `Ng` as `max(I)` against `I.size()`; on
  contiguous levels those are the same numbers. With 2 and 4 absent they are not, and the two sides
  still agree — PyRadiomics reports `Ng = 5` for three occupied levels, leaving rows 2 and 4 empty,
  and so does Nyxus.
- **`binWidth: 1` is the config match, not `binCount`.** Nyxus reads the volume's own values at
  `GLSZM_GREYDEPTH=0`; a bin width of 1 over integer levels leaves them where they are, so both sides
  index the same rows. A `binCount` here would renumber the three occupied levels into a contiguous
  range and destroy the property the fixture exists for.
- **`3GLSZM_ZE` asserts at `rel=2e-3` here rather than `rel=1e-3`.** Same `fast_log10` path, measured
  residual 1.1e-3 (1.7904872894287109 against 1.7924812503605767). Zone entropy sums `-p*log2(p)`
  over the matrix, so on six zones each term carries the approximation's error at full weight, while
  on the phantom's 860 zones it is spread over 186 terms and partly cancels.
- `test_3d_glszm_ibsi_equals_no_binning_mechanics` measures that this recipe and
  `glszm3d.regression_ut_phantom_nobinning` produce bit-identical values AND matrices on this
  fixture, which is why `matrix/glszm3d.md` carries one cell for the two.

## glszm3d.regression_ut_phantom
- The segmented phantom (`phantoms/ut_inten.nii` + `phantoms/ut_mask57.nii`, label 57) at
  `GREYDEPTH=64`, `IBSI=false` and `GLSZM_GREYDEPTH=64` (positive selects MATLAB-style binning into
  that many levels). **No oracle** — pinned Nyxus output as drift guards in
  `test_3d_glszm_regression.h`, regenerated by `test_3d_glszm_dump_regression()`.
- `GLSZM_GREYDEPTH=64` makes bin 1 the background level, so a voxel binned there starts no zone of
  its own. That is a different quantity from `glszm3d.pyradiomics_bincount20`, where nothing is
  background, which is why this fixture carries no oracle claim.
- Every recipe here states `GLSZM_GREYDEPTH` explicitly rather than leaning on the zero a settings
  vector starts at. Zero is this family's documented "no binning" default and it does produce
  numbers, but it is a third configuration again: the unwired version of
  `test_3d_glszm_regression.h` left it there, and at that setting every one of its sixteen pins is
  wrong by between 1.6x and 1.5e6x. It has a recipe of its own below.

## glszm3d.regression_ut_phantom_nobinning
- The same segmented phantom (`phantoms/ut_inten.nii` + `phantoms/ut_mask57.nii`, label 57) at
  `IBSI=false` and `GLSZM_GREYDEPTH=0` — no binning at all, the raw levels. **No oracle** — pinned
  Nyxus output in `test_3d_glszm_regression.h`, regenerated by `test_3d_glszm_dump_regression()`.
- **This is the configuration a run with no `--3glszm/greydepth` flag gets**, which is what earns it
  a recipe rather than a footnote: `compile_feature_settings()` zero-fills the family's settings
  vector and nothing writes `GLSZM_GREYDEPTH`, so 0 is the default reaching the feature.
  `test_3d_glszm_default_greydepth_mechanics` asserts that separately; the sixteen pins here are the
  numbers it produces, which finiteness alone was not saying.
- **No oracle is available at this point on this fixture.** PyRadiomics discretises before it counts
  zones and has no "read the raw levels" mode over a 2001-level phantom; the same *setting* does have
  an oracle on a fixture small enough to carry one, which is `glszm3d.pyradiomics_ibsi_gapped`.
- The matrix here is as wide as the phantom's largest raw level (`Ng` = 3024 against `binCount`'s 20),
  which is why the sixteen features share one gtest case: one phantom read answers all of them.
