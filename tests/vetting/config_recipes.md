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

## ih.mirp_fbn
- Fixed-bin-number discretised intensity histogram (IBSI IH family). Oracle: `mirp`.
- Vet `IH_*_IDX` (bin-index domain); `IH_*_VAL` (bin-center value) is analytic vs Nyxus definition.

## ih.ibsi_fbn
- Fixed-bin discretised intensity histogram (IBSI IH family 3.4) on the IBSI digital phantom,
  GREYDEPTH = <IH_PHANTOM_NBINS>, IBSI=true. Oracle: `ibsi`. Used by: `test_2d_intensity_histogram_ibsi.h`.
- `IH_*_IDX` vet directly against IBSI IH consensus (index/grey-level domain). `IH_*_VAL` are the
  same statistics over bin centers = affine transform of the IDX distribution (VAL = binWidth·IDX
  for spreads; +minVal offset for locations), so they are anchored to the IBSI-vetted IDX values.
  `ROBUST_MEAN_*` have no IBSI feature -> analytic (see test_ih_dispersion_robust_analytic).

## moments.skimage_regionprops
- scikit-image `regionprops` moments. Caveats: skimage transposes row/col indices (skimage m[i,j] =
  Nyxus m_{j,i}); weighted moments center on the intensity-weighted centroid; Hu returned raw (not log),
  2D only; normalized moments NaN for order < 2. Used by: `test_2d_moments_skimage.h`.

## radial.cellprofiler_8bin
- CellProfiler `MeasureObjectIntensityDistribution`, 8 radial bins/slices. Oracle: `cellprofiler`.
