# Regenerating the 2D GLCM goldens

Every number pinned in `tests/test_2d_glcm_{pyradiomics,mirp}.h` comes out of a checked-in
generator; nothing here is transcribed by hand. `test_2d_glcm_ibsi.h` is the exception and is
covered at the end. Run everything offline - CI never invokes a reference tool.

## 1. Stand the tools up

Conda is enough for both; no Docker (`tests/vetting/TOOLS.md` has the matrix and the gotchas).

```bash
conda create -n nyxus_oracle -c conda-forge python=3.9 pyradiomics simpleitk numpy
conda create -n nyxus_mirp   -c conda-forge python=3.11 mirp numpy
```

Record what you actually resolved, next to the goldens:

```bash
conda run -n nyxus_oracle python -c "import radiomics; print(radiomics.__version__)"   # v3.0.1
conda run -n nyxus_mirp   python -c "from importlib import metadata; print(metadata.version('mirp'))"   # 2.6.0
```

## 2. The fixture

Both generators build it themselves, so there is no file to prepare:

```
img[y,x] = ((y + 2x) % 8) + 1     8x8, one-pixel background border -> 10x10
mask     = 1 inside the 8x8, 0 on the border
```

`tests/test_2d_glcm_common.h` builds the identical array in C++ (`make_2d_glcm_dense_phantom`).
If you change one you must change the other, and the assertion is what will tell you: the goldens
are tied to these exact pixels.

## 3. Run the generators

```bash
conda run -n nyxus_oracle python tests/vetting/oracles/gen_glcm_pyradiomics.py
conda run -n nyxus_mirp   python tests/vetting/oracles/gen_glcm_mirp.py
```

Each prints a paste-ready table body plus a provenance header line. Replace the body of
`glcm_2d_pyradiomics_ref_vals` / `glcm_2d_mirp_ref_vals` with it, keep the header line, rebuild
`runAllTests`, run `--gtest_filter=*GLCM_FAMILY*`.

## 4. The configuration each tool is run at

Recipe `glcm.ibsi_identity` on the Nyxus side (`ibsi=true`, `GLCM_GREYDEPTH=0`, `GLCM_OFFSET=1`,
angles 0/45/90/135). The tool-side settings that make it the same configuration:

| | PyRadiomics | MIRP |
|---|---|---|
| discretisation | `binWidth=1` (identity on an integer image) | `base_discretisation_method="none"` |
| symmetry | `symmetricalGLCM=True` | symmetric by definition |
| distance | `distances=[1]` | `glcm_distance=1` |
| plane | `force2D=True`, `force2Ddimension=0` | `by_slice=True` |
| aggregation | one value over the angle set | `glcm_spatial_method="2d_average"` |
| weighting | `weightingNorm=None` | (none) |

Do **not** use recipe `glcm.pyradiomics_symmetric` (non-IBSI path, fixed bin count) for these: the
re-binning moves the absolute grey levels, and every feature that reads a level rather than a level
difference - `ACOR`, `SUMAVERAGE`, `IDN`, `IDMN` - stops being comparable.

## 5. Mapping tool names to Nyxus features

One tool quantity can cover several Nyxus names. The mapping lives in the generators
(`PYRADIOMICS_TO_NYXUS`, `MIRP_TO_NYXUS`); the equalities behind it:

| Nyxus | PyRadiomics | MIRP | note |
|---|---|---|---|
| `GLCM_ASM`, `GLCM_ENERGY` | `JointEnergy` | `cm_energy` | the two Nyxus names are one quantity |
| `GLCM_ID`, `GLCM_HOM1` | `Id` | `cm_inv_diff` | |
| `GLCM_IDM`, `GLCM_HOM2` | `Idm` | `cm_inv_diff_mom` | `GLCM_HOM2` has no `_AVE` twin |
| `GLCM_JE`, `GLCM_ENTROPY` | `JointEntropy` | `cm_joint_entr` | |
| `GLCM_JVAR`, `GLCM_VARIANCE` | `SumSquares` | `cm_joint_var` | IBSI joint variance UR99 |
| `GLCM_CLUTEND`, `GLCM_SUMVARIANCE` | `ClusterTendency` | `cm_clust_tend`, `cm_sum_var` | PyRadiomics dropped SumVariance as a duplicate; MIRP reports both |
| `GLCM_DIFAVE`, `GLCM_DIS` | `DifferenceAverage` | `cm_diff_avg`, `cm_dissimilarity` | PyRadiomics dropped Dissimilarity as a duplicate |
| `GLCM_SUMAVERAGE` | `SumAverage` | `cm_sum_avg` | PyRadiomics warns it is 2x`JointAverage` under symmetry |
| `<X>` and `<X>_AVE` | one value | one value | the tools report the angle-averaged value; the per-angle feature is checked as the mean of its 4 angles |

`MCC` (PyRadiomics) has no Nyxus counterpart and is ignored.

## 6. Convention differences to expect

- **Log-based features** - `DIFENTRO`, `JE`/`ENTROPY`, `SUMENTROPY`, `INFOMEAS1`, `INFOMEAS2` land
  1e-3..3e-3 off both tools because Nyxus sums through `fast_log10` with an `EPSILON` guard. Assert
  those at `rel=5e-3`; everything else holds at `rel=1e-9`.
- **`GLCM_VARIANCE` vs `GLCM_JVAR`** - different routines in Nyxus (marginal mean vs level index)
  reaching the same quantity. If a future change makes them differ, the fixture will show it: both
  are pinned to the same tool value.
- **Grey-level gaps** - at identity binning both tools build the matrix over the image's own levels,
  so a fixture with absent levels is fine; verified on the IBSI phantom (levels {1,3,4,6}), where
  all three agree to 1e-15. This is not true at a fixed bin count.

## 7. `test_2d_glcm_ibsi.h`

Its goldens are the published IBSI consensus (reference manual, "dig phantom", 2D-averaged), not a
tool run, so there is nothing to regenerate - and the phantom pixels are checked into
`tests/test_data.h`. To re-verify them, run either tool on that phantom at the configuration in
section 4 and compare; `glcm_2d_ibsi_vetting_report.md` is the record of doing exactly that, with
the parsing snippet needed to lift the phantom out of `test_data.h`.
