# Regenerating the 2D GLRLM goldens

Every number pinned in `tests/test_2d_glrlm_{pyradiomics,mirp}.h` comes out of a checked-in
generator; nothing here is transcribed by hand. `test_2d_glrlm_ibsi.h` is the exception and is
covered at the end. Run everything offline - CI never invokes a reference tool.

## 1. Stand the tools up

Conda is enough for both; no Docker.

```bash
conda create -n nyxus_oracle -c conda-forge python=3.9  pyradiomics simpleitk numpy
conda create -n nyxus_mirp   -c conda-forge python=3.11 mirp numpy
```

pyradiomics needs Python <= 3.9 on conda-forge; ask for the interpreter explicitly or the solver
picks a newer one and fails. Record what you resolved, next to the goldens:

```bash
conda run -n nyxus_oracle python -c "import radiomics; print(radiomics.__version__)"   # v3.0.1
conda run -n nyxus_mirp   python -c "from importlib import metadata; print(metadata.version('mirp'))"   # 2.6.0
```

## 2. The fixture

The IBSI digital phantom, read out of `tests/test_data.h` by
`tests/vetting/oracles/ibsi_phantom.py`. There is no separate copy to prepare and none to keep in
sync: the C++ tests read the same arrays through `test_2d_glrlm_common.h`.

```bash
python tests/vetting/oracles/ibsi_phantom.py
# z1: 5x6, 20 masked voxels, in-mask levels [1, 4, 6]
# z2: 5x6, 19 masked voxels, in-mask levels [1, 3, 4, 6]   ...
```

## 3. Run the generators

```bash
cd tests/vetting/oracles                     # the generators import ibsi_phantom from here
conda run -n nyxus_oracle python gen_glrlm_pyradiomics.py
conda run -n nyxus_mirp   python gen_glrlm_mirp.py
```

Each prints a paste-ready table body plus a provenance header line. Replace the body of
`glrlm_2d_pyradiomics_ref_vals` / `glrlm_2d_mirp_ref_vals` with it, keep the header line, rebuild
`runAllTests`, run `--gtest_filter=*GLRLM_FAMILY*`.

## 4. The configuration each tool is run at

Recipe `glrlm.ibsi_ng128` on the Nyxus side (`ibsi=true`, `GREYDEPTH=128`). The tool-side settings
that make it the same configuration:

| | PyRadiomics | MIRP |
|---|---|---|
| discretisation | `binWidth=1` (identity on this integer phantom) | `base_discretisation_method="none"` |
| distance | `distances=[1]` | (run length, no distance parameter) |
| plane | `force2D=True`, `force2Ddimension=0` | `by_slice=True` |
| aggregation | one value over the 4 in-slice directions | `glrlm_spatial_method="2d_average"` |
| weighting | `weightingNorm=None` | (none) |

Both report the direction-averaged value; averaging over the 4 phantom slices gives the IBSI
2D-averaged aggregation, which is what the assertions compare.

## 5. Mapping tool names to Nyxus features

One tool quantity per Nyxus feature - no aliases in this family, unlike GLCM. The mapping lives in
the generators (`PYRADIOMICS_TO_NYXUS`, `MIRP_TO_NYXUS`):

| Nyxus | PyRadiomics | MIRP |
|---|---|---|
| `GLRLM_SRE` / `GLRLM_LRE` | `ShortRunEmphasis` / `LongRunEmphasis` | `rlm_sre` / `rlm_lre` |
| `GLRLM_LGLRE` / `GLRLM_HGLRE` | `LowGrayLevelRunEmphasis` / `HighGrayLevelRunEmphasis` | `rlm_lgre` / `rlm_hgre` |
| `GLRLM_SRLGLE` / `GLRLM_SRHGLE` | `ShortRunLowGrayLevelEmphasis` / `ShortRunHighGrayLevelEmphasis` | `rlm_srlge` / `rlm_srhge` |
| `GLRLM_LRLGLE` / `GLRLM_LRHGLE` | `LongRunLowGrayLevelEmphasis` / `LongRunHighGrayLevelEmphasis` | `rlm_lrlge` / `rlm_lrhge` |
| `GLRLM_GLN` / `GLRLM_GLNN` | `GrayLevelNonUniformity` / `…Normalized` | `rlm_glnu` / `rlm_glnu_norm` |
| `GLRLM_RLN` / `GLRLM_RLNN` | `RunLengthNonUniformity` / `…Normalized` | `rlm_rlnu` / `rlm_rlnu_norm` |
| `GLRLM_RP` | `RunPercentage` | `rlm_r_perc` |
| `GLRLM_GLV` / `GLRLM_RV` | `GrayLevelVariance` / `RunVariance` | `rlm_gl_var` / `rlm_rl_var` |
| `GLRLM_RE` | `RunEntropy` | `rlm_rl_entr` |
| `<X>` and `<X>_AVE` | one value | one value |

## 6. Convention differences to expect

- **Run entropy** - `GLRLM_RE` lands 1.1e-3 off both tools because Nyxus sums logarithms through
  `fast_log10` with an `EPSILON` guard. Assert it at `rel=5e-3`; the other fifteen hold at
  `rel=1e-9`. It is the only logarithmic sum in the family.
- **Normalisation of GLN/RLN** - Nyxus reports the un-normalised counts alongside the normalised
  ones (`_NN` suffix), matching both tools; do not confuse `GLRLM_GLNN` with `GrayLevelNonUniformity`.
- **Grey-level gaps** - at identity binning both tools build the matrix over the image's own levels,
  so the phantom's absent levels (in-mask {1,3,4,6}) are not a problem; verified, all three agree to
  1e-15. This is not true at a fixed bin count.

## 7. `test_2d_glrlm_ibsi.h`

Its goldens are the published IBSI consensus (reference manual, "dig phantom", 2D-averaged), not a
tool run, so there is nothing to regenerate. To re-verify them, run either tool on the phantom at
the configuration in section 4 and compare; `glrlm_2d_ibsi_vetting_report.md` is the record of doing
exactly that.
