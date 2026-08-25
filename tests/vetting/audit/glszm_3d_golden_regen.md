# Regenerating the 3D GLSZM goldens

Two tables, two procedures. The PyRadiomics goldens and the size-zone matrices come from an offline
tool run; the regression pins come from Nyxus itself.

---

## 1. The PyRadiomics goldens and the two matrices

Feeds `glszm_3d_pyradiomics_ref_vals`, `glszm_3d_pyradiomics_matrix_ref_vals`,
`glszm_3d_pyradiomics_smallmatrix_ref_vals` and the six `*_ng` / `*_ns` / `*_nz` / `*_np` constants in
`tests/test_3d_glszm_pyradiomics.h`.

### Environment

pyradiomics 3.0.1 needs Python ≤ 3.9, so it does not install into the build env. Create a separate
one (`TOOLS.md` records this recipe for the other PyRadiomics families too):

```
conda create -n nyxus_oracle -c conda-forge python=3.8 pyradiomics simpleitk numpy
```

### Run

From the repository root:

```
python tests/vetting/oracles/gen_glszm3d_pyradiomics.py
```

It prints the paste-ready tables **and** re-verifies every pin already in the header, exiting
non-zero on a mismatch, on a pin it cannot produce, on a bound or identity violation, on a
cross-table disagreement, or on a feature PyRadiomics produces that nothing pins. A clean run ends
with `ALL CHECKS PASSED`.

### The recipe

| Nyxus | PyRadiomics |
|---|---|
| fixture `bench_compat_liver_3d`, label 1 | `compat_int/compat_int_mri.nii` + `compat_seg/compat_seg_liver.nii`, `label: 1` |
| `GLSZM_GREYDEPTH = -20` | `binCount: 20` |
| `IBSI = false` | — |
| `GREYDEPTH = 100` | — (inert: `D3_GLSZM_feature` never reads it) |
| — | `interpolator: sitkBSpline`, `resampledPixelSpacing:` empty, `weightingNorm:` empty, `imageType: Original` |

The equivalent YAML, if you would rather drive the `pyradiomics` CLI:

```yaml
setting:
  binCount: 20
  label: 1
  interpolator: 'sitkBSpline'
  resampledPixelSpacing:
  weightingNorm:
imageType:
  Original: {}
featureClass:
  glszm:
```

### Name mapping

`3GLSZM_<X>` ↔ `original_glszm_<Y>`, one to one for all sixteen; the table is `PYRAD` at the top of
the generator. No feature of this family is missing a PyRadiomics counterpart, and none of them
collide the way two 3D GLCM names do.

### Convention differences to account for

- **`3GLSZM_ZE` is not expected to match exactly.** `calc_sums_of_P()` uses
  `fast_log10(...)/LOG10_2` where PyRadiomics uses `numpy.log2`; the residual is 1.9e-4 and the
  assertion band is `rel=1e-3`. If you regenerate and it lands at 1e-16, the fast path was changed —
  that is a finding, not a fixed test.
- **A PyRadiomics column index is not a zone size.** `_calculateCoefficients()` deletes the columns
  for zone sizes nothing occupies and keeps the surviving sizes in `coefficients['jvector']`. Read
  the size from there. On this phantom the matrix goes from 634 columns to 46, so reading the index
  instead relabels almost every large zone — and the seven `j`-weighted features then miss by up to
  94% while the other nine still reproduce, which is exactly what makes the mistake easy to ship.
  `nonzero_cells()` does it correctly and the docstring says why.
- **`Ns` in the header is the largest zone size (634), not the number of distinct sizes (46).** That
  is the width Nyxus allocates.
- Nyxus' `to_grayscale_radiomix` and PyRadiomics' `binCount` produce the same partition of the ROI's
  min..max, so no grey-level correction is needed in either direction. Nothing like 3D NGTDM's
  zero-min shift applies here.

### The 4×4×3 volume

`DOC_VOLUME` in the generator is the same literal as `glszm_3d_pyradiomics_small_volume` in the
header, written `(z, y, x)` for SimpleITK. It is driven at `binWidth=1` with a mask of every non-zero
voxel, which maps its levels 1..4 to 1..4 — the same levels Nyxus reads straight off the volume at no
binning. Keep the two literals in step; the generator asserts the resulting cells against the header,
so they cannot drift silently.

---

## 2. The regression pins

Feeds `glszm_3d_regression_ref_vals` in `tests/test_3d_glszm_regression.h`. **No oracle** — these are
Nyxus' own output, pinned as drift guards, recipe `glszm3d.regression_ut_phantom`.

```
runAllTests --gtest_filter=*3D_GLSZM_DUMP_REGRESSION*
```

prints the whole table at `%.17g` in the shape the header wants, alongside what is currently pinned.
Paste it in. Its sibling `*3D_GLSZM_DUMP_PYRADIOMICS*` does the same for the oracle table, which is
how the residual against PyRadiomics is read without a debugger.

Full precision is the point: a value truncated to five digits eats a third of a `rel=1e-3` band
before the test starts, and these assert at `rel=1e-9`.

The recipe is `GREYDEPTH=64`, `IBSI=false`, `GLSZM_GREYDEPTH=64` on `bench_ut57_3d` (label 57) —
`make_glszm3d_settings(64, 64)`. That is what `make_3d_coverage_settings()` runs every 3D family at
on this same phantom. **State `GLSZM_GREYDEPTH`; do not leave it at the zero a settings vector starts
at.** Zero is a third configuration — no binning rather than 64 MATLAB levels — and the values differ
by up to five orders of magnitude between the two.

If a pin moves, the question is which side changed. The sixteen features are contractions of one
size-zone matrix, so run the PyRadiomics gate first: if the matrix assertion and the sixteen oracle
scalars still hold on the compat phantom, the arithmetic is intact and what moved is the ut phantom's
loading or binning.
