# Regenerating the 3D GLSZM goldens

Two procedures. The PyRadiomics goldens and the size-zone matrices come from an offline tool run; the
regression pins come from Nyxus itself. Each procedure feeds two tables — two oracle recipes and two
drift guards — and one dump case per side prints both of its tables.

---

## 1. The PyRadiomics goldens and the three matrices

Feeds `glszm_3d_pyradiomics_ref_vals`, `glszm_3d_pyradiomics_gapped_ref_vals`,
`glszm_3d_pyradiomics_matrix_ref_vals`, `glszm_3d_pyradiomics_smallmatrix_ref_vals`,
`glszm_3d_pyradiomics_gappedmatrix_ref_vals` and the twelve `*_ng` / `*_ns` / `*_nz` / `*_np`
constants in `tests/test_3d_glszm_pyradiomics.h`.

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

### The two literal volumes

`DOC_VOLUME` and `GAPPED_VOLUME` in the generator are the same literals as `glszm_3d_zcross_volume`
and `glszm_3d_gapped_volume` in `tests/test_3d_glszm_common.h`, written `(z, y, x)` for SimpleITK.
Both are driven at `binWidth=1` with a mask of every non-zero voxel, which leaves their integer
levels where they are — the same levels Nyxus reads straight off a volume at no binning. Keep each
pair in step; the generator asserts the resulting cells against the header, so they cannot drift
silently.

**`binWidth`, not `binCount`, for both.** A bin count would lay its edges over the volume's own
min..max and renumber the levels; for `GAPPED_VOLUME` that would close the gap between 1, 3 and 5,
which is the one property the fixture exists to carry.

**The recipe for `GAPPED_VOLUME` is `glszm3d.pyradiomics_ibsi_gapped`**: Nyxus runs it at
`IBSI=true`, which forces the family's binning to 0, and passes `GLSZM_GREYDEPTH=64` so that the
overwrite is exercised rather than assumed. Both sides report `Ng = 5` for three occupied levels.

---

## 2. The regression pins

Feeds `glszm_3d_regression_ref_vals` and `glszm_3d_regression_nobinning_ref_vals` in
`tests/test_3d_glszm_regression.h`. **No oracle** — these are Nyxus' own output, pinned as drift
guards, recipes `glszm3d.regression_ut_phantom` and `glszm3d.regression_ut_phantom_nobinning`.

```
runAllTests --gtest_filter=*3D_GLSZM_DUMP_REGRESSION*
```

prints both tables at `%.17g` in the shape the header wants, alongside what is currently pinned —
tagged `[3DGLSZM-REGR]` for the MATLAB-binning recipe and `[3DGLSZM-NOBIN]` for the default one.
Paste them in. Its sibling `*3D_GLSZM_DUMP_PYRADIOMICS*` does the same for the two oracle tables,
tagged `[3DGLSZM-PYRAD]` and `[3DGLSZM-GAPPED]`, which is how the residual against PyRadiomics is
read without a debugger.

Full precision is the point: a value truncated to five digits eats a third of a `rel=1e-3` band
before the test starts, and these assert at `rel=1e-9`.

The recipe is `GREYDEPTH=64`, `IBSI=false`, `GLSZM_GREYDEPTH=64` on `bench_ut57_3d` (label 57) —
`make_glszm3d_settings(64, 64)`. That is what `make_3d_coverage_settings()` runs every 3D family at
on this same phantom. **State `GLSZM_GREYDEPTH`; do not leave it at the zero a settings vector starts
at.** Zero is a third configuration — no binning rather than 64 MATLAB levels — and the values differ
by up to five orders of magnitude between the two.

The second recipe is the same phantom at `GLSZM_GREYDEPTH=0` — `make_glszm3d_settings(64, 0)` — which
is what a run passing no `--3glszm/greydepth` reaches the feature with. Its sixteen pins share one
gtest case, `TEST_3D_GLSZM_DEFAULT_GREYDEPTH_REGRESSION`, because one phantom read answers all
sixteen and the matrix at this setting is 3024 grey levels wide.
`TEST_3D_GLSZM_CONSTANT_ROI_REGRESSION` sits beside them and needs no phantom at all: it pins the
soft-NaN sentinel a constant-intensity ROI comes back as, which is a divergence rather than a value
(see `matrix/glszm3d.md`).

If a pin moves, the question is which side changed. The sixteen features are contractions of one
size-zone matrix, so run the PyRadiomics gate first: if the matrix assertion and the sixteen oracle
scalars still hold on the compat phantom, the arithmetic is intact and what moved is the ut phantom's
loading or binning.
