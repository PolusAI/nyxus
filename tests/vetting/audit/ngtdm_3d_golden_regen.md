# Regenerating the 3D NGTDM goldens

Two sources. The oracle tables come from PyRadiomics and are regenerated offline; the
drift guards are Nyxus' own output and are regenerated from the test binary.

| table | file | source |
|---|---|---|
| `ngtdm_3d_pyradiomics_ref_vals` | `tests/test_3d_ngtdm_pyradiomics.h` | PyRadiomics 3.0.1 |
| `ngtdm_3d_pyradiomics_matrix_ref_vals` | same | PyRadiomics `P_ngtdm` |
| `ngtdm_3d_pyradiomics_docmatrix_ref_vals` | same | PyRadiomics on the 4×4 docstring image |
| `ngtdm_3d_pyradiomics_ball_ref_vals` | same | PyRadiomics 3.0.1 on the ball phantom |
| `ngtdm_3d_pyradiomics_ball_matrix_ref_vals` | same | PyRadiomics `P_ngtdm` on the ball phantom |
| `NGTDM_3D_NONBOX_PYRADIOMICS` | `tests/python/test_3d_ngtdm_pyradiomics.py` | PyRadiomics 3.0.1 on the out-of-core ellipsoid |
| `ngtdm_3d_mirp_mixed_ref_vals` | `tests/test_3d_ngtdm_mirp.h` | MIRP 2.6.0 on label 59 of the ball phantom |
| `ngtdm_3d_mirp_mixed_matrix_ref_vals` | same | numpy NGTDM, cross-checked against the MIRP values |
| `NGTDM_3D_MIXED_MIRP` | `tests/python/test_3d_ngtdm_mirp.py` | MIRP 2.6.0 on the out-of-core ellipsoid plus a lone voxel |
| `ngtdm_3d_regression_ref_vals` | `tests/test_3d_ngtdm_regression.h` | Nyxus |

The ball phantom's two NIfTI files are themselves generated: `--write-ball` on
`gen_ngtdm3d_pyradiomics.py` writes them from `make_ball_phantom()` (labels 57, 58 and 59), and every
run checks the committed files against it.

## Environment

```
conda create -n nyxus_oracle -c conda-forge python=3.9 pyradiomics simpleitk numpy   # -> v3.0.1
```

PyRadiomics needs Python ≤ 3.9 on conda-forge; ask for the interpreter explicitly or the solver picks
a newer one and fails. See `tests/vetting/TOOLS.md`.

## PyRadiomics — all three oracle tables

```
conda run -n nyxus_oracle python tests/vetting/oracles/gen_ngtdm3d_pyradiomics.py
```

It prints all three tables paste-ready, then re-verifies every pin in the header it feeds and exits
non-zero on a mismatch, on a pin it cannot produce, or on a value it produces that the header pins
nothing for. It also runs the range/identity checks and the cross-table check (the five feature pins
recomputed from the matrix pins), so a table edited on its own does not survive.

**The extractor cannot load this fixture.** `compat_seg_ngtdm_3d.nii` is label 57 in all 48 voxels,
and `imageoperations.getMask()` rejects a mask whose `numpy.unique` has one entry — so
`RadiomicsFeatureExtractor.execute()` raises `No labels found in this mask`, and so does the
`pyradiomics <image> <mask> --param ...` CLI. The generator constructs `RadiomicsNGTDM` directly,
which reaches the same feature code without the loader check:

```python
f = ngtdm.RadiomicsNGTDM(img, sitk.Cast(msk, sitk.sitkUInt32), label=57, binWidth=1,
                         resampledPixelSpacing=None, force2D=False, distances=[1])
f._initCalculation()
f.P_ngtdm[0, :, 0]            # n_i
f.P_ngtdm[0, :, 1]            # s_i
f.P_ngtdm[0, :, 2]            # the grey levels
f.coefficients["p_i"][0]      # p_i
float(f.getBusynessFeatureValue()[0])
```

### Name mapping

These line up by name, unlike the GLCM family's — `3NGTDM_X` is PyRadiomics' `original_ngtdm_X` for
all five of `Busyness`, `Coarseness`, `Complexity`, `Contrast`, `Strength`.

### Convention differences to account for

- **Grey levels.** PyRadiomics' `binWidth=1` gives `floor(x/1) − floor(min/1) + 1`; on a fixture
  whose minimum is 0 that is `x + 1`. Nyxus at `NGTDM_GREYDEPTH=0` does not bin, but shifts every
  level by one when the minimum is zero. Same result here — but only because the minimum is zero and
  the values are integers. On any other fixture the two have to be matched deliberately.
- **Empty levels.** A level no ROI voxel carries is absent from `P_ngtdm` (PyRadiomics deletes it in
  `_calculateMatrix`) and absent from Nyxus' `I` (which is built from the set of values present).
  The 4×4 docstring image has no 4s, so its table has four rows and not five — the docstring's own
  table shows five. A level that only neighbourless voxels carry is the exception: Nyxus keeps its
  row with `n_i = 0` and leaves it out of `N_g,p` (see the MIRP section below).
- **The PyRadiomics NGTDM docstring's table and its worked arithmetic disagree** on `s_3`: the table
  says `2.63`, the text computes `3.03`, and a run agrees with the text (`91/30`). Pin the run.
- **`N_v,p` is the count of voxels with at least one neighbour**, which on every one of these
  fixtures is every ROI voxel. Nyxus computes it as the number of zones whose neighbourhood mean is
  `> 0`; those coincide here because no level is zero after the shift. On a fixture where they do
  not, this is the first thing to check.
- **A voxel with no neighbour** is a row with `s_i = 0` in PyRadiomics. Nyxus follows IBSI, as MIRP
  does: the voxel is in no row and counts towards none of `n_i`, `N_v,p`, `N_v,c` or `N_g,p`. So any
  ROI holding even one such voxel parts from PyRadiomics on all five features. Every PyRadiomics
  fixture here avoids it, and the generator fails if one does not; the mixed case is pinned to MIRP
  instead (below), and a ROI made only of such voxels is the empty-matrix case
  `test_3d_ngtdm_isolated_voxels_mechanics` covers.
- **Neighbours are ROI voxels.** Both tools take a voxel's neighbourhood from the mask, so on a ROI
  that does not fill its bounding box the background around it takes no part. The ball phantom is
  the fixture that would show otherwise.

### Independent reference

`reference_ngtdm()` in the same generator builds the NGTDM from the IBSI definition in exact rational
arithmetic with no `radiomics` import in its path, and the generator refuses to print anything if the
two disagree on levels or counts. Keep it: the oracle is being driven through a non-public entry
point, and this is what says the pins are the definition's values rather than one implementation's.

## MIRP — the mixed-ROI tables

```
conda run -n nyxus_mirp python tests/vetting/oracles/gen_ngtdm3d_mirp.py
```

Feeds `ngtdm_3d_mirp_mixed_ref_vals` and `ngtdm_3d_mirp_mixed_matrix_ref_vals` in
`tests/test_3d_ngtdm_mirp.h` (label 59 of the ball phantom) and `NGTDM_3D_MIXED_MIRP` in
`tests/python/test_3d_ngtdm_mirp.py` (the out-of-core ellipsoid plus a lone corner voxel at level 12).
It prints them paste-ready and re-verifies every pin. MIRP runs at `by_slice=False`, distance 1,
`fixed_bin_number` with `max − min + 1` bins, which maps the ROI's integer levels to the same
`x − min + 1` Nyxus' zero-min correction gives; the generator asserts that identity per ROI, runs
MIRP a second time with no discretisation on the lifted levels, and checks both against a numpy
NGTDM. The matrix table is the numpy one, in Nyxus' layout (the lone voxel's level as an empty row),
and the five MIRP values are recomputed from it. It also checks that MIRP reports NaN on label 58.
The mirp env has no NIfTI library, so the generator parses the uncompressed NIfTI header itself.

## Nyxus — the drift guards

```
runAllTests --gtest_filter=*3D_NGTDM_DUMP_REGRESSION*
```

Paste the `[3DNGTDM-REGEN]` lines over `ngtdm_3d_regression_ref_vals`. The dump uses
`make_ngtdm3d_regression_settings()`, the same helper the assertions use, so the two cannot drift
apart in their configuration.

Recipe `ngtdm3d.regression_ut_phantom`: `bench_ut57_3d`, `GREYDEPTH=64`, `NGTDM_GREYDEPTH=64`,
`NGTDM_RADIUS=1`, `IBSI=false`. **Not comparable to the PyRadiomics recipe** — at
`NGTDM_GREYDEPTH=64` the binning is MATLAB-style, which makes bin 1 the background level: a voxel
binned there is not a matrix row of its own but still counts towards its neighbours' neighbourhood
means. These pins claim no oracle.

`runAllTests --gtest_filter=*3D_NGTDM_DUMP_PYRADIOMICS*` prints the same for the oracle fixture,
alongside each pinned PyRadiomics value — that is the one to run when a residual needs reading
without a debugger.

## Coverage check

```
python tests/vetting/audit/scan_ngtdm3d_coverage.py --check   # acceptance check
python tests/vetting/report_features.py --write          # regenerate report_output.csv
```

The feature → test mapping is read out of the test sources, so it cannot drift from the tree, and
`report_features.py` joins it into `report_output.csv`. `--check` runs the acceptance check: every `vetted` row asserted by an oracle test, that test's
oracle equal to the row's, and `current_test` naming the file that defines the row's `test_name`.

## If a value moves

1. Run the generator. If PyRadiomics itself moved, the version is in its first line of output — the
   inverse-difference family taught this once already (`TOOLS.md`).
2. If PyRadiomics and `reference_ngtdm()` still agree and Nyxus does not, it is Nyxus. Read the
   matrix assertion first: it names the grey level, which localises the change to a level rather than
   to a feature.
3. Check `NGTDM_RADIUS` before anything else. At 0 the whole family is NaN, and that is a settings
   question, not a value question.
