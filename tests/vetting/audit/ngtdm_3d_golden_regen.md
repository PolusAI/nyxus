# Regenerating the 3D NGTDM goldens

Three tables, two sources. The oracle tables come from PyRadiomics and are regenerated offline; the
drift guards are Nyxus' own output and are regenerated from the test binary.

| table | file | source |
|---|---|---|
| `ngtdm_3d_pyradiomics_ref_vals` | `tests/test_3d_ngtdm_pyradiomics.h` | PyRadiomics 3.0.1 |
| `ngtdm_3d_pyradiomics_matrix_ref_vals` | same | PyRadiomics `P_ngtdm` |
| `ngtdm_3d_pyradiomics_docmatrix_ref_vals` | same | PyRadiomics on the 4×4 docstring image |
| `ngtdm_3d_regression_ref_vals` | `tests/test_3d_ngtdm_regression.h` | Nyxus |

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
- **Empty levels are dropped by both.** A level with `n_i = 0` is absent from `P_ngtdm` (PyRadiomics
  deletes it in `_calculateMatrix`) and absent from Nyxus' `I` (which is built from the set of values
  present). The 4×4 docstring image has no 4s, so its table has four rows and not five — the
  docstring's own table shows five.
- **The PyRadiomics NGTDM docstring's table and its worked arithmetic disagree** on `s_3`: the table
  says `2.63`, the text computes `3.03`, and a run agrees with the text (`91/30`). Pin the run.
- **`N_v,p` is the count of voxels with at least one neighbour**, which on both of these fixtures is
  every voxel. Nyxus computes it as the number of zones whose neighbourhood mean is `> 0`; those
  coincide here because no level is zero after the shift. On a fixture where they do not, this is
  the first thing to check.

### Independent reference

`reference_ngtdm()` in the same generator builds the NGTDM from the IBSI definition in exact rational
arithmetic with no `radiomics` import in its path, and the generator refuses to print anything if the
two disagree on levels or counts. Keep it: the oracle is being driven through a non-public entry
point, and this is what says the pins are the definition's values rather than one implementation's.

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

## Coverage artifact

```
python tests/vetting/audit/scan_ngtdm3d_coverage.py            # rewrite ngtdm_3d_coverage.csv
python tests/vetting/audit/scan_ngtdm3d_coverage.py --check    # report drift instead
```

The feature → test mapping is read out of the test sources, so the CSV cannot drift from the tree.
`--check` also runs the acceptance check: every `vetted` row asserted by an oracle test, that test's
oracle equal to the row's, and `current_test` naming the file that defines the row's `test_name`.

## If a value moves

1. Run the generator. If PyRadiomics itself moved, the version is in its first line of output — the
   inverse-difference family taught this once already (`TOOLS.md`).
2. If PyRadiomics and `reference_ngtdm()` still agree and Nyxus does not, it is Nyxus. Read the
   matrix assertion first: it names the grey level, which localises the change to a level rather than
   to a feature.
3. Check `NGTDM_RADIUS` before anything else. At 0 the whole family is NaN, and that is a settings
   question, not a value question.
