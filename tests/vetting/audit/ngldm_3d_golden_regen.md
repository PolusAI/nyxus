# Regenerating the 3D NGLDM goldens

Two golden tables on one benchmark: the MIRP oracle pins in `test_3d_ngldm_mirp.h` and the drift
guards in `test_3d_ngldm_regression.h`. Read `ngldm_3d_mirp_vetting_report.md` before touching
either — the oracle's scope is the NGLDM and not the discretisation, and that is what decides which
of the two a given value belongs in.

## Regression drift guards — `test_3d_ngldm_regression.h`

Recipe `ngldm3d.regression_ut_phantom`: the segmented phantom
(`tests/data/nifti/phantoms/ut_inten.nii` + `ut_mask57.nii`, label 57) at `GREYDEPTH=64`,
`IBSI=false`. No oracle — Nyxus' own values.

```
runAllTests --gtest_filter=*3D_NGLDM_DUMP_REGRESSION*
```

`test_3d_ngldm_dump_regression()` prints the whole table at 17 significant digits in the shape
`ngldm_3d_regression_ref_vals` wants; paste it over the table. It uses the same settings the
assertions use, so the two cannot drift apart.

Every value in the table moves if the NGLD matrix stops being built over the ROI, if the
neighbourhood stops holding all 26 Chebyshev-1 shifts, or if the dependence count stops counting the
centre voxel. That is what the `rel=1e-9` band is for.

## The retired coverage sweep

`test_3d_ngldm_coverage.h` instantiated the two generic `TEST_P` suites of
`test_3d_coverage_common.h` over the family's 19 features. It is gone, and this family is the
cleanest case of the retirement in the series: **its table was a bit-for-bit duplicate**.

- `ngldm_3d_regression_coverage_ref_vals` and `ngldm_3d_regression_ref_vals` held the same 19 keys
  with the same 19 doubles — verified by parsing both tables and comparing, not by eye.
- Both were taken at the same recipe. The sweep runs `GREYDEPTH=64`, `IBSI=false`; so does
  `test_3d_ngldm_regression.h`. Unlike 3D GLCM's grey64 table, there was no second configuration to
  preserve.
- The `NGLDM_WITH_3P_EMBEDDED_GT` half instantiated **zero** cases: no NGLDM table appears in
  `externally_vetted_3d_feature_names()`, because the family has no oracle-backed feature at all.
- All 19 features already had an individually named `*_regression` test and a `TEST()` registration,
  so nothing had to be ported.

What the sweep additionally checked, and where it lives now: the name-resolves-and-code-matches step
is done by every named test (`find_3D_FeatureByString` plus an assert on the returned code), and the
one-provider-per-`Feature3D`-code step by `FeatureManager::check_11_correspondence()`, in production
since the 3D GLCM sweep was retired.

The completeness guard in `test_3d_coverage_common.h` reads the family's pins straight off
`ngldm_3d_regression_ref_vals`, so the migration cost exactly one `add_keys()` line — which is what
that guard was rebuilt to make true. Deleting a pin from the table now fails
`TEST_3D_FEATURE_COVERAGE_COUNTS` by feature name.

## The MIRP comparison — `oracles/gen_ngldm3d_mirp.py`

Two recipes, one generator run:

- `ngldm3d.mirp_fbn64` — MIRP discretises the ROI itself at `fixed_bin_number` n=64, landing on
  levels 1-64 against Nyxus' 21-64. The table this produces measures that gap.
- `ngldm3d.mirp_samelevels` — MIRP is handed the levels Nyxus bins to, with
  `base_discretisation_method="none"`, so both compute the NGLDM over identical levels. Worst
  relative difference over the seventeen comparable features: **8.7e-16**.

`ngldm3d.mirp_samelevels` is the config-matched one and carries the family's 16 `vetted` rows, pinned
in `test_3d_ngldm_mirp.h` at `rel=1e-3`. `ngldm3d.mirp_fbn64` asserts nothing: it is the measurement
of the discretisation gap. The oracle's scope — the NGLDM given a set of levels, not the levels — is
in the vetting report under "What this agreement covers".

**It verifies every artifact it feeds, and every column of them.** For each of the two comparison
tables in `ngldm_3d_mirp_vetting_report.md` it checks the MIRP column against a fresh run, the Nyxus
column against the goldens pinned in the C++ headers, and the derived column recomputed from the two
— the ratio as a value, the samelevels residual as an upper bound. It then checks the 16 goldens in
`test_3d_ngldm_mirp.h` against the same run, fails on any feature MIRP can vet that the header does
not pin (bar `3NGLDM_DCP`, the one deliberate omission), checks the reverse direction for a feature
the report omits, and checks the measured grey-level span against the span the report states. Last
run: **50 verified, 0 failed, 0 unproducible, 0 unquoted**.

Verifying only the MIRP column would leave the other half of every published row unchecked — a
perturbed Nyxus value or a perturbed ratio would both survive. Negative control: changing
`3NGLDM_GLNU`'s Nyxus column to its pre-fix value and `3NGLDM_HDE`'s ratio to 9.99999 makes the run
report `FAIL ... [nyxus]` and `FAIL ... [derived]` respectively.

**`Ns` is readable off the pinned values, with nothing instrumented.** `GLNU/GLNUN` and `DCNU/DCNUN`
are both exactly `Ns` by construction (`f_GLNU /= Ns`, `f_GLNUN /= (Ns*Ns)`), and both give
**Ns = 274,432** — the ROI's voxel count, so every ROI voxel is a matrix entry and nothing outside the
ROI is. It is the cheapest check that the matrix is built over the right set of centres.

```
python tests/vetting/oracles/gen_ngldm3d_mirp.py
```

Needs mirp 2.6.0: `conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy`; the run's
header line prints the mirp version actually installed, so the provenance is the run's own rather
than this document's.

**Name mapping** — MIRP suffixes every NGLDM column with the neighbourhood and discretisation
(`_d1_a0.0_3d_fbn_n64` when it discretises, `_d1_a0.0_3d` when it does not). Match on the stem and
check the suffix separately, or a changed bin count silently reads a column computed at another
config:

| Nyxus | MIRP stem | | Nyxus | MIRP stem |
|---|---|---|---|---|
| `3NGLDM_LDE` | `ngl_lde` | | `3NGLDM_GLNU` | `ngl_glnu` |
| `3NGLDM_HDE` | `ngl_hde` | | `3NGLDM_GLNUN` | `ngl_glnu_norm` |
| `3NGLDM_LGLCE` | `ngl_lgce` | | `3NGLDM_DCNU` | `ngl_dcnu` |
| `3NGLDM_HGLCE` | `ngl_hgce` | | `3NGLDM_DCNUN` | `ngl_dcnu_norm` |
| `3NGLDM_LDLGLE` | `ngl_ldlge` | | `3NGLDM_DCP` | `ngl_dc_perc` |
| `3NGLDM_LDHGLE` | `ngl_ldhge` | | `3NGLDM_GLV` | `ngl_gl_var` |
| `3NGLDM_HDLGLE` | `ngl_hdlge` | | `3NGLDM_DCV` | `ngl_dc_var` |
| `3NGLDM_HDHGLE` | `ngl_hdhge` | | `3NGLDM_DCENT` | `ngl_dc_entr` |
| | | | `3NGLDM_DCENE` | `ngl_dc_energy` |

`3NGLDM_GLM` and `3NGLDM_DCM` have **no** MIRP counterpart — MIRP's NGLDM emits no `gl_mean` /
`dc_mean` column — so they cannot be vetted against this tool at any recipe.

It reads the `.nii` with no NIfTI library (the mirp env has neither SimpleITK nor nibabel) by parsing
the uncompressed NIfTI-1 header with numpy — the same approach as `gen_morphology3d_mirp.py`. Do not
reintroduce a two-env `.npy` hand-off.

## Sanity checks on any regenerated set

- `GLNU/GLNUN` and `DCNU/DCNUN` are both `Ns`, and `Ns` is the ROI's voxel count, 274,432 on this
  phantom. A set where they read the bounding box's 551,040 — or its 511,360-voxel interior — has the
  matrix built over the wrong set of centres.
- `3NGLDM_DCP` ≤ 1 by construction. It is exactly 1 here, which is why it is *not* useful as an
  oracle agreement — see the report.
- `ngldm3d.mirp_samelevels` is the check with teeth: re-run the generator and confirm the seventeen
  are still at machine precision before pinning anything.
- If the discretisation gap is ever closed, `ngldm3d.mirp_fbn64` becomes assertable and its rows go
  beside the samelevels ones rather than replacing them: SPEC 1 counts vetting per assertion, and the
  two recipes establish different things.

## Coverage artifact

```
python tests/vetting/audit/scan_ngldm3d_coverage.py           # rewrite
python tests/vetting/audit/scan_ngldm3d_coverage.py --check   # drift + acceptance check
```

`ORACLE_SUFFIX` maps `mirp` to the `_mirp` suffix, so `--check` credits `test_3d_ngldm_mirp.h`'s 16
assertions as oracle coverage and still rejects any row claiming `vetted` without one.
