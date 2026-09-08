# Regenerating the 3D GLDZM goldens

Two tables, two benchmarks, one generator. Read `gldzm_3d_mirp_vetting_report.md` before touching
anything here — in particular before assuming the two tables should agree with each other. They are
taken at different configurations on purpose, and only one of them is an oracle.

## Oracle goldens — `test_3d_gldzm_mirp.h`

Recipe `gldzm3d.mirp_compat_phantom`: `bench_compat_gldzm_3d`
(`tests/data/nifti/compat_int/compat_int_gldzm_3d.nii` + `compat_seg/compat_seg_gldzm_3d.nii`,
label 57) at `IBSI=true`, which switches the family's binning off. Oracle: MIRP 2.6.0 with
`by_slice=False` and `base_discretisation_method="none"`.

```
python tests/vetting/oracles/gen_gldzm3d_mirp.py
```

The run prints the sixteen values in the exact shape `gldzm_3d_mirp_ref_vals` wants, MIRP column
name in a trailing comment; paste them over the table. It then re-verifies every pin already in the
header against that same run at `rel<=1e-12` and exits non-zero on a mismatch, on a pin it cannot
produce, and on a MIRP feature the header pins nothing for.

Needs mirp 2.6.0: `conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy scipy`. The
run's header line prints the mirp version actually installed, so the provenance is the run's own
rather than this document's.

**Name mapping** — MIRP suffixes every GLDZM column with the dimensionality and the discretisation
it was computed at: `dzm_sde_3d` with no discretisation, `dzm_sde_3d_fbn_n64` at
`fixed_bin_number` n=64. Match on the stem and check the suffix separately, or a changed bin count
silently reads a column computed at another configuration. The full map is `MIRP` in the generator.

**Two features have no MIRP column at all.** `3GLDZM_GLM` and `3GLDZM_ZDM` — MIRP's GLDZM emits no
`dzm_gl_mean` / `dzm_zd_mean`, and IBSI defines neither. They are drift guards below and cannot be
regenerated from any oracle.

## Regenerating the compatibility phantom

The phantom is built by the generator, not by hand:

```
python tests/vetting/oracles/gen_gldzm3d_mirp.py --write-phantom
```

Every other run of the generator rebuilds the same two volumes in memory and fails if the checked-in
NIfTI pair no longer holds what `build_compat_phantom()` produces, so the fixture cannot drift away
from the rule that describes it. **Changing that rule changes all sixteen oracle goldens**, so the
two steps go together: rewrite the rule, `--write-phantom`, rerun the generator, paste the new
table.

## Regression drift guards — `test_3d_gldzm_regression.h`

Recipe `gldzm3d.regression_ut_phantom`: `bench_ut57_3d` (`tests/data/nifti/phantoms/ut_inten.nii` +
`ut_mask57.nii`, label 57) at `GREYDEPTH=64`, `IBSI=false` — the family's default mode. No oracle:
these are Nyxus' own values.

```
runAllTests --gtest_filter=*3D_GLDZM_DUMP_REGRESSION*
```

`test_3d_gldzm_dump_regression()` prints the whole table at 17 significant digits in the shape
`gldzm_3d_regression_ref_vals` wants; paste it over the table. It goes through the same
`extract_3d_gldzm` helper and the same settings the assertions use, so the two cannot drift apart.

**Do not regenerate this table to make a failure go away.** Every value in it moves if the zone map
stops being restricted to the ROI, if zone connectivity stops being 26, or if the distance stops
being the city-block transform to the ROI border — which is what the `rel=1e-9` band is for. A
movement here is a finding until it is explained.

**This table cannot be checked against MIRP**, and the reason is not the GLDZM. At `GREYDEPTH=64`
Nyxus places this ROI on levels 22-64 while MIRP's `fixed_bin_number` n=64 places it on 1-64, so a
comparison at that recipe measures the discretisation. `gldzm3d.mirp_fbn64` in the vetting report
records the residual; `matrix/gldzm3d.md` records the cell INVALID.

## The two measurements that pin nothing

The generator makes two further MIRP runs on `bench_ut57_3d`, and the artifact they feed is the
vetting report rather than a header:

- `gldzm3d.mirp_fbn64` — MIRP discretises the ROI itself. The table measures the discretisation gap.
- `gldzm3d.mirp_samelevels` — MIRP is handed the grey levels Nyxus bins to, with
  `base_discretisation_method="none"`, so both compute the GLDZM over identical levels. Worst
  relative difference over the sixteen comparable features: **1.0e-14**, on a 274,432-voxel ROI.

**The generator verifies the report, because that is the artifact those two runs feed.** It parses
both comparison tables in `gldzm_3d_mirp_vetting_report.md`, checks every MIRP value quoted there
against a fresh run at `rel<=1e-5` (the six significant figures the report quotes), checks the
reverse direction for a feature MIRP produces that the report omits, checks the measured grey-level
span against the span the report states, and exits non-zero on any of those. Last run: **48
verified, 0 failed, 0 unproducible, 0 unquoted.**

## Coverage artifact

```
python tests/vetting/audit/scan_gldzm3d_coverage.py           # rewrite
python tests/vetting/audit/scan_gldzm3d_coverage.py --check   # drift + acceptance check
```

`ORACLE_SUFFIX` is `{"mirp": "mirp"}`, so `--check` holds a `vetted` row to naming an oracle test
whose function suffix is that oracle, holds every row's `test_name` to a registered gtest case in
the file its `current_test` names, and holds every test function defined in the family to being
registered — the last of which is the check `3GLDZM_ZDM` needed and did not have.

Rerun it after any change to the family's registry rows or test files, and commit the rewritten
`gldzm_3d_coverage.csv` with them: CI runs `--check`, and a stale artifact fails it.

## Two cheap checks on a regenerated table, before believing it

- **`ZP` times the ROI voxel count is the zone count.** On the compatibility phantom that number is
  derivable without any tool: 189 bricks of one grey level each, less the seven the corner-touching
  implant absorbs, is 182 zones over 1512 voxels. If a regenerated `3GLDZM_ZP` is not 182/1512, the
  zone map changed and no amount of repinning is the right response.
- **`GLNU/GLNUN` and `ZDNU/ZDNUN` are each exactly the zone count**, by construction
  (`f_GLNUN = f_GLNU / Ns`). Two independent readings of the same number, and neither needs the
  program instrumented.
