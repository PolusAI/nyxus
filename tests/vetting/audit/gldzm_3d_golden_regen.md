# Regenerating the 3D GLDZM goldens

One benchmark, and — as with 3D NGLDM — no oracle goldens at all. Read
`gldzm_3d_mirp_vetting_report.md` before touching anything here: the pinned values are known to
disagree with the IBSI definition, and are pinned deliberately as a change detector.

## Regression drift guards — `test_3d_gldzm_regression.h`

Recipe `gldzm3d.regression_ut_phantom`: the segmented phantom
(`tests/data/nifti/phantoms/ut_inten.nii` + `ut_mask57.nii`, label 57) at `GREYDEPTH=64`,
`IBSI=false`, benchmark `bench_ut57_3d`. No oracle — Nyxus' own values, at full `%.17g` precision
and `rel=1e-9`.

**This family has no `*_dump_regression` test, and it is the only 3D family without one.** Every
other 3D family (`glcm`, `gldm`, `glrlm`, `glszm`, `morphology`, `ngldm`, `ngtdm`) carries a dump
test that prints its whole table at 17 significant digits in the shape the `ref_vals_map` wants,
under the same settings the assertions use, so the table and the assertions cannot drift apart.
Here the table has to be transcribed by hand, which is exactly the condition that let
`3GLDZM_ZDM` sit at `222` — a factor of 14.5 off — with nothing able to fail on it.

Adding `test_3d_gldzm_dump_regression()` is tracked in `PR/todo.md`; until it exists, regenerating
this table means reading the values out of a run by hand and re-checking each one.

**Expect to regenerate the whole table once the implementation is fixed.** The three defects in the
vetting report change every number in it. That is the intended trigger, not a surprise.

## The MIRP comparison — `oracles/gen_gldzm3d_mirp.py`

Recipe `gldzm3d.mirp_fbn64`. Nothing in the tree asserts against it today; it exists so the
divergence stays reproducible and so a promotion can be re-run against a fixed implementation.

```
python tests/vetting/oracles/gen_gldzm3d_mirp.py
```

Needs mirp 2.6.0: `conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy scipy`.

The generator does three things, and the second is what makes the conclusion safe:

1. Runs MIRP, which implements IBSI GLDZM. PyRadiomics has no GLDZM at all, so MIRP is the only
   mainstream oracle for this family.
2. Recomputes all 16 features from scratch — zones as the 26-connected components of each
   discretised grey level, distance to the ROI border as a city-block distance transform. A
   different route to the same definition, not a re-derivation of Nyxus' steps (SPEC 5.2). It
   reproduces MIRP to `rel=3.2e-16`, which is what licenses the claim that the definition is
   reachable and that Nyxus is not computing it.
3. Recomputes them once more with Nyxus' straight-ray distance substituted for the distance
   transform, holding the zone map fixed — which moves the features by well under a percent, and so
   locates the defect in the zone map rather than the distance metric.

`3GLDZM_GLM` and `3GLDZM_ZDM` have **no** MIRP or IBSI counterpart and never will — MIRP's GLDZM
emits no `dzm_gl_mean` / `dzm_zd_mean` column. They cannot be vetted against this tool even after
the fix.

## Sanity checks on any regenerated set

- `3GLDZM_ZP` ≤ 1 by construction.
- `3GLDZM_ZDV` is the variance of the same zone-distance distribution whose mean is `3GLDZM_ZDM`, so
  the two must stay consistent: `ZDM` far outside `ZDV`'s implied spread is the signature of the bad
  pin that this family already carried once.
- The divergence ratios in `gldzm_3d_mirp_vetting_report.md` are the numbers to watch. If a future
  change fixes the zone map, they should collapse toward 1 across the family — a quick way to
  confirm the fix landed.
- Promotion means adding `test_3d_gldzm_mirp.h`, setting `ORACLE_SUFFIX = {"mirp": "mirp"}` in
  `audit/scan_gldzm3d_coverage.py`, and moving the 16 comparable rows to `status=vetted`. Do not
  create that header until Nyxus actually agrees.

## Coverage artifact

```
python tests/vetting/audit/scan_gldzm3d_coverage.py           # rewrite
python tests/vetting/audit/scan_gldzm3d_coverage.py --check   # drift + acceptance check
```

Its `ORACLE_SUFFIX` is deliberately empty, so `--check` currently enforces that no row claims
`vetted` without an oracle test, that every row's `test_name` resolves to a registered gtest case in
the file its `current_test` names, and that every test function defined here is actually registered
— the last of which is the check `3GLDZM_ZDM` needed and did not have.
