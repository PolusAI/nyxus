# Regenerating the 3D NGLDM goldens

One benchmark, and — uniquely in this series — no oracle goldens at all. Read
`ngldm_3d_mirp_vetting_report.md` before touching anything here: the pinned values are known to be
wrong, and are pinned deliberately as a change detector.

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

**Expect to run this once the implementation is fixed.** Two defects — the NGLD matrix built over the
ROI bounding box instead of the ROI, and a 24-shift neighbourhood where 3D Chebyshev-1 has 26 — mean
every number in the table changes when they are corrected. That is the intended trigger, not a
surprise.

## The MIRP comparison — `oracles/gen_ngldm3d_mirp.py`

Recipe `ngldm3d.mirp_fbn64`. Nothing in the tree asserts against it today; it exists so the
divergence stays reproducible and so the promotion can be re-run against a fixed implementation.

```
python tests/vetting/oracles/gen_ngldm3d_mirp.py
```

Needs mirp 2.6.0: `conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy`. The generator
prints a paste-ready golden table and verifies pins if `test_3d_ngldm_mirp.h` exists — it does not
yet, and should only be created when Nyxus actually agrees.

**Name mapping** — MIRP suffixes every NGLDM column with the neighbourhood and discretisation
(`_d1_a0.0_3d_fbn_n64`). Match on the stem and check the suffix separately, or a changed bin count
silently reads a column computed at another config:

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

`3NGLDM_GLM` and `3NGLDM_DCM` have **no** MIRP counterpart and never will — MIRP's NGLDM emits no
`gl_mean` / `dc_mean` column. They cannot be vetted against this tool even after the fix.

It reads the `.nii` with no NIfTI library (the mirp env has neither SimpleITK nor nibabel) by parsing
the uncompressed NIfTI-1 header with numpy — the same approach as `gen_morphology3d_mirp.py`. Do not
reintroduce a two-env `.npy` hand-off.

## Sanity checks on any regenerated set

- `3NGLDM_DCP` ≤ 1 by construction. It is currently exactly 1, which is why it is *not* useful as an
  oracle agreement — see the report.
- The bounding-box-to-ROI ratio is the number to watch. On this phantom it is 551040/274432 = 2.008,
  and `3NGLDM_DCNU` currently sits 2.09× above MIRP. If a future change fixes the ROI masking, that
  ratio should collapse toward 1 across the family — a quick way to confirm the fix landed.
- After the fix, re-run **both** commands above and compare feature by feature before promoting
  anything. Promotion means adding `test_3d_ngldm_mirp.h`, setting `ORACLE_SUFFIX = {"mirp": "mirp"}`
  in `audit/scan_ngldm3d_coverage.py`, and moving the 17 comparable rows to `status=vetted`.

## Coverage artifact

```
python tests/vetting/audit/scan_ngldm3d_coverage.py           # rewrite
python tests/vetting/audit/scan_ngldm3d_coverage.py --check   # drift + acceptance check
```

Its `ORACLE_SUFFIX` is deliberately empty, so `--check` currently enforces only that no row claims
`vetted` without an oracle test. That is exactly the condition this PR restores.
