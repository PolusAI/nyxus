# 3D NGLDM vs MIRP — vetting report

The last family in the per-family series, and the only one that closes its gap row by **demoting**
it. `3NGLDM_DCP` read `status=vetted, oracle=mirp` from an offline run; running that oracle shows the
claim should never have been made, and shows why the other 18 rows are right to be `regression`.

## Tool and configuration

| | |
|---|---|
| Tool | mirp 2.6.0 (numpy 2.4.6, pandas 3.0.3) |
| Recipe | `ngldm3d.mirp_fbn64` |
| Fixture | the segmented phantom, `phantoms/ut_inten.nii` + `phantoms/ut_mask57.nii`, label 57 |
| MIRP config | `by_slice=False`, `fixed_bin_number` n=64, distance 1, difference level (alpha) 0 |
| Nyxus config | `GREYDEPTH=64`, `IBSI=false` — what `test_3d_ngldm_regression.h` sets |
| Generator | `tests/vetting/oracles/gen_ngldm3d_mirp.py` |
| Tolerance | n/a — nothing is asserted against MIRP |

**The configs are matched, and that is the point.** `D3_NGLDM_feature` bins on `STNGS_NGREYS`, the
generic grey depth, which the test sets to 64; MIRP is told `fixed_bin_number n=64`. Both use
distance 1 and alpha 0. There is no binning mismatch to explain the results below.

## Result

| feature | Nyxus | MIRP | Nyxus/MIRP |
|---|---|---|---|
| `3NGLDM_DCP` | 1 | 1 | 1x |
| `3NGLDM_LDE` | 0.1016 | 0.25594 | 0.397x |
| `3NGLDM_HDE` | 261.018 | 28.0738 | **9.3x** |
| `3NGLDM_LGLCE` | 0.000359684 | 0.0321849 | **0.0112x** |
| `3NGLDM_HGLCE` | 740.436 | 1323.96 | 0.559x |
| `3NGLDM_LDLGLE` | 5.83375e-05 | 0.000684901 | 0.0852x |
| `3NGLDM_LDHGLE` | 73.9199 | 474.82 | 0.156x |
| `3NGLDM_HDLGLE` | 0.0252015 | 8.71408 | **0.00289x** |
| `3NGLDM_HDHGLE` | 20099.8 | 14942.8 | 1.35x |
| `3NGLDM_GLNU` | 115443 | 4350.27 | **26.5x** |
| `3NGLDM_GLNUN` | 0.225757 | 0.0158519 | **14.2x** |
| `3NGLDM_DCNU` | 85056.8 | 40745 | 2.09x |
| `3NGLDM_DCNUN` | 0.166335 | 0.14847 | 1.12x |
| `3NGLDM_GLV` | 190.082 | 350.171 | 0.543x |
| `3NGLDM_DCV` | 86.1706 | 11.9476 | **7.21x** |
| `3NGLDM_DCENT` | 5.22774 | 8.67597 | 0.603x |
| `3NGLDM_DCENE` | 0.143484 | 0.00287482 | **49.9x** |

`3NGLDM_GLM` and `3NGLDM_DCM` are absent: MIRP's NGLDM emits no `gl_mean` / `dc_mean` column, so no
oracle exists for them at all.

## Two causes, both in the source

### 1. Off-ROI voxels are included in the NGLD matrix

`calc_ngld_matrix` iterates the whole ROI **bounding box** and explicitly declines to skip
background:

```cpp
// Do not skip off-ROI pixels
//	if (cpi == 0)
//		continue;
```

Measured on this phantom: the bounding box is 82×96×70 = **551,040** voxels against an ROI of
**274,432** — a factor of 2.008. IBSI NGLDM is defined over ROI voxels only, and MIRP computes it
that way.

This predicts the error pattern, which is the reason to believe it is the dominant cause rather than
a coincidence. Background voxels all bin to the same grey level and each sees ~24 identical
neighbours, so they pile into a single grey row at maximum dependence. That inflates a
sum-of-squares over grey rows far more than linearly (`GLNU` 26.5×), raises energy sharply
(`DCENE` 49.9×), lowers entropy (`DCENT` 0.60×), and pushes the dependence distribution to its top
end (`HDE` 9.3×, `LDE` 0.40×). `DCNU`, which scales closer to linearly with the voxel count, comes
out at 2.09× — almost exactly the bounding-box-to-ROI ratio.

### 2. The neighbourhood has 24 voxels, not 26

The `shifts` table lists 8 in-plane, 8 at `dz=+1` and 8 at `dz=-1`. Enumerating it against a full 3D
Chebyshev-distance-1 neighbourhood shows the two pure-axial neighbours missing:
`(dx,dy,dz) = (0,0,+1)` and `(0,0,-1)` — the voxels directly above and below the centre. So `nsh` is
24, and `int maxNr = nsh + 1;` carries a comment reading "max dependence 8 (due to 8 neighbors)",
which is the 2D count.

This is smaller in effect than cause 1 but not negligible: it changes every dependence count and
caps the matrix two columns short.

## Why `3NGLDM_DCP` is demoted rather than promoted

It is the family's only agreement, and it agrees at **1.0 on both sides**. Dependence-count
percentage is the fraction of voxels having at least one dependency; on any input where every voxel
has a same-binned neighbour it is exactly 1, which is true of this phantom under both the correct and
the incorrect neighbourhood, and with or without background voxels. It is an assertion that cannot
fail for the reasons we would want it to fail.

Promoting it would have produced the shape this whole series exists to remove: a family whose single
`vetted` row is a degenerate constant, standing next to sixteen features that disagree with the same
tool by up to 50×, and reading in the registry as though 3D NGLDM had been checked. The row now reads
`status=regression` with `candidate_oracle=mirp` and `flag=implementation-defect`, which is what is
actually true.

That leaves the family with **zero vetted rows** — the only one in the series. That is the honest
count, not a regression in coverage: it was zero before too, and the difference is that the registry
now says so.

## What this PR does and does not change

**Does not** change any feature value. Both causes are behaviour changes across 19 public features
and belong on their own branch; the fix plan is recorded outside this PR.

**Does** pin all 19 goldens at full precision and tighten the assertion from a 10% band to
`rel=1e-9`. The old pins were two- and three-significant-figure numbers (`0.1`, `261`, `740`,
`0.00036`) with a ±10% band — which cannot detect the very fix these values are waiting for. A drift
guard's job is to notice change, and these will change; when the implementation is corrected,
`test_3d_ngldm_dump_regression()` regenerates the table and this report's comparison should be re-run
to promote the family properly.

## Include hygiene and file-level observations

The family is two headers: `test_3d_ngldm_regression.h` and `test_3d_ngldm_coverage.h`. There is no
`_common.h` and no oracle file.

- **`test_3d_ngldm_regression.h` carried a dead `#if 0` block** — a superseded copy of the assert
  body. Removed. It also relied transitively on `<iomanip>`, `<iostream>`, `<string>`, `<tuple>`,
  `<vector>` and `helpers/fsystem.h` (for `fs::exists`); all now direct. Its golden lookup used
  `operator[]`, which default-inserts a missing key as 0 and then compares against a fabricated
  reference; now guarded with `find()`.
- Its header comment carried the 2026-07 MIRP comparison inline. That measurement is now in this
  report, at full precision and with causes attached, and the header keeps a short current-state
  statement plus a pointer.
- **`test_3d_ngldm_coverage.h`** keeps its single include of `test_3d_coverage_common.h`, as all
  eight 3D `_coverage.h` files do (SPEC §6.3.1).
- The 19 tests in `test_3d_ngldm_regression.h` **do** assert. An earlier note in this series claimed
  the file's body was `#if 0` and its tests therefore vacuous; that was wrong, and was verified by
  negative control — perturbing `3NGLDM_DCENE` from 0.14 to 0.99 makes
  `TEST_3D_NGLDM_DCENE_REGRESSION` fail at the assertion.

`tests/vetting/TOOLS.md` gains nothing here. The one trick worth recording — reading an uncompressed
NIfTI-1 phantom with numpy so a MIRP generator stays single-env — is added by the 3D morphology PR,
and `gen_ngldm3d_mirp.py` reuses it rather than duplicating the entry.

As with the other 3D families, `tests/vetting/matrix/ngldm.md` (SPEC §5.1) and
`tests/vetting/benchmarks.md` (SPEC §6.3) do not exist, and exist for no family but GLCM. A repo-wide
gap, not closed here.

## Reproduction

```
# MIRP side (conda env with mirp 2.6.0)
python tests/vetting/oracles/gen_ngldm3d_mirp.py

# Nyxus side
runAllTests --gtest_filter=*3D_NGLDM_DUMP_REGRESSION*

# coverage artifact
python tests/vetting/audit/scan_ngldm3d_coverage.py [--check]
```
