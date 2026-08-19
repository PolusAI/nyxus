# Regenerating the 3D morphology goldens

Two benchmarks on **one fixture** — the segmented phantom
(`tests/data/nifti/phantoms/ut_inten.nii` + `ut_mask57.nii`, label 57) through
`test_3d_morphology_common.h`'s settings (`IBSI=true`, `GREYDEPTH=128`, `PIXELSIZEUM=100`) — plus a
kernel check that reads no image at all. Because the Nyxus side is identical for both benchmarks, the
numbers *are* comparable to each other.

MIRP is the family's only oracle. The three volume rows were `oracle=matlab` until MIRP took them
over: MATLAB agreed, but it cannot be re-run here, and one oracle per assertion is the rule (SPEC §3).
The MATLAB values are kept as corroborating measurements in the headers and in this document; nothing
asserts against them.

## MIRP goldens — `test_3d_morphology_mirp.h`

Recipe `morphology3d.mirp_ibsi`. Covers the five PCA axis features and the three volume features.

```
python tests/vetting/oracles/gen_morphology3d_mirp.py
```

The generator prints a paste-ready table, re-verifies every pin (8 of them, last run all at rel=0),
checks the structural identities, and prints the cross-check quantities — including the MATLAB numbers
it replaced. It exits non-zero on any mismatch, any unproducible or unpinned golden, and any identity
violation. Needs mirp 2.6.0:
`conda create -n nyxus_mirp -c conda-forge python=3.11 mirp numpy`.

**Name mapping** — MIRP names the axes by role, Nyxus by size rank:

| Nyxus | MIRP |
|---|---|
| `3MAJOR_AXIS_LEN` | `morph_pca_maj_axis` |
| `3MINOR_AXIS_LEN` | `morph_pca_min_axis` |
| `3LEAST_AXIS_LEN` | `morph_pca_least_axis` |
| `3ELONGATION` | `morph_pca_elongation` |
| `3FLATNESS` | `morph_pca_flatness` |

The two orderings agree only because MAJOR is the largest eigenvalue — which is exactly the
correspondence a past defect broke, so the generator re-checks it every run.

**It reads the `.nii` with no NIfTI library.** The mirp env has neither SimpleITK nor nibabel. The
phantoms are uncompressed single-file NIfTI-1, so the header is parsed directly with `numpy` (`dim[8]`
at byte 40, `datatype` at 70, `pixdim[8]` at 76, `vox_offset` at 108, voxels x-fastest → reshape to
`(z,y,x)`). That keeps the generator single-env; do not reintroduce a two-step `.npy` hand-off.

**Sanity checks on any regenerated set** — all four are automated in the generator, and the family has
failed them before:

- `MAJOR ≥ MINOR ≥ LEAST > 0`. A `LEAST` above `MAJOR` means the eigenvalues were consumed in the
  wrong order.
- `ELONGATION` and `FLATNESS` in **[0,1]**, and each equal to its defining ratio (`MINOR/MAJOR`,
  `LEAST/MAJOR`). A `FLATNESS` above 1 is impossible, not surprising.
- `morph_vol_approx` must equal the ROI voxel count × voxel volume; the generator prints the voxel
  count so this is checkable by eye.

## Retired: the MATLAB goldens

The three volume goldens were produced by an offline MATLAB R2025b Image Processing Toolbox session:

```matlab
V = niftiread('ut_inten.nii');  M = niftiread('ut_mask57.nii') == 57;
s = regionprops3(M, 'Volume', 'ConvexVolume');
```

`Volume` → `3VOXEL_VOLUME` = 274432; `ConvexVolume` → `3VOLUME_CONVEXHULL` and, through the Nyxus
alias, `3MESH_VOLUME` = 497824.

**There was no in-repo generator for these and there could not be one here** — no MATLAB licence, and
Octave's `image` package has no `regionprops3` — which was the SPEC §6.4 gap tracked in
`not_covered.md` §C. MIRP computes the same quantities, is runnable from the tree, and agrees (exact
on the voxel volume, 0.17% on the hull), so the three rows moved to `oracle=mirp` and
`test_3d_morphology_matlab.h` was deleted. The MATLAB numbers survive as the corroborating
measurement quoted in `test_3d_morphology_mirp.h` and in the vetting report; the §6.4 gap is closed
rather than tracked.

`3AREA` was deliberately absent from the MATLAB table too: `regionprops3` `SurfaceArea` disagrees by
more than 10%, for the same reason MIRP does (see below).

## Covariance / eigenvalue kernel — `test_3d_morphology_mechanics.h`

Recipe `morphology3d.covmatrix_numpy`. No image and no feature: ten fixed voxel coordinates, their
sample covariance matrix and its eigenvalues — the arithmetic the PCA axis features are built on.

```
python tests/vetting/oracles/gen_morphology3d_covmatrix_numpy.py            # print
python tests/vetting/oracles/gen_morphology3d_covmatrix_numpy.py --check    # re-verify the pins
```

These were MATLAB `cov`/`eig` output quoted to five significant figures, from the same unrepeatable
session. numpy computes both quantities and agrees at every digit MATLAB printed;
`Nyxus::calc_covariance` normalises by n-1, which is what MATLAB `cov` and numpy `ddof=1` both
compute, so it is the same quantity rather than a near one. The pins now carry full precision and are
asserted at rel=1e-9.

The old assertions passed `frac_tolerance = 1.0` to `agrees_gt()`, which makes the tolerance the
ground truth itself — a **±100% band** on all twelve comparisons. A covariance off by a factor of two,
or a normalisation switched from n-1 to n (which moves these entries by 10%), would have passed. A
1e-7 relative perturbation of one eigenvalue now fails the test.

## Regression drift guards — `test_3d_morphology_regression.h`

Recipe `morphology3d.regression_ut_phantom`. No oracle — Nyxus' own values, at a 10% band. There is no
dump function; the table is small and hand-maintained. The same values are pinned a second time as the
coverage sweep's baseline in `test_3d_morphology_coverage.h`, so a change must be applied in both.

**Why these six are snapshot-only.** `3AREA` counts exposed voxel faces (59992) where MIRP and
pyradiomics integrate a marching-cubes mesh (46739) — a 28% *convention* difference. `3AREA_2_VOLUME`,
`3COMPACTNESS1`, `3COMPACTNESS2`, `3SPHERICITY` and `3SPHERICAL_DISPROPORTION` are all derived from
`3AREA` and inherit it. No tolerance turns that into an agreement; settling it means choosing a
convention, which changes six public feature values.

## Coverage artifact

```
python tests/vetting/audit/scan_morphology3d_coverage.py           # rewrite
python tests/vetting/audit/scan_morphology3d_coverage.py --check   # drift + acceptance check
```

`--check` asserts that every `vetted` row is backed by an oracle-suffixed test naming the oracle the
row names. That is what surfaced `3VOXEL_VOLUME` and `3VOLUME_CONVEXHULL` having a golden and a band
but no assertion of their own — worth re-running after any change to the family's test files.
