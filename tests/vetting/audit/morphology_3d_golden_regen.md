# Regenerating the 3D morphology goldens

Three benchmarks on **one fixture** — the segmented phantom
(`tests/data/nifti/phantoms/ut_inten.nii` + `ut_mask57.nii`, label 57) through
`test_3d_morphology_common.h`'s settings (`IBSI=true`, `GREYDEPTH=128`, `PIXELSIZEUM=100`). Because
the Nyxus side is identical for all three, the numbers *are* comparable to each other, which is what
makes the MIRP cross-check of the MATLAB goldens meaningful.

## MIRP goldens — `test_3d_morphology_mirp.h`

Recipe `morphology3d.mirp_ibsi`. Covers the five PCA axis features.

```
python tests/vetting/oracles/gen_morphology3d_mirp.py
```

The generator prints a paste-ready table, re-verifies every pin, checks the structural identities, and
prints the MATLAB cross-check quantities. It exits non-zero on any mismatch, any unproducible or
unpinned golden, and any identity violation. Needs mirp 2.6.0:
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

## MATLAB goldens — `test_3d_morphology_matlab.h`

Recipe `morphology3d.matlab_regionprops3`, MATLAB R2025b Image Processing Toolbox:

```matlab
V = niftiread('ut_inten.nii');  M = niftiread('ut_mask57.nii') == 57;
s = regionprops3(M, 'Volume', 'ConvexVolume');
```

`Volume` → `3VOXEL_VOLUME`; `ConvexVolume` → `3VOLUME_CONVEXHULL` and, through the Nyxus alias,
`3MESH_VOLUME`.

**There is no in-repo generator for these**, and there cannot be one here: no MATLAB licence, and
Octave's `image` package has no `regionprops3`. That is the SPEC §6.4 gap tracked in
`not_covered.md` §C. Until it is closed, `gen_morphology3d_mirp.py` prints the corresponding MIRP
quantities so the goldens can at least be checked against a second tool — currently exact on the
voxel volume and 0.17% on the hull volume.

`3AREA` is deliberately absent from the MATLAB table: `regionprops3` `SurfaceArea` disagrees by more
than 10%, for the same reason MIRP does (see below).

**Per-feature bands** live in `morphology_3d_matlab_ref_tols`, measured rather than assumed —
`3VOXEL_VOLUME` 0.1% (same definition, measured 2.3e-4%), the two hull volumes 5% (measured 3.88%).

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
row names. That is what surfaced `3VOXEL_VOLUME` and `3VOLUME_CONVEXHULL` having a MATLAB golden and a
band but no assertion of their own — worth re-running after any change to the family's test files.
