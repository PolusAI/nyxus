# 3D morphology vs MIRP — vetting report

Closes the five `3*_AXIS_LEN` / `3ELONGATION` / `3FLATNESS` rows that read `status=vetted,
oracle=mirp` with no in-tree oracle assertion, closes two more the plan did not count, and gives the
two volume rows a reference that can actually be re-run from this tree.

## Tool and configuration

| | |
|---|---|
| Tool | mirp 2.6.0 (numpy 2.4.6, pandas 3.0.3) |
| Recipe | `morphology3d.mirp_ibsi` |
| Fixture | the segmented phantom, `phantoms/ut_inten.nii` + `phantoms/ut_mask57.nii`, label 57 |
| MIRP config | `by_slice=False`, `base_feature_families="morphology"`, `base_discretisation_method="none"`, native 1×1×1 spacing |
| Nyxus config | `D3_SurfaceFeature`, `IBSI=true`, `GREYDEPTH=128`, `PIXELSIZEUM=100` (what `test_3d_morphology_common.h` sets) |
| Generator | `tests/vetting/oracles/gen_morphology3d_mirp.py` |
| Test | `test_3d_morphology_mirp.h`, with `test_3d_morphology_analytic.h` as a second, tool-independent oracle for `3AREA`, the two mesh volumes and the five area ratios |
| Tolerance | axes `rel=1e-9`; `3AREA`, `3MESH_VOLUME`, `3VOLUME_CONVEXHULL` and the five ratios `rel=1e-6` on measured residuals of 2.7e-08 to 2.4e-07; `3VOXEL_VOLUME` 0.1% band on a measured 2.338e-04% residual |

Morphology is computed from the mask geometry, so no grey-level binning applies on either side —
`GREYDEPTH` is set only because the shared fixture sets it, and MIRP is told `none` explicitly.

## Result

| feature | MIRP `morph_*` | fresh MIRP run | Nyxus | rel | verdict |
|---|---|---|---|---|---|
| `3MAJOR_AXIS_LEN` | `pca_maj_axis` | 104.70681271508683 | 104.70681271508683 | **0** | vetted |
| `3MINOR_AXIS_LEN` | `pca_min_axis` | 88.30145986864228 | 88.301459868642283 | **0** | vetted |
| `3LEAST_AXIS_LEN` | `pca_least_axis` | 71.51449974198198 | 71.514499741981993 | 2.0e-16 | vetted |
| `3ELONGATION` | `pca_elongation` | 0.8433210559938976 | 0.84332105599389762 | **0** | vetted |
| `3FLATNESS` | `pca_flatness` | 0.6829975804590384 | 0.68299758045903847 | 1.6e-16 | vetted |

Same definition on both sides — 4·√ of the mask covariance eigenvalues, and their ratios — so the
agreement is exact to double precision and the assertions are held at `rel=1e-9`, far tighter than
SPEC §7's `rel=1e-3` same-definition tier. A band wider than the divergence it absorbs hides drift.

**The Nyxus column is also the pre-existing hardcoded value.** These five features were already pinned
in `morphology_3d_regression_coverage_ref_vals` (`test_3d_morphology_coverage.h`) to 17 digits — as
Nyxus snapshots, with no oracle claim. Those pins are byte-identical to the Nyxus column above, so
what this PR changes is not the numbers but whether anything compares them to a tool.

**And that is the circularity check `revet` §3 asks for.** A golden that reproduces Nyxus' own
algorithm to 15 digits proves nothing on its own — it is exactly the trap the 2D first-order
percentile rows fell into. Here the MIRP column comes from a fresh `mirp.extract_features` run
performed for this PR, not from the snapshot and not from the 2026-07 harness notes; it independently
lands on the same values. The two are therefore genuinely concordant rather than one copied from the
other. (The fresh run does reproduce the July harness's numbers exactly, which is corroboration of
that harness, not the source of these goldens.)

## The generator re-checks the identities that the old defect broke

This family carried an eigenvalue-ordering defect: `calc_eigvals` returns eigenvalues **descending**,
but the axis lengths were assigned `MAJOR←L[1]`, `MINOR←L[2]`, `LEAST←L[0]`, which produced
`3LEAST_AXIS_LEN` 104.7 > `3MAJOR_AXIS_LEN` 88.3 and `3FLATNESS` 1.19 > 1 — both structurally
impossible. It is fixed in `src/nyx/features/3d_surface.cpp` and the values above are the corrected
ones.

`gen_morphology3d_mirp.py` therefore asserts, on the **oracle's own output**, that
`MAJOR ≥ MINOR ≥ LEAST > 0`, that `ELONGATION` and `FLATNESS` lie in [0,1], and that each equals its
defining ratio. All hold. This catches a misconfigured oracle run before anything is pinned — and it
is the same check that would have caught the original defect on the Nyxus side.

No separate `_invariant` file is added. The five oracle pins fix the exact values on this fixture, so
they are strictly stronger than the bounds would be here; the bounds earn their place in the
generator, where the oracle's output is not otherwise constrained.

## Two rows the plan did not count

The family plan lists five gap rows for 3D morphology. There were seven. `3VOXEL_VOLUME` and
`3VOLUME_CONVEXHULL` read `status=vetted, oracle=matlab` and had a MATLAB golden and a stated band in
`morphology_3d_matlab_ref_vals` / `_ref_tols` — but **no oracle-named test asserted them**.
`test_3d_morphology_matlab.h` registered exactly one function, for `3MESH_VOLUME`. The only thing
comparing the other two against MATLAB was the parameterized sweep in
`test_3d_morphology_coverage.h`, whose case names carry no oracle token, so the vetting claim rested
on a test that does not say what it is checking (SPEC §6.2).

The report generator missed this for the same reason it missed two rows in 2D morphology: a feature
name appearing in an oracle file counts as coverage, whether an assertion reads it or only a table
does. Both are now asserted by name, separately against MATLAB and MIRP, in their oracle headers.

## MATLAB and MIRP provide separate oracle assertions

Three rows claimed `oracle=matlab`, and `revet` step 3 says to run every oracle the family claims.
MATLAB is not available in CI, and Octave's `image` package has no `regionprops3`. The checked-in
`tests/vetting/oracles/gen_morphology3d_matlab.m` now makes the comparison reproducible with licensed
MATLAB R2026a and Image Processing Toolbox.

The two MATLAB values are pinned and asserted in `test_3d_morphology_matlab.h`. MIRP computes the
same quantities and is asserted separately in `test_3d_morphology_mirp.h`. The registry therefore
has two rows for each of these feature/config pairs, one per reference, as SPEC §3 permits:

| quantity | MIRP | MATLAB golden | agreement |
|---|---|---|---|
| voxel volume (`morph_vol_approx`) | 274432.0 | 274432.0 | **exact** |
| convex-hull volume (`morph_volume` / `morph_vol_dens_conv_hull`) | 496958.32 | 497824.0 | 0.17% |

`morphology_3d_mirp_volume_ref_vals` holds the voxel volume and `morphology_3d_mirp_mesh_ref_vals`
the mesh area, the mesh volume, the hull and the five ratios; one `test_3d_morphology_*_mirp` per
feature asserts them, and `gen_morphology3d_mirp.py` re-derives them on every run (14 pins). The
MATLAB generator produces its two built-in quantities (`Volume` and `ConvexVolume`) directly from the
[`PolusAI/nyxus` main fixture](https://github.com/PolusAI/nyxus/blob/main/tests/data/nifti/phantoms/ut_mask57.nii),
and the two MATLAB-named tests assert those pins against Nyxus.

The 0.17% between the two hull references is a difference of convention, not of noise. MIRP hulls the
vertices of the marching-cubes mesh; `ConvexVolume` is the number of voxels in `regionprops3`'s
`ConvexImage`, the hull rasterised back onto the lattice. Nyxus hulls the mesh vertices too, so it
matches MIRP (below) and sits 0.174% from MATLAB, which the MATLAB row holds to a 0.5% band.

### `3MESH_VOLUME` and `3VOLUME_CONVEXHULL` are the IBSI mesh quantities

MIRP reports IBSI's volume (mesh) as `morph_volume` = **274338.34375**, and its convex-hull volume
backs out of `morph_vol_dens_conv_hull` as **496958.32**. Nyxus builds the same surface — the
marching-cubes triangulation of the mask at the 0.5 isolevel, walked by `Nyxus::march_roi_surface` —
integrates it and hulls its vertices, and reproduces both:

| feature | MIRP | Nyxus | rel |
|---|---:|---:|---:|
| `3MESH_VOLUME` (`morph_volume`) | 274338.34375 | 274338.33333333331 | 3.8e-08 |
| `3VOLUME_CONVEXHULL` (`morph_volume / morph_vol_dens_conv_hull`) | 496958.3201121965 | 496958.33333333331 | 2.7e-08 |

These are the same quantities on both sides, so the residual is not a convention gap being absorbed:
it is MIRP carrying the mesh in `float32`. The pins are `rel=1e-6`, more than an order of magnitude
above the residuals. The hull of the voxel centres, by comparison, is 480651.67 here: 3.3% short,
because those points sit half a step inside the mesh on every face.

The mesh volume tracks the voxel-counting volume closely — 274338.34 against 274432 — which is what a
surface integral does on a solid ROI, and is the check that the quantity is a mesh volume at all.

### `3AREA` and its five ratios are the IBSI mesh quantities

MIRP reports IBSI's area (mesh) as `morph_area_mesh` = **46739.01953125**. Nyxus takes the area of
the same marching-cubes surface whose volume it integrates, triangle by triangle, and feeds it with
`3MESH_VOLUME` into IBSI's five ratios:

| feature | MIRP | Nyxus | rel |
|---|---:|---:|---:|
| `3AREA` (`morph_area_mesh`) | 46739.01953125 | 46739.022534087213 | 6.4e-08 |
| `3AREA_2_VOLUME` (`morph_av`) | 0.1703699827194214 | 0.17037000249358961 | 1.2e-07 |
| `3COMPACTNESS1` (`morph_comp_1`) | 0.015317650511860847 | 0.015317649265301225 | 8.1e-08 |
| `3COMPACTNESS2` (`morph_comp_2`) | 0.0833655446767807 | 0.083365524768728425 | 2.4e-07 |
| `3SPHERICAL_DISPROPORTION` (`morph_sph_dispr`) | 2.2891335487365723 | 2.2891337610474518 | 9.3e-08 |
| `3SPHERICITY` (`morph_sphericity`) | 0.4368464946746826 | 0.43684646874563782 | 5.9e-08 |

The residuals are again MIRP's `float32` mesh: every MIRP value above is an exactly representable
`float32`, and the ratio with the largest residual, `3COMPACTNESS2`, carries the area cubed. The
generator also checks that MIRP's five ratios equal IBSI's definitions over its own
`morph_area_mesh` and `morph_volume`, so a pin cannot hide a ratio MIRP computes some other way. The
pins are `rel=1e-6`. The ratios over the voxel-counting volume would sit 2.3e-04 to 6.8e-04 from
MIRP, and the exposed-voxel-face area 28% above it, so the band separates the definitions.

## A second, tool-independent oracle

`test_3d_morphology_analytic.h` pins `3MESH_VOLUME` and `3VOLUME_CONVEXHULL` against closed-form
geometry, so neither feature rests on MIRP alone. A lone voxel's 0.5-isolevel surface is the
octahedron with vertices half a lattice step out (volume 1/6) and a solid box is a bevelled box whose
volume is an exact function of its sides; both are convex, so each is its own hull, and both hold at
`rel=1e-12`. Octahedra and rhombic prisms of voxels pin the hull where it differs from the mesh. Those
shapes separate the conventions outright — counting voxels returns 1 for the single voxel against
1/6, and hulling the voxel centres returns 0.

The same file pins `3AREA` and the five ratios against closed forms at `rel=1e-12`: the lone voxel's
octahedron, of area √3, and bevelled boxes, whose area is a sum over faces, edge bevels and corner
facets, on three voxel grids, 1×1×1, 1.3×0.8×2.5 and 0.5×2×1, in both `SINGLEROI` settings. The
grids are unequal on every axis, so a spacing applied to the wrong axis misses the area. PyRadiomics
3.0.1 pins `3AREA`, `3AREA_2_VOLUME` and `3SPHERICITY` on the phantom at spacing (1.3, 0.8, 2.5)
(`test_3d_morphology_pyradiomics.h`, `rel=1e-9`; `SurfaceArea` 121066.46691610393).

## Include hygiene and file-level observations

The family uses five current headers. There is no pytest case for 3D morphology.

- **`test_3d_morphology_common.h` carried a dead `#if 0` block** — an entire superseded copy of the
  fixture, ending in a line of `*********************************` that is not valid C++ and only
  compiled because it was inside the disabled block. Removed. The header also included
  `test_ref_vals.h` while declaring no reference table (they were moved out to the per-oracle files),
  and relied transitively on `<string>`, `<tuple>`, `<vector>` and `helpers/fsystem.h` for `fs::exists`;
  the unused include is dropped and the four used ones are now direct.
- **`test_3d_morphology_matlab.h`** now contains only the three feature assertions supported by
  `regionprops3` (`Volume` and `ConvexVolume`). The former covariance/eigenvalue helper check lives
  in `test_3d_morphology_mechanics.h` and is not presented as MATLAB feature vetting.
- **`test_3d_morphology_regression.h`** relied transitively on `<string>`. Added.
- **`test_3d_morphology_coverage.h`** kept its single include of `test_3d_coverage_common.h`, as the
  3D `_coverage.h` files do; SPEC §6.3.1 sanctions a `_common.h` for fixture scaffolding. That file
  has since been retired — see `morphology_3d_golden_regen.md`, “The retired coverage sweep”.
- Three headers carried "Migrated from test_3d_shape.h (Wave 8)" provenance and a paragraph on which
  functions used to live where. That is history, which `revet` step 6 puts in this report rather than
  in the test files; trimmed to current-state descriptions.

As with 3D GLRLM, `tests/vetting/matrix/morphology.md` (SPEC §5.1) and `tests/vetting/benchmarks.md`
(SPEC §6.3) do not exist — and exist for no family but GLCM. Not created here for the same reason:
one family's copy would leave the series inconsistent. A repo-wide gap worth its own PR.

## Reproduction

```
# oracle goldens + identity checks + the MATLAB cross-check (conda env with mirp 2.6.0)
python tests/vetting/oracles/gen_morphology3d_mirp.py

# coverage check
python tests/vetting/audit/scan_morphology3d_coverage.py --check
```
