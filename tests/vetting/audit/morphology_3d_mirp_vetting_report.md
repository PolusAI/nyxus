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
| Test | `test_3d_morphology_mirp.h` |
| Tolerance | axes `rel=1e-9`; volumes measured per feature — `3VOXEL_VOLUME` 0.1% band on a measured 2.3e-4%, `3VOLUME_CONVEXHULL` 5% band on a measured 3.41% |

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

The report generator misses this for the same reason it missed two rows in 2D morphology: a feature
name appearing in an oracle file counts as coverage, whether an assertion reads it or only a table
does. Both are now asserted by name — against MIRP, in `test_3d_morphology_mirp.h`, for the reason
the next section gives.

## MATLAB cannot be re-run — so MIRP replaces it

Three rows claimed `oracle=matlab`, and `revet` step 3 says to run every oracle the family claims.
That is not possible here: there is no MATLAB licence, and Octave's `image` package has no
`regionprops3`. The goldens had no in-repo generator either, which is the SPEC §6.4 gap tracked in
`not_covered.md` §C.

What is possible is a reference from a tool that computes the same quantities and *is* runnable from
the tree. All three volume rows now read `oracle=mirp` and are asserted in `test_3d_morphology_mirp.h`;
`test_3d_morphology_matlab.h` is gone. Keeping both would also have left two oracles asserting one
feature value, which SPEC §3 does not allow — one oracle per assertion — and the one to keep is the
one the tree can re-run. The MATLAB numbers stay in the header as the corroborating measurement they
are:

| quantity | MIRP | MATLAB golden | agreement |
|---|---|---|---|
| voxel volume (`morph_vol_approx`) | 274432.0 | 274432.0 | **exact** |
| convex-hull volume (`morph_volume` / `morph_vol_dens_conv_hull`) | 496958.32 | 497824.0 | 0.17% |

`morphology_3d_mirp_volume_ref_vals` holds all three, `test_3d_morphology_{voxel_volume,
volume_convex_hull,mesh_volume}_mirp` assert them, and `gen_morphology3d_mirp.py` re-derives them on
every run (8 pins, all at rel=0). The MATLAB session cannot be repeated, but the *values* it produced
are now reproducible from the tree — which is what SPEC §6.4 is protecting.

That also relocates the disagreement: Nyxus' `3VOLUME_CONVEXHULL` is 479997.83, which is 3.58% from
MATLAB and 3.41% from MIRP, while the two tools sit 0.17% apart. The difference is on the Nyxus side —
a **discrete voxel hull against two independently triangulated ones** — and that sentence is the
citation SPEC §7 asks for behind the 5% band: a definitional difference between a voxelised and a
triangulated hull, measured against two tools rather than asserted.

### `3MESH_VOLUME` is not a mesh volume

The same run says something the MATLAB comparison alone could not. MIRP reports IBSI's volume (mesh)
as `morph_volume` = **274338.34** — a surface-mesh integral, necessarily close to the voxel volume
274432 for a solid ROI. Nyxus' `3MESH_VOLUME` reports **479997.83**, the convex-hull volume, because the
implementation aliases the two:

| quantity | value | vs Nyxus `3MESH_VOLUME` |
|---|---|---|
| IBSI volume (mesh), MIRP `morph_volume` | 274338.34 | **75% low** |
| convex-hull volume, MIRP | 496958.32 | 3.5% |
| Nyxus `3MESH_VOLUME` | 479997.83 | — |

So the feature is measuring a hull, under a name that says mesh, and the MATLAB pin it used to carry
(`ConvexVolume`, 497824) was consistent only because the same alias was applied when the golden was
chosen. Its MIRP golden is the hull volume for exactly that reason: the assertion judges the alias
against the quantity the alias computes, and pinning it against `morph_volume` instead would encode a
75% miss as agreement. The naming/definition gap is filed as a defect for its own branch, not
absorbed by the 5% band.

## The surface-area convention gap stays open

`3AREA` counts exposed voxel faces (59992); MIRP integrates a marching-cubes mesh
(`morph_area_mesh` = 46739.02), a **28% difference**. It is a convention difference, not a numerical
one, so no amount of tolerance makes it an agreement.

`3AREA` and the five features derived from it — `3AREA_2_VOLUME`, `3COMPACTNESS1`, `3COMPACTNESS2`,
`3SPHERICITY`, `3SPHERICAL_DISPROPORTION` — therefore stay `status=regression` on
`morphology3d.regression_ut_phantom`, with the reason recorded per row rather than left as a bare
absence. Settling it means choosing between the IBSI mesh convention and the documented voxel one,
which changes six public feature values and belongs on its own branch.

## Include hygiene and file-level observations

The family is four headers plus the new one. There is no pytest case for 3D morphology.

- **`test_3d_morphology_common.h` carried a dead `#if 0` block** — an entire superseded copy of the
  fixture, ending in a line of `*********************************` that is not valid C++ and only
  compiled because it was inside the disabled block. Removed. The header also included
  `test_ref_vals.h` while declaring no reference table (they were moved out to the per-oracle files),
  and relied transitively on `<string>`, `<tuple>`, `<vector>` and `helpers/fsystem.h` for `fs::exists`;
  the unused include is dropped and the four used ones are now direct.
- **`test_3d_morphology_matlab.h`** names `Pixel3` and `Nyxus::calc_eigvals` but included neither
  `features/pixel.h` nor `helpers/helpers.h`, and used `std::abs`/`std::vector` without `<cmath>` or
  `<vector>`. All added.
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

# coverage artifact
python tests/vetting/audit/scan_morphology3d_coverage.py [--check]
```
