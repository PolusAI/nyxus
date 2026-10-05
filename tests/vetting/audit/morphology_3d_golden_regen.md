# Regenerating the 3D morphology goldens

Two benchmarks on **one fixture** — the segmented phantom
(`tests/data/nifti/phantoms/ut_inten.nii` + `ut_mask57.nii`, label 57) through
`test_3d_morphology_common.h`'s settings (`IBSI=true`, `GREYDEPTH=128`, `PIXELSIZEUM=100`) — plus a
kernel check that reads no image at all. Because the Nyxus side is identical for both benchmarks, the
numbers *are* comparable to each other.

The voxel-counting and convex-hull volumes have two separate oracle assertions at the shared Nyxus
config, MATLAB `regionprops3` and MIRP; the mesh volume has MIRP's alone. This is intentional redundancy under SPEC §3, whose registry is one row per
feature × config recipe × oracle assertion and whose rollup asks whether at least one row is vetted.

## MIRP goldens — `test_3d_morphology_mirp.h`

Recipe `morphology3d.mirp_ibsi`. Covers the five PCA axis features, the voxel-counting volume, the
mesh volume and the convex hull of the mesh.

```
python tests/vetting/oracles/gen_morphology3d_mirp.py
```

The generator prints three paste-ready tables, re-verifies every pin (8 of them, last run all at
rel=0), checks the structural identities, and prints the cross-check quantities — including the MATLAB
numbers asserted separately. It exits non-zero on any mismatch, any unproducible or unpinned golden,
and any identity violation. Needs mirp 2.6.0:
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

**The mesh volume and its hull.** They are pinned in `morphology_3d_mirp_mesh_ref_vals` at
`rel=1e-6`:

| Nyxus | MIRP | MIRP value | Nyxus | rel |
|---|---|---:|---:|---:|
| `3MESH_VOLUME` | `morph_volume` | 274338.34375 | 274338.33333333331 | 3.8e-08 |
| `3VOLUME_CONVEXHULL` | `morph_volume / morph_vol_dens_conv_hull` | 496958.3201121965 | 496958.33333333331 | 2.7e-08 |

Both sides build the same surface — the marching-cubes triangulation of the mask at the 0.5
isolevel — integrate it and hull its vertices, so these are the same quantities, not near ones, and
the residual is MIRP carrying the mesh in `float32`. `rel=1e-6` sits more than an order of magnitude
above both.

What MIRP reports and nothing pins: `morph_area_mesh` and the five ratios built from it, which
`3AREA` does not share a convention with (see the regression section below), and the part Nyxus
implements no feature for — the density family (`morph_*_dens_*`), `morph_diam`, `morph_integ_int`,
`morph_moran_i` and `morph_geary_c`. The generator fails on an unpinned mesh or volume column, so that
list cannot silently grow.

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

The two volume goldens were produced by an offline MATLAB R2026a Image Processing Toolbox session:

```matlab
M = niftiread('ut_mask57.nii') == 57;
s = regionprops3(M, 'Volume', 'ConvexVolume');
```

`Volume` → `3VOXEL_VOLUME` = 274432; `ConvexVolume` → `3VOLUME_CONVEXHULL` = 497824.

`ConvexVolume` is the number of voxels in `regionprops3`'s `ConvexImage`: the hull rasterised back onto
the lattice, where Nyxus and MIRP integrate the hull of the mesh vertices. The two conventions sit
0.174% apart on this phantom (497824 against 496958.33), and the MATLAB row holds a 0.5% band on that.

`regionprops3` has no mesh-volume property, so it holds no row for `3MESH_VOLUME`. That feature is an
integral of the ROI surface mesh and is covered by MIRP's `morph_volume` and by the closed-form
solids in `test_3d_morphology_analytic.h`.

The checked-in generator is `tests/vetting/oracles/gen_morphology3d_matlab.m`. It downloads the mask
from the [`PolusAI/nyxus` main fixture](https://github.com/PolusAI/nyxus/blob/main/tests/data/nifti/phantoms/ut_mask57.nii)
and calls MATLAB's `niftiread` and `regionprops3` built-ins directly. It
requires licensed MATLAB R2026a with Image Processing Toolbox. Octave's `image` package has no
`regionprops3`. The two values are pinned and asserted in
`test_3d_morphology_matlab.h`; MIRP supplies a separate second oracle for the same feature/config
pairs. The checked-in generator closes the SPEC §6.4 provenance gap without making MATLAB a CI
runtime dependency.

`3AREA` was deliberately absent from the MATLAB table too: `regionprops3` `SurfaceArea` disagrees by
more than 10%, for the same reason MIRP does (see below).

## Analytic goldens — `test_3d_morphology_analytic.h`

Recipe `morphology3d.analytic_lattice_solids`. Nothing to regenerate: the goldens are closed-form
geometry computed in the assertions themselves, from voxel clouds the file builds.

A lone voxel's 0.5-isolevel surface is the octahedron with vertices half a lattice step out along each
axis, so `3MESH_VOLUME` = 1/6 exactly. A solid *w*×*h*×*d* box comes out bevelled — each of the
4(*w*+*h*+*d*−3) cells along an interior edge run, and each of the 8 corner cells, cuts a fixed amount
off the staircase — so its volume is an exact function of the three sides:

```
volume = whd − (w+h+d−3)/2 − 5/6
```

Both shapes hold at `rel=1e-12`. The same formula is what `D3_SurfaceFeature`'s whole-volume
(`SINGLEROI`) branch evaluates in place of meshing every voxel. The box assertions run once per
`SINGLEROI` setting against the same closed form, so the shortcut is held to the value the general path
produces.

`3VOLUME_CONVEXHULL` is the volume of the convex hull of the mesh vertices. Every vertex is an in-ROI
voxel moved half a step towards an out-of-ROI neighbour, so a lattice solid whose extreme voxels are
known pins it exactly. The single voxel and the bevelled box are convex, so their hull is the mesh
volume above, in both `SINGLEROI` settings. The octahedron |x|+|y|+|z| ≤ R gives the octahedron of
radius R + ½, 4/3·(R + ½)³. The rhombic prism |x|+|y| ≤ R, |z| ≤ H gives a prism of rhombus radius
R + ½ and height 2H, capped at each end by a frustum of height ½ narrowing to radius R. Four boxes,
three octahedra and two prisms, placed hundreds of voxels from the origin, hold at `rel=1e-12`; scipy's
qhull over the same vertices reproduces every one of them to 5e-16.

An ROI one slice thick still has a surface with thickness, so it has a hull: the bevelled
*w*×*h*×1 slab. And the mask alone decides the surface: a box whose voxels are all 0, or whose two
faces are 0, gives the same bevelled box as a uniform one.

## In-RAM and out-of-core

An ROI too large for memory is featurized by `D3_SurfaceFeature::osized_calculate`, which streams the
disk-backed voxel cloud one Z-plane at a time. `3MESH_VOLUME` goes through the same
`Nyxus::roi_mesh_volume` on both paths — the marching-cubes walk reads two planes at a time from
whichever source the path has — so an ROI gets the same triangles, summed in the same order, either
way. The whole-volume (`SINGLEROI`) closed forms live in one helper both paths call.
The same walk collects the hull points, so the hull is shared too. `test_3d_morphology_invariant.h`
featurizes a non-convex ROI both ways, in both `SINGLEROI` settings, and requires `3MESH_VOLUME`,
`3AREA` and `3VOLUME_CONVEXHULL` to be equal exactly.

The same file holds two more invariants. A ragged ROI translated by thousands of voxels must keep all
three values bit for bit: both volumes are summed relative to a point on the body from integer
(doubled) coordinates, so every term is exact. And a lattice ball's mesh volume must sit within 1% of
4/3·π·r³ (−3.6% at r=5, −0.15% by r=15) and strictly below the voxel count, since on a convex body the
surface through the face centres cuts every boundary corner off the union of voxel cubes.

## The marching-cubes case table

The mesh volume reads `MC_TRIANGLES` in `src/nyx/features/3d_mesh.cpp`, a 256-entry table derived
and re-verified by `tests/vetting/audit/derive_marching_cubes_table.py`:

```
python tests/vetting/audit/derive_marching_cubes_table.py            # derive and verify
python tests/vetting/audit/derive_marching_cubes_table.py --emit     # print the C++ table
```

The script contours each cube's six faces with marching squares and chains the segments into the
closed loops the surface has to span. An ambiguous face — two diagonal corners inside — is resolved by
separating them, a rule that reads only that face's own four corners, so two cubes sharing a face
always cut it the same way. For the 134 masks with no ambiguous face it adopts the classic
triangulation of those loops, taken from `skimage.measure.marching_cubes(method='lorensen')` and
accepted only after checking that it spans exactly the loops derived here; that settles an arbitrary
diagonal choice the loops leave open, worth a few parts in a thousand of area, in favour of the one
MIRP and pyradiomics integrate. The other 120 masks keep the derived triangulation, because the
classic table leaves those cubes open. scikit-image is a generation-time reference only
(BSD-3-Clause), never a build or CI dependency.

The script's own checks: the surface closes on random volumes exercising all 256 masks, a lone voxel
gives exactly 1/6 and √3, and every solid box matches the closed form above.

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

Recipe `morphology3d.regression_ut_phantom`. No oracle — Nyxus' own values, pinned to 17 digits and
asserted at `rel=1e-9`, the band the family's MIRP PCA pins and covariance-kernel pins already use.

```
runAllTests --gtest_filter=*3D_MORPHOLOGY_DUMP_REGRESSION*
```

prints the whole table in paste-ready form, read out of the same fixture the assertions use.
`D3_SurfaceFeature::calculate()` reads one setting, `SINGLEROI`, and every 3D morphology fixture sets
it `false`, so the recipe's `GREYDEPTH`/`IBSI`/`PIXELSIZEUM` do not reach these numbers — which is why
the retired sweep's pins, taken at a different recipe, were byte-identical to these.

**Why the band is `rel=1e-9` and not 10%.** A 10% band passes a value off by a factor of 1.1, and two
of the eight goldens had once drifted inside it unnoticed: `3AREA` read 58457 against the then-actual
59992 (2.6%), and `3VOLUME_CONVEXHULL` read 478516 — a pin dating to #279 — against 479997.83 (0.31%).

`rel=1e-9` is not a guess at what toolchains might agree on. Every value in the table is
double-precision arithmetic with no approximation left in the path, so the band has the whole double
range as headroom. Seven of the eight were already **bit-for-bit identical** on MSVC Release, Linux
gcc `RelWithDebInfo -O1` under `-fsanitize=address,undefined`, and Apple clang Release on `macos-14`;
`3VOLUME_CONVEXHULL` was the exception, and the next section is why it no longer is.

**Why these six are snapshot-only.** `3AREA` counts exposed voxel faces (59992) where MIRP and
pyradiomics integrate a marching-cubes mesh (46739) — a 28% *convention* difference. `3AREA_2_VOLUME`,
`3COMPACTNESS1`, `3COMPACTNESS2`, `3SPHERICITY` and `3SPHERICAL_DISPROPORTION` are all derived from
`3AREA` and inherit it. No tolerance turns that into an agreement; settling it means choosing a
convention, which changes six public feature values.

## The convex hull: mesh vertices, in double, with eps scaled to the ROI

The hull is taken over the mesh vertices, which `Nyxus::roi_mesh_volume` collects while it walks the
surface. Of the vertices on each lattice row along x it keeps the two ends -- a vertex between two
others on one line is never a hull vertex -- and hands them over in doubled coordinates, so the
half-integer vertices become integer lattice points. `D3_SurfaceFeature::convex_hull_volume`, which
the in-core and out-of-core paths share, hulls them and divides the volume by 8. It derives
`quick_hull`'s epsilon from the cloud's own coordinate extent:

```cpp
using Points = std::vector<std::array<double, dim>>;
...
const double eps = 16.0 * std::numeric_limits<double>::epsilon() * maxcoord;
quick_hull<typename Points::const_iterator> qh{ dim, eps };
```

`quick_hull` deduces its arithmetic from the point's element type, so the element type is the whole
precision decision. `eps` is the tolerance on "is this point outside the facet plane", and it has to
sit above the rounding error of the distances it judges and below the smallest real gap between a
lattice point and a facet. Both scale with the ROI, which is why it is derived rather than fixed.

**The margin.** The doubled mesh vertices are lattice points, so a point that is not exactly coplanar
with a facet stands at least |det|/|normal| away from it, with `det` a non-zero integer. For integer
coordinates under 10³ that is upwards of 1e-7.

The rounding error of a distance depends on how the facet plane is built. `quick_hull`'s generic
path takes each normal component and the offset `D` as determinants of the raw coordinates, and `D`
is of order |point|³ — ~1e6 on the phantom, which leaves ~1e-10 of rounding in every distance, three
orders of magnitude above any `eps` scaled to the coordinates. For `dimension_ == 3` the plane is
instead the cross product of two edges taken through the first vertex: every product is between
coordinate differences, exact for lattice vertices, and a distance carries rounding of order
|point| × `DBL_EPSILON`, ~1e-14 on the phantom. `eps` (under 1e-12 there) sits between that and the
lattice gap.

`eps` is also the one tolerance every decision uses: `partition()` puts a point outside a facet when
its distance exceeds `eps`, `process_visibles()` counts a facet visible from the apex by the same
test, and `steal_best()` extends the initial simplex only with a point further than `eps` from the
subspace so far. A facet coplanar with the apex is therefore never removed, no zero-area facet is
created, and points confined to a plane yield no simplex and a hull volume of 0; fewer than four
points never reach `quick_hull` at all. With no decision left to rounding, the facet set encloses the
exact hull whatever order a standard library's hash sets visit the facets in. The triangulation of a
hull face with more than three vertices still follows that order: the hash sets are keyed on point
addresses, so the same cloud can come back as 48, 50 or 52 facets that enclose the same volume.

**The volume sum.** Each facet contributes the triple product of its vertices taken relative to one
hull vertex. Every coordinate difference is an integer, so every term is exact while it stays under
2^53, and so is their sum, in any order and wherever the ROI sits in the image. That is what lets the
regression pin hold at `rel=1e-9`, and what `TEST_3D_MORPHOLOGY_TRANSLATED_ROI_INVARIANT` holds to
bit-for-bit equality. A sum of `det4` terms on absolute coordinates against a non-integer centre is
neither: its rounding grows with the distance from the origin, 9.5e-9 at 2000 voxels out.

Each of the three predicates has a test that fails without it, and the degenerate cases are driven
on the kernel directly in `test_3d_morphology_mechanics.h` (recipe `morphology3d.quick_hull_scipy`):
`TEST_3D_MORPHOLOGY_QUICK_HULL_LATTICE_ELLIPSOID_MECHANICS` hulls three lattice ellipsoids at six
placements each and requires no zero-area facet, no point more than `eps` outside a facet, and the
volume qhull gives; `TEST_3D_MORPHOLOGY_QUICK_HULL_DEGENERATE_BASIS_MECHANICS` requires
`get_affine_basis()` to stop at three points on a plane and two on a line; and
`TEST_3D_MORPHOLOGY_CONVEX_HULL_VOLUME_DEGENERATE_MECHANICS` requires `convex_hull_volume` to return 0
for no point, one, three and a coplanar set.

| edit, reverted alone | fails |
|---|---|
| cofactor plane in place of the cross product | `VOLUME_CONVEX_HULL_MIRP`, `LATTICE_HULL_VOLUME_ANALYTIC`, `TRANSLATED_ROI_INVARIANT`, `QUICK_HULL_LATTICE_ELLIPSOID_MECHANICS`, `VOLUME_CONVEX_HULL_REGRESSION` |
| `steal_best()` accepting any distance above 0 | `CONVEX_HULL_VOLUME_DEGENERATE_MECHANICS` aborts on `assert(check())`; `QUICK_HULL_DEGENERATE_BASIS_MECHANICS` |
| `process_visibles()` testing `zero <` | `VOLUME_CONVEX_HULL_MIRP`, `QUICK_HULL_LATTICE_ELLIPSOID_MECHANICS`, `VOLUME_CONVEX_HULL_REGRESSION` |

A coplanar set's volume alone cannot tell the second edit apart: without the asserts, a flat simplex
also integrates to 0, which is why the basis is asserted directly. The third edit is invisible to the
closed-form solids, whose small hulls never put an apex within `eps` of a neighbouring facet; the
ellipsoids and the phantom do, and which ellipsoid placement depends on the hash order, so the test
runs all eighteen.

Regenerate or re-verify the ellipsoid volumes with

```
python tests/vetting/oracles/gen_morphology3d_quick_hull_scipy.py            # print
python tests/vetting/oracles/gen_morphology3d_quick_hull_scipy.py --check    # re-verify the pins
```

scipy's `ConvexHull` is qhull, an implementation independent of `quick_hull`; a hull of lattice points
has a volume that is a multiple of 1/6, and each pin is that multiple.

**Why that matters here.** In single precision it did not hold. With the points in `float` and a
hard-coded `eps = 1e-10f`, coordinates of ~100 put consecutive representable values ~7.6e-6 apart, so
the plane distances carried rounding error six orders of magnitude above `eps` and the predicate was
decided by noise: *which* nearly-coplanar boundary voxels became hull vertices depended on how a given
compiler rounded and contracted that arithmetic, and a different facet set integrates to a different
volume. Measured on the segmented phantom at the time:

| toolchain | `3VOLUME_CONVEXHULL`, float hull |
|---|---|
| MSVC 19.44, Release, Ninja (local) | 479997.83333333186 |
| MSVC, Release, `windows-latest` (CI) | 479997.83333333186 |
| gcc, Release, `ubuntu-latest` (CI) | 479997.83333333186 |
| gcc 13, RelWithDebInfo `-O1`, ASan+UBSan (local) | 479997.83333333186 |
| Apple clang, Release, `macos-14` (CI) | **480308.33333333244** |

Four platforms, one outlier, rel **6.5e-4** — five to six orders of magnitude above what double
arithmetic on a fixed algorithm leaves, and the reason the pin carried `rel=1e-3` while its seven
neighbours carried `rel=1e-9`. That hull was also taken over a different point set, the centres of
the contour voxels, which sits half a step inside the mesh on every face: 3.3% below MIRP on the
phantom, with the gap read at the time as voxelised-against-triangulated. Hulled over the mesh
vertices instead, the phantom returns **496958.33333333331**, which is 1490875/3, the exact hull
volume qhull computes from the same vertices, 2.7e-08 from MIRP, and the pin is back at `rel=1e-9`
with its neighbours.

## The retired coverage sweep

`test_3d_morphology_coverage.h` instantiated the two generic `TEST_P` suites of
`test_3d_coverage_common.h` over the family's 14 features — 3 with a MIRP golden, 11 pinned in a local
`morphology_3d_regression_coverage_ref_vals` table. It is gone, and nothing was lost with it:

| what the sweep did | where it lives now |
|---|---|
| MIRP band check on `3VOXEL_VOLUME`, `3VOLUME_CONVEXHULL`, `3MESH_VOLUME` | `test_3d_morphology_mirp.h`, through named tests; `3MESH_VOLUME` and `3VOLUME_CONVEXHULL` are now pinned at `rel=1e-6`, against `morph_volume` and MIRP's hull, rather than both against the hull at 5% |
| full-precision pins on `3ELONGATION`, `3FLATNESS`, `3LEAST_AXIS_LEN`, `3MAJOR_AXIS_LEN`, `3MINOR_AXIS_LEN` | `test_3d_morphology_mirp.h` — vetted against MIRP at `rel=1e-9`, which is strictly stronger than a self-pin |
| full-precision pins on `3AREA`, `3AREA_2_VOLUME`, `3COMPACTNESS1`, `3COMPACTNESS2`, `3SPHERICAL_DISPROPORTION`, `3SPHERICITY` | `morphology_3d_regression_ref_vals`, byte-identical, now at `rel=1e-9` instead of 10% |
| “the name resolves and the feature code matches” | every named test does it: `calculate_3d_morphology_feature_value()` calls `find_3D_FeatureByString` and asserts the returned code |
| “every registered `Feature3D` code has exactly one provider” | `FeatureManager::check_11_correspondence()`, in production since the 3D GLCM sweep was retired, and unit-tested in `test_feature_manager_mechanics.h` |

The completeness guard in `test_3d_coverage_common.h` reads the retired families' pins straight off
the tables their named tests assert against, so a family leaving the sweep is one `add_keys()` line
and no hand-kept counts.

## Coverage check

```
python tests/vetting/audit/scan_morphology3d_coverage.py --check   # acceptance check
python tests/vetting/report_features.py --write          # regenerate report_output.csv
```

`--check` asserts that every `vetted` row is backed by an oracle-suffixed test naming the oracle the
row names. That is what surfaced `3VOXEL_VOLUME` and `3VOLUME_CONVEXHULL` having a golden and a band
but no assertion of their own — worth re-running after any change to the family's test files.
