# 3D mesh volume vs PyRadiomics — comparison report

`3MESH_VOLUME` and PyRadiomics `MeshVolume` agree bit for bit, in every orientation, on every mask
tested whose surface crosses no ambiguous lattice face — a cube face with two diagonal corners inside
and two outside. On masks with such faces they differ, and the difference depends on orientation:
PyRadiomics is 0.42–0.82% below Nyxus on a pitted ball and 1.6–11.3% above on random noise. Across
the four orientations PyRadiomics' own volume spreads by up to 9.6% and Nyxus' by up to 0.65%.
PyRadiomics is therefore a usable oracle for `3MESH_VOLUME`, beside MIRP and the analytic solids,
only for ROIs whose surface has no ambiguous face. It is registered as one in that role:
`test_3d_morphology_pyradiomics.h` pins `3MESH_VOLUME` against PyRadiomics on the segmented phantom
at voxel spacing (1.3, 0.8, 2.5) (recipe `morphology3d.pyradiomics_anisotropic`, `rel=1e-9`). This
report covers the masks that pin cannot: its unit-spacing comparison is about the triangulation, not
the spacing.

## Tool and configuration

| | |
|---|---|
| Tool | PyRadiomics 3.0.1 (SimpleITK 2.3.1, NumPy 1.23.5, Python 3.8.20) |
| PyRadiomics call | `radiomics.cShape.calculate_coefficients`, the C routine `shape.py` calls, at unit spacing, on the mask cropped to its bounding box, transposed to `[z, y, x]` and padded by one voxel — the array `shape.py` builds from what the extractor hands it |
| Nyxus side | `MC_TRIANGLES` read from `src/nyx/features/3d_mesh.cpp`, walked as `march_roi_surface()` walks it and integrated from the first vertex as `roi_mesh_volume()` does |
| Script | `tests/vetting/audit/compare_mesh_volume_pyradiomics.py` |

```
python tests/vetting/audit/compare_mesh_volume_pyradiomics.py
```

The script reads the shipped table, so it always checks the table in the tree, but it does not run
the Nyxus binary. It runs PyRadiomics at unit spacing, so both sides report volume in voxel units.
The extractor itself refuses a one-voxel mask, so that row is a direct call. PyRadiomics
(BSD-3-Clause) is a reference only, never a CI runtime dependency (SPEC §4).

## What both implementations share

- Marching cubes over the binary mask at the 0.5 isolevel, with one voxel of empty padding, so the
  surface also closes where the ROI touches its bounding box.
- Every vertex at the midpoint of a cube edge; a binary mask needs no interpolation. Both meshes have
  the same vertices.
- Volume by the divergence theorem: the signed volumes of the tetrahedra each triangle spans with a
  reference point, summed and divided by 6.

PyRadiomics also integrates the mesh area, as half the norm of each triangle's cross product. Nyxus
computes no mesh area: `3AREA` sums the areas of the exposed voxel faces. The script computes the same mesh area on
the Nyxus triangles only to compare the two meshes.

## Where they differ

| aspect | PyRadiomics 3.0.1 | Nyxus | consequence |
|---|---|---|---|
| case table | its own 128-entry table; a cube whose corner 7 (in `cshape.c`'s numbering) is inside is served by complementing its mask and flipping the volume's sign | all 256 masks, derived by `derive_marching_cubes_table.py`; an ambiguous face always separates its inside corners | a complemented PyRadiomics cube joins the inside corners of an ambiguous face, where Nyxus separates them. On the 134 masks with no ambiguous face the two tables span the same contour loops, but triangulate 98 of them with different diagonals; the volume of every one of those cubes is identical |
| volume reference point | the corner of the padded array, one voxel outside the ROI's bounding box | the first vertex of the surface | over a closed surface the choice does not matter; over PyRadiomics' open one it does (see Result) |
| voxel spacing | each vertex scaled by the image spacing, so the volume is in the spacing's units (mm³ for a typical medical image) | the mesh is built on the voxels as acquired and its volume multiplied by sx·sy·sz, the spacing being the `--aniso*` factors as given or, with physical spacing on, the spacing ratios with the smallest axis 1 | scaling every vertex is a linear map, so both scale the volume by sx·sy·sz: they agree when Nyxus is given the image's spacing as `--aniso*`, which `test_3d_morphology_pyradiomics.h` checks. With physical spacing on instead, Nyxus reports volume in units of the finest axis's voxel |
| convex-hull volume | no 3D hull feature | `3VOLUME_CONVEXHULL`, the hull of the mesh vertices | no PyRadiomics counterpart |
| surface area and its ratios | `SurfaceArea` is the mesh area; `Sphericity`, `SurfaceVolumeRatio` and the deprecated `Compactness1`, `Compactness2`, `SphericalDisproportion` use mesh area and mesh volume | `3AREA` sums the areas of the exposed voxel faces; `3AREA_2_VOLUME`, `3COMPACTNESS1/2`, `3SPHERICAL_DISPROPORTION`, `3SPHERICITY` use `3AREA` and `3VOXEL_VOLUME` | different definitions, not comparable |

## Result

The script's output, verbatim. Volumes (V) and areas (A) are in voxel units.

- The **4 orientations** are the mask as built and mirrored along each of its three axes; a single
  value means all four agree.
- The **gap** is (PyRadiomics − Nyxus) / Nyxus, per orientation.
- **Moved** calls `cShape` directly on the mask placed 7 voxels further from the array's corner
  along each axis. PyRadiomics' extractor never does this, since it crops first; the column only
  exposes the open surface.
- **Nyxus mesh A** is listed because the script computes it; the Nyxus `3AREA` feature is the
  exposed-face area, not this area.

| mask | voxels | Nyxus V | Nyxus V, 4 orientations | PyRadiomics V | PyRadiomics V, 4 orientations | gap, 4 orientations | PyRadiomics V moved | Nyxus mesh A | PyRadiomics A | Nyxus mesh closed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| single voxel | 1 | 0.166667 | 0.166667 | 0.166667 | 0.166667 | +0.00% | 0.166667 | 1.732051 | 1.732051 | yes |
| box 3x4x5 | 60 | 54.666667 | 54.666667 | 54.666667 | 54.666667 | +0.00% | 54.666667 | 79.187895 | 79.187895 | yes |
| ball r=6 | 925 | 911.500000 | 911.500000 | 911.500000 | 911.500000 | +0.00% | 911.500000 | 499.688711 | 499.688711 | yes |
| ellipsoid 8/5/3 | 477 | 465.166667 | 465.166667 | 465.166667 | 465.166667 | +0.00% | 465.166667 | 357.821622 | 357.821622 | yes |
| two voxels, edge contact | 2 | 0.333333 | 0.333333 | 0.333333 | 0.333333 | +0.00% | 0.333333 | 3.464102 | 3.464102 | yes |
| two voxels, corner contact | 2 | 0.333333 | 0.333333 | 0.333333 | 0.333333 | +0.00% | 0.333333 | 3.464102 | 3.464102 | yes |
| pitted ball r=10 | 4018 | 3994.625000 | 3994.375000 to 3995.666667 | 3977.708333 | 3961.708333 to 3977.708333 | -0.82% to -0.42% | 3984.708333 | 1535.250855 | 1514.763251 | yes |
| random 30% fill, 8^3 | 164 | 92.208333 | 91.750000 to 92.208333 | 96.833333 | 93.250000 to 102.500000 | +1.63% to +11.21% | 78.166667 | 385.967632 | 419.125472 | yes |
| random 50% fill, 8^3 | 249 | 198.208333 | 198.000000 to 199.291667 | 203.166667 | 203.166667 to 220.416667 | +2.50% to +11.30% | 185.666667 | 562.485921 | 570.553568 | yes |
| random 70% fill, 8^3 | 361 | 346.791667 | 346.791667 to 348.083333 | 364.416667 | 355.416667 to 364.416667 | +2.25% to +5.08% | 376.083333 | 594.229507 | 562.946064 | yes |

The ellipsoid has semi-axes 8, 5 and 3. The pitted ball is the r=10 digital ball (4169 voxels) with
each voxel at 9 < r ≤ 10 removed with probability 0.15; that removed 151 of the 1098 such voxels.

**Masks without ambiguous faces agree.** The four solid shapes cross no ambiguous face. Their volumes
agree bit for bit in all four orientations, and their areas agree to 1e-14 relative, despite the
different diagonals. The two-voxel masks agree too. The edge-contact pair crosses one ambiguous face;
PyRadiomics serves both cubes sharing it without complementing, so it separates the corners just as
Nyxus does. The corner-contact pair touches only across a cube's body diagonal and has no ambiguous
face.

**Masks with ambiguous faces differ, for three reasons.** Every cube whose triangles differ between
the two has an ambiguous face.

- Where both cubes sharing an ambiguous face are complemented, PyRadiomics joins the inside corners
  and Nyxus separates them. Both surfaces are closed there; they enclose different volumes.
- Where one of the two is complemented and the other is not, they cut the face differently and
  PyRadiomics' surface has a hole.
- In a cube PyRadiomics serves without complementing, both separate the corners of the ambiguous
  face, but the triangles inside the cube still differ. This is the smallest share.

The script does not separate the three; the per-cube breakdown below was a one-off check.

**PyRadiomics' open surface makes its volume depend on orientation.** With holes, the volume depends
on the integration reference point. Through the extractor that point is fixed relative to the ROI,
so moving the ROI within the image does not change the reported value. Mirroring the image does:
across the four orientations PyRadiomics' volume spreads by 0.40% of its as-built value on the
pitted ball and by up to 9.6% on the noise masks. The "moved" column shows the reference-point
effect directly. Placed 7 voxels further out, the pitted ball's volume changes by 7.0 voxel³, and
the noise masks' by up to 19%.

**Nyxus' volume also depends on orientation, much less.** Its surface is closed on every mask, so
the reference point does not matter and translating the ROI cannot change the volume. Mirroring
can, because the table is not mirror-symmetric for every mask with an ambiguous face. For 86 of the
360 pairs of such a mask and a mirror axis, the triangles the table gives the mirrored mask are not
the mirror image of the triangles it gives the mask. For every mask without an ambiguous face they
are. Across the four orientations the Nyxus volume spreads by up to 0.65% of its as-built value, on
the 50% noise mask. On the solid shapes it does not move.

**How the causes were established.** These checks were run once outside the script, which cannot do
them: the pip package ships `cShape` compiled, without the C source that holds its tables.

- Rebuilding PyRadiomics' triangles from the `gridAngles`, `triTable` and `vertList` tables in
  3.0.1's `radiomics/src/cshape.c` reproduces `cShape`'s volume exactly on all ten masks.
- The rebuilt mesh is closed on the four solid shapes and the two two-voxel masks.
- On the pitted ball and the 30%, 50% and 70% noise masks, 184, 184, 172 and 240 directed edges have
  no reverse. That is 4 for each of the 46, 46, 43 and 60 ambiguous faces shared by a complemented
  cube and a plain one.
- A further 17, 21, 26 and 24 ambiguous faces are shared by two complemented cubes.
- Split by cube at PyRadiomics' reference point, the cubes PyRadiomics serves without complementing
  account for 0.33–0.75 voxel³ of gaps of 4.6–17.6 voxel³. The complemented cubes account for the
  rest.
- The mirror-symmetry count comes from comparing, for each of the 254 non-trivial masks and each
  axis, the table's triangles for the mirrored mask with the mirrored triangles of the mask.

"Closed" in the table means every directed edge of the Nyxus mesh is traversed as often in reverse.
Where two sheets of the surface touch along a lattice edge, four triangles share that edge, two in
each direction. In the Nyxus mesh this happens on 2 edges of the 50% noise mask and 12 of the 70% one,
and leaves the surface closed; every other edge of every Nyxus mesh is shared by exactly two
triangles.

## Performance

Measured in one session with a standalone harness that links `src/nyx/features/3d_mesh.cpp` and
PyRadiomics 3.0.1's `cshape.c`, both built with MSVC 2022 `/O2`, on one Windows 11 workstation. The
harness is not in the tree. The inputs are digital balls. Each number is the best of 3–10 runs,
except PyRadiomics as called by `shape.py` at r=40, which ran once. PyRadiomics' C routine always
computes the mesh diameters as well, so it was timed with and without that step. These numbers are
not in a test and will drift with hardware and compiler.

| ball | voxels | Nyxus volume only | Nyxus volume + hull points | PyRadiomics mesh only | PyRadiomics as called by `shape.py` |
|---|---:|---:|---:|---:|---:|
| r=10 | 4,169 | 0.12 ms | 0.64 ms | 0.10 ms | 3.7 ms |
| r=40 | 267,761 | 4.2 ms | 12.4 ms | 3.6 ms | 760 ms |
| r=100 | 4,187,857 | 61 ms | 135 ms | 50 ms | not run |

- The mesh passes are within about 20% of each other, PyRadiomics' being the faster. Neither was
  profiled. They differ in structure: PyRadiomics reads a dense, padded mask; Nyxus starts from the
  voxel list, keeps two z-planes in memory and hands each triangle to a callback, so that the same
  walker serves in-memory and out-of-core ROIs.
- `roi_mesh_volume()` with hull points, the call `3d_surface.cpp` makes, costs 2.2–5.3× the volume
  alone over balls of r=10 to 100. The extra work is collecting the two ends of each lattice row of
  vertices, which `RowEnds` keeps in a `std::map`.
- PyRadiomics' diameter step compares every pair of surface vertices, so its cost grows with the
  square of the surface size. With it, the call takes 37× the mesh pass at r=10 and 214× at r=40.
