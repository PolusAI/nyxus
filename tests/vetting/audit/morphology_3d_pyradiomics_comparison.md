# 3D mesh volume vs PyRadiomics — comparison report

`3MESH_VOLUME` and PyRadiomics `MeshVolume` agree exactly on every solid shape tested. They differ
where a face of the lattice is ambiguous (two diagonal corners inside, two outside) and PyRadiomics
cuts it differently in the two cubes that share it. There, PyRadiomics' mesh is open, and its volume
depends on where the ROI sits in the image. The Nyxus mesh is closed on every mask tested, and its
volume does not depend on placement. PyRadiomics is therefore a valid second oracle for
`3MESH_VOLUME` on solid shapes, at unit spacing. No assertion in the tree pins a PyRadiomics value.

## Tool and configuration

| | |
|---|---|
| Tool | PyRadiomics 3.0.1 (SimpleITK 2.3.1, NumPy 1.23.5, Python 3.8.20) |
| PyRadiomics call | `radiomics.cShape.calculate_coefficients`, the C routine `shape.py` calls, on a `[z, y, x]` mask padded by one voxel, unit spacing |
| Nyxus side | `MC_TRIANGLES` read from `src/nyx/features/3d_mesh.cpp`, walked as `march_roi_surface()` walks it and integrated from the first vertex as `roi_mesh_volume()` does |
| Script | `tests/vetting/audit/compare_mesh_volume_pyradiomics.py` |

```
python tests/vetting/audit/compare_mesh_volume_pyradiomics.py
```

The script reads the shipped table, so it always checks the table in the tree, but it does not run
the Nyxus binary. It runs PyRadiomics at unit spacing, so both sides report volume in voxel units.
PyRadiomics is a reference only, never a build or CI dependency (BSD-3-Clause; SPEC 6.4).

## What both implementations share

- Marching cubes over the binary mask at the 0.5 isolevel, with one voxel of empty padding, so the
  surface also closes where the ROI touches its bounding box.
- Every vertex at the midpoint of a cube edge; a binary mask needs no interpolation.
- Volume by the divergence theorem: the signed volumes of the tetrahedra each triangle spans with a
  reference point, summed and divided by 6.

PyRadiomics also integrates the mesh area, as half the norm of each triangle's cross product. Nyxus
computes no mesh area: `3AREA` counts exposed voxel faces. The script computes the same mesh area on
the Nyxus triangles only to compare the two meshes.

## Where they differ

| aspect | PyRadiomics 3.0.1 | Nyxus | consequence |
|---|---|---|---|
| case table | classic 128-entry table; a cube whose eighth corner is inside is served by complementing its mask and flipping the volume's sign | all 256 masks, derived by `derive_marching_cubes_table.py`; ambiguous faces always separate the inside corners | PyRadiomics' surface is open on an ambiguous face shared by a complemented cube and a plain one, which cut it differently; elsewhere the triangles are the same |
| volume reference point | the image origin, after padding | the first vertex of the surface | an open mesh makes PyRadiomics' volume depend on placement; Nyxus terms are exact products of half-integer coordinates and do not depend on placement |
| voxel spacing | each vertex scaled by the physical spacing; volume in physical units (mm³) | never physical units: anisotropic data is resampled by the spacing ratios, smallest axis 1, and the volume is in units of that voxel | equal only at unit spacing; at an isotropic spacing *s*, PyRadiomics' volume is *s*³ times Nyxus' |
| convex-hull volume | no 3D hull feature | `3VOLUME_CONVEXHULL`, the hull of the mesh vertices | no PyRadiomics counterpart |
| surface area and its ratios | `SurfaceArea` is the mesh area; `Sphericity`, `SurfaceVolumeRatio` and the deprecated `Compactness1`, `Compactness2`, `SphericalDisproportion` use mesh area and mesh volume | `3AREA` counts exposed voxel faces; `3AREA_2_VOLUME`, `3COMPACTNESS1/2`, `3SPHERICAL_DISPROPORTION`, `3SPHERICITY` use `3AREA` and `3VOXEL_VOLUME` | different definitions, not comparable |

## Result

Volumes (V) and areas (A) in voxel units. "Moved" is the same mask placed 7 voxels further from the
image origin along each axis. The mesh area is listed for both sides because the script computes it,
but the Nyxus `3AREA` feature is the face count, not this mesh area.

| mask | voxels | Nyxus V | Nyxus V moved | PyRadiomics V | PyRadiomics V moved | Nyxus mesh A | PyRadiomics A | Nyxus mesh closed |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| single voxel | 1 | 0.166667 | 0.166667 | 0.166667 | 0.166667 | 1.732051 | 1.732051 | yes |
| box 3×4×5 | 60 | 54.666667 | 54.666667 | 54.666667 | 54.666667 | 79.187895 | 79.187895 | yes |
| ball r=6 | 925 | 911.500000 | 911.500000 | 911.500000 | 911.500000 | 499.688711 | 499.688711 | yes |
| ellipsoid, semi-axes 8/5/3 | 477 | 465.166667 | 465.166667 | 465.166667 | 465.166667 | 357.821622 | 357.821622 | yes |
| two voxels, edge contact | 2 | 0.333333 | 0.333333 | 0.333333 | 0.333333 | 3.464102 | 3.464102 | yes |
| two voxels, corner contact | 2 | 0.333333 | 0.333333 | 0.333333 | 0.333333 | 3.464102 | 3.464102 | yes |
| ball r=10, 15% of surface voxels removed | 4018 | 3994.625000 | 3994.625000 | 3977.708333 | 3984.708333 | 1535.250855 | 1514.763251 | yes |
| random 30% fill, 8³ | 164 | 92.208333 | 92.208333 | 96.833333 | 78.166667 | 385.967632 | 419.125472 | yes |
| random 50% fill, 8³ | 249 | 198.208333 | 198.208333 | 203.166667 | 185.666667 | 562.485921 | 570.553568 | yes |
| random 70% fill, 8³ | 361 | 346.791667 | 346.791667 | 364.416667 | 376.083333 | 594.229507 | 562.946064 | yes |

**The solid shapes tested agree exactly**, area as well as volume. The two-voxel masks agree too. The
edge-contact pair has an ambiguous face, but both cubes that share it hold the same two corners and
cut it the same way. The corner-contact pair touches only across a cube's body diagonal and has no
ambiguous face.

**Ambiguous faces split the two.** On the pitted ball PyRadiomics sits 0.42% below Nyxus, and its
volume changes by 7.0 voxel³ when the ROI moves. On the noise masks the gap is up to 5.1% in place
and up to 15% after the move, on either side of the Nyxus value. The Nyxus volume is the same in both
placements on every mask.

**PyRadiomics' mesh is open, and that is why.** This was checked once outside the script, which
cannot do it: the pip package ships `cShape` compiled, without its tables. Rebuilding PyRadiomics'
triangles from the `gridAngles`, `triTable` and `vertList` tables in 3.0.1's `radiomics/src/cshape.c`
reproduces `cShape`'s volume exactly on all ten masks. The rebuilt mesh is closed on the four solid
shapes and the two two-voxel masks. On the pitted ball and the 30%, 50% and 70% noise masks, 184, 184,
172 and 240 directed edges have no reverse. Every one of them lies on an ambiguous face shared by a
complemented cube and a plain one.

"Closed" means every directed edge of the mesh is traversed as often in reverse. Where two sheets of
the surface touch along a lattice edge, four triangles share that edge, two in each direction. This
happens a few times in the 50% and 70% noise masks and leaves the surface closed.

## Performance

Measured in one session with a standalone harness that links `src/nyx/features/3d_mesh.cpp` and
PyRadiomics 3.0.1's `cshape.c`, both built with MSVC 2022 `/O2`, on one Windows 11 workstation. The
inputs are digital balls. Each number is the best of 3–10 runs, except PyRadiomics as called by
`shape.py` at r=40, which ran once. PyRadiomics' C routine always computes the mesh diameters as
well, so it was timed with and without that step. These numbers are not in a test and will drift
with hardware and compiler.

| ball | voxels | Nyxus volume only | Nyxus volume + hull points | PyRadiomics mesh only | PyRadiomics as called by `shape.py` |
|---|---:|---:|---:|---:|---:|
| r=10 | 4,169 | 0.12 ms | 0.64 ms | 0.10 ms | 3.7 ms |
| r=40 | 267,761 | 4.2 ms | 12.4 ms | 3.6 ms | 760 ms |
| r=100 | 4,187,857 | 61 ms | 135 ms | 50 ms | not run |

- The mesh passes are within about 20% of each other, PyRadiomics' being the faster. PyRadiomics
  reads a dense, padded mask; Nyxus starts from the voxel list, keeps two z-planes in memory and
  hands each triangle to a callback. The same walker then serves in-memory and out-of-core ROIs.
- `roi_mesh_volume()` with hull points, the call `3d_surface.cpp` makes, costs 2.2–5.3× the volume
  alone over balls of r=10 to 100. The extra work is collecting the two ends of each lattice row of
  vertices, which `RowEnds` keeps in a `std::map`.
- PyRadiomics' diameter step compares every pair of surface vertices, so its cost grows with the
  square of the surface size. It is 37× the mesh pass at r=10 and 210× at r=40.
