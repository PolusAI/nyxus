3D shape features
=================

3D shape features describe the size and form of a ROI in a volume: its surface, its volume
measured three ways, the volume of its convex hull, and the lengths of its principal axes. They use
only the ROI's mask; voxel intensities play no part.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - 14 codes, one value and one column each: ``3AREA``, ``3MESH_VOLUME``, ``3VOXEL_VOLUME``,
       ``3VOLUME_CONVEXHULL``, the five ratios ``3AREA_2_VOLUME``, ``3COMPACTNESS1``,
       ``3COMPACTNESS2``, ``3SPHERICAL_DISPROPORTION`` and ``3SPHERICITY``, the three axis lengths
       ``3MAJOR_AXIS_LEN``, ``3MINOR_AXIS_LEN`` and ``3LEAST_AXIS_LEN``, and ``3ELONGATION`` and
       ``3FLATNESS``.
   * - **Group**
     - ``*3D_ALL_MORPHOLOGY*``; ``*3D_ALL*`` includes it.
   * - **Dimensions**
     - 3D only. The 2D shape features are on the :doc:`morphology page <f_morphology>`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes. Out of core the volume is streamed one z-plane at a time and gives the same values as
       in memory; the axis features can differ in the last digits, because out of core the
       covariance is formed from running sums of the coordinates and their products rather than
       from coordinates centred on their mean. When one read of the file returns the whole volume
       (NIfTI, or OME-Zarr chunked over all of Z), an oversized ROI stops the run with an error
       instead (see :doc:`/large_images`).
   * - **IBSI**
     - ``3AREA`` (C0JK), ``3MESH_VOLUME`` (RNU0), ``3VOXEL_VOLUME`` (YEKZ), ``3AREA_2_VOLUME``
       (2PR5), ``3COMPACTNESS1`` (SKGS), ``3COMPACTNESS2`` (BQWJ), ``3SPHERICAL_DISPROPORTION``
       (KRCK), ``3SPHERICITY`` (QCFX), the three axis lengths (TDIC, P9VJ, 7J51),
       ``3ELONGATION`` (Q3CK) and ``3FLATNESS`` (N17B). ``3VOLUME_CONVEXHULL`` has no identifier
       of its own; IBSI uses it only inside the volume density (R3ER).

.. tip::

   Three different volumes are on offer. ``3MESH_VOLUME`` is IBSI's volume (mesh) and is the one
   to compare with MIRP or PyRadiomics. ``3VOXEL_VOLUME`` is essentially the voxel count.
   ``3AREA`` is the area of the same mesh, and the five ratios combine it with ``3MESH_VOLUME``,
   as IBSI defines them.


How the features are computed
-----------------------------

The features start from the ROI's voxels, the voxels of the mask that carry the ROI's label, at
integer positions :math:`(x_k, y_k, z_k)`: :math:`x` along the columns, :math:`y` down the rows
and :math:`z` across the slices. :math:`N` is their number. From them Nyxus builds:

- **The surface mesh:** the marching-cubes surface of the binary mask at the 0.5 isolevel, built on
  the ROI's bounding box padded by one empty voxel on every side, so the surface is closed. The
  mask is binary, so every mesh vertex sits halfway between an ROI voxel centre and a neighbouring
  non-ROI voxel centre: one of its three coordinates is a half-integer. ``3AREA`` is the area of
  this surface and ``3MESH_VOLUME`` the volume it encloses; the surface around a cavity inside the
  ROI counts towards both. The case table splits the two inside corners of an ambiguous cube face
  (two diagonal corners inside, two outside), so the surface stays closed in every case.
- **The convex hull:** the convex hull of the mesh vertices. ``3VOLUME_CONVEXHULL`` is its volume.
  The hull is computed in doubled, integer coordinates, so the volume is exact.
- **The covariance matrix:** the 3 × 3 sample covariance (divided by :math:`N-1`) of the voxel
  coordinates. Its eigenvalues give the axis lengths, ``3ELONGATION`` and ``3FLATNESS``.

The values are:

- **Units:** lengths in voxels, ``3AREA`` in squared voxels, volumes in cubic
  voxels. No pixel size is applied. With anisotropy settings the coordinates are scaled by the
  anisotropy, so the units are those of the scaled voxel (see `Settings`_).
- **Intensities:** not used. A ROI whose voxels are all 0, or all equal, gives the same values as
  one with varying intensities. ``--coarseGrayDepth`` and ``--ibsi`` change nothing.
- **Which volume where:** ``3AREA_2_VOLUME``, ``3COMPACTNESS1``, ``3COMPACTNESS2``,
  ``3SPHERICAL_DISPROPORTION`` and ``3SPHERICITY`` combine ``3AREA`` with ``3MESH_VOLUME``.
  ``3VOXEL_VOLUME`` and ``3VOLUME_CONVEXHULL`` enter no other feature.
- **One ROI per volume:** in whole-volume mode, and with ``single_roi=True``, Nyxus does not trace
  the ROI's shape: it takes the ROI to be a solid box and uses closed forms; the axis features are
  then 0. With ``single_roi=True`` this is a known defect (`#494 <https://github.com/PolusAI/nyxus/issues/494>`__); see
  :ref:`shape-3d-possible-issues`. In whole-volume mode, on the command line
  (``--segDir`` equal to ``--intDir``) or through ``Nyxus3D.featurize_directory`` with the
  intensity directory as the label directory, the box is one voxel larger than the volume on
  every axis, :math:`(w+1) \times (h+1) \times (d+1)`, a known defect (`#493 <https://github.com/PolusAI/nyxus/issues/493>`__). ``featurize_files(volumes, volumes,
  single_roi=True)`` is not whole-volume mode: it merges the nonzero voxels of each mask, here
  the volume itself, into one ROI, and the box is the bounding box of the volume's nonzero
  voxels.

On a mask with no ambiguous cube face (see `Possible issues`_),
``skimage.measure.marching_cubes`` at level 0.5 on the padded mask, followed by
``skimage.measure.mesh_surface_area`` and the divergence theorem, reproduces ``3AREA`` and
``3MESH_VOLUME`` to the float32 precision of its vertices, and
``scipy.spatial.ConvexHull`` on those vertices reproduces ``3VOLUME_CONVEXHULL`` to double
precision. Where a face is ambiguous, skimage's table splits it differently and its area and
volume differ from Nyxus'.


Worked example
^^^^^^^^^^^^^^

Take a ROI of 44 voxels on three slices, placed with a margin of one voxel in a 5 × 7 × 8 volume.
Slice by slice, rows :math:`y` and columns :math:`x`, with background ``.``:

.. code-block:: text

   z = 0 (21 voxels)    z = 1 (16 voxels)    z = 2 (7 voxels)
   .111..               ..11..               ......
   11111.               .1111.               ..11..
   111111               .11111               ..111.
   .11111               ..1111               ...11.
   ...11.               ....1.               ......

The ROI's 0.5-isolevel mesh has an area of 66.366 squared voxels, so ``3AREA`` is 66.366, and it
encloses :math:`475/12 = 39.583` cubic voxels; the convex hull of the mesh vertices has a volume
of :math:`122/3 = 40.667`. ``3VOXEL_VOLUME`` is :math:`44 \times 0.9999977 = 43.9999`. The ratios
follow from the mesh area and volume: ``3AREA_2_VOLUME`` is :math:`66.366/39.583 = 1.677` and
``3SPHERICITY`` is :math:`(36\pi \times 39.583^2)^{1/3}/66.366 = 0.8463`.

The covariance matrix of the voxel coordinates, in the order :math:`x, y, z`, is

.. math::

   \textbf{K} = \begin{bmatrix}
   1.7611 & 0.5137 & 0.1586\\
   0.5137 & 1.2474 & 0.0169\\
   0.1586 & 0.0169 & 0.5476 \end{bmatrix}

with eigenvalues :math:`\lambda_1 = 2.0920`, :math:`\lambda_2 = 0.94139` and
:math:`\lambda_3 = 0.52260`. The major axis points along :math:`(0.85, 0.52, 0.09)`, mostly along
:math:`x`; the least axis points almost along :math:`z`. So ``3MAJOR_AXIS_LEN`` is
:math:`4\sqrt{2.0920} = 5.786`, ``3ELONGATION`` is :math:`\sqrt{0.94139/2.0920} = 0.6708` and
``3FLATNESS`` is :math:`\sqrt{0.52260/2.0920} = 0.4998`.

The *Example* value of each feature in the :ref:`index <shape-3d-index>` is what Nyxus returns for
this ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/shape_3d_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --dim=3 --features="*3D_ALL_MORPHOLOGY*" \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N`
     - Number of ROI voxels.
   * - :math:`(x_k, y_k, z_k)`
     - Integer position of voxel :math:`k`; :math:`\bar x, \bar y, \bar z` are their means.
   * - :math:`A_m`
     - Area of the marching-cubes mesh, ``3AREA``.
   * - :math:`V_v`
     - Voxel-counting volume, ``3VOXEL_VOLUME``.
   * - :math:`V_m`
     - Volume enclosed by the marching-cubes mesh, ``3MESH_VOLUME``.
   * - :math:`T`
     - The mesh's triangles; a triangle has vertices :math:`\mathbf a, \mathbf b, \mathbf c`, and
       :math:`\mathbf o` is the first vertex of the first triangle.
   * - :math:`s_x, s_y, s_z`
     - The anisotropy factors, the voxel's size along each axis; 1 without anisotropy.
   * - :math:`w, h, d`
     - The width, height and depth of a box, in voxels along :math:`x`, :math:`y` and :math:`z`.
   * - :math:`\textbf{K}`
     - Sample covariance matrix of the voxel coordinates, divided by :math:`N-1`.
   * - :math:`\lambda_1 \ge \lambda_2 \ge \lambda_3`
     - Eigenvalues of :math:`\textbf{K}`, in descending order.


Settings
--------

The 3D shape features have no settings of their own. These options change their values:

- ``--anisox``, ``--anisoy`` and ``--anisoz`` (Python: ``anisotropy_x``, ``anisotropy_y``,
  ``anisotropy_z``) give the voxel's size along each axis. The volume is featurized on its voxels
  as acquired, and the shape features scale their coordinates by the anisotropy: with
  ``--anisoz=2`` the worked example's ``3VOXEL_VOLUME``, ``3MESH_VOLUME`` and
  ``3VOLUME_CONVEXHULL`` double, ``3AREA`` is the area of the mesh stretched along z, 103.438
  rather than twice 66.366, and the axis lengths are those of the coordinates with z doubled.
- ``--use-physical-spacing=true`` (Python: ``use_physical_spacing``) takes the factors from the
  volume's OME ``PhysicalSize`` values, scaled so that the smallest is 1. The values are then in
  units of the smallest spacing, not in physical units. Any ``--aniso*`` value other than 1
  replaces the physical spacing on all three axes; the axes not given then use 1.
- Whole-volume mode, ``--segDir`` equal to ``--intDir`` (Python: ``featurize_directory`` with
  the intensity directory as the label directory), treats each volume as one ROI and measures a
  box in its place; ``featurize_files(..., single_roi=True)`` measures the box of the nonzero
  voxels, a known defect (`#494 <https://github.com/PolusAI/nyxus/issues/494>`__). See `How the features are computed`_.
- ``--ramLimit`` (Python: ``ram_limit``) decides which ROIs are processed out of core; the values
  stay the same, except as noted in the *Out-of-core* row above.

``--coarseGrayDepth``, ``--ibsi`` and the intensity map options have no effect. All options are
listed in the :ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named with its code. There are no ``_AVE`` columns.


.. _shape-3d-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example.

.. include:: shape_3d_index.inc


Features
--------

.. _feature-3area:

Surface area
^^^^^^^^^^^^

:Code: ``3AREA``
:Range: positive; in squared voxels

The area of the ROI's marching-cubes surface at the 0.5 isolevel, the surface ``3MESH_VOLUME``
integrates: the sum of the areas of its triangles. Each triangle's area is taken on the voxel
grid, with its edges scaled by the anisotropy along :math:`x`, :math:`y` and :math:`z` before the
cross product.

.. math::

   A_m = \frac{1}{2} \sum_{(\mathbf a, \mathbf b, \mathbf c) \in T}
   \big|\, S(\mathbf b - \mathbf a) \times S(\mathbf c - \mathbf a) \,\big|,
   \qquad S = \mathrm{diag}(s_x, s_y, s_z)

Under anisotropy the area is therefore not the unit-spacing area times one factor, unlike the
volumes, which are the unit-spacing volumes times :math:`s_x s_y s_z`.

**Relations.** A single voxel gives the octahedron of area :math:`\sqrt 3`, or
:math:`\sqrt{s_x^2 s_y^2 + s_y^2 s_z^2 + s_x^2 s_z^2}` with anisotropy. A solid
:math:`w \times h \times d` box gives the area of the box with its edges and corners bevelled,

.. math::

   A_m = 2\big[(h-1)(d-1)\, s_y s_z + (w-1)(d-1)\, s_x s_z + (w-1)(h-1)\, s_x s_y\big]
   + 2\big[(w-1)\, s_x \sqrt{s_y^2+s_z^2} + (h-1)\, s_y \sqrt{s_x^2+s_z^2}
   + (d-1)\, s_z \sqrt{s_x^2+s_y^2}\big] + \sqrt{s_x^2 s_y^2 + s_y^2 s_z^2 + s_x^2 s_z^2}

which on unit voxels is :math:`2(wh + hd + wd) - 2(w+h+d-3)(2-\sqrt 2) - (6-\sqrt 3)`: 129.673
for a 6 × 4 × 5 box. A cavity adds the area of its own part of the surface: a solid 5 × 5 × 5
cube gives 131.673, and the same cube with its centre voxel removed :math:`131.673 + \sqrt 3 =
133.405`.

**Possible issues.** The mesh is a faceted surface, and its area stays above that of the smooth
surface it approximates however large the ROI: see ``3SPHERICITY``.


.. _feature-3mesh-volume:

Volume (mesh)
^^^^^^^^^^^^^

:Code: ``3MESH_VOLUME``
:Range: positive; in cubic voxels

The volume enclosed by the ROI's marching-cubes surface at the 0.5 isolevel, by the divergence
theorem: the sum of the signed volumes of the tetrahedra each triangle spans with a fixed point
:math:`\mathbf o` on the surface.

.. math::

   V_m = \frac{1}{6} \left| \sum_{(\mathbf a, \mathbf b, \mathbf c) \in T}
   (\mathbf a - \mathbf o) \cdot \big[(\mathbf b - \mathbf o) \times (\mathbf c - \mathbf o)\big]
   \right|

**Relations.** A single voxel gives the octahedron of volume :math:`1/6`. A solid
:math:`w \times h \times d` box gives :math:`whd - (w+h+d-3)/2 - 5/6`, the box with its edges and
corners bevelled. A cavity is enclosed by its own part of the surface and is subtracted: a
solid 5 × 5 × 5 cube gives 118.167, and the same cube with its centre voxel removed 118.0.


.. _feature-3voxel-volume:

Volume (voxel counting)
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3VOXEL_VOLUME``
:Range: positive; in cubic voxels

The volume from the voxel count. The code multiplies the count by the volume of a ball of diameter
1 voxel and divides by a packing density written as 0.5236, a truncation of :math:`\pi/6`, so the
result is the voxel count times a constant slightly below 1.

.. math::

   V_v = N \, \frac{\tfrac{4}{3}\pi \left(\tfrac{1}{2}\right)^3}{0.5236}
   = N \, \frac{\pi/6}{0.5236} = 0.99999766\, N

**Possible issues.** The factor 0.9999976616 makes the value :math:`2.3 \times 10^{-6}` relative
below the voxel count, for example 43.99990 for 44 voxels. In whole-volume mode the value is the
exact volume of the box Nyxus substitutes, without the factor.


.. _feature-3volume-convexhull:

Convex hull volume
^^^^^^^^^^^^^^^^^^

:Code: ``3VOLUME_CONVEXHULL``
:Range: at least ``3MESH_VOLUME``; in cubic voxels

The volume of the convex hull of the vertices of the ROI's marching-cubes mesh. Every nonempty ROI
has a hull of positive volume: even a single voxel or a one-slice ROI gives the hull of its
bevelled surface.

.. math::

   V_{\mathrm{hull}} = \mathrm{vol}\big(\mathrm{conv}\{\text{mesh vertices}\}\big)

**Relations.** A convex digital solid, such as a box or a single voxel, gives
``3VOLUME_CONVEXHULL`` equal to ``3MESH_VOLUME``. IBSI's volume density (convex hull), R3ER, is
:math:`V_m / V_{\mathrm{hull}}`; Nyxus does not report it, but it can be computed from the two
columns.


.. _feature-3area-2-volume:

Surface to volume ratio
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3AREA_2_VOLUME``
:Range: positive; in inverse voxels

.. math::

   \mathrm{3AREA\_2\_VOLUME} = \frac{A_m}{V_m}

The ratio depends on the ROI's size: it falls as the ROI grows, 1.677 on the worked example and
0.331 on a digital ball of radius 10.


.. _feature-3compactness1:

Compactness 1
^^^^^^^^^^^^^

:Code: ``3COMPACTNESS1``
:Range: positive, dimensionless, at most :math:`1/(6\pi) = 0.0531`, the value of a sphere

.. math::

   \mathrm{3COMPACTNESS1} = \frac{V_m}{\sqrt{\pi A_m^{3}}}

A solid 5 × 5 × 5 cube gives 0.0441 and a digital ball of radius 10 gives 0.0460.


.. _feature-3compactness2:

Compactness 2
^^^^^^^^^^^^^

:Code: ``3COMPACTNESS2``
:Range: 0 to 1, dimensionless; 1 for a sphere

.. math::

   \mathrm{3COMPACTNESS2} = 36\pi \, \frac{V_m^{2}}{A_m^{3}}

A solid 5 × 5 × 5 cube gives 0.6918 and a digital ball of radius 10 gives 0.7533.


.. _feature-3spherical-disproportion:

Spherical disproportion
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3SPHERICAL_DISPROPORTION``
:Range: at least 1, dimensionless; 1 for a sphere

The mesh area divided by the surface area of a sphere with volume :math:`V_m`.

.. math::

   \mathrm{3SPHERICAL\_DISPROPORTION} = \frac{A_m}{\left(36\pi V_m^{2}\right)^{1/3}}

**Relations.** ``3SPHERICAL_DISPROPORTION`` is :math:`1/` ``3SPHERICITY``. A solid 5 × 5 × 5
cube gives 1.131 and a digital ball of radius 10 gives 1.099.


.. _feature-3sphericity:

Sphericity
^^^^^^^^^^

:Code: ``3SPHERICITY``
:Range: 0 to 1, dimensionless; 1 for a sphere

The surface area of a sphere with volume :math:`V_m`, divided by the mesh area.

.. math::

   \mathrm{3SPHERICITY} = \frac{\left(36\pi V_m^{2}\right)^{1/3}}{A_m}

**Possible issues.** A digital ball does not reach 1: a ball of radius 10 gives 0.9099, against
0.8844 for a solid 5 × 5 × 5 cube, and on balls of radius 5 to 30 the value stays between 0.90
and 0.93. The facets of the mesh keep its area above that of the sphere as the ball grows.


.. _feature-3major-axis-len:
.. _feature-3minor-axis-len:
.. _feature-3least-axis-len:

Major, minor and least axis lengths
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``3MAJOR_AXIS_LEN``, ``3MINOR_AXIS_LEN``, ``3LEAST_AXIS_LEN``
:Range: non-negative, ``3MAJOR_AXIS_LEN`` :math:`\ge` ``3MINOR_AXIS_LEN`` :math:`\ge`
        ``3LEAST_AXIS_LEN``; in voxels

Four times the square root of the largest, middle and smallest eigenvalue of the covariance
matrix of the voxel coordinates: the full axis lengths of the ellipsoid with the same second
moments as the ROI. The axes are the eigenvectors of :math:`\textbf{K}`, not the image axes; the
major axis is the direction in which the ROI is most spread out.

.. math::

   K_{ij} = \frac{1}{N-1} \sum_{k=1}^{N} (u_{k,i} - \bar u_i)(u_{k,j} - \bar u_j),
   \qquad \mathbf u_k = (x_k, y_k, z_k)

.. math::

   \mathrm{3MAJOR\_AXIS\_LEN} = 4\sqrt{\lambda_1}, \qquad
   \mathrm{3MINOR\_AXIS\_LEN} = 4\sqrt{\lambda_2}, \qquad
   \mathrm{3LEAST\_AXIS\_LEN} = 4\sqrt{\lambda_3}

All three are 0 for a single voxel, and ``3LEAST_AXIS_LEN`` is 0 for a ROI in one slice. In
whole-volume mode all three are 0 whatever the volume (see :ref:`shape-3d-possible-issues`).


.. _feature-3elongation:
.. _feature-3flatness:

Elongation and flatness
^^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``3ELONGATION``, ``3FLATNESS``
:Range: 0 to 1, dimensionless

The minor and the least axis length relative to the major axis length. 1 means the ROI is spread
equally in those directions, as a ball is; values near 0 mean a needle (elongation) or a plate
(flatness).

.. math::

   \mathrm{3ELONGATION} = \sqrt{\lambda_2 / \lambda_1}, \qquad
   \mathrm{3FLATNESS} = \sqrt{\lambda_3 / \lambda_1}

Both are 0 for a single voxel and in whole-volume mode, and ``3FLATNESS`` is 0 for a ROI in one
slice.


.. _shape-3d-possible-issues:

Possible issues
---------------

The mesh area does not converge to the smooth surface's area
   ``3AREA`` is the area of a faceted surface, and refining the ROI does not bring it down to the
   area of the smooth surface the ROI samples. A digital ball therefore keeps a ``3SPHERICITY``
   below 1, between 0.90 and 0.93 for radii 5 to 30, and the other four ratios do not reach their
   sphere values either. IBSI, MIRP and PyRadiomics compute the same mesh area and share this.

``3VOXEL_VOLUME`` is scaled by 0.9999977
   The packing density 0.5236 truncates :math:`\pi/6 = 0.5235988`, so the value is the voxel
   count times 0.9999976616. No other feature uses it.

Whole-volume mode differs from a segmented ROI that fills the volume
   With no mask, the features come from closed forms of a box (see `How the features are
   computed`_ for which box). ``3MAJOR_AXIS_LEN``, ``3MINOR_AXIS_LEN``, ``3LEAST_AXIS_LEN``,
   ``3ELONGATION`` and ``3FLATNESS`` are 0, and ``3VOXEL_VOLUME`` is the exact box volume, without
   the packing factor. A 6 × 4 × 5 volume with no zero voxels gives 0 and 120 through
   ``featurize_files(..., single_roi=True)``, 0 and 210 (a 7 × 5 × 6 box, a known defect, `#493 <https://github.com/PolusAI/nyxus/issues/493>`__) on the command line or
   through ``featurize_directory``, and 6.860 for the major axis and 119.99972 for the volume when
   a mask marks the same box as a ROI. ``3AREA``, ``3MESH_VOLUME``, ``3VOLUME_CONVEXHULL`` and the
   five ratios of the ``featurize_files`` run agree with the masked run; on the command line they
   are those of the larger box (``3AREA`` 192.158 instead of 129.673). The box values come from
   the closed forms given under ``3AREA`` and ``3MESH_VOLUME``. Through ``featurize_files`` zero
   voxels are left out: a 3 × 2 × 2 block of nonzero voxels in the same volume gives 12 and a
   ``3AREA`` of 23.046.

``single_roi=True`` measures the bounding box, not the mask
   With ``featurize_files(..., single_roi=True)`` every shape feature describes the solid
   bounding box of the merged mask, and the axis features are 0, whatever the mask's shape. This
   is a known defect (`#494 <https://github.com/PolusAI/nyxus/issues/494>`__). A ball of radius 5 (515 voxels) in a 15 × 15 × 15 volume
   gives ``3VOXEL_VOLUME`` 514.999 and ``3AREA`` 340.08 as a segmented ROI, and 1331 and 686.58, the
   values of its 11 × 11 × 11 box, with ``single_roi=True``; ``3MAJOR_AXIS_LEN`` is 8.909 and 0.
   The values are right only for a mask that fills its box.

Degenerate ROIs give 0, not NaN
   A single voxel reports 0 for the three axis lengths, ``3ELONGATION`` and ``3FLATNESS``. A ROI
   in one slice reports 0 for ``3LEAST_AXIS_LEN`` and ``3FLATNESS``.

Ambiguous cube faces
   Where a cube face of the marching-cubes grid has two diagonal corners inside the ROI and two
   outside, the case table separates the inside corners. This keeps the surface closed, but it
   departs from the classic marching-cubes table, and these cases are not covered by a reference
   tool (see `Validation`_). Solid, smooth ROIs rarely produce them; the worked example has none.


Validation
----------

The test suite checks the 3D shape features against four references:

- **MIRP** 2.6.0, on a segmented phantom: ``3AREA``, ``3MESH_VOLUME``, ``3VOLUME_CONVEXHULL``,
  ``3AREA_2_VOLUME``, ``3COMPACTNESS1``, ``3COMPACTNESS2``, ``3SPHERICAL_DISPROPORTION`` and
  ``3SPHERICITY`` to a relative 10\ :sup:`-6` (measured 2.7 × 10\ :sup:`-8` to 2.4 × 10\ :sup:`-7`,
  ``3AREA`` 46739.0225 against MIRP's 46739.0195, MIRP carrying the mesh in float32);
  ``3VOXEL_VOLUME`` to a relative 10\ :sup:`-3`
  (measured 2.3 × 10\ :sup:`-6`, the packing factor); ``3MAJOR_AXIS_LEN``, ``3MINOR_AXIS_LEN``,
  ``3LEAST_AXIS_LEN``, ``3ELONGATION`` and ``3FLATNESS`` to a relative 10\ :sup:`-9`.
- **PyRadiomics** 3.0.1, on a phantom with anisotropic voxel spacing (1.3, 0.8, 2.5) given as
  ``--aniso*``: ``3AREA`` (``SurfaceArea``), ``3AREA_2_VOLUME`` (``SurfaceVolumeRatio``),
  ``3SPHERICITY`` (``Sphericity``), ``3MESH_VOLUME``, ``3MAJOR_AXIS_LEN``, ``3MINOR_AXIS_LEN``,
  ``3LEAST_AXIS_LEN``, ``3ELONGATION`` and ``3FLATNESS`` to a relative 10\ :sup:`-9`.
  PyRadiomics divides the covariance by :math:`N` where Nyxus divides by :math:`N - 1`, so its
  axis lengths are multiplied by :math:`\sqrt{N/(N-1)}` before the comparison.
- **MATLAB** (R2026a, ``regionprops3``), on the same phantom: ``3VOXEL_VOLUME`` to a relative
  10\ :sup:`-3`, and ``3VOLUME_CONVEXHULL`` within 0.5 % of ``ConvexVolume`` (measured 0.17 %),
  which counts the voxels of the rasterised hull rather than integrating it.
- **Analytic** values on lattice solids: ``3MESH_VOLUME`` on a single voxel and boxes,
  ``3VOLUME_CONVEXHULL`` also on octahedra and rhombic prisms, ``3AREA`` and the five ratios on
  the single-voxel octahedron and on bevelled boxes with unit spacing and with spacings
  (1.3, 0.8, 2.5) and (0.5, 2, 1), all also in whole-volume mode, to a relative 10\ :sup:`-12`.

The mesh rows hold for ROIs without ambiguous cube faces; the phantom has none. Out-of-core
results are checked against in-memory results of the same build. The Nyxus values quoted on this
page (the worked example, the single-voxel, one-slice, ball, box, cube, whole-volume and
anisotropy values) are pinned by the worked-example script, except the sphericities of the balls
of radius 5, 20 and 30. The covariance entries, the major-axis direction and the skimage and
scipy reproduction are not pinned.


References
----------

1. Zwanenburg A, Vallières M, et al. *The Image Biomarker Standardization Initiative: standardized
   quantitative radiomics for high-throughput image-based phenotyping.* Radiology.
   2020;295(2):328–338. doi:10.1148/radiol.2020191145
2. Lorensen WE, Cline HE. *Marching cubes: a high resolution 3D surface construction algorithm.*
   Computer Graphics (SIGGRAPH '87). 1987;21(4):163–169. doi:10.1145/37402.37422
3. Barber CB, Dobkin DP, Huhdanpaa H. *The quickhull algorithm for convex hulls.* ACM Transactions
   on Mathematical Software. 1996;22(4):469–483. doi:10.1145/235815.235821
4. van der Walt S, et al. *scikit-image: image processing in Python.* PeerJ. 2014;2:e453.
