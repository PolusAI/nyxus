3D grey-level size-zone matrix (3D GLSZM)
=========================================

3D GLSZM features describe the zones of a 3D ROI: groups of connected voxels that share one grey
level. The matrix counts zones by grey level and by size in voxels, regardless of direction. The
features are those of the :doc:`2D GLSZM <f_glszm>`, computed on the 3D matrix. This page covers
what is specific to 3D: how zones are connected, how grey levels are set, and which voxels take
part.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``3GLSZM_*``: 16 features, one value per ROI
   * - **Group**
     - ``*3D_GLSZM*``; also part of ``*3D_ALL_TEXTURE*`` and ``*3D_ALL*``
   * - **Dimensions**
     - 3D. The 2D family, ``GLSZM_*``, is described in :doc:`f_glszm`.
   * - **GPU**
     - No. The features are computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory.
       When one read of the file returns the whole volume (NIfTI, or OME-Zarr chunked over all of
       Z), an oversized ROI stops the run with an error (see :doc:`/large_images`).
   * - **IBSI**
     - All 16 features have an IBSI identifier. Zones are 26-connected, as IBSI defines them in 3D,
       and the matrix is the single 3D matrix of IBSI aggregation method KOBO.

.. tip::

   To reproduce IBSI values, run in IBSI mode or with ``--coarseGrayDepth=0``: each integer
   intensity is then a grey level. By default the family bins into 64 MATLAB-style levels, like
   every texture family; the ``3glszm/greydepth`` metaparameter, from Python only, overrides the
   depth for this family (see `Settings`_).


How the matrix is built
-----------------------

Nyxus takes the ROI's bounding box, bins the ROI's voxels and gives the voxels outside the mask
level 0, the background level. It then visits the voxels slice by slice, row by row. Each voxel
that is not yet in a zone and is not at level 0 starts a new zone. The zone grows through every voxel of the same
grey level that touches a voxel already in it, by a face, an edge or a corner.
:math:`\textbf{P}(i,j)` counts the zones of grey level :math:`i` and size :math:`j` voxels.

- **Connectivity:** 26-connected. All 26 neighbours of a voxel are searched: 8 in its own slice and
  9 in each adjacent slice. A zone is the full 26-connected component, whatever the scan order. This
  differs from the 2D family, whose zones grow only to the east, south-east, south and south-west.
- **Directions:** none. One matrix is built per ROI, and each feature has one value.
- **Grey levels:** set by ``--coarseGrayDepth`` (``coarse_gray_depth``, 64 by default), or by
  the ``3glszm/greydepth`` metaparameter, which overrides it for this family.

  - 0, and always in IBSI mode: no binning. A voxel of intensity :math:`x` has grey level
    :math:`x`, and :math:`\textbf{P}` has one row for every level from 1 to the ROI's largest
    intensity, including levels no voxel has.
  - Positive :math:`n` (MATLAB-style): level :math:`\lfloor n\,x / x_{\max} + 1 \rfloor`, clipped
    to :math:`[1, n]`, where :math:`x_{\max}` is the ROI's largest intensity. Intensity 0 is level
    1.
  - Negative :math:`-n` (PyRadiomics-style): level
    :math:`\lfloor n\,(x - x_{\min}) / (x_{\max} - x_{\min}) \rfloor + 1`, capped at :math:`n`,
    over the ROI's range. Intensity 0 is a level like any other.

  With binning, :math:`\textbf{P}` has one row per level present in the ROI.
- **Background:** level 0. Only the bounding-box voxels outside the mask have it with binning, since
  every ROI voxel bins to level 1 or more; without binning, ROI voxels of intensity 0 have it too.
  Voxels at level 0 belong to no zone.
- **Voxel count:** :math:`N_v` is the number of voxels in the ROI mask, whatever their intensity.
  The 2D family counts the non-zero pixels of the bounding box instead.


Worked example
^^^^^^^^^^^^^^

Take a 3 × 3 × 3 ROI that fills its bounding box and already has four grey levels. The slices are
:math:`z = 0, 1, 2`, each with rows :math:`y` and columns :math:`x` from 0:

.. math::

   \textbf{G}_{z=0} = \begin{bmatrix}
   3 & 3 & 4\\
   2 & 2 & 4\\
   4 & 3 & 1 \end{bmatrix},
   \qquad
   \textbf{G}_{z=1} = \begin{bmatrix}
   4 & 2 & 1\\
   3 & 2 & 2\\
   1 & 1 & 3 \end{bmatrix},
   \qquad
   \textbf{G}_{z=2} = \begin{bmatrix}
   1 & 2 & 3\\
   4 & 4 & 4\\
   4 & 3 & 4 \end{bmatrix}

In IBSI mode the matrix is below; rows are grey levels 1 to 4 and columns are zone sizes 1 to 6.
The default 64-level binning moves the levels to 17, 33, 49 and 64 and keeps the zones, so the
features that use only zone sizes and counts are the same and the grey-level-weighted ones change
(grey level variance 389.4 instead of 1.58).

.. math::

   \textbf{P} = \begin{bmatrix}
   2 & 0 & 1 & 0 & 0 & 0\\
   0 & 0 & 0 & 0 & 0 & 1\\
   1 & 0 & 0 & 0 & 0 & 1\\
   1 & 1 & 0 & 0 & 0 & 1 \end{bmatrix},
   \qquad N_z = 9, \qquad N_v = 27

The six voxels of level 2 form one zone. The level-3 zone of size 6 joins two groups of three, the 3s
at :math:`(z,y,x) = (0,0,0), (0,0,1), (1,1,0)` and at :math:`(0,2,1), (1,2,2), (2,2,1)`. Only the
corner shared by :math:`(1,1,0)` and :math:`(0,2,1)` connects them. With 18-connectivity
(faces and edges) the volume has 10 zones; with 6-connectivity (faces only) it has 15.

The *Example* value of each feature in the :ref:`index <glszm-3d-index>` is what Nyxus returns for
this ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/glszm_3d_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --dim=3 --features="*3D_GLSZM*" --ibsi=true \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

The symbols are those of the :doc:`2D page <f_glszm>`, with voxels in place of pixels:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N_g`
     - Number of rows of :math:`\textbf{P}`. Without binning, the ROI's largest intensity; with
       binning, the number of levels present in the ROI.
   * - :math:`g_i`
     - Grey level of row :math:`i`. Without binning, :math:`g_i = i`; with binning, the
       :math:`i`-th smallest level present.
   * - :math:`N_s`
     - Number of columns of :math:`\textbf{P}`: the size of the largest zone, in voxels.
   * - :math:`\textbf{P}(i,j)`
     - Number of zones of grey level :math:`g_i` and size :math:`j` voxels.
   * - :math:`N_z`
     - Number of zones, :math:`N_z = \sum_{i,j}\textbf{P}(i,j)`.
   * - :math:`N_v`
     - Number of voxels in the ROI mask, including voxels that belong to no zone.
   * - :math:`p(i,j)`
     - Zone probability, :math:`p(i,j) = \textbf{P}(i,j) / N_z`.
   * - :math:`s_i,\; s_j`
     - Zones per grey level, :math:`s_i = \sum_j \textbf{P}(i,j)`, and per size,
       :math:`s_j = \sum_i \textbf{P}(i,j)`.

Every feature uses the grey level :math:`g_i`, never the row position. Zone entropy uses the same
fast single-precision base-2 logarithm as the 2D family.


Settings
--------

.. list-table::
   :widths: 16 28 26 30
   :header-rows: 1

   * - Setting
     - Command line
     - Python
     - Effect
   * - Grey levels
     - ``--coarseGrayDepth=64``
     - ``Nyxus3D(..., coarse_gray_depth=64)``
     - Positive (the default, 64): MATLAB-style bins from 0 to the ROI maximum. 0: no binning.
       Negative: PyRadiomics-style bins over the ROI range.
   * - Grey levels of this family only
     - none
     - ``nyx.set_metaparam("3glszm/greydepth=8")`` on a ``Nyxus3D`` object
     - Overrides ``--coarseGrayDepth`` for 3D GLSZM, with the same meaning of the sign.
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus3D(..., ibsi=True)``
     - No binning, whatever ``--coarseGrayDepth`` and ``3glszm/greydepth`` hold. Voxels of
       intensity 0 are in no zone.
   * - Missing value
     - ``--noval=0``
     - —
     - Value of every feature of a constant ROI.
   * - Memory budget
     - ``--ramLimit``
     - ``Nyxus3D(..., ram_limit=...)``
     - ROIs above it are processed out of core, with the same values, unless one read of the file
       returns the whole volume: then such a ROI stops the run with an error.

The family has no direction, distance or connectivity setting. All options are listed in the
:ref:`cli-reference`.

On the worked example, ``3glszm/greydepth=-3`` bins levels 1, 2, 3 and 4 to 1, 2, 3 and 3: 5 zones,
zone percentage 0.1852, small area emphasis 0.4286 and grey level variance 0.64.


Output columns
--------------

Each feature produces one column, named with its code: ``3GLSZM_SAE``, ``3GLSZM_ZP`` and so on.
There are no per-direction columns and no ``_AVE`` columns.


.. _glszm-3d-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example.

.. include:: glszm_3d_index.inc


Features
--------

Each feature is the 2D feature of the same name, computed on the 3D matrix with zone sizes in
voxels. Only zone percentage uses a different denominator.

.. _feature-3glszm-gln:

3GLSZM_GLN: grey level non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_GLN``
:Range: :math:`[1, N_z]`, zones

Defined as :ref:`GLSZM_GLN <feature-glszm-gln>`.


.. _feature-3glszm-glnn:

3GLSZM_GLNN: grey level non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_GLNN``
:Range: :math:`(0, 1]`, dimensionless

Defined as :ref:`GLSZM_GLNN <feature-glszm-glnn>`.


.. _feature-3glszm-glv:

3GLSZM_GLV: grey level variance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_GLV``
:Range: :math:`[0, \infty)`, squared grey levels

Defined as :ref:`GLSZM_GLV <feature-glszm-glv>`. Without binning it is in squared intensity units.


.. _feature-3glszm-hglze:

3GLSZM_HGLZE: high grey level zone emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_HGLZE``
:Range: :math:`[g_1^{\,2}, g_{N_g}^{\,2}]`, squared grey levels

Defined as :ref:`GLSZM_HGLZE <feature-glszm-hglze>`.


.. _feature-3glszm-lae:

3GLSZM_LAE: large area emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_LAE``
:Range: :math:`[1, N_v^{\,2}]`, squared voxels

Defined as :ref:`GLSZM_LAE <feature-glszm-lae>`. IBSI calls it *large zone emphasis*.


.. _feature-3glszm-lahgle:

3GLSZM_LAHGLE: large area high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_LAHGLE``
:Range: :math:`(0, \infty)`, squared voxels times squared grey levels

Defined as :ref:`GLSZM_LAHGLE <feature-glszm-lahgle>`.


.. _feature-3glszm-lalgle:

3GLSZM_LALGLE: large area low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_LALGLE``
:Range: :math:`(0, \infty)`, squared voxels over squared grey levels

Defined as :ref:`GLSZM_LALGLE <feature-glszm-lalgle>`.


.. _feature-3glszm-lglze:

3GLSZM_LGLZE: low grey level zone emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_LGLZE``
:Range: :math:`(0, 1]`, inverse squared grey levels

Defined as :ref:`GLSZM_LGLZE <feature-glszm-lglze>`.


.. _feature-3glszm-sae:

3GLSZM_SAE: small area emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_SAE``
:Range: :math:`(0, 1]`, inverse squared voxels

Defined as :ref:`GLSZM_SAE <feature-glszm-sae>`. IBSI calls it *small zone emphasis*.


.. _feature-3glszm-sahgle:

3GLSZM_SAHGLE: small area high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_SAHGLE``
:Range: :math:`(0, g_{N_g}^{\,2}]`, squared grey levels over squared voxels

Defined as :ref:`GLSZM_SAHGLE <feature-glszm-sahgle>`.


.. _feature-3glszm-salgle:

3GLSZM_SALGLE: small area low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_SALGLE``
:Range: :math:`(0, 1]`, inverse squared voxels and grey levels

Defined as :ref:`GLSZM_SALGLE <feature-glszm-salgle>`.


.. _feature-3glszm-szn:

3GLSZM_SZN: size zone non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_SZN``
:Range: :math:`[1, N_z]`, zones

Defined as :ref:`GLSZM_SZN <feature-glszm-szn>`.


.. _feature-3glszm-sznn:

3GLSZM_SZNN: size zone non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_SZNN``
:Range: :math:`(0, 1]`, dimensionless

Defined as :ref:`GLSZM_SZNN <feature-glszm-sznn>`.


.. _feature-3glszm-ze:

3GLSZM_ZE: zone entropy
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_ZE``
:Range: :math:`[0, \log_2(N_g N_s)]` bits

Defined as :ref:`GLSZM_ZE <feature-glszm-ze>`, with the same approximate logarithm.

**Possible issues.** On the worked example Nyxus returns 2.9485, while an exact logarithm gives
2.9477, a relative difference of 2.9 × 10\ :sup:`-4`.


.. _feature-3glszm-zp:

3GLSZM_ZP: zone percentage
^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_ZP``
:Range: :math:`(0, 1]`, dimensionless

The number of zones divided by the number of voxels in the ROI mask. High for fine texture, where
most zones are small.

.. math::

   \mathrm{ZP} = \frac{N_z}{N_v}

**Possible issues.** Without binning, :math:`N_v` includes the ROI voxels of intensity 0, which
belong to no zone. The value is then lower than for the same zones in a ROI without such voxels
(see :ref:`glszm-3d-possible-issues`).


.. _feature-3glszm-zv:

3GLSZM_ZV: zone variance
^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLSZM_ZV``
:Range: :math:`[0, \infty)`, squared voxels

Defined as :ref:`GLSZM_ZV <feature-glszm-zv>`. IBSI calls it *zone size variance*.


.. _glszm-3d-possible-issues:

Possible issues
---------------

Binning merges levels into fewer, larger zones
   With few levels, neighbouring levels share a bin and their zones merge. On the worked example
   with ``3glszm/greydepth=2``, the five voxels of intensity 1 bin to level 1 and the other 22 to
   level 2: 4 zones, zone percentage 0.1481, small area emphasis 0.5283, large area emphasis 123.75
   and low grey level zone emphasis 0.8125, in memory and out of core. The 2D family also puts the
   bounding-box pixels outside the mask in zones of level 1 under MATLAB-style binning; the 3D
   family does not.

Voxels of intensity 0 inside the mask
   Without binning, in IBSI mode or with ``--coarseGrayDepth=0``, a voxel of intensity 0 belongs to
   no zone but is counted in :math:`N_v`. With binning it gets a level like any other voxel. On the worked example with two voxels set to 0, Nyxus
   finds the same 9 zones and zone percentage 9/27 = 0.3333; over the 25 non-zero voxels it would be
   0.36.

ROI that does not fill its bounding box
   The bounding-box voxels outside the mask have intensity 0 and belong to no zone; they are not
   counted in :math:`N_v` either. Removing three voxels of the worked example from the mask gives 9
   zones over 24 voxels, zone percentage 0.375, in memory and out of core.

``get_metaparam("3glszm/greydepth")`` fails
   Reading the metaparameter back raises an "unrecognized feature" error, although setting it
   works.

Constant ROI
   When all of a ROI's voxels have the same intensity, every 3D GLSZM feature returns the
   ``--noval`` substitute (0 by default), although the ROI has a well-defined matrix of one zone
   per connected component.

Memory use
   :math:`\textbf{P}` has :math:`N_g` rows and :math:`N_s` columns of 4 bytes. Without binning
   :math:`N_g` is the ROI's largest intensity, so a 12-bit volume (intensities up to 4095) whose
   largest zone has 100 000 voxels needs a 1.6 GB matrix. Binning to a small :math:`N_g` keeps it
   small.


Validation
----------

The test suite checks every 3D GLSZM feature against PyRadiomics in two configurations. On a 3D
liver phantom with PyRadiomics-style binning into 20 levels, fifteen features agree to a relative
10\ :sup:`-9`, and the size-zone matrix agrees cell by cell. On a 3 × 3 × 3 volume with grey levels 1,
3 and 5 in IBSI mode, the same fifteen agree to a relative 10\ :sup:`-9`. Zone entropy agrees to a
relative 10\ :sup:`-3` on the phantom and 2 × 10\ :sup:`-3` on the small volume, the error of the
approximate logarithm. MATLAB-style binning, the default setting on a larger phantom and the
constant ROI are covered by regression tests only. On a volume whose ROI holds the volume's
lowest intensity, and on a control without it, fifteen features agree with PyRadiomics binning
into 8 levels to a relative 10\ :sup:`-9`, in memory and out of core.


References
----------

1. Thibault G, Fertil B, Navarro C, et al. *Texture indexes and gray level size zone matrix:
   application to cell nuclei classification.* Pattern Recognition and Information Processing
   (PRIP). 2009:140–145.
2. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
3. van Griethuysen JJM, et al. *Computational radiomics system to decode the radiographic
   phenotype.* Cancer Research. 2017;77(21):e104–e107.
