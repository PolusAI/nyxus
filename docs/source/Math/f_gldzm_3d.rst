Grey-level distance-zone matrix, 3D (3GLDZM)
============================================

The 3D GLDZM features describe the zones of a volumetric ROI, groups of connected voxels that share
one grey level, by how far each zone lies from the surface of the ROI. The features are those of the
2D family: this page covers what is specific to volumes, and every formula is on the
:doc:`2D GLDZM page <f_gldzm>`.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``3GLDZM_*``: 18 features, one value per ROI
   * - **Group**
     - ``*3D_GLDZM*``; also part of ``*3D_ALL_TEXTURE*`` and ``*3D_ALL*``
   * - **Dimensions**
     - 3D. The 2D family, ``GLDZM_*``, is on the :doc:`2D GLDZM page <f_gldzm>`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory. A ROI whose bounding box is more than 131,068 voxels
       along its shortest side is refused. When one read of the file returns the whole volume
       (NIfTI, or OME-Zarr chunked over all of Z), an oversized ROI stops the run with an error
       (see :doc:`/large_images`).
   * - **IBSI**
     - 16 of the 18 features have an IBSI identifier. Grey level mean and zone distance mean have
       none.

.. tip::

   The 3D family follows the IBSI definition of the matrix in every mode: zones are 26-connected,
   a zone's distance is the city-block distance to the nearest voxel outside the ROI, and voxels
   outside the mask never form zones. The 2D family differs on the last two points. To reproduce
   IBSI or MIRP values, run in IBSI mode (``--ibsi=true``, ``ibsi=True``), which leaves the
   intensities unbinned.


How the matrix is built
-----------------------

Nyxus scans the ROI's bounding box plane by plane, row by row. Each ROI voxel that is not yet in a
zone starts a new zone, which grows through every ROI voxel of the same grey level among its 26
neighbours. Each voxel of the zone has a distance to the ROI surface, and the zone's distance is the
smallest of them. :math:`\textbf{P}(i,j)` counts the zones of grey level :math:`g_i` and distance
:math:`j`.

- **Connectivity:** 26-connected, as IBSI defines for 3D. Two voxels of one grey level belong to the
  same zone when they share a face, an edge or a corner.
- **Distance:** the city-block distance from the voxel to the nearest voxel that is not in the ROI,
  through the 6 face neighbours. A voxel with a face neighbour outside the ROI, or on a face of the
  bounding box, has distance 1; the next layer inwards has distance 2, and so on. Every voxel outside
  the mask counts as outside the ROI, whatever its intensity, and so does everything beyond the
  bounding box. Nyxus computes the distances in one breadth-first pass from the surface inwards,
  which is IBSI's erosion with the 6-connected element.
- **Directions:** none. One matrix is built per ROI, and each feature has one value.
- **Grey levels:** intensities are first binned into :math:`N_g` grey levels, as set by
  ``--coarseGrayDepth`` (64 by default). The 3D family has no metaparameter of its own. A voxel of
  intensity :math:`x` gets level :math:`\lfloor N_g\,x / x_{\max} \rfloor + 1`, capped at
  :math:`N_g`, where :math:`x_{\max}` is the ROI's largest intensity, and intensity 0 gets level 1.
  In IBSI mode, or with ``--coarseGrayDepth=0``, they are not binned: each integer intensity is a
  grey level. A negative ``--coarseGrayDepth`` selects PyRadiomics-style bins (see
  `Settings`_).
- **Background:** only the voxels inside the mask form zones, in every mode. Voxels of the bounding
  box outside the mask are skipped, and they count as outside the ROI for the distances.
- **Level 0:** a GLDZM grey level starts at 1. When a ROI voxel would get level 0 (intensity 0 in
  IBSI mode, or with PyRadiomics-style bins), every level of the ROI is raised by one, so that voxel
  still forms a zone (see :ref:`gldzm-3d-possible-issues`).


Worked example
^^^^^^^^^^^^^^

Take a 3 × 4 × 4 ROI that fills its bounding box and already has five grey levels. The three slices,
from :math:`z = 0` to :math:`z = 2`, are:

.. math::

   \begin{bmatrix}
   5 & 1 & 1 & 1\\
   1 & 1 & 1 & 1\\
   2 & 2 & 2 & 2\\
   2 & 2 & 2 & 2 \end{bmatrix}
   \qquad
   \begin{bmatrix}
   1 & 1 & 1 & 1\\
   1 & 3 & 4 & 1\\
   2 & 3 & 4 & 2\\
   2 & 2 & 2 & 2 \end{bmatrix}
   \qquad
   \begin{bmatrix}
   1 & 1 & 2 & 3\\
   1 & 1 & 2 & 2\\
   2 & 2 & 2 & 4\\
   2 & 2 & 2 & 5 \end{bmatrix}

The ROI is three voxels thick, so only the 2 × 2 centre of the middle slice has distance 2; every
other voxel lies on the surface and has distance 1. The 26-connected zones are:

- all the 1s form one zone, and all the 2s another, both at distance 1;
- the two 3s in the centre of the middle slice touch no other 3, so they form a zone at distance 2;
  the 3 in the corner of the last slice is a zone of its own at distance 1;
- the two 4s in the centre of the middle slice share an edge with the 4 of the last slice, which is
  on the surface, so the three form one zone at distance 1;
- the two 5s lie in opposite corners and form two zones at distance 1.

In IBSI mode (no binning) the matrix is below; rows are grey levels 1 to 5 and columns are
distances 1 and 2.

.. math::

   \textbf{P} = \begin{bmatrix}
   1 & 0\\
   1 & 0\\
   1 & 1\\
   1 & 0\\
   2 & 0 \end{bmatrix},
   \qquad N_s = \sum_{i,j} \textbf{P}(i,j) = 7, \qquad N_v = 48

The *Example* value of each feature in the :ref:`index <gldzm-3d-index>` is what Nyxus returns for
this ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/gldzm_3d_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --dim=3 --features="*3D_GLDZM*" --ibsi=true \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv

Leave the voxel at :math:`z = 0`, row 1, column 1 out of the mask and the 3 behind it, in the middle
slice, gets a face neighbour outside the ROI. Its distance drops to 1, and so does the distance of
its zone: small distance emphasis is 1 instead of 0.8929, zone distance mean 1 instead of 1.143, and
zone percentage 7/47 = 0.1489. The values are the same in IBSI mode and with the default binning,
and whatever the intensity of the voxel left out.


Notation
--------

The symbols are those of the :doc:`2D page <f_gldzm>`, with voxels in place of pixels.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N_g`
     - Number of rows of :math:`\textbf{P}`. With binning, the number of grey levels present among
       the ROI's voxels; in IBSI mode, the ROI's largest intensity (one more when a voxel has
       intensity 0).
   * - :math:`g_i`
     - Grey level of row :math:`i`. With binning, the :math:`i`-th smallest level present; in IBSI
       mode, :math:`g_i = i`.
   * - :math:`N_d`
     - Number of columns of :math:`\textbf{P}`: the largest zone distance, in voxels. It is at most
       half the shortest side of the ROI's bounding box, rounded up.
   * - :math:`\textbf{P}(i,j)`
     - Number of zones of grey level :math:`g_i` and distance :math:`j`.
   * - :math:`N_s`
     - Number of zones, :math:`N_s = \sum_{i,j}\textbf{P}(i,j)`.
   * - :math:`N_v`
     - Number of ROI voxels: the voxels inside the mask, including those of intensity 0.

Every feature uses the grey level :math:`g_i`, never the row position. Zone distance entropy uses
the exact base-2 logarithm in double precision, with :math:`\varepsilon = 2.2 \times 10^{-16}`
inside it.


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
     - Number of grey levels. Positive: equal-width bins from 0 to the ROI maximum; 0: no binning;
       negative: :math:`|n|` PyRadiomics-style bins of width :math:`(x_{\max}-x_{\min})/|n|` from
       the ROI minimum, with intensity 0 kept at level 0. Python accepts only a positive value.
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus3D(..., ibsi=True)``
     - No binning, whatever the grey-level setting.
   * - Anisotropy
     - ``--anisox``, ``--anisoy``, ``--anisoz``, or ``--use-physical-spacing=true``
     - ``Nyxus3D(..., anisotropy_x=..., anisotropy_y=..., anisotropy_z=...)``, or
       ``use_physical_spacing=True``
     - No effect on these features: zones and distances are taken on the voxels as acquired.

Zone connectivity and the distance are fixed. No ``set_metaparam`` parameter applies to
``3GLDZM_*``: unlike ``3glcm``, ``3glrlm`` or ``3glszm`` there is no ``3gldzm/greydepth``, and
``set_metaparam("3gldzm/greydepth=8")`` stops with an error. All options are listed in the
:ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named with its code: ``3GLDZM_SDE``, ``3GLDZM_ZP`` and so on.
There are no per-direction columns and no ``_AVE`` columns.


.. _gldzm-3d-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with the tolerance; see the :doc:`/featurelist` for what the categories mean. *Example* is
the value on the worked example.

.. include:: gldzm_3d_index.inc


Features
--------

Every 3D feature applies the formula of its 2D counterpart to the 3D matrix, with voxels in place of
pixels.

.. _feature-3gldzm-glm:

Grey level mean
^^^^^^^^^^^^^^^

:Code: ``3GLDZM_GLM``
:Range: :math:`[g_1, g_{N_g}]`, grey levels

As :ref:`GLDZM_GLM <feature-gldzm-glm>`. Depends on the binning: 3.286 in IBSI mode and 42.43 with
the default 64-level binning on the worked example.


.. _feature-3gldzm-glnu:

Grey level non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_GLNU``
:Range: :math:`[1, N_s]`, zones

As :ref:`GLDZM_GLNU <feature-gldzm-glnu>`.


.. _feature-3gldzm-glnun:

Grey level non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_GLNUN``
:Range: :math:`(0, 1]`, dimensionless

As :ref:`GLDZM_GLNUN <feature-gldzm-glnun>`.


.. _feature-3gldzm-glv:

Grey level variance
^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_GLV``
:Range: :math:`[0, \infty)`, squared grey levels

As :ref:`GLDZM_GLV <feature-gldzm-glv>`. It scales with the square of the grey-level spacing: 1.918
in IBSI mode and 311.67 with the default binning on the worked example, which maps the 5 levels to
13, 26, 39, 52 and 64.


.. _feature-3gldzm-hglze:

High grey level zone emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_HGLZE``
:Range: :math:`[g_1^{\,2}, g_{N_g}^{\,2}]`, squared grey levels

As :ref:`GLDZM_HGLZE <feature-gldzm-hglze>`. 12.71 in IBSI mode and 2111.9 with the default
binning on the worked example.


.. _feature-3gldzm-lde:

Large distance emphasis
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_LDE``
:Range: :math:`[1, N_d^{\,2}]`, squared voxels

As :ref:`GLDZM_LDE <feature-gldzm-lde>`.


.. _feature-3gldzm-ldhgle:

Large distance high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_LDHGLE``
:Range: :math:`[g_1^{\,2}, g_{N_g}^{\,2} N_d^{\,2}]`, squared grey levels times squared voxels

As :ref:`GLDZM_LDHGLE <feature-gldzm-ldhgle>`.


.. _feature-3gldzm-ldlgle:

Large distance low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_LDLGLE``
:Range: :math:`(0, N_d^{\,2}]`, squared voxels over squared grey levels

As :ref:`GLDZM_LDLGLE <feature-gldzm-ldlgle>`.


.. _feature-3gldzm-lglze:

Low grey level zone emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_LGLZE``
:Range: :math:`[1/g_{N_g}^{\,2}, 1/g_1^{\,2}]`, inverse squared grey levels

As :ref:`GLDZM_LGLZE <feature-gldzm-lglze>`. 0.2307 in IBSI mode and 0.001367 with the default
binning on the worked example. Voxels outside the mask form no zones, so they do not weigh on it.


.. _feature-3gldzm-sde:

Small distance emphasis
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_SDE``
:Range: :math:`[1/N_d^{\,2}, 1]`, inverse squared voxels

As :ref:`GLDZM_SDE <feature-gldzm-sde>`.


.. _feature-3gldzm-sdhgle:

Small distance high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_SDHGLE``
:Range: :math:`(0, g_{N_g}^{\,2}]`, squared grey levels over squared voxels

As :ref:`GLDZM_SDHGLE <feature-gldzm-sdhgle>`.


.. _feature-3gldzm-sdlgle:

Small distance low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_SDLGLE``
:Range: :math:`(0, 1/g_1^{\,2}]`, inverse squared grey levels times inverse squared voxels

As :ref:`GLDZM_SDLGLE <feature-gldzm-sdlgle>`.


.. _feature-3gldzm-zde:

Zone distance entropy
^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_ZDE``
:Range: :math:`[0, \log_2(N_g N_d)]` bits

As :ref:`GLDZM_ZDE <feature-gldzm-zde>`, with the exact logarithm: 2.5216 on the worked example.


.. _feature-3gldzm-zdm:

Zone distance mean
^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_ZDM``
:Range: :math:`[1, N_d]`, voxels

As :ref:`GLDZM_ZDM <feature-gldzm-zdm>`.


.. _feature-3gldzm-zdnu:

Zone distance non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_ZDNU``
:Range: :math:`[1, N_s]`, zones

As :ref:`GLDZM_ZDNU <feature-gldzm-zdnu>`.


.. _feature-3gldzm-zdnun:

Zone distance non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_ZDNUN``
:Range: :math:`(0, 1]`, dimensionless

As :ref:`GLDZM_ZDNUN <feature-gldzm-zdnun>`.


.. _feature-3gldzm-zdv:

Zone distance variance
^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDZM_ZDV``
:Range: :math:`[0, \infty)`, squared voxels

As :ref:`GLDZM_ZDV <feature-gldzm-zdv>`.


.. _feature-3gldzm-zp:

Zone percentage
^^^^^^^^^^^^^^^

:Code: ``3GLDZM_ZP``
:Range: :math:`(0, 1]`, dimensionless

As :ref:`GLDZM_ZP <feature-gldzm-zp>`, with :math:`N_v` the number of ROI voxels. Only ROI voxels
form zones, so the value never exceeds 1, with or without binning: 0.1458 on the worked example in
both modes.


.. _gldzm-3d-possible-issues:

Possible issues
---------------

Thin ROIs have no depth to measure
   A voxel's distance is at most half the shortest side of the ROI's bounding box, rounded up. In a ROI two voxels
   thick along any axis, as in a stack of two slices, every voxel is on the surface: all zones have
   distance 1, small and large distance emphasis are 1 and zone distance variance is 0. The worked
   example, three voxels thick, reaches only distance 2. On volumes with few, thick slices, compare
   the distance features only between ROIs of similar thickness.

ROI voxels of intensity 0 raise every grey level by one
   In IBSI mode a voxel of intensity 0 would have grey level 0, which a GLDZM cannot hold. Nyxus
   then adds 1 to the level of every ROI voxel, so the voxel keeps its zone and :math:`N_v` keeps
   counting it, but every grey level is one more than the intensity. Setting the voxel at the centre
   of the worked example (:math:`z = 1`, row 1, column 1) to 0 gives 8 zones over levels 1 to 6;
   grey level mean is 3.875, where the intensities of those zones average 2.875, and zone percentage
   is 0.1667. The grey-level-weighted features of a ROI shift when a single voxel turns 0, and do
   not match an implementation that leaves such voxels out. The same lift applies with
   PyRadiomics-style bins, which keep intensity 0 at level 0. Default binning sends intensity 0 to
   level 1 and is not affected.

Default binning starts at 0, not at the ROI minimum
   The positive ``--coarseGrayDepth`` bins span 0 to the ROI maximum, so a ROI whose intensities
   sit far from 0 uses only the top bins. This is the MATLAB-style rule that every texture family
   except NGLDM uses with a positive grey depth (see :ref:`conventions-grey-levels`). MIRP's and
   PyRadiomics' fixed bin number spans the ROI's own range, so the grey-level-weighted features
   differ from theirs with binning; the distance features do not depend on the bins as long as the
   zones stay the same.

Python offers only the default binning scheme
   ``Nyxus3D`` accepts only a positive ``coarse_gray_depth``. ``set_params(coarse_gray_depth=0)``
   leaves the setting unchanged, and a negative value raises a ``TypeError``. Use ``ibsi=True`` for
   unbinned grey levels, and the command line for PyRadiomics-style bins.

Constant ROI
   When all of a ROI's voxels have the same intensity, every 3D GLDZM feature returns the
   ``--noval`` substitute (0 by default).


Validation
----------

The test suite checks the 16 IBSI features against MIRP on a 3D phantom built to separate the
family's definition choices, to an absolute 10\ :sup:`-9`, in two configurations: IBSI mode, and
PyRadiomics-style binning with 8 bins, a bin count that leaves the phantom's levels 1 to 8
unchanged. The largest difference is 5.7 × 10\ :sup:`-15`, on zone distance entropy; 13 of the 16
values are bit-identical. The phantom is negative-controlled: 6- or 18-connected zones fail all 16
checks, and a distance taken as the shortest of the six axis-aligned walks fails 9 of them. Grey
level mean and zone distance mean have no independent implementation and are covered by
regression tests. The default binning, and PyRadiomics-style binning at a bin count that merges
levels, are covered by regression tests only.


References
----------

1. Thibault G, Angulo J, Meyer F. *Advanced statistical matrices for texture characterization:
   application to cell classification.* IEEE Transactions on Biomedical Engineering.
   2014;61(3):630–637.
2. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
