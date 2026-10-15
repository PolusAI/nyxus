Grey-level dependence matrix, 3D (GLDM 3D)
==========================================

3D GLDM features describe how many voxels of the same grey level surround each voxel of an ROI.
They are the volumetric form of the :doc:`2D GLDM family <f_gldm>`: the matrix counts voxels by
grey level and by dependence over the full 26-voxel neighbourhood, and the fourteen features are
the same contractions of it. This page covers what is specific to 3D; the formulas are defined on
the 2D page.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``3GLDM_*``: 14 features, one value per ROI
   * - **Group**
     - ``*3D_GLDM*``; also part of ``*3D_ALL_TEXTURE*`` and ``*3D_ALL*``
   * - **Dimensions**
     - 3D. The 2D family, ``GLDM_*``, is defined on :doc:`its own page <f_gldm>`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory.
       When one read of the file returns the whole volume (NIfTI, or OME-Zarr chunked over all of
       Z), an oversized ROI stops the run with an error (see :doc:`/large_images`).
   * - **IBSI**
     - None. IBSI defines no GLDM; the family follows PyRadiomics. IBSI's neighbouring grey level
       dependence matrix with :math:`\alpha = 0` and :math:`\delta = 1` counts the same
       dependences; see also the :doc:`NGLDM family <f_ngldm>`.

.. tip::

   By default the 3D family does not bin intensities: each integer intensity is a grey level, in
   IBSI mode and out of it, and ``--coarseGrayDepth`` has no effect. These values match
   PyRadiomics with ``binWidth=1`` and ``gldm_a=0`` on volumes whose intensities start at 1. To
   bin, set the ``3gldm/greydepth`` metaparameter (see :ref:`gldm-3d-settings`).


How the matrix is built
-----------------------

Nyxus visits every voxel of the ROI's bounding box that carries a nonzero grey level and counts its
*dependence*: the voxel itself plus each of its 26 neighbours that has exactly the same grey level.
A voxel of grey level :math:`i` and dependence :math:`j` adds 1 to :math:`\textbf{P}(i,j)`.

- **Neighbourhood:** the 26 voxels at Chebyshev distance 1: 8 in the voxel's own slice and 9 in
  each of the slices above and below. Neighbours outside the bounding box do not exist. The family
  has no directions and no per-direction values.
- **Dependence:** a neighbour is dependent only when its grey level equals the centre's
  (:math:`\alpha = 0`, PyRadiomics ``gldm_a = 0``). The centre voxel counts, so :math:`j` runs from
  1 to 27. A voxel with no dependent neighbour is counted with :math:`j = 1`.
- **Grey levels:** set by ``3gldm/greydepth``, 0 by default.

  - 0, or IBSI mode: no binning. Each integer intensity is a grey level, and the matrix has rows
    1 to the ROI maximum.
  - :math:`N > 0`: intensity :math:`x` maps to :math:`\min(\lfloor N x / x_{\max} + 1 \rfloor, N)`,
    where :math:`x_{\max}` is the ROI's maximum intensity.
  - :math:`N < 0`: :math:`|N|` equal-width bins between the ROI minimum and maximum. Intensity
    :math:`x > 0` maps to :math:`\min(\lfloor (x - x_{\min})/w + 1 \rfloor, |N|)` with
    :math:`w = (x_{\max} - x_{\min})/|N|`.

  Formulas use the grey level's value: the intensity without binning, the level otherwise.
- **Background:** the bounding box holds 0 for every voxel outside the ROI's mask, including
  voxels of other ROIs. A voxel whose grey level is the background level, 0 (or 1 when
  :math:`N > 0`), is not counted, and no counted voxel has that level, so background never counts
  as a neighbour. The mask itself is not consulted: a ROI voxel of intensity 0 is treated as
  background too (see :ref:`gldm-3d-possible-issues`).


Worked example
^^^^^^^^^^^^^^

Take a 3 × 4 × 4 ROI (slices, rows, columns) that already has four grey levels, inside a
3 × 6 × 6 volume of zeros:

.. math::

   \textbf{G}_1 = \begin{bmatrix}
   1 & 1 & 2 & 3\\
   1 & 2 & 2 & 3\\
   4 & 4 & 2 & 1\\
   4 & 3 & 3 & 1 \end{bmatrix},\quad
   \textbf{G}_2 = \begin{bmatrix}
   1 & 2 & 2 & 3\\
   1 & 1 & 3 & 3\\
   4 & 2 & 2 & 1\\
   4 & 4 & 3 & 3 \end{bmatrix},\quad
   \textbf{G}_3 = \begin{bmatrix}
   2 & 2 & 3 & 3\\
   1 & 1 & 3 & 4\\
   4 & 4 & 1 & 1\\
   2 & 4 & 4 & 3 \end{bmatrix}

With the default settings, or in IBSI mode, the matrix is below, with grey levels 1 to 4 as rows
and dependences 1 to 10 as columns; no voxel has a dependence above 10. The 4 in slice 2, row 4,
column 2 has nine neighbours of level 4 (three in slice 1, two in its own slice, four in slice 3),
so its dependence is 10. The 4 in slice 3, row 2, column 4 has none, so its dependence is 1.

.. math::

   \textbf{P} = \begin{bmatrix}
   0 & 0 & 3 & 0 & 3 & 4 & 0 & 2 & 1 & 0\\
   0 & 1 & 1 & 1 & 3 & 2 & 1 & 2 & 0 & 0\\
   0 & 0 & 2 & 2 & 3 & 3 & 0 & 3 & 0 & 0\\
   1 & 0 & 0 & 1 & 0 & 4 & 2 & 0 & 2 & 1 \end{bmatrix},
   \qquad N_z = \sum_{i,j} \textbf{P}(i,j) = 48

The *Example* value of each feature in the :ref:`index <gldm-3d-index>` is what Nyxus returns for
this ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/gldm_3d_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --dim=3 --features="*3D_GLDM*" --ibsi=true \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

The symbols are those of the :doc:`2D page <f_gldm>`, with voxels in place of pixels:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N_g`
     - Number of grey levels: the ROI maximum without binning, otherwise the number of distinct
       binned levels in the bounding box.
   * - :math:`N_d`
     - Largest possible dependence, 27.
   * - :math:`i,\; j`
     - Grey level of a voxel (:math:`i`) and its dependence (:math:`j`).
   * - :math:`\textbf{P}(i,j)`
     - Number of ROI voxels with grey level :math:`i` and dependence :math:`j`.
   * - :math:`N_z`
     - Number of counted voxels, :math:`N_z = \sum_{i,j}\textbf{P}(i,j)`: every ROI voxel that is
       not at the background level.
   * - :math:`p_{ij}`
     - Joint probability, :math:`p_{ij} = \textbf{P}(i,j) / N_z`.
   * - :math:`s_{i\cdot},\; s_{\cdot j}`
     - Voxels per grey level and voxels per dependence.
   * - :math:`\mu_i,\; \mu_j`
     - Mean grey level and mean dependence over all counted voxels.

Sums over :math:`j` run from 1 to :math:`N_d`. Empty rows and columns of :math:`\textbf{P}` add
nothing, so the levels between the ROI minimum and maximum that no voxel has do not change any
feature.

Logarithms are computed with the same fast base-2 approximation as in 2D. Dependence entropy on the
worked example is 4.3631, where an exact logarithm, and PyRadiomics, give 4.3656 (relative
5.7 × 10\ :sup:`-4`).


.. _gldm-3d-settings:

Settings
--------

.. list-table::
   :widths: 16 22 32 30
   :header-rows: 1

   * - Setting
     - Command line
     - Python
     - Effect
   * - Grey levels
     - none
     - ``Nyxus3D.set_metaparam("3gldm/greydepth=0")``
     - 0 (default): no binning. Positive :math:`N`: equal-width levels from 0 to the ROI maximum.
       Negative :math:`N`: :math:`|N|` PyRadiomics-style bins between the ROI minimum and
       maximum. Read back with ``Nyxus3D.get_metaparam("3gldm/greydepth")``.
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus3D(..., ibsi=True)``
     - No binning, whatever ``3gldm/greydepth`` holds.

``--coarseGrayDepth`` and the ``coarse_gray_depth`` argument of ``Nyxus3D`` do not affect this
family. The neighbourhood, the distance and :math:`\alpha` are fixed. The metaparameter applies to
in-memory and out-of-core ROIs alike. All options are listed in the :ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named by its code, for example ``3GLDM_SDE``. There are no
per-direction columns and no ``_AVE`` columns.


.. _gldm-3d-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with the tolerance; see the :doc:`/featurelist` for what the categories mean. *Example* is
the value on the worked example.

.. include:: gldm_3d_index.inc


Features
--------

Every 3D feature is the 2D formula with the 3D matrix :math:`\textbf{P}` substituted and
:math:`N_d = 27`. Only the ranges change where they depend on :math:`N_d`.

.. _feature-3gldm-de:

Dependence entropy (3D)
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_DE``
:Range: :math:`[0, \log_2(27 N_g)]` bits

Defined as :ref:`GLDM_DE <feature-gldm-de>`, with the same approximate logarithm.

.. _feature-3gldm-dn:

Dependence non-uniformity (3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_DN``
:Range: :math:`[1, N_z]`, voxels

Defined as :ref:`GLDM_DN <feature-gldm-dn>`.

.. _feature-3gldm-dnn:

Dependence non-uniformity normalized (3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_DNN``
:Range: :math:`(0, 1]`, dimensionless

Defined as :ref:`GLDM_DNN <feature-gldm-dnn>`.

.. _feature-3gldm-dv:

Dependence variance (3D)
^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_DV``
:Range: :math:`[0, 169]`, squared dependences

Defined as :ref:`GLDM_DV <feature-gldm-dv>`.

.. _feature-3gldm-gln:

Grey level non-uniformity (3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_GLN``
:Range: :math:`[1, N_z]`, voxels

Defined as :ref:`GLDM_GLN <feature-gldm-gln>`.

.. _feature-3gldm-glv:

Grey level variance (3D)
^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_GLV``
:Range: :math:`[0, \infty)`, squared grey levels

Defined as :ref:`GLDM_GLV <feature-gldm-glv>`. Without binning it is the variance of the raw
intensities over the counted voxels.

.. _feature-3gldm-hgle:

High grey level emphasis (3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_HGLE``
:Range: :math:`[1, \infty)`, squared grey levels

Defined as :ref:`GLDM_HGLE <feature-gldm-hgle>`.

.. _feature-3gldm-lde:

Large dependence emphasis (3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_LDE``
:Range: :math:`[1, 729]`, squared dependences

Defined as :ref:`GLDM_LDE <feature-gldm-lde>`.

.. _feature-3gldm-ldhgle:

Large dependence high grey level emphasis (3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_LDHGLE``
:Range: :math:`[1, \infty)`, squared grey levels times squared dependences

Defined as :ref:`GLDM_LDHGLE <feature-gldm-ldhgle>`.

.. _feature-3gldm-ldlgle:

Large dependence low grey level emphasis (3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_LDLGLE``
:Range: :math:`(0, 729]`, squared dependences over squared grey levels

Defined as :ref:`GLDM_LDLGLE <feature-gldm-ldlgle>`.

.. _feature-3gldm-lgle:

Low grey level emphasis (3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_LGLE``
:Range: :math:`(0, 1]`, inverse squared grey levels

Defined as :ref:`GLDM_LGLE <feature-gldm-lgle>`.

.. _feature-3gldm-sde:

Small dependence emphasis (3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_SDE``
:Range: :math:`[1/729, 1]`, inverse squared dependences

Defined as :ref:`GLDM_SDE <feature-gldm-sde>`.

.. _feature-3gldm-sdhgle:

Small dependence high grey level emphasis (3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_SDHGLE``
:Range: :math:`(0, \infty)`, squared grey levels over squared dependences

Defined as :ref:`GLDM_SDHGLE <feature-gldm-sdhgle>`.

.. _feature-3gldm-sdlgle:

Small dependence low grey level emphasis (3D)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLDM_SDLGLE``
:Range: :math:`(0, 1]`, inverse squared grey levels times inverse squared dependences

Defined as :ref:`GLDM_SDLGLE <feature-gldm-sdlgle>`.


.. _gldm-3d-possible-issues:

Possible issues
---------------

Positive ``3gldm/greydepth`` drops the dimmest voxels
   With :math:`N > 0`, intensity 0 and every intensity below :math:`x_{\max}/N` share level 1,
   and level 1 is treated as background: those ROI voxels are neither counted nor counted as
   neighbours. On the worked example with ``3gldm/greydepth=2`` the 13 voxels of intensity 1
   leave the matrix, :math:`N_z` drops from 48 to 35, a single level remains, grey level variance
   is 0, and small dependence emphasis is 0.01297 instead of 0.06470. With
   ``3gldm/greydepth=1`` every voxel is dropped and every feature returns the ``--noval``
   substitute. Zero or negative values of the metaparameter do not drop ROI voxels.

Grey-level values follow the raw intensities by default
   Without binning, ``GLV`` and the six emphasis features that weight voxels by grey level scale
   with the intensities. Multiplying the worked example by 250 leaves every dependence feature
   unchanged and raises high grey level emphasis from 7.292 to 455729; with
   ``3gldm/greydepth=64`` it is 1916.8. Compare these features only between volumes of the same
   intensity scale and the same ``3gldm/greydepth``.

Voxels of intensity 0 inside the mask
   A ROI voxel of intensity 0 is treated as background in every mode: it is not counted and does
   not count as a neighbour. PyRadiomics keeps it as a grey level. On the worked example with the
   voxel in slice 2, row 2, column 2 set to 0, Nyxus counts 47 voxels and returns small dependence
   emphasis 0.06801 and large dependence emphasis 33.64, while PyRadiomics (``binWidth=1``) counts
   48 and returns 0.08743 and 32.96.

Voxels outside the mask
   Voxels of the bounding box outside the mask, including voxels of other ROIs, are excluded in
   every mode. On the worked example with three voxels left out of the mask, small dependence
   emphasis is 0.07001 and large dependence emphasis 32.87, and the values are the same when those
   three voxels belong to a second ROI.

Isolated voxels
   A ROI voxel with no ROI neighbour of its own level is still counted, with dependence 1. A ROI of
   two separate voxels returns small and large dependence emphasis 1.

Constant ROI
   When all of a ROI's voxels have one intensity, every 3D GLDM feature returns the ``--noval``
   substitute. PyRadiomics returns values for such a ROI.


Validation
----------

The test suite checks every 3D GLDM feature against PyRadiomics, named in the
:ref:`index <gldm-3d-index>`. On a 3D phantom volume with 20 PyRadiomics-style bins
(``3gldm/greydepth=-20`` against PyRadiomics ``binCount=20``, ``gldm_a=0``, distance 1), the
thirteen features without a logarithm agree to an absolute 10\ :sup:`-9` (measured
7 × 10\ :sup:`-15`), and the test also checks every cell of the dependence matrix. Dependence
entropy agrees to an absolute 4 × 10\ :sup:`-3` because of the approximate logarithm. The
default (no binning) and the positive ``3gldm/greydepth`` binning are covered by regression tests
only, and so is the constant-ROI result.


References
----------

1. Sun C, Wee WG. *Neighboring gray level dependence matrix for texture classification.* Computer
   Vision, Graphics, and Image Processing. 1983;23(3):341–352.
2. van Griethuysen JJM, et al. *Computational radiomics system to decode the radiographic
   phenotype.* Cancer Research. 2017;77(21):e104–e107.
3. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
