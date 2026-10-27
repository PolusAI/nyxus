Neighbouring grey-level dependence matrix, 3D (3D NGLDM)
========================================================

The 3D NGLDM features describe how many of each voxel's neighbours share its grey level. Every ROI
voxel is the centre of a 3 × 3 × 3 neighbourhood, and the matrix counts the voxels by grey level and
by dependence count, regardless of direction. The features are those of the 2D family, computed on
the 3D matrix; their formulas are on the :doc:`2D NGLDM page <f_ngldm>`. This page covers what is
specific to 3D.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``3NGLDM_*``: 19 features, one value per ROI
   * - **Group**
     - ``*3D_NGLDM*``; also part of ``*3D_ALL_TEXTURE*`` and ``*3D_ALL*``
   * - **Dimensions**
     - 3D. The 2D family, ``NGLDM_*``, is on :doc:`its own page <f_ngldm>`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory. When one read of the file returns the whole volume
       (NIfTI, or OME-Zarr chunked over all of Z), an oversized ROI stops the run with an error
       (see :doc:`/large_images`).
   * - **IBSI**
     - 17 of the 19 features have an IBSI identifier. Grey level mean and dependence count mean
       have none.

.. tip::

   To reproduce IBSI or MIRP values, run in IBSI mode: ``--ibsi=true`` on the command line,
   ``ibsi=True`` in Python. With the default settings a voxel's grey level is
   :math:`\lfloor 64\,x / x_{\max} \rfloor`, so the eight features that use the grey level's value
   differ from those tools (see :ref:`ngldm-3d-possible-issues`). Voxels outside the mask are left
   out in every mode.


How the matrix is built
-----------------------

Nyxus visits every ROI voxel. It counts the voxel's 26 neighbours that lie inside the ROI and have
exactly the same grey level, and adds 1 for the voxel itself: that is the voxel's dependence count
:math:`j`, from 1 to 27. A voxel with grey level :math:`g_i` and dependence count :math:`j` adds 1
to :math:`\textbf{S}(i,j)`.

- **Neighbourhood:** the 26 voxels at Chebyshev distance 1: the 8 in the voxel's own slice and the 9
  in each of the slices above and below. A neighbour that shares a face, an edge or a corner counts
  the same. The distance cannot be changed.
- **Coarseness parameter:** two voxels are dependent only when their grey levels are equal: IBSI's
  :math:`\alpha = 0`. :math:`\alpha` cannot be changed.
- **Directions:** none. One matrix is built per ROI, and each feature has one value.
- **Grey levels:** for a positive ``--coarseGrayDepth``, the same rule as the 2D family, not the
  MATLAB-style binning of the other texture families. With ``--coarseGrayDepth`` :math:`N` (64 by
  default), intensity :math:`x` becomes
  grey level :math:`\lfloor N x / x_{\max} \rfloor`, where :math:`x_{\max}` is the ROI's largest
  intensity. Levels run from 0 to :math:`N`: the ROI maximum gets level :math:`N`, and every
  intensity below :math:`x_{\max}/N`, including 0, gets level 0. In IBSI mode and with
  ``--coarseGrayDepth=0`` the intensities are not converted: each integer intensity is a grey
  level. A negative value gives :math:`|N|` PyRadiomics-style bins over the ROI's range, the shared
  rule, under which every ROI voxel has level 1 or more (see :ref:`conventions-grey-levels`). The
  3D family has no ``greydepth`` metaparameter of its own.
- **Rows:** :math:`\textbf{S}` has one row per grey level present in the ROI, in increasing order,
  in both modes. Levels with no voxel have no row, so the matrix has at most :math:`N + 1` rows with
  the default conversion. Columns run from dependence count 1 to the largest one in the ROI, at most
  27.
- **Out-of-ROI neighbours:** voxels outside the mask are neither centres nor neighbours, in every
  mode, whatever their intensity. A neighbour outside the volume counts the same way. ROI voxels of
  intensity 0 are centres with grey level 0 (see :ref:`ngldm-3d-possible-issues`).


Worked example
^^^^^^^^^^^^^^

Take a 3 × 3 × 3 ROI that already has four grey levels, shown slice by slice:

.. math::

   \textbf{G}_{z=1} = \begin{bmatrix}
   1 & 1 & 2\\
   1 & 2 & 2\\
   4 & 3 & 3 \end{bmatrix},
   \qquad
   \textbf{G}_{z=2} = \begin{bmatrix}
   1 & 1 & 2\\
   3 & 2 & 2\\
   4 & 4 & 3 \end{bmatrix},
   \qquad
   \textbf{G}_{z=3} = \begin{bmatrix}
   1 & 2 & 2\\
   3 & 3 & 4\\
   4 & 4 & 1 \end{bmatrix}

In IBSI mode the matrix is below; rows are grey levels 1 to 4 and columns are dependence counts 1
to 8. The 1 in the last corner of slice 3 touches no other 1, so its dependence count is 1. The 1 in
the first corner of slice 1 has four 1s around it in its own slice and the slice below, so its
dependence count is 5.

.. math::

   \textbf{S} = \begin{bmatrix}
   1 & 0 & 1 & 0 & 3 & 2 & 0 & 0\\
   0 & 0 & 0 & 0 & 2 & 3 & 0 & 3\\
   0 & 0 & 2 & 4 & 0 & 0 & 0 & 0\\
   0 & 0 & 2 & 1 & 2 & 1 & 0 & 0 \end{bmatrix},
   \qquad N_s = \sum_{i,j} \textbf{S}(i,j) = 27

With the default settings the grey levels are 16, 32, 48 and 64, and :math:`\textbf{S}` is the
same. The *Example* value of each feature in the :ref:`index <ngldm-3d-index>` is what Nyxus
returns in IBSI mode for this ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/ngldm_3d_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --dim=3 --features="*3D_NGLDM*" --ibsi=true \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

The symbols are those of the :doc:`2D page <f_ngldm>`, with voxels in place of pixels:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N_g`
     - Number of rows of :math:`\textbf{S}`: the number of grey levels present in the ROI.
   * - :math:`g_i`
     - Grey level of row :math:`i`: the :math:`i`-th smallest level present. In IBSI mode it is the
       intensity; otherwise it is :math:`\lfloor N x / x_{\max} \rfloor`.
   * - :math:`j`
     - Dependence count: 1 plus the number of same-level neighbours inside the ROI, from 1 to 27.
   * - :math:`N_r`
     - Largest dependence count in the ROI.
   * - :math:`\textbf{S}(i,j)`
     - Number of ROI voxels with grey level :math:`g_i` and dependence count :math:`j`.
   * - :math:`N_s`
     - Number of neighbourhoods, :math:`N_s = \sum_{i,j}\textbf{S}(i,j)`. Every ROI voxel is one, so
       :math:`N_s` is the number of ROI voxels.
   * - :math:`p_{ij}`
     - Joint probability, :math:`p_{ij} = \textbf{S}(i,j) / N_s`.

As in 2D, every feature uses the grey level :math:`g_i`, never the row position. The three features
that divide by :math:`g_i^2` skip the row of grey level 0, if there is one, but its voxels stay in
:math:`N_s`. Dependence count entropy uses an exact base-2 logarithm.


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
     - Positive: :math:`N` in :math:`\lfloor N x / x_{\max} \rfloor`, giving levels 0 to :math:`N`;
       0: no conversion; negative: :math:`|N|` PyRadiomics-style bins over the ROI range.
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus3D(..., ibsi=True)``
     - No conversion: each integer intensity is a grey level.

The neighbourhood is fixed at the 26 neighbours with :math:`\alpha = 0`. The family reads the global
``--coarseGrayDepth``, not a metaparameter: ``set_metaparam("3ngldm/greydepth=N")`` is refused with
an unrecognized-feature error, and the ``greydepth`` metaparameters of the other 3D texture families
(``3gldm/greydepth`` and so on) do not change this matrix. All options are listed in the
:ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named with its code: ``3NGLDM_LDE``, ``3NGLDM_DCENT`` and so on.
There are no per-direction columns and no ``_AVE`` columns.


.. _ngldm-3d-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example, in IBSI mode.

.. include:: ngldm_3d_index.inc


Features
--------

Each 3D feature is the 2D formula of the same name, evaluated on the 3D matrix. Only the ranges
change, because the dependence count runs to 27 instead of 9.

.. _feature-3ngldm-dcene:

Dependence count energy
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_DCENE``
:Range: :math:`(0, 1]`, dimensionless

The sum of the squared joint probabilities. Formula: :ref:`NGLDM_DCENE <feature-ngldm-dcene>`.


.. _feature-3ngldm-dcent:

Dependence count entropy
^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_DCENT``
:Range: :math:`[0, \log_2(27 N_g)]` bits

The base-2 entropy of the joint distribution of grey level and dependence count. Formula:
:ref:`NGLDM_DCENT <feature-ngldm-dcent>`.

**Relations.** In IBSI mode ``3GLDM_DE`` is computed on a matrix that, on the worked example, is
the same as this one, but with an approximate logarithm: 3.557, against 3.560 here.


.. _feature-3ngldm-dcm:

Dependence count mean
^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_DCM``
:Range: :math:`[1, 27]`, voxels

The mean dependence count over all ROI voxels. Formula: :ref:`NGLDM_DCM <feature-ngldm-dcm>`. IBSI
defines no such feature.


.. _feature-3ngldm-dcnu:

Dependence count non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_DCNU``
:Range: :math:`[N_s/N_r, N_s]`, voxels

Measures how evenly the voxels are spread over the dependence counts. Formula:
:ref:`NGLDM_DCNU <feature-ngldm-dcnu>`.


.. _feature-3ngldm-dcnun:

Dependence count non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_DCNUN``
:Range: :math:`[1/N_r, 1]`, dimensionless

Dependence count non-uniformity divided by the number of voxels once more. Formula:
:ref:`NGLDM_DCNUN <feature-ngldm-dcnun>`.


.. _feature-3ngldm-dcp:

Dependence count percentage
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_DCP``
:Range: 1, dimensionless

The fraction of ROI voxels that are neighbourhood centres. Every ROI voxel is one, so the value is
always 1, and it carries no information about the ROI. Formula:
:ref:`NGLDM_DCP <feature-ngldm-dcp>`.


.. _feature-3ngldm-dcv:

Dependence count variance
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_DCV``
:Range: :math:`[0, 169]`, squared voxels

The variance of the dependence count over all ROI voxels. Formula:
:ref:`NGLDM_DCV <feature-ngldm-dcv>`.


.. _feature-3ngldm-glm:

Grey level mean
^^^^^^^^^^^^^^^

:Code: ``3NGLDM_GLM``
:Range: :math:`[g_1, g_{N_g}]`, grey levels

The mean grey level over all ROI voxels. Formula: :ref:`NGLDM_GLM <feature-ngldm-glm>`. IBSI
defines no such feature. With the default settings it is 38.52 on the worked example, against 2.407
in IBSI mode.


.. _feature-3ngldm-glnu:

Grey level non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_GLNU``
:Range: :math:`[N_s/N_g, N_s]`, voxels

Measures how evenly the voxels are spread over the grey levels. Formula:
:ref:`NGLDM_GLNU <feature-ngldm-glnu>`.


.. _feature-3ngldm-glnun:

Grey level non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_GLNUN``
:Range: :math:`[1/N_g, 1]`, dimensionless

Grey level non-uniformity divided by the number of voxels once more. Formula:
:ref:`NGLDM_GLNUN <feature-ngldm-glnun>`.


.. _feature-3ngldm-glv:

Grey level variance
^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_GLV``
:Range: :math:`[0, \infty)`, squared grey levels

The variance of the grey level over all ROI voxels. Formula: :ref:`NGLDM_GLV <feature-ngldm-glv>`.
It scales with the square of the grey-level spacing: 1.204 in IBSI mode and 308.3 with the default
settings on the worked example.


.. _feature-3ngldm-hde:

High dependence emphasis
^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_HDE``
:Range: :math:`[1, 729]`, squared voxels

The mean over all ROI voxels of the squared dependence count. Formula:
:ref:`NGLDM_HDE <feature-ngldm-hde>`.


.. _feature-3ngldm-hdhgle:

High dependence high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_HDHGLE``
:Range: :math:`[0, \infty)`, squared grey levels times squared voxels

The mean over all ROI voxels of the squared dependence count times the squared grey level. Formula:
:ref:`NGLDM_HDHGLE <feature-ngldm-hdhgle>`.


.. _feature-3ngldm-hdlgle:

High dependence low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_HDLGLE``
:Range: :math:`[0, 729]` when :math:`g_i \ge 1`, squared voxels over squared grey levels

The mean over all ROI voxels of the squared dependence count over the squared grey level. Formula:
:ref:`NGLDM_HDLGLE <feature-ngldm-hdlgle>`. Voxels of grey level 0 add nothing to the sum but count
in :math:`N_s`.


.. _feature-3ngldm-hglce:

High grey level count emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_HGLCE``
:Range: :math:`[0, \infty)`, squared grey levels

The mean over all ROI voxels of the squared grey level. Formula:
:ref:`NGLDM_HGLCE <feature-ngldm-hglce>`. It depends on the grey-level conversion: 7 in IBSI mode
and 1792 with the default settings on the worked example.


.. _feature-3ngldm-lde:

Low dependence emphasis
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_LDE``
:Range: :math:`[1/729, 1]`, inverse squared voxels

The mean over all ROI voxels of one over the squared dependence count. Formula:
:ref:`NGLDM_LDE <feature-ngldm-lde>`.

**Relations.** In IBSI mode ``3GLDM_SDE`` returns the same value on the worked example.


.. _feature-3ngldm-ldhgle:

Low dependence high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_LDHGLE``
:Range: :math:`[0, \infty)`, squared grey levels over squared voxels

The mean over all ROI voxels of the squared grey level over the squared dependence count. Formula:
:ref:`NGLDM_LDHGLE <feature-ngldm-ldhgle>`.


.. _feature-3ngldm-ldlgle:

Low dependence low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_LDLGLE``
:Range: :math:`[0, 1]`, inverse squared grey levels times inverse squared voxels

The mean over all ROI voxels of one over the squared dependence count times the squared grey level.
Formula: :ref:`NGLDM_LDLGLE <feature-ngldm-ldlgle>`. Voxels of grey level 0 add nothing to the sum
but count in :math:`N_s`.


.. _feature-3ngldm-lglce:

Low grey level count emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3NGLDM_LGLCE``
:Range: :math:`[0, 1]`, inverse squared grey levels

The mean over all ROI voxels of one over the squared grey level. Formula:
:ref:`NGLDM_LGLCE <feature-ngldm-lglce>`. It depends on the grey-level conversion: 0.3719 in IBSI
mode and 0.001453 with the default settings on the worked example. Voxels of grey level 0 add
nothing to the sum but count in :math:`N_s`.


.. _ngldm-3d-possible-issues:

Possible issues
---------------

Grey level 0 is counted but not weighted
   ``LGLCE``, ``LDLGLE`` and ``HDLGLE`` divide by :math:`g_i^2`, so they skip the row of grey level
   0. Its voxels still count in :math:`N_s`, which lowers these three features instead of leaving
   them as if those voxels were absent. Grey level 0 comes from ROI voxels of intensity 0, in every
   mode, and with the default conversion from every intensity below :math:`x_{\max}/N`. On the
   worked example with its centre voxel set to 0, IBSI mode gives ``LGLCE`` 0.3627; leaving that
   voxel out of :math:`N_s` too would give 0.3766. With the worked example's intensities times 20
   and one voxel of intensity 1, which the default conversion maps to level 0, ``LGLCE`` is
   0.001308. The other features treat level 0 as an ordinary grey level.

The grey levels differ from the other 3D texture families
   For a positive ``--coarseGrayDepth`` the 3D NGLDM converts intensities with the 2D NGLDM rule,
   :math:`\lfloor N x / x_{\max} \rfloor`. The other 3D texture families bin the same setting with
   the shared rule, MATLAB-style for a positive value, unless their own ``greydepth``
   metaparameter is set. The same positive setting therefore gives different grey levels in this
   family and the others. The
   eight features that use :math:`g_i` (``GLM``, ``GLV``, ``LGLCE``, ``HGLCE`` and the four
   combined emphases) therefore depend on ``--coarseGrayDepth``; compare them only between runs
   with the same setting. The features that use only counts (``LDE``, ``HDE``, ``GLNU``, ``GLNUN``,
   ``DCNU``, ``DCNUN``, ``DCM``, ``DCV``, ``DCENT``, ``DCENE``) change only when the conversion
   merges or splits levels. On the worked example ``--coarseGrayDepth=8`` gives levels 2, 4, 6 and
   8 and leaves every count-only feature unchanged.

Negative or zero ``--coarseGrayDepth``
   These follow the shared rule. ``--coarseGrayDepth=0`` gives the IBSI-mode values. A negative
   value bins into :math:`|N|` PyRadiomics-style bins over the ROI's range: on the worked example
   ``--coarseGrayDepth=-8`` gives intensities 1 to 4 the levels 1, 3, 6 and 8, ``GLM`` 4.259 and
   ``HGLCE`` 25.15, while the count-only features keep their IBSI-mode values.

Constant ROI
   When all of a ROI's voxels have one intensity, every 3D NGLDM feature returns the ``--noval``
   substitute (0 by default), in memory and out of core.

The ROI is defined by the mask, not the bounding box
   Neighbours outside the mask never count, whatever their intensity. On the worked example placed
   inside a 5 × 5 × 5 volume whose other voxels all have intensity 4, every feature keeps its value.
   Leaving three voxels of the worked example out of the mask gives ``LDE`` 0.06111 and ``HDE``
   20.75 in both modes.

No IBSI counterpart for two features
   ``GLM`` and ``DCM`` are not IBSI features. They are the means that IBSI uses inside grey level
   variance and dependence count variance.


Validation
----------

The test suite checks 16 of the 19 3D NGLDM features against MIRP at two settings, asserted at a
relative 10\ :sup:`-3`:

- IBSI mode, with MIRP given the same raw intensities and no discretisation. Nyxus matches MIRP to
  a relative 7.5 × 10\ :sup:`-15`.
- The default conversion with 64 levels, with MIRP given the grey levels Nyxus produces. Nyxus
  matches MIRP to a relative 8.7 × 10\ :sup:`-16`. This check covers the matrix and the feature
  formulas, not the conversion itself.

Dependence count percentage is pinned by a regression test only: Nyxus always returns 1, and MIRP
returns 1 on the test volume too, so an agreement could not fail. Grey level mean and dependence count mean have no MIRP
counterpart and are covered by regression tests only. The validation of each feature is named in
the :ref:`index <ngldm-3d-index>`.


References
----------

1. Sun C, Wee WG. *Neighboring gray level dependence matrix for texture classification.* Computer
   Vision, Graphics, and Image Processing. 1983;23(3):341–352.
2. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
3. MIRP, the medical image radiomics processor: https://github.com/oncoray/mirp
