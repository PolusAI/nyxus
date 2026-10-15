3D grey-level co-occurrence matrix (3D GLCM)
============================================

3D GLCM features describe how often pairs of grey levels occur next to each other inside a 3D ROI,
along each of the 13 directions of the 26-voxel neighbourhood. Every formula is the one of the
:doc:`2D GLCM <f_glcm>`, applied to a 3D co-occurrence matrix. This page covers what is specific to
3D: the directions, which direction a plain code reports, binning, background and the settings.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``3GLCM_*``: 30 features, each a single column, and 29 ``3GLCM_*_AVE`` direction averages
   * - **Group**
     - ``*3D_GLCM*``; also part of ``*3D_ALL_TEXTURE*`` and ``*3D_ALL*``
   * - **Dimensions**
     - 3D. The 2D family, ``GLCM_*``, is on the :doc:`GLCM page <f_glcm>`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes. The out-of-core path streams the volume plane by plane and builds the same 13 matrices.
       When one read of the file returns the whole volume (NIfTI, or OME-Zarr chunked over all of
       Z), an oversized ROI stops the run with an error (see :doc:`/large_images`).
   * - **IBSI**
     - 25 of the 30 features have an IBSI identifier. Four of the other 5 return the same value as
       one that does; ``3GLCM_VARIANCE`` equals IBSI joint variance, which ``3GLCM_JVAR`` matches
       only on a symmetric matrix. The ``_AVE`` codes follow the IBSI *3D:avg* aggregation (ITBB).

.. tip::

   A plain code such as ``3GLCM_CONTRAST`` reports **one direction only**, the diagonal step
   :math:`(1,1,1)`. It is not an average and it changes when the volume is rotated or mirrored. For
   a value over all directions, as IBSI, PyRadiomics and MIRP report it, use the ``_AVE`` code, and
   run in IBSI mode (``ibsi=True``).


How the matrix is built
-----------------------

For a direction :math:`\mathbf{d} = (d_x, d_y, d_z)`, Nyxus visits every voxel of the ROI's
bounding box and pairs it with the voxel one step along :math:`\mathbf{d}`, where :math:`x` is the
column, :math:`y` the row and :math:`z` the plane. If both voxels are inside the bounding box, the
pair of their grey levels is counted.

- **Directions:** the 13 directions that cover the 26-voxel neighbourhood once each, in this order:

  .. list-table::
     :widths: 10 90
     :header-rows: 1

     * - #
       - :math:`(d_x, d_y, d_z)`
     * - 1–9
       - :math:`(1,1,1)`, :math:`(1,1,0)`, :math:`(1,1,-1)`, :math:`(1,0,1)`, :math:`(1,0,0)`,
         :math:`(1,0,-1)`, :math:`(1,-1,1)`, :math:`(1,-1,0)`, :math:`(1,-1,-1)`
     * - 10–13
       - :math:`(0,1,1)`, :math:`(0,1,0)`, :math:`(0,1,-1)`, :math:`(0,0,1)`

  Each direction gets its own matrix and its own feature values; the matrices are not merged.
- **Which direction a plain code reports:** the first one, :math:`(1,1,1)`. The other 12 values
  enter only the ``_AVE`` codes.
- **Numbering:** the :doc:`3D GLRLM <f_glrlm_3d>` page lists the same 13 triples in the same
  order, but written :math:`(\Delta z, \Delta y, \Delta x)`, so a direction number names a
  different step in the two families. Direction 5, :math:`(1,0,0)`, is a step along :math:`x`
  here and a step across planes in 3D GLRLM. Directions 1, 4, 7 and 11, among them direction 1,
  :math:`(1,1,1)`, which the plain codes report, are the same step in both.
- **Distance:** 1 voxel, along each axis of the direction. It cannot be changed (see
  `Settings`_).
- **Grey levels:** intensities are binned into :math:`N_g` equal-width levels spanning 0 to the
  ROI's maximum, as set by ``coarse_gray_depth`` (64 by default) or the ``3glcm/greydepth``
  metaparameter. In IBSI mode they are not binned: each integer intensity is a grey level and
  :math:`N_g` is the ROI's maximum intensity.
- **Symmetry:** in IBSI mode, and with a negative or zero grey depth (PyRadiomics-style bins or no
  binning), each pair is counted in both orders, so :math:`\textbf{P}` is symmetric. With a
  positive grey depth, as by default, each ordered pair is counted once.
- **Background:** in IBSI mode and with PyRadiomics-style binning, pairs that include a voxel
  outside the ROI (or of intensity 0) are not counted. With the default binning they are counted,
  with the background voxel at grey level 1 (see :ref:`glcm-3d-possible-issues`).


Worked example
^^^^^^^^^^^^^^

Take a 3 × 4 × 4 ROI that already has four grey levels, shown plane by plane (rows :math:`y`,
columns :math:`x`):

.. math::

   z=0:\ \begin{bmatrix}
   1 & 2 & 4 & 3\\
   2 & 1 & 3 & 4\\
   4 & 4 & 1 & 2\\
   3 & 1 & 2 & 2 \end{bmatrix}
   \qquad
   z=1:\ \begin{bmatrix}
   2 & 3 & 1 & 1\\
   4 & 2 & 2 & 3\\
   1 & 3 & 4 & 4\\
   2 & 2 & 3 & 1 \end{bmatrix}
   \qquad
   z=2:\ \begin{bmatrix}
   3 & 1 & 2 & 4\\
   1 & 1 & 4 & 2\\
   2 & 4 & 3 & 3\\
   4 & 3 & 1 & 2 \end{bmatrix}

Along :math:`(1,1,1)` each voxel of planes 0 and 1 that is not in the last row or column has a
neighbour, which gives :math:`2 \times 3 \times 3 = 18` pairs. The voxel at :math:`z=0, y=0, x=0`
(level 1) pairs with :math:`z=1, y=1, x=1` (level 2), for example. In IBSI mode each pair is also
counted reversed, and the matrix the plain codes are computed from is:

.. math::

   \textbf{P}_{(1,1,1)} = \begin{bmatrix}
   2 & 3 & 2 & 1\\
   3 & 2 & 3 & 2\\
   2 & 3 & 0 & 4\\
   1 & 2 & 4 & 2 \end{bmatrix},
   \qquad \sum_{i,j} \textbf{P}(i,j) = 36

The other directions give other matrices, with sums of 54 for :math:`(1,1,0)`, 72 for
:math:`(1,0,0)`, 64 for :math:`(0,0,1)`, and so on. The per-direction values differ widely on this ROI. Contrast runs from
1.778 along :math:`(1,-1,0)` to 3.25 along :math:`(0,1,1)`. ``3GLCM_CONTRAST`` is 1.944, the
:math:`(1,1,1)` value, and ``3GLCM_CONTRAST_AVE`` is 2.519, the mean of all 13. Correlation even
changes sign: ``3GLCM_CORRELATION`` is 0.1855 and ``3GLCM_CORRELATION_AVE`` is −0.0387.

The *Example* value of each feature in the :ref:`index <glcm-3d-index>` is what Nyxus returns for
this ROI in the plain column, that is for :math:`\textbf{P}_{(1,1,1)}` above. The test suite runs
this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/glcm_3d_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --dim=3 --features="*3D_GLCM*" --ibsi=true \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

The symbols are those of the :doc:`2D GLCM page <f_glcm>` (:math:`N_g`, :math:`\textbf{P}(i,j)`,
:math:`p_{ij}`, the marginals, :math:`p_{x+y}`, :math:`p_{x-y}`, the entropies and
:math:`\varepsilon`), with one addition:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\mathbf{d}`
     - One of the 13 directions; :math:`\textbf{P}_{\mathbf{d}}` is its co-occurrence matrix and
       :math:`F_{\mathbf{d}}` the value of feature :math:`F` computed from it.
   * - :math:`i,\; j`
     - Grey level of the reference voxel (:math:`i`) and of its neighbour one step along
       :math:`\mathbf{d}` (:math:`j`).

Logarithms use the same fast base-2 approximation as in 2D. On the worked example the entropy
features differ from an exact logarithm by a relative 2 × 10\ :sup:`-4` (joint entropy) to
2 × 10\ :sup:`-3` (information measure of correlation 1).


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
     - Number of grey levels. Positive: equal-width bins; negative: PyRadiomics-style bins (command
       line only; ``Nyxus3D`` rejects a value below 1).
   * - Grey levels, 3D GLCM only
     - not available
     - ``set_metaparam("3glcm/greydepth=N")``
     - Overrides the grey levels for this family only, with the same sign convention; a negative
       value selects PyRadiomics-style bins from Python. On the worked example
       ``3glcm/greydepth=8`` gives the same values as ``coarse_gray_depth=8``.
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus3D(..., ibsi=True)``
     - No binning, pairs counted in both orders, background pairs skipped.
   * - Distance
     - not available
     - ``set_metaparam("3glcm/offset=N")``
     - Accepted but has no effect: the distance is always 1.

``set_metaparam`` also accepts ``3glcm/numang`` and ``3glcm/sparseintensities``; neither changes
the result. ``get_metaparam("3glcm/greydepth")`` and ``get_metaparam("3glcm/offset")`` return the
2D GLCM settings, not the values set with ``3glcm/...``. ``--glcmAngles`` and ``--glcmOff`` apply to
the 2D family only. All options are listed in the :ref:`cli-reference`.


Output columns
--------------

Each feature produces a plain column and, except ``3GLCM_HOM2``, an average column:

- ``3GLCM_CONTRAST``: the value for the direction :math:`(1,1,1)`. Unlike the 2D family, the
  column has no direction suffix, and there are no columns for the other 12 directions.
- ``3GLCM_CONTRAST_AVE``: the mean over the 13 directions,

  .. math::

     F_{\mathrm{AVE}} = \frac{1}{13}\sum_{\mathbf{d}} F_{\mathbf{d}}

Every ``_AVE`` column is named ``3GLCM_<feature>_AVE``. ``3GLCM_HOM2`` has no ``_AVE`` column;
``3GLCM_IDM_AVE`` has the same value. The output lists the 30 plain columns first, then the 29
averages, in the order of the C++ feature set (``3GLCM_VARIANCE_AVE`` comes after
``3GLCM_INFOMEAS2_AVE``, for example), not alphabetically.


.. _glcm-3d-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with the relative tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the plain-column value, direction :math:`(1,1,1)`, on the worked example.

.. include:: glcm_3d_index.inc


Features
--------

Each feature is computed from one directional matrix exactly as its 2D counterpart, with
:math:`\textbf{P}_{\mathbf{d}}` in place of the 2D matrix. The plain code reports
:math:`F_{(1,1,1)}`; the ``_AVE`` code reports the mean over the 13 directions. Ranges are those of
the 2D feature.

.. _feature-3glcm-acor:

Autocorrelation
^^^^^^^^^^^^^^^

:Code: ``3GLCM_ACOR``
:Range: :math:`[1, N_g^2]`, squared grey levels

Defined as :ref:`GLCM_ACOR <feature-glcm-acor>`.


.. _feature-3glcm-asm:

Angular second moment
^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLCM_ASM``
:Range: :math:`(0, 1]`, dimensionless

Defined as :ref:`GLCM_ASM <feature-glcm-asm>`. ``3GLCM_ENERGY`` returns the same value.


.. _feature-3glcm-cluprom:

Cluster prominence
^^^^^^^^^^^^^^^^^^

:Code: ``3GLCM_CLUPROM``
:Range: :math:`[0, \infty)`, grey levels to the fourth power

Defined as :ref:`GLCM_CLUPROM <feature-glcm-cluprom>`, centred on :math:`2\mu_y`. This equals the
IBSI centre :math:`\mu_x + \mu_y` only on a symmetric matrix.


.. _feature-3glcm-clushade:

Cluster shade
^^^^^^^^^^^^^

:Code: ``3GLCM_CLUSHADE``
:Range: :math:`(-\infty, \infty)`, grey levels cubed

Defined as :ref:`GLCM_CLUSHADE <feature-glcm-clushade>`, centred on :math:`2\mu_y`.


.. _feature-3glcm-clutend:

Cluster tendency
^^^^^^^^^^^^^^^^

:Code: ``3GLCM_CLUTEND``
:Range: :math:`[0, \infty)`, squared grey levels

Defined as :ref:`GLCM_CLUTEND <feature-glcm-clutend>`, centred on :math:`2\mu_y`.
``3GLCM_SUMVARIANCE`` returns the same value.


.. _feature-3glcm-contrast:

Contrast
^^^^^^^^

:Code: ``3GLCM_CONTRAST``
:Range: :math:`[0, (N_g-1)^2]`, squared grey levels

Defined as :ref:`GLCM_CONTRAST <feature-glcm-contrast>`. It scales with the square of the
grey-level spacing: on the worked example it is 1.944 in IBSI mode and 478.61 with the default
64-level binning.


.. _feature-3glcm-correlation:

Correlation
^^^^^^^^^^^

:Code: ``3GLCM_CORRELATION``
:Range: :math:`[-1, 1]`, dimensionless

Defined as :ref:`GLCM_CORRELATION <feature-glcm-correlation>`. When a marginal has zero variance,
as on a constant ROI, the result is the ``--noval`` substitute, 0 by default (``-0.0`` in Arrow
and Parquet files; see :ref:`conventions-missing`).


.. _feature-3glcm-difave:

Difference average
^^^^^^^^^^^^^^^^^^

:Code: ``3GLCM_DIFAVE``
:Range: :math:`[0, N_g-1]`, grey levels

Defined as :ref:`GLCM_DIFAVE <feature-glcm-difave>`. Equal to ``3GLCM_DIS``.


.. _feature-3glcm-difentro:

Difference entropy
^^^^^^^^^^^^^^^^^^

:Code: ``3GLCM_DIFENTRO``
:Range: :math:`[0, \log_2 N_g]` bits

Defined as :ref:`GLCM_DIFENTRO <feature-glcm-difentro>`.


.. _feature-3glcm-difvar:

Difference variance
^^^^^^^^^^^^^^^^^^^

:Code: ``3GLCM_DIFVAR``
:Range: :math:`[0, \infty)`, squared grey levels

Defined as :ref:`GLCM_DIFVAR <feature-glcm-difvar>`.


.. _feature-3glcm-dis:

Dissimilarity
^^^^^^^^^^^^^

:Code: ``3GLCM_DIS``
:Range: :math:`[0, N_g-1]`, grey levels

Defined as :ref:`GLCM_DIS <feature-glcm-dis>`. Equal to ``3GLCM_DIFAVE``.


.. _feature-3glcm-energy:

Energy
^^^^^^

:Code: ``3GLCM_ENERGY``
:Range: :math:`(0, 1]`, dimensionless

The same value as ``3GLCM_ASM``; see :ref:`GLCM_ENERGY <feature-glcm-energy>`.


.. _feature-3glcm-entropy:

Entropy
^^^^^^^

:Code: ``3GLCM_ENTROPY``
:Range: :math:`[0, 2\log_2 N_g]` bits

The same value as ``3GLCM_JE``; see :ref:`GLCM_ENTROPY <feature-glcm-entropy>`.


.. _feature-3glcm-hom1:

Homogeneity 1
^^^^^^^^^^^^^

:Code: ``3GLCM_HOM1``
:Range: :math:`(0, 1]`, dimensionless

Defined as :ref:`GLCM_HOM1 <feature-glcm-hom1>`. Equal to ``3GLCM_ID``.


.. _feature-3glcm-hom2:

Homogeneity 2
^^^^^^^^^^^^^

:Code: ``3GLCM_HOM2`` (no ``_AVE`` column)
:Range: :math:`(0, 1]`, dimensionless

Defined as :ref:`GLCM_HOM2 <feature-glcm-hom2>`. Equal to ``3GLCM_IDM``; for the direction average
use ``3GLCM_IDM_AVE``.


.. _feature-3glcm-id:

Inverse difference
^^^^^^^^^^^^^^^^^^

:Code: ``3GLCM_ID``
:Range: :math:`(0, 1]`, dimensionless

Defined as :ref:`GLCM_ID <feature-glcm-id>`.


.. _feature-3glcm-idm:

Inverse difference moment
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLCM_IDM``
:Range: :math:`(0, 1]`, dimensionless

Defined as :ref:`GLCM_IDM <feature-glcm-idm>`.


.. _feature-3glcm-idmn:

Inverse difference moment normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLCM_IDMN``
:Range: :math:`(0, 1]`, dimensionless

Defined as :ref:`GLCM_IDMN <feature-glcm-idmn>`.


.. _feature-3glcm-idn:

Inverse difference normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLCM_IDN``
:Range: :math:`(0, 1]`, dimensionless

Defined as :ref:`GLCM_IDN <feature-glcm-idn>`.


.. _feature-3glcm-infomeas1:

Information measure of correlation 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLCM_INFOMEAS1``
:Range: :math:`[-1, 0]`, dimensionless

Defined as :ref:`GLCM_INFOMEAS1 <feature-glcm-infomeas1>`: it divides by :math:`HY`, the entropy of
the neighbour marginal. When :math:`HY = 0`, as on a constant ROI, the result is the ``--noval``
substitute, 0 by default (``-0.0`` in Arrow and Parquet files; see :ref:`conventions-missing`).


.. _feature-3glcm-infomeas2:

Information measure of correlation 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLCM_INFOMEAS2``
:Range: :math:`[0, 1)`, dimensionless

Defined as :ref:`GLCM_INFOMEAS2 <feature-glcm-infomeas2>`.


.. _feature-3glcm-iv:

Inverse variance
^^^^^^^^^^^^^^^^

:Code: ``3GLCM_IV``
:Range: :math:`[0, 1]`, dimensionless

Defined as :ref:`GLCM_IV <feature-glcm-iv>`.


.. _feature-3glcm-jave:

Joint average
^^^^^^^^^^^^^

:Code: ``3GLCM_JAVE``
:Range: :math:`[1, N_g]`, grey levels

Defined as :ref:`GLCM_JAVE <feature-glcm-jave>`.


.. _feature-3glcm-je:

Joint entropy
^^^^^^^^^^^^^

:Code: ``3GLCM_JE``
:Range: :math:`[0, 2\log_2 N_g]` bits

Defined as :ref:`GLCM_JE <feature-glcm-je>`. ``3GLCM_ENTROPY`` returns the same value.


.. _feature-3glcm-jmax:

Joint maximum
^^^^^^^^^^^^^

:Code: ``3GLCM_JMAX``
:Range: :math:`(0, 1]`, dimensionless

Defined as :ref:`GLCM_JMAX <feature-glcm-jmax>`.


.. _feature-3glcm-jvar:

Joint variance
^^^^^^^^^^^^^^

:Code: ``3GLCM_JVAR``
:Range: :math:`[0, \infty)`, squared grey levels

Defined as :ref:`GLCM_JVAR <feature-glcm-jvar>`: the neighbour's level :math:`j` around the
reference mean :math:`\mu_x`. It equals ``3GLCM_VARIANCE`` on a symmetric matrix; with the default
counting the worked example gives 257.66 against 334.03.


.. _feature-3glcm-sumaverage:

Sum average
^^^^^^^^^^^

:Code: ``3GLCM_SUMAVERAGE``
:Range: :math:`[2, 2N_g]`, grey levels

Defined as :ref:`GLCM_SUMAVERAGE <feature-glcm-sumaverage>`.


.. _feature-3glcm-sumentropy:

Sum entropy
^^^^^^^^^^^

:Code: ``3GLCM_SUMENTROPY``
:Range: :math:`[0, \log_2(2N_g-1)]` bits

Defined as :ref:`GLCM_SUMENTROPY <feature-glcm-sumentropy>`.


.. _feature-3glcm-sumvariance:

Sum variance
^^^^^^^^^^^^

:Code: ``3GLCM_SUMVARIANCE``
:Range: :math:`[0, \infty)`, squared grey levels

Returns the ``3GLCM_CLUTEND`` value, as :ref:`GLCM_SUMVARIANCE <feature-glcm-sumvariance>` does in
2D.


.. _feature-3glcm-variance:

Variance
^^^^^^^^

:Code: ``3GLCM_VARIANCE``
:Range: :math:`[0, \infty)`, squared grey levels

Defined as :ref:`GLCM_VARIANCE <feature-glcm-variance>`; it matches the IBSI joint variance on
every matrix.


.. _glcm-3d-possible-issues:

Possible issues
---------------

Plain codes report one direction
   ``3GLCM_<feature>`` is the value along :math:`(1,1,1)` only. It depends on the orientation of
   the volume, and on the worked example it differs from the 13-direction mean by 23 % for
   contrast (1.944 against 2.519) and in sign for correlation (0.1855 against −0.0387). Use the
   ``_AVE`` codes for orientation-averaged values.

Directions without pairs enter the average
   A direction whose matrix is empty, because the ROI is too thin along it, reports the
   ``--noval`` substitute (0 by default), and that value is averaged with the others. For a ROI
   one plane thick, the 9 directions with :math:`d_z \neq 0`, :math:`(1,1,1)` among them, have no
   pairs: every plain code is 0, and on the middle plane of the worked example
   ``3GLCM_CONTRAST_AVE`` is 0.673, while the mean of the 4 in-plane directions is 2.1875.

Background counted at grey level 1 with the default binning
   With the default (positive) grey depth, voxels of the bounding box outside the ROI are binned to
   level 1 and their pairs are counted. Removing three voxels from the worked example's mask gives
   ``3GLCM_CONTRAST`` 745.28; counting only pairs of ROI voxels would give 493.2. IBSI mode and
   PyRadiomics-style binning skip those pairs (2.0 in IBSI mode on the same mask). The 2D family
   skips them in every mode.

Asymmetric matrix by default
   As in 2D, the default counting takes each ordered pair once, so features centred on
   :math:`2\mu_y` (``CLUPROM``, ``CLUSHADE``, ``CLUTEND``, ``SUMVARIANCE``), ``JVAR`` and
   ``INFOMEAS1`` differ from IBSI and can change when the volume is mirrored. See
   :ref:`glcm-possible-issues` on the 2D page.

Constant ROI
   When all voxels share one grey level the matrix is a single diagonal entry and the features take
   their limiting values (``3GLCM_ASM`` 1, ``3GLCM_CONTRAST`` 0, ``3GLCM_JAVE`` the level).
   ``3GLCM_CORRELATION`` and ``3GLCM_INFOMEAS1`` divide by zero; the result is the ``--noval``
   substitute in CSV files and pandas, and ``-0.0`` in Arrow and Parquet files. This differs from
   the 2D family, which returns 0 for every feature in every format, whatever ``--noval`` is (see
   :ref:`glcm-possible-issues`).

Grey-level values and indices
   As in 2D, with PyRadiomics-style binning some features use positions 1 to :math:`N_g` in the
   list of grey levels present, and others the levels themselves.


Validation
----------

The test suite checks every 3D GLCM feature against PyRadiomics 3.0.1 with PyRadiomics-style
binning (20 bins), where both matrices are symmetric. Each plain feature is compared direction by
direction, all 13 values against PyRadiomics' own per-direction values, and each ``_AVE`` code
against PyRadiomics' direction-averaged value. Entropy features are compared at a looser tolerance
because of the approximate logarithm. Values with the default binning are covered by regression
tests only, and IBSI mode only by the worked-example script on this page.


References
----------

1. Haralick RM, Shanmugam K, Dinstein I. *Textural features for image classification.* IEEE
   Transactions on Systems, Man, and Cybernetics. 1973;SMC-3(6):610–621.
2. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
3. van Griethuysen JJM, et al. *Computational radiomics system to decode the radiographic
   phenotype.* Cancer Research. 2017;77(21):e104–e107.
