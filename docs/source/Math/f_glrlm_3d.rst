Grey-level run-length matrix in 3D (3GLRLM)
===========================================

3D GLRLM features describe the runs in a volumetric ROI: straight lines of consecutive voxels that
share one grey level. They count runs by grey level and length along each of 13 directions, and
capture whether a texture is fine (many short runs) or coarse (few long runs). The features are
those of the 2D :doc:`GLRLM family <f_glrlm>`, computed on 3D run-length matrices; this page covers
what is specific to 3D.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``3GLRLM_*``: 16 features, each reporting one direction, and their ``3GLRLM_*_AVE``
       averages over the 13 directions
   * - **Group**
     - ``*3D_GLRLM*``; also part of ``*3D_ALL_TEXTURE*`` and ``*3D_ALL*``
   * - **Dimensions**
     - 3D. The 2D family, ``GLRLM_*``, is described in :doc:`f_glrlm`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory.
       When one read of the file returns the whole volume (NIfTI, or OME-Zarr chunked over all of
       Z), an oversized ROI stops the run with an error (see :doc:`/large_images`).
   * - **IBSI**
     - All 16 features have an IBSI identifier. The ``_AVE`` codes follow the IBSI aggregation
       "averaged over the 3D directions" (ITBB).

.. tip::

   The plain codes, such as ``3GLRLM_SRE``, report a single direction, not an average. Use the
   ``_AVE`` codes to compare with IBSI, PyRadiomics or MIRP values, which aggregate over all
   directions. By default the 3D family does not bin grey levels, so the default settings and IBSI
   mode give the same values; ``--coarseGrayDepth`` does not apply to it (see
   :ref:`glrlm-3d-settings`).


How the matrix is built
-----------------------

For a direction :math:`\theta`, Nyxus scans the voxels of the ROI's bounding box slice by slice and
row by row. Each voxel that does not yet belong to a run starts a new run, which grows along
:math:`\theta` while the next voxel has the same grey level. The finished run, with grey level
:math:`i` and length :math:`j` voxels, adds 1 to :math:`\textbf{P}(i,j)`. Each direction gets its
own matrix and its own feature values; the matrices are not merged.

- **Directions:** the 13 directions of the 26-voxel neighbourhood, one of each opposite pair, as a
  step :math:`(\Delta z, \Delta y, \Delta x)` in voxels, with slices, rows and columns numbered from
  0. They are computed in this order:

  .. list-table::
     :widths: 10 30 10 30
     :header-rows: 1

     * - #
       - Step :math:`(\Delta z, \Delta y, \Delta x)`
       - #
       - Step :math:`(\Delta z, \Delta y, \Delta x)`
     * - 1
       - :math:`(1, 1, 1)`
       - 8
       - :math:`(1, -1, 0)`
     * - 2
       - :math:`(1, 1, 0)`
       - 9
       - :math:`(1, -1, -1)`
     * - 3
       - :math:`(1, 1, -1)`
       - 10
       - :math:`(0, 1, 1)`
     * - 4
       - :math:`(1, 0, 1)`
       - 11
       - :math:`(0, 1, 0)`
     * - 5
       - :math:`(1, 0, 0)`
       - 12
       - :math:`(0, 1, -1)`
     * - 6
       - :math:`(1, 0, -1)`
       - 13
       - :math:`(0, 0, 1)`
     * - 7
       - :math:`(1, -1, 1)`
       -
       -

  Directions 1 to 9 cross slices; directions 10 to 13 stay within a slice and are the four 2D
  directions. Run lengths count voxels, whatever the voxel spacing. The 13 directions are always
  all computed; no setting selects them.
  The :doc:`3D GLCM <f_glcm_3d>` page lists the same 13 triples in the same order, but written
  :math:`(d_x, d_y, d_z)`, so a direction number names a different step in the two families:
  direction 5, :math:`(1, 0, 0)`, crosses slices here and is a step along :math:`x` in 3D GLCM.
  Directions 1, 4, 7 and 11, among them direction 1, :math:`(1, 1, 1)`, the one the plain codes
  report, are the same step in both.
- **Grey levels:** by default intensities are not binned: each integer intensity is a grey level,
  from 1 to the ROI's maximum. The ``3glrlm/greydepth`` metaparameter selects a binning (see
  :ref:`glrlm-3d-settings`).
- **Background:** voxels with grey level 0 belong to no run. Without binning, in IBSI mode and with
  PyRadiomics-style binning, this excludes the voxels of the bounding box outside the mask and ROI
  voxels of intensity 0. MATLAB-style binning (positive ``3glrlm/greydepth``) maps intensity 0 to
  level 1, so there such voxels do form runs (see :ref:`glrlm-3d-possible-issues`).


Worked example
^^^^^^^^^^^^^^

Take a 3 × 3 × 3 ROI that already has four grey levels, shown slice by slice
(:math:`z = 0, 1, 2`), with rows :math:`y` from the top and columns :math:`x` from the left:

.. math::

   \textbf{G}_{z=0} = \begin{bmatrix}
   1 & 2 & 4\\
   3 & 1 & 2\\
   3 & 3 & 1 \end{bmatrix},
   \qquad
   \textbf{G}_{z=1} = \begin{bmatrix}
   2 & 2 & 3\\
   4 & 1 & 3\\
   2 & 1 & 1 \end{bmatrix},
   \qquad
   \textbf{G}_{z=2} = \begin{bmatrix}
   3 & 4 & 2\\
   2 & 2 & 3\\
   1 & 2 & 1 \end{bmatrix}

For direction 1, :math:`(1, 1, 1)`, the matrix is below, with grey levels 1 to 4 as rows and run
lengths 1 to 3 as columns. Voxels are written :math:`(z, y, x)`. The main diagonal, level 1 at :math:`(0,0,0)`, :math:`(1,1,1)` and
:math:`(2,2,2)`, is the one run of length 3; the 2 at :math:`(1,0,0)` continues to the 2 at
:math:`(2,1,1)`, a run of length 2.

.. math::

   \textbf{P} = \begin{bmatrix}
   3 & 1 & 1\\
   7 & 1 & 0\\
   7 & 0 & 0\\
   3 & 0 & 0 \end{bmatrix},
   \qquad N_s = \sum_{i,j} \textbf{P}(i,j) = 23, \qquad N_v = 27

The *Example* value of each feature in the :ref:`index <glrlm-3d-index>` is what Nyxus returns for this ROI in the plain code, which is
direction 1. The test suite runs this example and checks every one of them, in IBSI mode and with
the default settings, which give the same values.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/glrlm_3d_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --dim=3 --features="*3D_GLRLM*" --ibsi=true \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

The symbols are those of the :doc:`2D page <f_glrlm>`, with voxels in place of pixels:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N_g`
     - Number of grey levels: the maximum grey level without binning or in IBSI mode, otherwise the
       number of distinct binned levels in the bounding box.
   * - :math:`N_r`
     - Length of the longest run along :math:`\theta`.
   * - :math:`i,\; j`
     - Grey level of a run (:math:`i`) and its length in voxels (:math:`j`). Formulas use the grey
       level's value: the intensity without binning, the binned level otherwise.
   * - :math:`\textbf{P}(i,j)`
     - Number of runs with grey level :math:`i` and length :math:`j` along one :math:`\theta`.
   * - :math:`N_s`
     - Number of runs along :math:`\theta`, :math:`N_s = \sum_{i,j}\textbf{P}(i,j)`.
   * - :math:`N_v`
     - Number of ROI voxels, including voxels of intensity 0.
   * - :math:`p_{ij},\; r_{i\cdot},\; r_{\cdot j},\; \mu_i,\; \mu_j,\; \varepsilon`
     - As on the :doc:`2D page <f_glrlm>`.

Logarithms are computed with the same fast base-2 approximation as in 2D. On the worked example
run entropy is 2.3975 for direction 1, while an exact logarithm gives 2.4013, a relative
1.6 × 10\ :sup:`-3`.


.. _glrlm-3d-settings:

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
     - ``Nyxus3D.set_metaparam("3glrlm/greydepth=0")``
     - 0 (the default): no binning, each integer intensity is a grey level. Positive :math:`N`:
       MATLAB-style binning into :math:`N` levels from 0 to the ROI maximum, intensity
       :math:`x` going to :math:`\lfloor N x / x_{\max} + 1 \rfloor`, capped at :math:`N`, and 0 to
       level 1. Negative :math:`-N`: PyRadiomics-style binning into :math:`N` equal-width bins
       between the ROI minimum and maximum, with 0 kept as background.
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus3D(..., ibsi=True)``
     - No binning, whatever ``3glrlm/greydepth`` says.

``3glrlm/greydepth`` is set and read with ``set_metaparam`` and ``get_metaparam`` on a ``Nyxus3D``
object; the command line has no option for it. It applies to this family only. ``--coarseGrayDepth``
(``coarse_gray_depth``) does not change the 3D GLRLM: on the worked example it gives the same values
at 2 levels as at 64. ``--glcmAngles`` does not apply either. All options are listed in the
:ref:`cli-reference`.

On the worked example, ``3glrlm/greydepth=2`` maps the levels 1, 2, 3 and 4 to 1, 2, 2 and 2, which
merges runs: run percentage drops from 0.8519 to 0.7407 and grey level variance from 0.9225 to
0.1875.


Output columns
--------------

Each feature produces two columns:

- ``3GLRLM_SRE``: the value for direction 1, :math:`(1, 1, 1)`, only. The other twelve directions
  are computed but not written to the output.
- ``3GLRLM_SRE_AVE``: the mean of the 13 per-direction values.

On the worked example ``3GLRLM_SRE`` is 0.8961 and ``3GLRLM_SRE_AVE`` is 0.9033; ``3GLRLM_RP`` is
0.8519 and ``3GLRLM_RP_AVE`` is 0.8803.


.. _glrlm-3d-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the direction-1 value on the worked example.

.. include:: glrlm_3d_index.inc


Features
--------

Each feature is defined on the :doc:`2D page <f_glrlm>`, with the 3D matrix of one direction in
place of the 2D one and :math:`N_v` in place of :math:`N_p`. The formulas are computed the same way
in 3D, with the differences in integer arithmetic noted below.

.. _feature-3glrlm-gln:

Grey level non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLRLM_GLN``
:Range: :math:`[1, N_s]`, runs

Defined as :ref:`GLRLM_GLN <feature-glrlm-gln>`.

.. _feature-3glrlm-glnn:

Grey level non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLRLM_GLNN``
:Range: :math:`(0, 1]`, dimensionless

Defined as :ref:`GLRLM_GLNN <feature-glrlm-glnn>`.

.. _feature-3glrlm-glv:

Grey level variance
^^^^^^^^^^^^^^^^^^^

:Code: ``3GLRLM_GLV``
:Range: :math:`[0, \infty)`, squared grey levels

Defined as :ref:`GLRLM_GLV <feature-glrlm-glv>`.

.. _feature-3glrlm-hglre:

High grey level run emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLRLM_HGLRE``
:Range: :math:`[1, \infty)`, squared grey levels

Defined as :ref:`GLRLM_HGLRE <feature-glrlm-hglre>`. Unlike the 2D code, the 3D code computes
:math:`i^2` in floating point, so it does not wrap for large grey levels.

.. _feature-3glrlm-lglre:

Low grey level run emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLRLM_LGLRE``
:Range: :math:`(0, 1]`, inverse squared grey levels

Defined as :ref:`GLRLM_LGLRE <feature-glrlm-lglre>`.

.. _feature-3glrlm-lre:

Long run emphasis
^^^^^^^^^^^^^^^^^

:Code: ``3GLRLM_LRE``
:Range: :math:`[1, N_r^2]`, squared voxels

Defined as :ref:`GLRLM_LRE <feature-glrlm-lre>`.

.. _feature-3glrlm-lrhgle:

Long run high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLRLM_LRHGLE``
:Range: :math:`[1, \infty)`, squared grey levels times squared voxels

Defined as :ref:`GLRLM_LRHGLE <feature-glrlm-lrhgle>`.

**Possible issues.** :math:`i^2 j^2` is computed in 32-bit unsigned integers and wraps once it
reaches :math:`2^{32}`. An 8 × 8 × 8 ROI at intensity 10000, with one voxel at 9999, has a run of 8
along direction 1 and gives :math:`1.0529 \times 10^9` instead of :math:`1.2308 \times 10^9` in IBSI
mode (see :ref:`glrlm-3d-possible-issues`).

.. _feature-3glrlm-lrlgle:

Long run low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLRLM_LRLGLE``
:Range: :math:`(0, \infty)`, squared voxels over squared grey levels

Defined as :ref:`GLRLM_LRLGLE <feature-glrlm-lrlgle>`.

.. _feature-3glrlm-re:

Run entropy
^^^^^^^^^^^

:Code: ``3GLRLM_RE``
:Range: :math:`[0, \log_2(N_g N_r)]` bits

Defined as :ref:`GLRLM_RE <feature-glrlm-re>`, with the same approximate logarithm.

.. _feature-3glrlm-rln:

Run length non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLRLM_RLN``
:Range: :math:`[1, N_s]`, runs

Defined as :ref:`GLRLM_RLN <feature-glrlm-rln>`.

.. _feature-3glrlm-rlnn:

Run length non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLRLM_RLNN``
:Range: :math:`(0, 1]`, dimensionless

Defined as :ref:`GLRLM_RLNN <feature-glrlm-rlnn>`.

.. _feature-3glrlm-rp:

Run percentage
^^^^^^^^^^^^^^

:Code: ``3GLRLM_RP``
:Range: :math:`(0, 1]`, dimensionless

The number of runs along the direction per ROI voxel.

.. math::

   \mathrm{RP} = \frac{N_s}{N_v}

:math:`N_v` counts every ROI voxel, including voxels of intensity 0 that form no run.

**Possible issues.** With MATLAB-style binning, runs of voxels outside the mask are counted in
:math:`N_s` but not in :math:`N_v`, so the value can exceed 1 (see
:ref:`glrlm-3d-possible-issues`).

.. _feature-3glrlm-rv:

Run variance
^^^^^^^^^^^^

:Code: ``3GLRLM_RV``
:Range: :math:`[0, \infty)`, squared voxels

Defined as :ref:`GLRLM_RV <feature-glrlm-rv>`.

.. _feature-3glrlm-sre:

Short run emphasis
^^^^^^^^^^^^^^^^^^

:Code: ``3GLRLM_SRE``
:Range: :math:`(0, 1]`, inverse squared voxels

Defined as :ref:`GLRLM_SRE <feature-glrlm-sre>`.

.. _feature-3glrlm-srhgle:

Short run high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLRLM_SRHGLE``
:Range: :math:`(0, \infty)`, squared grey levels over squared voxels

Defined as :ref:`GLRLM_SRHGLE <feature-glrlm-srhgle>`.

.. _feature-3glrlm-srlgle:

Short run low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3GLRLM_SRLGLE``
:Range: :math:`(0, 1]`, inverse squared grey levels times inverse squared voxels

Defined as :ref:`GLRLM_SRLGLE <feature-glrlm-srlgle>`.

**Possible issues.** :math:`i^2 j^2` is computed in 32-bit unsigned integers. When it wraps to 0
for any cell of the matrix, empty or not, the result is 0: the 8 × 8 × 8 ROI described under long
run high grey level emphasis gives 0 instead of :math:`3.374 \times 10^{-9}` (see
:ref:`glrlm-3d-possible-issues`).


.. _glrlm-3d-possible-issues:

Possible issues
---------------

The plain codes report one direction
   ``3GLRLM_SRE`` and the other plain codes hold direction 1, :math:`(1, 1, 1)`, only: a diagonal
   through the slices. They are not comparable with a single direction of another tool unless that
   tool's direction is the same step, and they change if the volume is transposed. The ``_AVE``
   codes use all 13 directions and do not depend on their order.

Voxels outside the mask with MATLAB-style binning
   A positive ``3glrlm/greydepth`` maps intensity 0 to grey level 1. Voxels of the bounding box
   outside the mask, and ROI voxels of intensity 0, then form runs of level 1 and join runs of ROI
   voxels at that level. On the worked example with two corner voxels of the bounding box left out
   of the mask and ``3glrlm/greydepth=2``, run percentage is 0.8 instead of 0.72 and low grey level
   run emphasis is 0.5125 instead of 0.4583, the values with those voxels excluded. The default
   (no binning), IBSI mode and PyRadiomics-style binning exclude these voxels.

   These runs can make run percentage exceed 1. The worked-example script builds an ellipsoid with
   semi-axes 12, 30 and 30 voxels, which fills 45,224 of the 86,400 voxels of its 24 × 60 × 60
   bounding box, with intensities drawn uniformly from 1 to 255. ``3GLRLM_RP`` is 1.131 with
   ``3glrlm/greydepth=16``, 1.182 with 64 and 1.191 with 128, and ``3GLRLM_RP_AVE`` is 1.130 with
   64. Without binning they are 0.9962 and 0.9965. A ROI that fills its bounding box cannot exceed
   1: a 20 × 20 × 20 cube of the same kind of intensities gives 0.9863 with
   ``3glrlm/greydepth=64``.

Grey-level values depend on the binning
   As in 2D, ``GLV`` and the six emphasis features that weight runs by grey level use the grey
   level's value: the intensity by default, the bin number with a ``3glrlm/greydepth``. Compare
   them only between runs with the same setting.

Integer overflow with large grey levels or long runs
   The products :math:`i^2` and :math:`i^2 j^2` are computed in 32-bit unsigned integers before
   conversion to floating point, except :math:`i^2` in ``HGLRE``, which is computed in floating
   point:

   .. list-table::
      :widths: 40 60
      :header-rows: 1

      * - Feature
        - Effect
      * - ``LGLRE``, ``SRHGLE``, ``LRLGLE``
        - Wrong for grey levels of 65536 and above, which 8- and 16-bit images do not reach
      * - ``LRHGLE``
        - Wrong when :math:`i^2 j^2 \ge 2^{32}` for a run
      * - ``SRLGLE``
        - 0 when :math:`i^2 j^2` wraps to 0 for any cell of the matrix

   Without binning, the default, this happens with 16-bit intensities and short runs. A 2 × 2 × 2
   ROI with maximum intensity 65535 and runs of length 1 is still exact.

Constant ROI
   When all of a ROI's voxels have one intensity, every 3D GLRLM feature, including the ``_AVE``
   columns, returns the ``--noval`` substitute.


Validation
----------

The test suite checks every 3D GLRLM feature against PyRadiomics on a test phantom with
PyRadiomics-style binning into 20 bins (``3glrlm/greydepth=-20``), named in the
:ref:`index <glrlm-3d-index>`. Each of the 13 per-direction values is compared with PyRadiomics'
value for the same direction, and each ``_AVE`` with PyRadiomics' value aggregated over the directions, to a relative
10\ :sup:`-9`; run entropy agrees to 2 × 10\ :sup:`-3` because of the approximate logarithm.
Out-of-core results are checked against the in-memory ones. MATLAB-style binning is covered by
regression tests only.


References
----------

1. Galloway MM. *Texture analysis using gray level run lengths.* Computer Graphics and Image
   Processing. 1975;4(2):172–179.
2. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
3. van Griethuysen JJM, et al. *Computational radiomics system to decode the radiographic
   phenotype.* Cancer Research. 2017;77(21):e104–e107.
