Grey-level run-length matrix (GLRLM)
====================================

GLRLM features describe the runs in an ROI: straight lines of consecutive pixels that share one grey
level. They count runs by grey level and length along a fixed direction, and capture whether a
texture is fine (many short runs) or coarse (few long runs).

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``GLRLM_*``: 16 features, each with a column per direction, and their ``GLRLM_*_AVE``
       direction averages
   * - **Group**
     - ``*ALL_GLRLM*``; also part of ``*ALL*`` and ``*WHOLESLIDE*``
   * - **Dimensions**
     - 2D. The 3D family, ``3GLRLM_*``, has its own page, :doc:`f_glrlm_3d`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory on an image that Nyxus reads as one tile. On other
       images out-of-core processing of this family raises an error or returns wrong
       values, depending on the ROI (see :ref:`large-images-tiles`).
   * - **IBSI**
     - All 16 features have an IBSI identifier.

.. tip::

   To reproduce IBSI, PyRadiomics or MIRP values, run in IBSI mode: ``--ibsi=true`` on the command
   line, ``ibsi=True`` in Python. With the default settings the grey levels are binned from 0, so
   every feature that weights runs by grey level differs from those tools, and pixels of the
   bounding box outside the mask take part in runs (see :ref:`glrlm-possible-issues`).


How the matrix is built
-----------------------

For a direction :math:`\theta`, Nyxus scans the ROI's bounding box row by row. Each pixel that does
not yet belong to a run starts a new run, which grows along :math:`\theta` while the next pixel has
the same grey level. The finished run, with grey level :math:`i` and length :math:`j` pixels, adds 1
to :math:`\textbf{P}(i,j)`.

- **Directions:** :math:`\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}`, always all four.
  With rows numbered from the top, a run steps along a row for 0°, down a column for 90°, down and
  to the right for 45°, and down and to the left for 135°. Each direction gets its own matrix and
  its own feature values.
- **Grey levels:** intensities are first binned into grey levels, as set by ``--coarseGrayDepth``
  (64 by default). The default binning maps intensity :math:`x` to
  :math:`\lfloor N x / x_{\max} + 1 \rfloor`, capped at :math:`N`, where :math:`N` is the number of
  levels and :math:`x_{\max}` the ROI's maximum intensity. In IBSI mode they are not binned: each
  integer intensity is a grey level.
- **Background:** pixels with grey level 0 belong to no run. In IBSI mode and with PyRadiomics-style
  binning, this excludes pixels outside the mask and ROI pixels of intensity 0. The default binning
  maps intensity 0 to level 1, so there such pixels do form runs.


Worked example
^^^^^^^^^^^^^^

Take a 5 × 5 ROI that already has five grey levels:

.. math::

   \textbf{G} = \begin{bmatrix}
   5 & 2 & 5 & 4 & 4\\
   3 & 3 & 3 & 1 & 3\\
   2 & 1 & 1 & 1 & 3\\
   4 & 2 & 2 & 2 & 3\\
   3 & 5 & 3 & 3 & 2 \end{bmatrix}

With :math:`\theta = 0^\circ` and IBSI mode (no binning), the matrix is below, with grey levels 1 to
5 as rows and run lengths 1 to 3 as columns. Row 1 of the image contains four runs: a 5, a 2 and a 5
of length 1, and two 4s that form one run of length 2.

.. math::

   \textbf{P} = \begin{bmatrix}
   1 & 0 & 1\\
   3 & 0 & 1\\
   4 & 1 & 1\\
   1 & 1 & 0\\
   3 & 0 & 0 \end{bmatrix},
   \qquad N_s = \sum_{i,j} \textbf{P}(i,j) = 17, \qquad N_p = 25

The *Example* value of each feature in the :ref:`index <glrlm-index>` is what Nyxus returns for this
ROI in the ``_0`` (0°) column. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/glrlm_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features="*ALL_GLRLM*" --ibsi=true \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N_g`
     - Number of grey levels: the maximum intensity in IBSI mode, otherwise the number of distinct
       binned levels in the bounding box (with the default binning, including level 1 of the pixels
       outside the mask).
   * - :math:`N_r`
     - Length of the longest run along :math:`\theta`.
   * - :math:`i,\; j`
     - Grey level of a run (:math:`i`) and its length in pixels (:math:`j`). Formulas use the grey
       level's value: the intensity in IBSI mode, the binned level otherwise.
   * - :math:`\textbf{P}(i,j)`
     - Number of runs with grey level :math:`i` and length :math:`j` along one :math:`\theta`.
   * - :math:`N_s`
     - Number of runs, :math:`N_s = \sum_{i,j}\textbf{P}(i,j)`.
   * - :math:`N_p`
     - Number of ROI pixels with nonzero intensity.
   * - :math:`p_{ij}`
     - Joint probability, :math:`p_{ij} = \textbf{P}(i,j) / N_s`.
   * - :math:`r_{i\cdot},\; r_{\cdot j}`
     - Runs per grey level, :math:`r_{i\cdot} = \sum_j \textbf{P}(i,j)`, and runs per length,
       :math:`r_{\cdot j} = \sum_i \textbf{P}(i,j)`.
   * - :math:`\mu_i,\; \mu_j`
     - Mean grey level and mean run length over all runs, :math:`\mu_i = \sum_{i,j} i\,p_{ij}` and
       :math:`\mu_j = \sum_{i,j} j\,p_{ij}`.
   * - :math:`\varepsilon`
     - :math:`2.2 \times 10^{-16}`, added inside the logarithm to avoid :math:`\log 0`.

Sums over :math:`i` run over the grey levels and sums over :math:`j` from 1 to :math:`N_r`.

Logarithms are computed with a fast base-2 approximation. Run entropy therefore differs from an
exact logarithm by a relative 2 × 10\ :sup:`-4` to 1.2 × 10\ :sup:`-3` on the worked example, which
is why it is validated at a relative tolerance of 5 × 10\ :sup:`-3`.


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
     - ``Nyxus(..., coarse_gray_depth=64)``
     - Number of grey levels. Positive: equal-width bins from 0 to the ROI maximum; 0: no binning;
       negative: PyRadiomics-style bins between the ROI minimum and maximum.
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus(..., ibsi=True)``
     - No binning: each integer intensity is a grey level.

The four directions are always computed; ``--glcmAngles`` does not apply to GLRLM. No
``set_metaparam`` parameter changes the 2D matrix; ``3glrlm/greydepth`` applies to the 3D family
only. All options are listed in the :ref:`cli-reference`.


Output columns
--------------

Each feature produces one column per direction and one average:

- ``GLRLM_SRE_0``, ``GLRLM_SRE_45``, ``GLRLM_SRE_90``, ``GLRLM_SRE_135``: the value for each
  direction.
- ``GLRLM_SRE_AVE``: the mean of the four per-direction values.

Each direction has its own matrix; the matrices are not merged.


.. _glrlm-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the 0° value on the worked example.

.. include:: glrlm_index.inc


Features
--------

.. _feature-glrlm-gln:

Grey level non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLRLM_GLN``
:Range: :math:`[1, N_s]`, runs

Measures how evenly the runs are spread over the grey levels. Low values mean the runs are spread
evenly.

.. math::

   \mathrm{GLN} = \frac{1}{N_s}\sum_{i} r_{i\cdot}^{\,2}


.. _feature-glrlm-glnn:

Grey level non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLRLM_GLNN``
:Range: :math:`(0, 1]`, dimensionless

Grey level non-uniformity divided by the number of runs once more, so values are comparable across
ROIs of different size.

.. math::

   \mathrm{GLNN} = \frac{1}{N_s^{\,2}}\sum_{i} r_{i\cdot}^{\,2}


.. _feature-glrlm-glv:

Grey level variance
^^^^^^^^^^^^^^^^^^^

:Code: ``GLRLM_GLV``
:Range: :math:`[0, \infty)`, squared grey levels

The variance of the grey level over all runs.

.. math::

   \mathrm{GLV} = \sum_{i}\sum_{j=1}^{N_r} (i-\mu_i)^2\, p_{ij}

**Possible issues.** Grey level variance scales with the square of the grey-level spacing. On the
worked example it is 1.529 in IBSI mode and 249.44 with the default 64-level binning, which maps
the levels 1 to 5 to 13, 26, 39, 52 and 64.


.. _feature-glrlm-hglre:

High grey level run emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLRLM_HGLRE``
:Range: :math:`[1, \infty)`, squared grey levels

The mean over all runs of the squared grey level. High values mean bright runs dominate.

.. math::

   \mathrm{HGLRE} = \frac{1}{N_s}\sum_{i}\sum_{j=1}^{N_r} i^2\,\textbf{P}(i,j)

**Possible issues.** Depends on the binning: 10.53 in IBSI mode and 1756.7 with the default
settings on the worked example.


.. _feature-glrlm-lglre:

Low grey level run emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLRLM_LGLRE``
:Range: :math:`(0, 1]`, inverse squared grey levels

The mean over all runs of one over the squared grey level. High values mean dark runs dominate.

.. math::

   \mathrm{LGLRE} = \frac{1}{N_s}\sum_{i}\sum_{j=1}^{N_r} \frac{\textbf{P}(i,j)}{i^2}

**Possible issues.** Depends on the binning: 0.2301 in IBSI mode and 0.001363 with the default
settings on the worked example.


.. _feature-glrlm-lre:

Long run emphasis
^^^^^^^^^^^^^^^^^

:Code: ``GLRLM_LRE``
:Range: :math:`[1, N_r^2]`, squared pixels

The mean over all runs of the squared run length. High values mean long runs dominate, as in a
coarse texture.

.. math::

   \mathrm{LRE} = \frac{1}{N_s}\sum_{i}\sum_{j=1}^{N_r} j^2\,\textbf{P}(i,j)


.. _feature-glrlm-lrhgle:

Long run high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLRLM_LRHGLE``
:Range: :math:`[1, \infty)`, squared grey levels times squared pixels

The mean over all runs of the squared run length times the squared grey level. High values mean
long bright runs dominate.

.. math::

   \mathrm{LRHGLE} = \frac{1}{N_s}\sum_{i}\sum_{j=1}^{N_r} i^2 j^2\,\textbf{P}(i,j)


.. _feature-glrlm-lrlgle:

Long run low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLRLM_LRLGLE``
:Range: :math:`(0, \infty)`, squared pixels over squared grey levels

The mean over all runs of the squared run length over the squared grey level. High values mean
long dark runs dominate.

.. math::

   \mathrm{LRLGLE} = \frac{1}{N_s}\sum_{i}\sum_{j=1}^{N_r} \frac{j^2\,\textbf{P}(i,j)}{i^2}


.. _feature-glrlm-re:

Run entropy
^^^^^^^^^^^

:Code: ``GLRLM_RE``
:Range: :math:`[0, \log_2(N_g N_r)]` bits

Measures the randomness of the joint distribution of run grey level and run length.

.. math::

   \mathrm{RE} = -\sum_{i}\sum_{j=1}^{N_r} p_{ij}\,\log_2\!\big(p_{ij}+\varepsilon\big)

**Possible issues.** The logarithm is approximate: on the worked example Nyxus returns 3.0582 at
0°, while an exact logarithm gives 3.0575.


.. _feature-glrlm-rln:

Run length non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLRLM_RLN``
:Range: :math:`[1, N_s]`, runs

Measures how evenly the runs are spread over the run lengths. Low values mean the run lengths are
spread evenly.

.. math::

   \mathrm{RLN} = \frac{1}{N_s}\sum_{j=1}^{N_r} r_{\cdot j}^{\,2}


.. _feature-glrlm-rlnn:

Run length non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLRLM_RLNN``
:Range: :math:`(0, 1]`, dimensionless

Run length non-uniformity divided by the number of runs once more, so values are comparable across
ROIs of different size.

.. math::

   \mathrm{RLNN} = \frac{1}{N_s^{\,2}}\sum_{j=1}^{N_r} r_{\cdot j}^{\,2}


.. _feature-glrlm-rp:

Run percentage
^^^^^^^^^^^^^^

:Code: ``GLRLM_RP``
:Range: :math:`(0, 1]`, dimensionless

The number of runs per ROI pixel. Fine textures, with many short runs, give values near 1.

.. math::

   \mathrm{RP} = \frac{N_s}{N_p}

**Possible issues.** With the default binning, runs of pixels outside the mask are counted in
:math:`N_s` but not in :math:`N_p`, so the value can exceed 1 (see :ref:`glrlm-possible-issues`).


.. _feature-glrlm-rv:

Run variance
^^^^^^^^^^^^

:Code: ``GLRLM_RV``
:Range: :math:`[0, \infty)`, squared pixels

The variance of the run length over all runs.

.. math::

   \mathrm{RV} = \sum_{i}\sum_{j=1}^{N_r} (j-\mu_j)^2\, p_{ij}


.. _feature-glrlm-sre:

Short run emphasis
^^^^^^^^^^^^^^^^^^

:Code: ``GLRLM_SRE``
:Range: :math:`(0, 1]`, inverse squared pixels

The mean over all runs of one over the squared run length. High values mean short runs dominate,
as in a fine texture.

.. math::

   \mathrm{SRE} = \frac{1}{N_s}\sum_{j=1}^{N_r} \frac{r_{\cdot j}}{j^2}


.. _feature-glrlm-srhgle:

Short run high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLRLM_SRHGLE``
:Range: :math:`(0, \infty)`, squared grey levels over squared pixels

The mean over all runs of the squared grey level over the squared run length. High values mean
short bright runs dominate.

.. math::

   \mathrm{SRHGLE} = \frac{1}{N_s}\sum_{i}\sum_{j=1}^{N_r} \frac{i^2\,\textbf{P}(i,j)}{j^2}


.. _feature-glrlm-srlgle:

Short run low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLRLM_SRLGLE``
:Range: :math:`(0, 1]`, inverse squared grey levels times inverse squared pixels

The mean over all runs of one over the squared run length times the squared grey level. High
values mean short dark runs dominate.

.. math::

   \mathrm{SRLGLE} = \frac{1}{N_s}\sum_{i}\sum_{j=1}^{N_r} \frac{\textbf{P}(i,j)}{i^2 j^2}


.. _glrlm-possible-issues:

Possible issues
---------------

Pixels outside the mask with the default binning
   The default (positive) ``--coarseGrayDepth`` binning maps intensity 0 to grey level 1. Pixels of
   the bounding box outside the mask, and ROI pixels of intensity 0, then form runs of level 1 and
   join runs of ROI pixels at that level. Every feature changes. On the worked example with three
   pixels of the bounding box left out of the mask, run percentage is 0.7273 instead of 0.6364 and
   low grey level run emphasis is 0.1262 instead of 0.001426, the values with those pixels
   excluded. Rectangular ROIs with no pixel of intensity 0 are not affected. IBSI mode and
   PyRadiomics-style binning (negative ``--coarseGrayDepth``) exclude these pixels.

Grey-level values depend on the binning
   ``GLV`` and the six emphasis features that weight runs by grey level use the grey level's
   value. The default binning counts levels from intensity 0, not from the ROI minimum, so a ROI
   whose intensities span a narrow range uses only the top levels. Compare these features only
   between runs with the same ``--coarseGrayDepth``. The features that use only run lengths and
   counts (``SRE``, ``LRE``, ``RLN``, ``RLNN``, ``RP``, ``RV``) change only when binning merges or
   splits runs. ``GLN``, ``GLNN`` and ``RE`` group the runs by grey level, so they also change when
   binning merges grey levels.

Direction labels
   45° runs down and to the right with rows numbered from the top. Tools that measure angles
   counter-clockwise with the row axis pointing up call this diagonal 135°. Compare per-direction
   values by step, not by label; the ``_AVE`` columns are unaffected.

Constant ROI
   When all of a ROI's pixels have one intensity, every GLRLM feature, including the ``_AVE``
   columns, returns the ``--noval`` substitute.


Validation
----------

The test suite checks every 2D GLRLM feature and its ``_AVE`` against an independent
implementation, named in the :ref:`index <glrlm-index>`. In IBSI mode Nyxus matches PyRadiomics
on the IBSI digital phantom to a relative 10\ :sup:`-9`; run entropy agrees to 5 × 10\ :sup:`-3`
because of the approximate logarithm. The test suite also compares the results with IBSI's
published consensus values at a relative 10\ :sup:`-2`, the precision those values are published
to; that comparison is a consistency check, not one of the validations in the index. The default binning and the treatment of pixels outside the mask are covered by
regression tests only.


References
----------

1. Galloway MM. *Texture analysis using gray level run lengths.* Computer Graphics and Image
   Processing. 1975;4(2):172–179.
2. Chu A, Sehgal CM, Greenleaf JF. *Use of gray value distribution of run lengths for texture
   analysis.* Pattern Recognition Letters. 1990;11(6):415–419.
3. Dasarathy BV, Holder EB. *Image characterizations based on joint gray level–run length
   distributions.* Pattern Recognition Letters. 1991;12(8):497–502.
4. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
5. van Griethuysen JJM, et al. *Computational radiomics system to decode the radiographic
   phenotype.* Cancer Research. 2017;77(21):e104–e107.
