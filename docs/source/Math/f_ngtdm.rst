Neighbourhood grey-tone difference matrix (NGTDM)
=================================================

NGTDM features compare each pixel with the average of its neighbours. For every grey level the
matrix holds how often the level occurs and how far, in total, its pixels differ from their
neighbourhood averages. The features describe how coarse, busy or contrasted a texture looks.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``NGTDM_*``: 5 features, one value per ROI
   * - **Group**
     - ``*ALL_NGTDM*``; also part of ``*ALL*`` and ``*WHOLESLIDE*``
   * - **Dimensions**
     - 2D. The 3D family, ``3NGTDM_*``, has its own page, :doc:`f_ngtdm_3d`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory on an image that Nyxus reads as one tile, except in
       IBSI mode: there only the first ROI that a ``Nyxus`` object or a command-line run computes
       out of core matches, and the later ones are wrong (a fix is coming). On other
       images out-of-core processing of this family raises an error or returns wrong
       values, depending on the ROI (see :ref:`large-images-tiles`).
   * - **IBSI**
     - All 5 features have an IBSI identifier.

.. tip::

   To reproduce IBSI, PyRadiomics or MIRP values, run in IBSI mode: ``--ibsi=true`` on the command
   line, ``ibsi=True`` in Python. With the default settings the pixels of the bounding box that lie
   outside the mask become grey level 1, and take part both as pixels and as neighbours, which
   changes every feature (see :ref:`ngtdm-possible-issues`).


How the matrix is built
-----------------------

Nyxus visits every pixel of the ROI's bounding box that has a non-zero grey level. Its neighbourhood
is the 8 pixels around it, at a distance of 1. Neighbours outside the bounding box or with grey
level 0 are left out. If at least one neighbour remains, the pixel is *valid*: it adds 1 to
:math:`n_i` for its grey level :math:`g_i`, and adds :math:`|g_i - \bar{A}|` to :math:`s_i`, where
:math:`\bar{A}` is the mean grey level of its remaining neighbours. A pixel with no remaining
neighbour is left out of the matrix entirely.

- **Neighbourhood:** always the 8-neighbourhood at distance 1, the one IBSI defines for 2D.
  ``--pixelDistance`` does not apply to NGTDM.
- **Directions:** none. One matrix is built per ROI, and each feature has one value.
- **Grey levels:** intensities are first binned into grey levels, as set by ``--coarseGrayDepth``
  (64 by default). A pixel of intensity :math:`x` gets level
  :math:`\lfloor N\,x / x_{\max} \rfloor + 1`, capped at :math:`N`, where :math:`N` is the number
  of levels and :math:`x_{\max}` the ROI's largest intensity. In IBSI mode they are not binned: each
  integer intensity is a grey level.
- **Background:** in IBSI mode, and with PyRadiomics-style binning (a negative
  ``--coarseGrayDepth``), pixels of intensity 0 are neither counted nor used as neighbours, whether
  they lie outside the mask or inside it. With the default binning, intensity 0 becomes grey level 1,
  so the bounding-box pixels outside the mask are counted and are averaged into their neighbours'
  :math:`\bar{A}`.
- **Probabilities:** :math:`p_i = n_i / N_{v,p}`, where :math:`N_{v,p}` is the number of valid
  pixels, not the number of ROI pixels. This is the IBSI definition.


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

Every pixel has at least one neighbour, so all 25 are valid. A corner pixel has 3 neighbours, an
edge pixel 5 and an inner pixel 8. The four pixels of level 1 have neighbourhood averages
:math:`24/8`, :math:`20/8`, :math:`15/8` and :math:`18/8`, so

.. math::

   s_1 = |1 - 3| + |1 - 2.5| + |1 - 1.875| + |1 - 2.25| = 5.625

In IBSI mode (no binning) the matrix is below. Its rows are grey levels 1 to 5 and its columns are
:math:`n_i`, :math:`p_i` and :math:`s_i`.

.. math::

   \begin{bmatrix}
   4 & 0.16 & 5.625\\
   6 & 0.24 & 4.1917\\
   9 & 0.36 & 5.0667\\
   3 & 0.12 & 3.5333\\
   3 & 0.12 & 6.9333 \end{bmatrix},
   \qquad N_{v,p} = 25, \qquad \sum_i s_i = 25.35

The exact values are :math:`s_2 = 503/120`, :math:`s_3 = 76/15`, :math:`s_4 = 53/15` and
:math:`s_5 = 104/15`. The *Example* value of each feature in the :ref:`index <ngtdm-index>` is what
Nyxus returns for this ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/ngtdm_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features="*ALL_NGTDM*" --ibsi=true \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N_g`
     - Number of rows of the matrix. With binning, the number of grey levels present in the bounding
       box; in IBSI mode, the ROI's largest intensity plus one (row 0 is always empty).
   * - :math:`g_i`
     - Grey level of row :math:`i`. With binning, the :math:`i`-th smallest level present; in IBSI
       mode, the intensity itself.
   * - :math:`\bar{A}`
     - Mean grey level of a pixel's neighbours that have a non-zero grey level.
   * - :math:`n_i`
     - Number of valid pixels (at least one neighbour) of grey level :math:`g_i`.
   * - :math:`N_{v,p}`
     - Number of valid pixels, :math:`N_{v,p} = \sum_i n_i`.
   * - :math:`p_i`
     - Grey-level probability among valid pixels, :math:`p_i = n_i / N_{v,p}`.
   * - :math:`s_i`
     - Neighbourhood grey-tone difference, :math:`s_i = \sum |g_i - \bar{A}|` over the valid
       pixels of grey level :math:`g_i`; 0 when :math:`n_i = 0`.
   * - :math:`N_{g,p}`
     - Number of distinct non-zero grey levels in the bounding box, including levels whose pixels
       are all invalid (see :ref:`ngtdm-possible-issues`).

Sums over :math:`i` and :math:`j` run over the rows of the matrix. Rows with :math:`p_i = 0` add
nothing. Every feature uses the grey level :math:`g_i`, never the row position.


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
     - Number of grey levels. Positive: equal-width bins from 0 to the ROI maximum, with background
       as level 1; negative: PyRadiomics-style bins over the ROI range (command line only).
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus(..., ibsi=True)``
     - No binning; pixels of intensity 0 are excluded, as pixels and as neighbours.

The 2D family has no direction, distance or metaparameter settings. ``3ngtdm/greydepth`` and
``3ngtdm/radius`` apply to ``3NGTDM_*`` only. All options are listed in the :ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named with its code: ``NGTDM_COARSENESS``, ``NGTDM_CONTRAST`` and
so on. There are no per-direction columns and no ``_AVE`` columns.


.. _ngtdm-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with the relative tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example.

.. include:: ngtdm_index.inc


Features
--------

.. _feature-ngtdm-busyness:

Busyness
^^^^^^^^

:Code: ``NGTDM_BUSYNESS``
:Range: :math:`[0, \infty)`, dimensionless

Compares the neighbourhood differences with the differences between probability-weighted grey
levels. High values mean intensity changes quickly from pixel to pixel.

.. math::

   \mathrm{Busyness} = \frac{\sum_i p_i s_i}{\sum_{i}\sum_{j} \left| g_i p_i - g_j p_j \right|},
   \qquad p_i \neq 0,\; p_j \neq 0

The value is 0 when :math:`N_{g,p} = 1`, and also whenever the denominator is 0: when only one grey
level has :math:`p_i > 0`, where IBSI also sets busyness to 0, or when every :math:`g_i p_i` is
equal, where IBSI leaves the value undefined.


.. _feature-ngtdm-coarseness:

Coarseness
^^^^^^^^^^

:Code: ``NGTDM_COARSENESS``
:Range: :math:`(0, \infty)`, inverse grey levels

One over the probability-weighted sum of the neighbourhood differences. High values mean intensity
changes slowly, as in large uniform patches.

.. math::

   \mathrm{Coarseness} = \frac{1}{\sum_i p_i s_i}

**Possible issues.** No term guards the denominator. When every valid pixel equals its neighbourhood
average, as in a constant ROI, the division gives infinity and the output is the ``--noval``
substitute, 0 by default (``-0.0`` in Arrow and Parquet files; see :ref:`conventions-missing`).
IBSI defines the value as :math:`10^6` there, so the smoothest ROIs report
the lowest coarseness (see :ref:`ngtdm-possible-issues`).


.. _feature-ngtdm-complexity:

Complexity
^^^^^^^^^^

:Code: ``NGTDM_COMPLEXITY``
:Range: :math:`[0, \infty)`, squared grey levels

Grey-level differences between pairs of levels, weighted by their neighbourhood differences. High
values mean many abrupt intensity changes.

.. math::

   \mathrm{Complexity} = \frac{1}{N_{v,p}}\sum_{i}\sum_{j} \left| g_i - g_j \right|
   \frac{p_i s_i + p_j s_j}{p_i + p_j},
   \qquad p_i \neq 0,\; p_j \neq 0


.. _feature-ngtdm-contrast:

Contrast
^^^^^^^^

:Code: ``NGTDM_CONTRAST``
:Range: :math:`[0, \infty)`, cubed grey levels

The spread of the grey levels times the mean neighbourhood difference per valid pixel. High values
mean both a wide dynamic range and strong local variation.

.. math::

   \mathrm{Contrast} = \left(\frac{1}{N_{g,p}\,(N_{g,p}-1)}\sum_{i}\sum_{j} p_i\,p_j\,(g_i - g_j)^2\right)
   \left(\frac{1}{N_{v,p}}\sum_i s_i\right)

When :math:`N_{g,p} = 1` the first factor divides by 1 instead, and the value is 0.

**Possible issues.** :math:`N_{g,p}` counts every grey level in the bounding box, while IBSI counts
only the levels with :math:`p_i > 0`. A level whose pixels have no neighbour therefore lowers
contrast. Adding one isolated pixel of intensity 7 beside the worked example gives 0.09734 in place
of 0.1460; the other four features do not change.


.. _feature-ngtdm-strength:

Strength
^^^^^^^^

:Code: ``NGTDM_STRENGTH``
:Range: :math:`[0, \infty)`, grey levels

The probability-weighted squared differences between grey levels, divided by the summed
neighbourhood differences. High values mean large, distinct regions of different intensity.

.. math::

   \mathrm{Strength} = \frac{\sum_{i}\sum_{j} (p_i + p_j)(g_i - g_j)^2}{\sum_i s_i},
   \qquad p_i \neq 0,\; p_j \neq 0

**Possible issues.** When :math:`\sum_i s_i = 0` the division is 0/0 and the output is the
``--noval`` substitute (``-0.0`` in Arrow and Parquet files). With the default ``--noval=0`` this
equals the IBSI value, 0.


.. _ngtdm-possible-issues:

Possible issues
---------------

Background as grey level 1 with the default binning
   With a positive ``--coarseGrayDepth``, intensity 0 bins to level 1. The bounding-box pixels
   outside the mask then become valid pixels of level 1, and they are averaged into the
   neighbourhood of every ROI pixel next to them. ROI pixels whose intensity is below
   :math:`x_{\max} / N` also get level 1. On the worked example with three pixels of the bounding
   box left out of the mask, coarseness is 0.01324 and contrast 351.2; the same binning without
   those pixels gives 0.01685 and 277.7. IBSI mode excludes them and gives a contrast of 0.1302.
   Rectangular ROIs with no pixel of intensity 0 are not affected.

Grey-level values depend on the binning
   Every feature except busyness scales with the grey-level spacing. On the worked example contrast
   is 0.1460 in IBSI mode and 309.1 with the default 64-level binning, which maps the levels 1 to 5
   to 13, 26, 39, 52 and 64. Coarseness is 0.2006 and 0.01551, complexity 8.251 and 1359.7,
   strength 1.373 and 17.47. Busyness changes only slightly, from 0.6360 to 0.6357, because the
   default binning does not keep the levels equally spaced. Compare values only between runs with
   the same ``--coarseGrayDepth``.

Pixels with no neighbour
   A pixel whose 8 neighbours are all outside the ROI, or all of grey level 0, is not counted in
   :math:`n_i` or :math:`N_{v,p}`. Its grey level is still counted in :math:`N_{g,p}`, which changes
   contrast (see :ref:`Contrast <feature-ngtdm-contrast>`). When no pixel of the ROI has a
   neighbour, every feature returns the ``--noval`` substitute; in IBSI mode busyness is 0 instead
   when the ROI has a single grey level. Except for a single grey level outside IBSI mode, the
   substitute is applied when the value is written, so Arrow and Parquet files hold ``-0.0``.

Zero neighbourhood differences
   When every valid pixel equals the mean of its neighbours, :math:`\sum_i p_i s_i = 0`. Coarseness
   returns the ``--noval`` substitute (0 by default) where IBSI gives :math:`10^6`. Busyness returns
   0 and strength the ``--noval`` substitute, both 0 by default, as in IBSI. Both substitutes are
   ``-0.0`` in Arrow and Parquet files, whatever ``--noval`` is (see :ref:`conventions-missing`).
   For a block of four 1s with one isolated pixel of intensity 2, Nyxus returns 0 for all five
   features; the IBSI values are 10\ :sup:`6` for coarseness and 0 for the other four.

Constant ROI
   When all of a ROI's pixels have one intensity and the ROI fills its bounding box, every NGTDM
   feature returns 0 by default. With the default binning the ROI has a single grey level and every
   feature returns the ``--noval`` substitute. In IBSI mode the matrix is built; coarseness and
   strength return the ``--noval`` substitute (``-0.0`` in Arrow and Parquet files) and the other
   three return 0. IBSI gives
   :math:`10^6` for coarseness. With the default binning, a constant ROI that does not fill its
   bounding box gains the background as a second level and its features are computed.

Pixels of intensity 0 inside the mask
   In IBSI mode a pixel of intensity 0 is not counted and is not used as a neighbour, even inside
   the mask.

Run time with large intensities in IBSI mode
   In IBSI mode the matrix has one row per integer from 0 to the ROI's largest intensity, and
   contrast, busyness, complexity and strength each loop over every pair of rows, empty or not;
   all five are computed whichever ones are selected. One 40 × 40 ROI takes 0.03 s with
   intensities up to 4095, 0.28 s up to 16383 and 4.3 s up to 65535. Binning to a small number of
   levels avoids this.


Validation
----------

The test suite checks every 2D NGTDM feature against MIRP on the four slices of the IBSI digital
phantom in IBSI mode, on each slice and on their mean. All five agree to an absolute
10\ :sup:`-9`; the largest difference measured is 3.6 × 10\ :sup:`-15`. The four-slice means are
also checked against IBSI's published consensus values at a relative 10\ :sup:`-2`, the precision
those values are published to. The default mode is
covered by regression tests only. The values quoted under Possible issues are pinned by the
worked-example script, except the run times, which are measured only. None is compared with a
reference tool.


References
----------

1. Amadasun M, King R. *Textural features corresponding to textural properties.* IEEE Transactions
   on Systems, Man, and Cybernetics. 1989;19(5):1264–1274. doi:10.1109/21.44046
2. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
3. Zwanenburg A, Leger S, Agolli L, et al. *Assessing robustness of radiomic features by image
   perturbation.* Scientific Reports. 2019;9:614. (MIRP)
