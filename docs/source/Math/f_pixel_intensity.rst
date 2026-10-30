Pixel intensity features
========================

Intensity features are first-order statistics of the intensities of a ROI's pixels. They describe
the level, spread and shape of the intensity distribution, and ignore where in the ROI each pixel
lies.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - 36 single-value features, from ``COV`` to ``VARIANCE_BIASED``, and ``HISTOGRAM``, which has
       one column per histogram bin
   * - **Group**
     - ``*ALL_INTENSITY*``, which does not include ``HISTOGRAM``. ``*ALL*`` includes both.
   * - **Dimensions**
     - 2D. The 3D family, ``3MEAN``, ``3MEDIAN`` and so on, has its own page,
       :doc:`f_intensity_3d`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory on tiled TIFF files and on striped files of up to
       1024 × 1024 pixels; see :ref:`intensity-possible-issues` for larger striped files and
       :ref:`large-images-tiles`.
   * - **IBSI**
     - 18 features have an identifier from the IBSI intensity-based statistics, which list 18
       features. ``COV`` follows a different convention from IBSI; its entry says how.

.. tip::

   To get ``HISTOGRAM``, request it together with at least one other intensity feature, for
   example ``["*ALL_INTENSITY*", "HISTOGRAM"]``. Requested on its own, every bin is 0, a known defect
   (`#489 <https://github.com/PolusAI/nyxus/issues/489>`__; see :ref:`intensity-possible-issues`). The IBSI intensity-histogram features, ``IH_*``, are a
   separate family with its own page, :doc:`f_intensity_histogram`.


How the features are computed
-----------------------------

Every pixel of the mask that belongs to the ROI is used once, with its intensity. Nothing else is
used: not the pixel's position, not its neighbours, not the pixels of the bounding box outside the
mask.

- **Which pixels:** all pixels of the ROI, including pixels of intensity 0 inside the mask.
  :math:`N_v` is the number of ROI pixels.
- **Values:** the intensities themselves. Nothing is binned except for ``ENTROPY``,
  ``UNIFORMITY`` and ``HISTOGRAM``, which use a histogram of ``--coarseGrayDepth`` bins. The
  median and the percentiles are taken from the sorted intensities.
- **Histogram:** the histogram has equal-width bins over the ROI's own range, from
  :math:`X_{\min}` to :math:`X_{\max}`. The maximum goes into the last bin. A constant ROI puts
  every pixel into the first bin.
- **Units:** the features are reported in the image's own units when the image is read from a
  file; see :ref:`intensity-units`. Location features (minimum, maximum, mean, median, mode,
  percentiles, robust mean, integrated intensity, energy, root mean square) go through the whole
  intensity map, spread features and ``COVERED_IMAGE_INTENSITY_RANGE`` through its slope,
  variances through the square of its slope, and the ratios ``COV``, ``QCOD`` and
  ``UNIFORMITY_PIU`` are recomputed from the mapped parts.
  Skewness, the kurtoses, the 5th and 6th moments, entropy, uniformity and the histogram do not
  change under the map.
- **In-memory arrays:** ``Nyxus.featurize`` records no map. The features are in the grey levels
  Nyxus stores, which equal the input when it holds non-negative integers. An array with negative
  values is shifted up by its minimum, and fractions are truncated.
- **IBSI mode:** ``--ibsi=true`` does not change any intensity feature.


Worked example
^^^^^^^^^^^^^^

Take the 5 × 5 ROI of the texture pages, with every pixel in the mask:

.. math::

   \textbf{X} = \begin{bmatrix}
   5 & 2 & 5 & 4 & 4\\
   3 & 3 & 3 & 1 & 3\\
   2 & 1 & 1 & 1 & 3\\
   4 & 2 & 2 & 2 & 3\\
   3 & 5 & 3 & 3 & 2 \end{bmatrix}

It holds four 1s, six 2s, nine 3s, three 4s and three 5s, so :math:`N_v = 25`,
:math:`\sum x = 70` and :math:`\mu = 2.8`. The squared deviations sum to 36, so ``VARIANCE`` is
:math:`36/24 = 1.5` and ``VARIANCE_BIASED`` is :math:`36/25 = 1.44`. The median is the 13th
value, 3, and the mode is 3.

The percentiles interpolate between the sorted intensities :math:`x_{(0)} \le \dots \le x_{(24)}`.
For ``P90`` the rank is :math:`h = (25 - 1) \times 0.9 = 21.6`, between :math:`x_{(21)} = 4`, the
last 4, and :math:`x_{(22)} = 5`, the first 5, so

.. math::

   P_{90} = 4 + 0.6 \times (5 - 4) = 4.6

the value of ``numpy.percentile``. ``P10`` has rank 2.4, among the 1s, and is 1, so
``ROBUST_MEAN`` is the mean of the 1s to 4s, :math:`55/22 = 2.5`.

With the default 64 bins, ``HISTOGRAM_BIN_0`` to ``HISTOGRAM_BIN_63`` are 0 except bins 0, 16,
32, 48 and 63, which hold 4, 6, 9, 3 and 3 pixels. Entropy is therefore the entropy of the
probabilities 0.16, 0.24, 0.36, 0.12 and 0.12, 2.1819 bits.

The *Example* value of each feature in the :ref:`index <intensity-index>` is what Nyxus returns for
this ROI; for ``HISTOGRAM`` it is ``HISTOGRAM_BIN_0``. The test suite runs this example and checks
every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/intensity_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features="*ALL_INTENSITY*,HISTOGRAM" \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`x_k`
     - Intensity of the :math:`k`-th pixel of the ROI.
   * - :math:`N_v`
     - Number of ROI pixels.
   * - :math:`X_{\min},\; X_{\max}`
     - Smallest and largest intensity in the ROI.
   * - :math:`\mu`
     - Mean intensity, :math:`\mu = \frac{1}{N_v}\sum_k x_k`.
   * - :math:`m_r`
     - Central moment of order :math:`r`, :math:`m_r = \frac{1}{N_v}\sum_k (x_k - \mu)^r`.
   * - :math:`\sigma`
     - Standard deviation with :math:`N_v - 1`,
       :math:`\sigma = \sqrt{\frac{1}{N_v - 1}\sum_k (x_k - \mu)^2}`; 0 when :math:`N_v = 1`.
   * - :math:`\sigma_b`
     - Standard deviation with :math:`N_v`, :math:`\sigma_b = \sqrt{m_2}`.
   * - :math:`M`
     - Median intensity.
   * - :math:`P_q`
     - The ``Pqq`` feature: the :math:`q`-th percentile of the ROI's intensities.
   * - :math:`x_{(i)}`
     - The :math:`i`-th smallest intensity of the ROI, counting from :math:`x_{(0)} = X_{\min}`.
   * - :math:`N_b`
     - Number of bins of the coarse histogram, the absolute value of ``--coarseGrayDepth``, or 24
       when it is 0.
   * - :math:`h_j,\; p_j`
     - Count of ROI pixels in bin :math:`j` of the coarse histogram, and :math:`p_j = h_j / N_v`.

Sums over :math:`k` run over the ROI pixels, and sums over :math:`j` over the :math:`N_b` bins.
A pixel of intensity :math:`x` falls in bin
:math:`j = \lfloor N_b (x - X_{\min}) / (X_{\max} - X_{\min}) \rfloor`, and :math:`x = X_{\max}`
falls in bin :math:`N_b - 1`.


Settings
--------

.. list-table::
   :widths: 16 28 26 30
   :header-rows: 1

   * - Setting
     - Command line
     - Python
     - Effect
   * - Histogram bins
     - ``--coarseGrayDepth=64``
     - ``Nyxus(..., coarse_gray_depth=64)``
     - :math:`N_b`, the number of bins of ``ENTROPY``, ``UNIFORMITY`` and ``HISTOGRAM``: the
       absolute value, or 24 at 0. No other intensity feature depends on it.
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus(..., ibsi=True)``
     - No effect on these features.
   * - Image intensity map
     - ``--fpimgdr``, ``--fpimgmin``, ``--fpimgmax``, ``--preserve-hu``
     - ``Nyxus(..., preserve_hu=True)`` and the matching keywords
     - Decide how a file's intensities become grey levels, and so the units the features are
       reported in (see :ref:`intensity-units`).

No setting changes the median or the percentiles. All options are listed in the
:ref:`cli-reference`.


Output columns
--------------

Each single-value feature produces one column, named with its code: ``MEAN``, ``P25`` and so on.

``HISTOGRAM`` produces :math:`N_b` columns, ``HISTOGRAM_BIN_0`` to ``HISTOGRAM_BIN_63`` with the
default 64 bins. Column ``HISTOGRAM_BIN_j`` holds :math:`h_j`, the number of ROI pixels in bin
:math:`j`. The columns hold counts, not probabilities, and the bin edges are not written; bin
:math:`j` starts at :math:`X_{\min} + j\,(X_{\max} - X_{\min})/N_b`, so ``MIN`` and ``MAX``
recover them.


.. _intensity-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example.

.. include:: intensity_index.inc


Features
--------

.. _feature-cov:

Coefficient of variation
^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``COV``
:Range: :math:`[0, \infty)` for non-negative intensities, dimensionless

The standard deviation relative to the mean. High values mean the intensities spread widely
around a low mean.

.. math::

   \mathrm{COV} = \frac{\sigma}{\mu}

The value is 0 when :math:`\mu = 0`.

**Possible issues.** :math:`\sigma` uses :math:`N_v - 1`; IBSI divides :math:`\sigma_b` by the
mean. On the worked example Nyxus gives 0.4374 and the IBSI definition 0.4286. On images with
negative values, such as CT, the mean can be near 0 or negative and the ratio has no useful
meaning.


.. _feature-covered-image-intensity-range:

Covered image intensity range
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``COVERED_IMAGE_INTENSITY_RANGE``
:Range: :math:`[0, 1]`, dimensionless

The ROI's intensity range as a fraction of the intensity range of all ROIs in the image.

.. math::

   \mathrm{COVERED\_IMAGE\_INTENSITY\_RANGE} = \frac{X_{\max} - X_{\min}}{I_{\max} - I_{\min}}

:math:`I_{\min}` and :math:`I_{\max}` are the smallest and largest intensity of the pixels that
belong to any ROI of the image. Pixels outside every ROI are not used.

**Possible issues.** For in-memory arrays (``Nyxus.featurize``) no image range is recorded; the
division is by 0 and the output is the substitute, 0 in pandas, as on the worked example, and
``-0.0`` in Arrow and Parquet files. When every ROI pixel of the image has one intensity, the
division is 0/0 and the output is the ``--noval`` substitute, 0 by default (``-0.0`` in Arrow and
Parquet files; see :ref:`conventions-missing`).


.. _feature-energy:

Energy
^^^^^^

:Code: ``ENERGY``
:Range: :math:`[0, \infty)`, squared intensity units

The sum of the squared intensities. It grows with both the brightness and the size of the ROI.

.. math::

   \mathrm{ENERGY} = \sum_k x_k^2

**Relations.** ``ROOT_MEAN_SQUARED`` is :math:`\sqrt{\mathrm{ENERGY}/N_v}`. Under an intensity
map :math:`x = o + s\,u` from stored grey levels :math:`u`, energy is
:math:`N_v o^2 + 2 o s \sum u + s^2 \sum u^2`, so it depends on the offset of the map.

**Possible issues.** Each pixel's square is formed as a 32-bit unsigned integer before it is
summed, a known defect (`#490 <https://github.com/PolusAI/nyxus/issues/490>`__), so a stored grey level above 65535 wraps modulo :math:`2^{32}`. Multiplying the worked
example by 20000 gives 5.415 × 10\ :sup:`10` where the sum of squares is 9.28 × 10\ :sup:`10`.
Images of 8- and 16-bit integers are not affected.


.. _feature-entropy:

Entropy
^^^^^^^

:Code: ``ENTROPY``
:Range: :math:`[0, \log_2 N_b]`, bits

The Shannon entropy of the coarse histogram. High values mean the intensities spread evenly over
many bins.

.. math::

   \mathrm{ENTROPY} = -\sum_j p_j \log_2\left(p_j + 2.2 \times 10^{-16}\right)

The logarithm is exact; empty bins add nothing.

**Possible issues.** The value depends on :math:`N_b`. On the worked example it is 2.182 with 64
bins and 1.942 with 4, where the 4s and 5s share a bin. Because the bins span the ROI's own range,
the value does not change when all intensities are scaled. The IBSI intensity-histogram entropy is
``IH_ENTROPY_IDX`` in the ``IH_*`` family.


.. _feature-excess-kurtosis:

Excess kurtosis
^^^^^^^^^^^^^^^

:Code: ``EXCESS_KURTOSIS``
:Range: :math:`[-2, \infty)`, dimensionless

Kurtosis minus 3, so that a normal distribution scores 0. Negative values mean a flat
distribution with light tails.

.. math::

   \mathrm{EXCESS\_KURTOSIS} = \frac{m_4}{m_2^{2}} - 3

The value is 0 when :math:`N_v < 5` or :math:`m_2 = 0`. This is the IBSI definition.


.. _feature-histogram:

Histogram
^^^^^^^^^

:Code: ``HISTOGRAM``
:Range: :math:`[0, N_v]` per column, pixel counts

The coarse histogram itself: column ``HISTOGRAM_BIN_j`` is :math:`h_j`, the number of ROI pixels
in bin :math:`j`, for :math:`j = 0, \dots, N_b - 1`. The columns sum to :math:`N_v`.

.. math::

   \mathrm{HISTOGRAM\_BIN\_}j = h_j

**Possible issues.** ``HISTOGRAM`` is computed only when another intensity feature is computed
too. Requested alone, as ``["HISTOGRAM"]``, all :math:`N_b` columns are 0. This is a known
defect (`#489 <https://github.com/PolusAI/nyxus/issues/489>`__).


.. _feature-hyperflatness:

Hyperflatness
^^^^^^^^^^^^^

:Code: ``HYPERFLATNESS``
:Range: :math:`[0, \infty)`, dimensionless

The sixth standardized moment. Like kurtosis it grows with the weight of the tails, but rare
extreme values count even more.

.. math::

   \mathrm{HYPERFLATNESS} = \frac{m_6}{\sigma^{6}}

The value is 0 when :math:`\sigma = 0`. There is no minimum ROI size.

**Possible issues.** The moment :math:`m_6` divides by :math:`N_v`, but :math:`\sigma` uses
:math:`N_v - 1`, while ``SKEWNESS`` and ``KURTOSIS`` standardize with :math:`\sigma_b`. On the
worked example the value is 5.769; standardizing with :math:`\sigma_b` gives 6.520. The two
converge as :math:`N_v` grows.


.. _feature-hyperskewness:

Hyperskewness
^^^^^^^^^^^^^

:Code: ``HYPERSKEWNESS``
:Range: :math:`(-\infty, \infty)`, dimensionless

The fifth standardized moment. Its sign tells which tail holds the more extreme values.

.. math::

   \mathrm{HYPERSKEWNESS} = \frac{m_5}{\sigma^{5}}

The value is 0 when :math:`\sigma = 0`. There is no minimum ROI size: a ROI of the three
intensities 1, 2 and 4 gives 0.3464, where ``SKEWNESS`` is 0.

**Possible issues.** As for ``HYPERFLATNESS``, :math:`m_5` divides by :math:`N_v` and
:math:`\sigma` by :math:`N_v - 1`. On the worked example the value is 1.227; standardizing with
:math:`\sigma_b` gives 1.359.


.. _feature-integrated-intensity:

Integrated intensity
^^^^^^^^^^^^^^^^^^^^

:Code: ``INTEGRATED_INTENSITY``
:Range: :math:`[0, \infty)` for non-negative intensities, intensity units

The sum of the intensities, the total signal of the ROI.

.. math::

   \mathrm{INTEGRATED\_INTENSITY} = \sum_k x_k = N_v\,\mu


.. _feature-interquartile-range:

Interquartile range
^^^^^^^^^^^^^^^^^^^

:Code: ``INTERQUARTILE_RANGE``
:Range: :math:`[0, X_{\max} - X_{\min}]`, intensity units

The width of the middle half of the distribution.

.. math::

   \mathrm{INTERQUARTILE\_RANGE} = P_{75} - P_{25}

Both quartiles are those of :ref:`Percentiles <feature-p25>`. On the worked example
:math:`P_{75} = 3` and :math:`P_{25} = 2`, and the value is 1.


.. _feature-kurtosis:

Kurtosis
^^^^^^^^

:Code: ``KURTOSIS``
:Range: :math:`[1, \infty)`, dimensionless

The fourth standardized moment, without subtracting 3; a normal distribution scores 3. This is
Pearson's kurtosis, the value of MATLAB's ``kurtosis`` and of PyRadiomics.

.. math::

   \mathrm{KURTOSIS} = \frac{m_4}{m_2^{2}}

The value is 0 when :math:`N_v < 5` or :math:`m_2 = 0`.


.. _feature-max:

Maximum
^^^^^^^

:Code: ``MAX``
:Range: intensity units

The largest intensity in the ROI, :math:`X_{\max}`.


.. _feature-mean:

Mean
^^^^

:Code: ``MEAN``
:Range: :math:`[X_{\min}, X_{\max}]`, intensity units

The average intensity.

.. math::

   \mathrm{MEAN} = \mu = \frac{1}{N_v}\sum_k x_k


.. _feature-mean-absolute-deviation:

Mean absolute deviation
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``MEAN_ABSOLUTE_DEVIATION``
:Range: :math:`[0, \infty)`, intensity units

The average distance of the intensities from their mean.

.. math::

   \mathrm{MEAN\_ABSOLUTE\_DEVIATION} = \frac{1}{N_v}\sum_k \left|x_k - \mu\right|


.. _feature-median:

Median
^^^^^^

:Code: ``MEDIAN``
:Range: :math:`[X_{\min}, X_{\max}]`, intensity units

The middle intensity. It is exact: the middle value when :math:`N_v` is odd, and the average of the
two middle values when it is even. It is the 50th percentile of the rule the
:ref:`percentiles <feature-p25>` use.


.. _feature-median-absolute-deviation:

Median absolute deviation
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``MEDIAN_ABSOLUTE_DEVIATION``
:Range: :math:`[0, \infty)`, intensity units

The average distance of the intensities from their median.

.. math::

   \mathrm{MEDIAN\_ABSOLUTE\_DEVIATION} = \frac{1}{N_v}\sum_k \left|x_k - M\right|

This is the IBSI definition. It is a mean, not the median of the absolute deviations that
``scipy.stats.median_abs_deviation`` and MATLAB's ``mad(x, 1)`` return: on the worked example
Nyxus gives 0.92 and the median of the absolute deviations is 1.


.. _feature-min:

Minimum
^^^^^^^

:Code: ``MIN``
:Range: intensity units

The smallest intensity in the ROI, :math:`X_{\min}`. A pixel of intensity 0 inside the mask makes
it 0.


.. _feature-mode:

Mode
^^^^

:Code: ``MODE``
:Range: :math:`[X_{\min}, X_{\max}]`, intensity units

The most frequent intensity. When several intensities share the highest count, the smallest of
them. It counts exact intensities, not histogram bins, so on an image with many distinct values it
can rest on very few pixels.


.. _feature-p01:
.. _feature-p10:
.. _feature-p25:
.. _feature-p75:
.. _feature-p90:
.. _feature-p99:

Percentiles
^^^^^^^^^^^

:Codes: ``P01``, ``P10``, ``P25``, ``P75``, ``P90``, ``P99``
:Range: :math:`[X_{\min}, X_{\max}]`, intensity units

The 1st, 10th, 25th, 75th, 90th and 99th percentiles of the ROI's intensities, interpolated
linearly between the sorted values :math:`x_{(0)} \le \dots \le x_{(N_v - 1)}`. For
:math:`q \in \{1, 10, 25, 75, 90, 99\}` the rank is :math:`h = (N_v - 1)\,q / 100`, and

.. math::

   P_q = x_{(\lfloor h \rfloor)} + (h - \lfloor h \rfloor)
   \left(x_{(\lfloor h \rfloor + 1)} - x_{(\lfloor h \rfloor)}\right)

This is the default (``'linear'``) method of ``numpy.percentile``, the one PyRadiomics and MIRP
use, and the one that gives the IBSI digital phantom's reference values. A whole-number rank
gives that sorted value itself, and a constant ROI gives :math:`X_{\min}` for every percentile. On the worked example the six values are 1, 1, 2, 3, 4.6
and 5.

**Possible issues.** Other tools place the ranks differently. MATLAB's ``prctile`` by default
interpolates at the midpoint ranks :math:`(i - 0.5)/N_v`, numpy's ``'hazen'`` method;
``Method="inclusive"`` selects the linear rule above. The two rules give the
same value when both ranks fall among equal intensities, and otherwise differ by a fraction of the
gap between two neighbouring intensities, most for the extreme percentiles of a small ROI: on the
worked example MATLAB gives 5 for P90 and 3.25 for P75. For 5 ones and 15 threes P25 is 2.5, where
MATLAB gives 2; with 6 ones and 14 threes both give 1.


.. _feature-qcod:

Quartile coefficient of dispersion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``QCOD``
:Range: :math:`[0, 1]` for non-negative intensities, dimensionless

The interquartile range relative to the sum of the quartiles, a robust counterpart of ``COV``.

.. math::

   \mathrm{QCOD} = \frac{P_{75} - P_{25}}{P_{75} + P_{25}}

The value is 0 when :math:`P_{75} + P_{25} = 0`. On images with negative values it has no
useful meaning.


.. _feature-range:

Range
^^^^^

:Code: ``RANGE``
:Range: :math:`[0, \infty)`, intensity units

.. math::

   \mathrm{RANGE} = X_{\max} - X_{\min}


.. _feature-robust-mean:

Robust mean
^^^^^^^^^^^

:Code: ``ROBUST_MEAN``
:Range: :math:`[P_{10}, P_{90}]`, intensity units

The mean of the intensities between the 10th and the 90th percentile, both included.

.. math::

   \mathrm{ROBUST\_MEAN} = \bar{x}_{10\text{-}90} = \frac{1}{N_{10\text{-}90}}
   \sum_{P_{10} \le x_k \le P_{90}} x_k

where :math:`N_{10\text{-}90}` is the number of pixels in that interval. The value is 0 when the
interval holds no pixel.

The bounds are those of :ref:`Percentiles <feature-p25>`. On the worked example
:math:`P_{10} = 1` and :math:`P_{90} = 4.6` keep the 1s and exclude the 5s, and the mean is 2.5.


.. _feature-robust-mean-absolute-deviation:

Robust mean absolute deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``ROBUST_MEAN_ABSOLUTE_DEVIATION``
:Range: :math:`[0, \infty)`, intensity units

The mean absolute deviation of the intensities between :math:`P_{10}` and :math:`P_{90}` from
their own mean.

.. math::

   \mathrm{ROBUST\_MEAN\_ABSOLUTE\_DEVIATION} = \frac{1}{N_{10\text{-}90}}
   \sum_{P_{10} \le x_k \le P_{90}} \left|x_k - \bar{x}_{10\text{-}90}\right|

This is the IBSI definition. The value is 0 when the interval holds no pixel.

As for ``ROBUST_MEAN``, the bounds are those of :ref:`Percentiles <feature-p25>`. On the worked
example the value is 0.8182.


.. _feature-root-mean-squared:

Root mean square
^^^^^^^^^^^^^^^^

:Code: ``ROOT_MEAN_SQUARED``
:Range: :math:`[0, \infty)`, intensity units

.. math::

   \mathrm{ROOT\_MEAN\_SQUARED} = \sqrt{\frac{1}{N_v}\sum_k x_k^2}

It inherits the wrap-around of ``ENERGY`` for grey levels above 65535, a known defect
(`#490 <https://github.com/PolusAI/nyxus/issues/490>`__): 46538 instead of 60926 for
the worked example times 20000.


.. _feature-skewness:

Skewness
^^^^^^^^

:Code: ``SKEWNESS``
:Range: :math:`(-\infty, \infty)`, dimensionless

The third standardized moment. Positive values mean a longer tail towards high intensities.

.. math::

   \mathrm{SKEWNESS} = \frac{m_3}{m_2^{3/2}}

The value is 0 when :math:`N_v < 4` or :math:`m_2 = 0`. Apart from the minimum size, this is the
IBSI definition.


.. _feature-standard-deviation:

Standard deviation
^^^^^^^^^^^^^^^^^^

:Code: ``STANDARD_DEVIATION``
:Range: :math:`[0, \infty)`, intensity units

The sample standard deviation, with :math:`N_v - 1`.

.. math::

   \mathrm{STANDARD\_DEVIATION} = \sigma = \sqrt{\frac{1}{N_v - 1}\sum_k (x_k - \mu)^2}

The value is 0 when :math:`N_v = 1`.


.. _feature-standard-deviation-biased:

Biased standard deviation
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``STANDARD_DEVIATION_BIASED``
:Range: :math:`[0, \infty)`, intensity units

The population standard deviation, with :math:`N_v`.

.. math::

   \mathrm{STANDARD\_DEVIATION\_BIASED} = \sigma_b = \sqrt{\frac{1}{N_v}\sum_k (x_k - \mu)^2}


.. _feature-standard-error:

Standard error
^^^^^^^^^^^^^^

:Code: ``STANDARD_ERROR``
:Range: :math:`[0, \infty)`, intensity units

The standard error of the mean.

.. math::

   \mathrm{STANDARD\_ERROR} = \frac{\sigma}{\sqrt{N_v}}


.. _feature-uniformity:

Uniformity
^^^^^^^^^^

:Code: ``UNIFORMITY``
:Range: :math:`[1/N_b, 1]`, dimensionless

The sum of the squared bin probabilities of the coarse histogram. High values mean the
intensities gather in few bins; a constant ROI gives 1.

.. math::

   \mathrm{UNIFORMITY} = \sum_j p_j^2

**Possible issues.** Like ``ENTROPY``, it depends on :math:`N_b`: 0.2416 with 64 bins and 0.2704
with 4 on the worked example.


.. _feature-uniformity-piu:

Percent image uniformity
^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``UNIFORMITY_PIU``
:Range: :math:`[0, 100]` for non-negative intensities, percent

The percent image uniformity used in MRI quality control. 100 means a constant ROI.

.. math::

   \mathrm{UNIFORMITY\_PIU} = 100\left(1 - \frac{X_{\max} - X_{\min}}{X_{\max} + X_{\min}}\right)

The value is 0 when :math:`X_{\max} + X_{\min} = 0`. On images with negative values it can leave
the range, for example 102.1 for a ROI from −99 to −95.


.. _feature-variance:

Variance
^^^^^^^^

:Code: ``VARIANCE``
:Range: :math:`[0, \infty)`, squared intensity units

The sample variance, with :math:`N_v - 1`.

.. math::

   \mathrm{VARIANCE} = \sigma^2 = \frac{1}{N_v - 1}\sum_k (x_k - \mu)^2

The value is 0 when :math:`N_v = 1`. IBSI's intensity variance divides by :math:`N_v`; that is
``VARIANCE_BIASED``.


.. _feature-variance-biased:

Biased variance
^^^^^^^^^^^^^^^

:Code: ``VARIANCE_BIASED``
:Range: :math:`[0, \infty)`, squared intensity units

The population variance, with :math:`N_v`. This is the IBSI intensity variance and the
PyRadiomics ``Variance``.

.. math::

   \mathrm{VARIANCE\_BIASED} = \sigma_b^2 = m_2 = \frac{1}{N_v}\sum_k (x_k - \mu)^2

The value is 0 when :math:`N_v = 1`.


.. _intensity-possible-issues:

Possible issues
---------------

Percentile ranks
   The six percentiles, and ``INTERQUARTILE_RANGE``, ``QCOD``, ``ROBUST_MEAN`` and
   ``ROBUST_MEAN_ABSOLUTE_DEVIATION`` through them, interpolate at numpy's default linear ranks
   :math:`(N_v - 1)\,q`, as PyRadiomics and MIRP do. MATLAB's ``prctile`` takes midpoint
   ranks by default, and its values can differ by a fraction of the gap between two neighbouring
   intensities (see :ref:`Percentiles <feature-p25>`); with ``Method="inclusive"`` it takes the
   same ranks as Nyxus. ``MEDIAN`` is the same under both rules.

Mixed standardizations
   ``SKEWNESS``, ``KURTOSIS`` and ``EXCESS_KURTOSIS`` standardize with :math:`\sigma_b`, as IBSI and
   PyRadiomics do. ``HYPERSKEWNESS`` and ``HYPERFLATNESS`` standardize with :math:`\sigma`, which
   uses :math:`N_v - 1`, and ``COV`` divides :math:`\sigma` by the mean where IBSI uses
   :math:`\sigma_b`. The differences shrink as :math:`N_v` grows; on the 25-pixel worked example
   they are 2 % for ``COV`` and 10 % and 12 % for the 5th and 6th moments.

Small ROIs
   ``SKEWNESS`` is 0 for ROIs of fewer than 4 pixels, and ``KURTOSIS`` and ``EXCESS_KURTOSIS`` for
   fewer than 5. ``HYPERSKEWNESS`` and ``HYPERFLATNESS`` have no minimum. ``VARIANCE``,
   ``VARIANCE_BIASED`` and both standard deviations are 0 for a single pixel.

Constant ROI
   When all of a ROI's pixels have one intensity, the location features, the percentiles and the
   robust mean return that intensity, ``UNIFORMITY`` is 1, ``UNIFORMITY_PIU`` is 100 and
   ``HISTOGRAM_BIN_0`` holds every pixel. The dispersions, ``COV``, ``QCOD``, the moments and
   entropy are 0; entropy can print as a tiny negative number, about :math:`-3 \times 10^{-16}`.

``HISTOGRAM`` requested alone
   ``HISTOGRAM`` is not part of ``*ALL_INTENSITY*``, and requesting it does not by itself start the
   intensity computation. Requested alone, all its columns are 0, a known defect
   (`#489 <https://github.com/PolusAI/nyxus/issues/489>`__). Request it with ``*ALL_INTENSITY*``
   or any other intensity feature, or use ``*ALL*``.

Covered range of in-memory images
   ``COVERED_IMAGE_INTENSITY_RANGE`` is not computed for arrays passed to ``Nyxus.featurize``: it
   is 0 in pandas and ``-0.0`` in Arrow and Parquet files. Write the image to a file to get it.

Out-of-core ROIs of large striped TIFF files
   A ROI processed out of core is read from the file again. For a striped (not tiled) TIFF more
   than 1024 pixels high or wide, the pixels read back do not match the image, and the features
   computed from them differ from the in-memory values (``MIN``, ``MAX``, ``RANGE`` and
   ``UNIFORMITY_PIU`` come from the first pass over the image and stay correct): on a
   1200 × 1200 image of four quadrants of 100, 1100, 2100 and 3100, ``MEAN`` was 4764.0 in place
   of 1600. Tiled TIFF files of the same size, and striped files of 1024 × 1024 pixels, give the
   in-memory values.

Grey levels above 65535
   ``ENERGY`` and ``ROOT_MEAN_SQUARED`` square each pixel as a 32-bit unsigned integer, a known
   defect (`#490 <https://github.com/PolusAI/nyxus/issues/490>`__), which
   wraps for grey levels above 65535 (32-bit integer images, or floating-point images read with
   ``--fpimgdr`` above 65535). The other features are computed in floating point and are not
   affected.

In-memory arrays are not in the input's units
   ``Nyxus.featurize`` shifts an array with negative values up by its minimum and truncates
   fractions, and does not map the features back. For the worked example minus 10, ``MIN`` is 0
   and ``MEAN`` 1.8, not −9 and −7.2. To get features in the original units, read the image from a
   file.

Pixels of intensity 0 inside the mask
   They are ROI pixels and take part in every feature. Setting the top-left pixel of the worked
   example, a 5, to 0 lowers ``MIN`` to 0 and ``MEAN`` to 2.6.

Negative intensities
   Read from a file, an image with negative values, such as a CT, is reported in its own units.
   ``COV``, ``QCOD`` and ``UNIFORMITY_PIU`` are ratios meant for non-negative intensities; with
   negative values they can be negative or exceed their usual range.

Anisotropy on 2D images read from files
   ``--anisox``/``--anisoy`` (Python ``anisotropy_x``/``anisotropy_y``) do not change any
   intensity feature: they are computed on the pixels as acquired. For 3D anisotropy see
   :doc:`f_intensity_3d`.


Validation
----------

The test suite checks the intensity features against three references and a closed form:

- **MATLAB** (R2026a), on test ROIs, for every feature except ``ENTROPY``, ``UNIFORMITY``,
  ``ROBUST_MEAN``, ``ROBUST_MEAN_ABSOLUTE_DEVIATION``, ``MEDIAN_ABSOLUTE_DEVIATION`` and
  ``HISTOGRAM``: to a relative 10\ :sup:`-3`, and to 10\ :sup:`-9` for ``P01`` to ``P99``,
  ``INTERQUARTILE_RANGE`` and ``QCOD``, which take ``prctile(..., Method="inclusive")``, the same
  linear ranks as Nyxus. Those percentile references are computed on GNU Octave 11.3.0 with the
  same rule (``quantile(x, p, 1, 7)``); the MATLAB run of the generator is pending.
- **PyRadiomics** 3.0.1, for ``ENERGY``, ``ENTROPY``, ``INTERQUARTILE_RANGE``, ``KURTOSIS``,
  ``MAX``, ``MEAN``, ``MEAN_ABSOLUTE_DEVIATION``, ``MEDIAN``, ``MIN``, ``P10``, ``P90``,
  ``RANGE``, ``ROBUST_MEAN_ABSOLUTE_DEVIATION``, ``ROOT_MEAN_SQUARED``, ``SKEWNESS``,
  ``UNIFORMITY`` and ``VARIANCE_BIASED`` to a relative 10\ :sup:`-6`; ``VARIANCE`` to
  10\ :sup:`-2`, because PyRadiomics divides by :math:`N_v`.
- **IBSI** published values on the IBSI digital phantom, at their three significant figures
  (relative 10\ :sup:`-2`), for ``ENERGY``, ``EXCESS_KURTOSIS``, ``INTERQUARTILE_RANGE``, ``MEAN``,
  ``MEAN_ABSOLUTE_DEVIATION``, ``MEDIAN``, ``MIN``, ``P10``, ``P90``, ``RANGE``,
  ``ROBUST_MEAN_ABSOLUTE_DEVIATION``, ``ROOT_MEAN_SQUARED`` and ``SKEWNESS``.
- **The closed form** of the linear percentile rule, worked by hand on two small sample sets,
  for the six percentiles, ``INTERQUARTILE_RANGE``, ``QCOD`` and
  ``ROBUST_MEAN_ABSOLUTE_DEVIATION``, to an absolute 10\ :sup:`-12`.

``HISTOGRAM`` is checked against hand-computed bin counts to a relative 10\ :sup:`-4`.
``MEDIAN_ABSOLUTE_DEVIATION`` and ``ROBUST_MEAN`` are regression-only: MATLAB's ``mad`` and
trimmed mean use different definitions. Out-of-core results are checked against in-memory results
of the same build. The worked-example values quoted on this page, the comparisons with the IBSI
standardizations, the MATLAB percentiles and the CT ``UNIFORMITY_PIU`` example among them, are
pinned by the worked-example script; the out-of-core example was measured once and is not pinned.
None of them is compared with a reference tool.


References
----------

1. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io, section *Intensity-based statistical
   features*.
2. van Griethuysen JJM, et al. *Computational radiomics system to decode the radiographic
   phenotype.* Cancer Research. 2017;77(21):e104–e107.
3. Goerner FL, et al. *A comparison of five standard methods for evaluating image intensity
   uniformity in partially parallel imaging MRI.* Medical Physics. 2013;40(8):082302.
   https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3745492/ (Percent image uniformity)
