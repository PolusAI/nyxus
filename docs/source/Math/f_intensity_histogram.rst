IBSI intensity histogram features
=================================

The intensity histogram features are first-order statistics of a ROI's intensities after they are
sorted into a fixed number of equal-width bins. Each statistic comes in two forms: over the bin
indices, as IBSI defines it, and over the bin-centre intensities, in the image's own units.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``IH_*``: 46 features, one value per ROI. 20 end in ``_VAL`` (intensities), 20 are
       statistics of the bin indices ending in ``_IDX``, and 6 describe the histogram's gradient
       and its bins (``IH_MAX_GRADIENT``, ``IH_MAX_GRADIENT_IDX``, ``IH_MIN_GRADIENT``,
       ``IH_MIN_GRADIENT_IDX``, ``IH_NUM_BINS``, ``IH_BIN_SIZE``).
   * - **Group**
     - ``*ALL_IH*``; ``*ALL*`` also includes it. Computed only in IBSI mode.
   * - **Dimensions**
     - 2D only; there is no 3D family.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory on tiled TIFF files and on striped files of up to
       1024 × 1024 pixels; see :ref:`intensity-histogram-possible-issues` for larger striped files
       and :ref:`large-images-tiles`.
   * - **IBSI**
     - 23 features have an identifier from the IBSI intensity histogram features, which list 23
       features: the 19 ``_IDX`` statistics other than ``IH_ROBUST_MEAN_IDX`` and the 4 gradient
       features. The IBSI reference manual gives the identifier 1PR8 to both the minimum and the
       10th percentile; the index repeats it as published.

.. tip::

   Request the family in IBSI mode: ``--ibsi=true`` on the command line, ``ibsi=True`` in Python.
   Without it the ``IH_*`` columns are left out of the output, even when ``*ALL*`` is requested.
   The other first-order statistics, computed from the intensities themselves, are on the
   :doc:`pixel intensity page <f_pixel_intensity>`.


How the features are computed
-----------------------------

Nyxus builds one histogram per ROI and computes every feature from it. Nothing else is used: not
the pixels' positions, not their neighbours, not the pixels of the bounding box outside the mask.

- **Which pixels:** all pixels of the ROI, including pixels of intensity 0 inside the mask.
  :math:`N_v` is the number of ROI pixels.
- **Bins:** :math:`N` equal-width bins over the ROI's own range, from :math:`X_{\min}` to
  :math:`X_{\max}`. :math:`N` is ``--coarseGrayDepth``, 64 by default. The bin width is
  :math:`w = (X_{\max} - X_{\min}) / N`. A pixel of intensity :math:`x` falls in bin
  :math:`b(x) = \min\left(\lfloor (x - X_{\min}) / w \rfloor,\; N - 1\right) + 1`, so bins are
  numbered from 1 and :math:`X_{\max}` falls in bin :math:`N`. This is the IBSI fixed bin number
  discretisation. IBSI mode does not turn the binning off, as it does for most texture families.
- **Two kinds of value:** an ``_IDX`` feature is computed over the bin index :math:`i`, the
  discretised intensity of IBSI. A ``_VAL`` feature is the same statistic over the bin centre
  :math:`c_i = X_{\min} + (i - \tfrac{1}{2})\,w`, except the minimum and maximum, which are the
  ROI's own extremes, and the percentiles, which are interpolated inside their bin.
- **Units:** the ``_VAL`` features are reported in the image's own units when the image is read
  from a file; see :ref:`intensity-units`. The histogram is built after the image's intensity map
  is undone, so bins, bin centres and ``IH_BIN_SIZE`` are in those units too. Location features
  are in intensity units, spreads in intensity units, the variance in squared units, and the
  ratios, skewness, kurtosis, entropy and uniformity are dimensionless. The ``_IDX`` features are
  in bins.
- **In-memory arrays:** ``Nyxus.featurize`` records no map. The ``_VAL`` features are in the grey
  levels Nyxus stores, which equal the input when it holds non-negative integers. An array with
  negative values is shifted up by its minimum, and fractions are truncated.
- **No histogram:** a constant ROI has no range to bin, and fewer than 2 bins make no histogram.
  Every ``IH_*`` feature then returns the ``--noval`` substitute, 0 by default.


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

With :math:`N = 5` bins, :math:`w = (5 - 1)/5 = 0.8`. The bin edges are 1, 1.8, 2.6, 3.4, 4.2 and
5, so intensity :math:`k` falls in bin :math:`k` and the bin indices equal the intensities. The
counts are :math:`h = (4, 6, 9, 3, 3)`, the probabilities :math:`p = (0.16, 0.24, 0.36, 0.12,
0.12)` and the cumulative fractions :math:`F = (0.16, 0.40, 0.76, 0.88, 1)`. The bin centres are
1.4, 2.2, 3.0, 3.8 and 4.6.

``IH_MEAN_IDX`` is :math:`\sum_i i\,p_i = 2.8`, the mean of the intensities. ``IH_MEAN_VAL`` is
:math:`\sum_i c_i\,p_i = 2.84`: the bin centres do not sit on the intensities.

For the 90th percentile, the last bin with :math:`F_{i-1} \le 0.9` is bin 5, with
:math:`F_4 = 0.88`, so

.. math::

   Q(0.9) = 4.2 + 0.8 \times \frac{0.9 - 0.88}{0.12} = 4.3333

``IH_P90_VAL`` is 4.3333 and ``IH_P90_IDX`` is the bin holding it, 5. The 10th percentile is
:math:`1 + 0.8 \times 0.1/0.16 = 1.5`, in bin 1. Every bin lies between bins 1 and 5, so the robust
features equal their plain counterparts here.

The histogram gradient is :math:`H' = (6-4,\; (9-4)/2,\; (3-6)/2,\; (3-9)/2,\; 3-3) =
(2, 2.5, -1.5, -3, 0)`, so ``IH_MAX_GRADIENT`` is 2.5 at bin 2 and ``IH_MIN_GRADIENT`` is
:math:`-3` at bin 4.

The *Example* value of each feature in the :ref:`index <intensity-histogram-index>` is what Nyxus
returns for this ROI with 5 bins. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/intensity_histogram_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features="*ALL_IH*" --ibsi=true --coarseGrayDepth=5 \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N_v`
     - Number of ROI pixels.
   * - :math:`X_{\min},\; X_{\max}`
     - Smallest and largest intensity in the ROI.
   * - :math:`N`
     - Number of bins, ``--coarseGrayDepth``.
   * - :math:`w`
     - Bin width, :math:`w = (X_{\max} - X_{\min}) / N`.
   * - :math:`i`
     - Bin index, from 1 to :math:`N`.
   * - :math:`b(x)`
     - Bin of a value :math:`x`,
       :math:`b(x) = \min\left(\max\left(\lfloor (x - X_{\min}) / w \rfloor, 0\right), N - 1\right) + 1`.
   * - :math:`c_i`
     - Centre of bin :math:`i`, :math:`c_i = X_{\min} + (i - \tfrac{1}{2})\,w`.
   * - :math:`h_i,\; p_i`
     - Number of ROI pixels in bin :math:`i`, and :math:`p_i = h_i / N_v`.
   * - :math:`F_i`
     - Cumulative fraction, :math:`F_i = \sum_{k \le i} p_k`, with :math:`F_0 = 0`.
   * - :math:`v_i`
     - The value a bin stands for: :math:`c_i` for a ``_VAL`` feature, :math:`i` for an ``_IDX``
       feature.
   * - :math:`\mu`
     - Mean, :math:`\mu = \sum_i p_i v_i`.
   * - :math:`m_r`
     - Central moment of order :math:`r`, :math:`m_r = \sum_i p_i (v_i - \mu)^r`.
   * - :math:`Q(q)`
     - The histogram quantile at fraction :math:`q`, interpolated inside its bin (see
       :ref:`Percentiles <feature-ih-p10-val>`).

Sums over :math:`i` run over all :math:`N` bins; empty bins add nothing. Where an entry gives one
formula in :math:`v_i` for a ``_VAL`` and an ``_IDX`` code, each code uses its own :math:`v_i`.


Settings
--------

.. list-table::
   :widths: 16 28 26 30
   :header-rows: 1

   * - Setting
     - Command line
     - Python
     - Effect
   * - Bins
     - ``--coarseGrayDepth=64``
     - ``Nyxus(..., coarse_gray_depth=64)``
     - :math:`N`, the number of bins. Below 2 there is no histogram and every feature returns the
       ``--noval`` substitute. The only such value Python accepts is 1, since it refuses 0 and
       negative values; on the command line, 1, 0 or any negative value. The same option sets the
       grey levels of the texture families.
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus(..., ibsi=True)``
     - Required. Without it the family is removed from the request and no ``IH_*`` column is
       written. It does not change the binning.
   * - Image intensity map
     - ``--fpimgdr``, ``--fpimgmin``, ``--fpimgmax``, ``--preserve-hu``
     - ``Nyxus(..., preserve_hu=True)`` and the matching keywords
     - Decide how a file's intensities become grey levels, and so the units of the ``_VAL``
       features (see :ref:`intensity-units`).

The family has no metaparameters; ``--coarseGrayDepth`` is its only bin setting, and a fixed bin
width cannot be requested. All options are listed in the :ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named with its code: ``IH_MEAN_VAL``, ``IH_MEAN_IDX`` and so on.
There are no per-bin columns and no ``_AVE`` columns. The bin counts themselves are not written;
``HISTOGRAM``, on the :doc:`pixel intensity page <f_pixel_intensity>`, writes the counts of a
histogram with the same bins.


.. _intensity-histogram-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example with 5 bins.

.. include:: intensity_histogram_index.inc


Features
--------

.. _feature-ih-num-bins:
.. _feature-ih-bin-size:

Bin count and bin width
^^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``IH_NUM_BINS``, ``IH_BIN_SIZE``
:Range: :math:`N \ge 2`, bins; :math:`w > 0`, intensity units

The number of bins and the width of one bin. They record the binning the other features used.

.. math::

   \mathrm{IH\_NUM\_BINS} = N, \qquad \mathrm{IH\_BIN\_SIZE} = w = \frac{X_{\max} - X_{\min}}{N}


.. _feature-ih-coefficient-of-variation-val:
.. _feature-ih-coefficient-of-variation-idx:

Coefficient of variation
^^^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``IH_COEFFICIENT_OF_VARIATION_VAL``, ``IH_COEFFICIENT_OF_VARIATION_IDX``
:Range: :math:`[0, \infty)` for positive intensities, dimensionless

The standard deviation over the mean. ``IH_COEFFICIENT_OF_VARIATION_IDX`` is the IBSI feature.

.. math::

   \mathrm{CoV} = \frac{\sqrt{m_2}}{\mu}

**Possible issues.** The ``_VAL`` form divides by the mean intensity. With negative intensities,
such as a CT in Hounsfield units, it can be negative; with a mean of 0 it divides by zero and
returns the ``--noval`` substitute.


.. _feature-ih-entropy-val:
.. _feature-ih-entropy-idx:

Entropy
^^^^^^^

:Codes: ``IH_ENTROPY_VAL``, ``IH_ENTROPY_IDX``
:Range: :math:`[0, \log_2 N]`, bits

The Shannon entropy of the bin probabilities. High values mean the pixels spread evenly over many
bins. Both codes return the same value; ``IH_ENTROPY_IDX`` is the IBSI feature.

.. math::

   \mathrm{Entropy} = -\sum_{p_i > 10^{-7}} p_i \log_2 p_i

The logarithm is exact. A bin holding at most one ten-millionth of the pixels is left out, which
can matter only for ROIs of more than :math:`10^7` pixels.


.. _feature-ih-excess-kurtosis-val:
.. _feature-ih-excess-kurtosis-idx:

Excess kurtosis
^^^^^^^^^^^^^^^

:Codes: ``IH_EXCESS_KURTOSIS_VAL``, ``IH_EXCESS_KURTOSIS_IDX``
:Range: :math:`[-2, \infty)`, dimensionless

How heavy the tails of the binned distribution are, relative to a normal distribution.

.. math::

   \mathrm{Kurtosis} = \frac{m_4}{m_2^2} - 3

**Relations.** The bin centres are a linear map of the bin indices, so both codes return the same
value.


.. _feature-ih-interquantile-range-val:
.. _feature-ih-interquantile-range-idx:

Interquartile range
^^^^^^^^^^^^^^^^^^^

:Codes: ``IH_INTERQUANTILE_RANGE_VAL``, ``IH_INTERQUANTILE_RANGE_IDX``
:Range: :math:`[0, X_{\max} - X_{\min}]`, intensity units; :math:`[0, N - 1]`, bins

The spread of the middle half of the pixels.

.. math::

   \mathrm{IH\_INTERQUANTILE\_RANGE\_VAL} = Q(0.75) - Q(0.25), \qquad
   \mathrm{IH\_INTERQUANTILE\_RANGE\_IDX} = b\left(Q(0.75)\right) - b\left(Q(0.25)\right)

On the worked example :math:`Q(0.25) = 2.1` and :math:`Q(0.75) = 3.3778`, in bins 2 and 3.


.. _feature-ih-maximum-val:
.. _feature-ih-maximum-idx:

Maximum
^^^^^^^

:Codes: ``IH_MAXIMUM_VAL``, ``IH_MAXIMUM_IDX``
:Range: intensity units; bins

The largest intensity of the ROI, and its bin.

.. math::

   \mathrm{IH\_MAXIMUM\_VAL} = X_{\max}, \qquad \mathrm{IH\_MAXIMUM\_IDX} = b(X_{\max}) = N

``IH_MAXIMUM_VAL`` is the intensity itself, not the centre of the last bin.


.. _feature-ih-max-gradient:
.. _feature-ih-max-gradient-idx:

Maximum histogram gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``IH_MAX_GRADIENT``, ``IH_MAX_GRADIENT_IDX``
:Range: pixels per bin; bins

The steepest rise of the histogram, and the bin where it occurs. The gradient uses one-sided
differences at the first and last bins and central differences inside, as IBSI defines it:

.. math::

   H'_i = \begin{cases}
   h_2 - h_1 & i = 1\\
   (h_{i+1} - h_{i-1})/2 & 1 < i < N\\
   h_N - h_{N-1} & i = N
   \end{cases}
   \qquad
   \mathrm{IH\_MAX\_GRADIENT} = \max\left(\max_i H'_i,\; \varepsilon\right)

where :math:`\varepsilon = 2.2251 \times 10^{-308}`, the smallest positive normalized double
(``DBL_MIN``).
``IH_MAX_GRADIENT_IDX`` is the lowest :math:`i` at which :math:`H'_i` reaches the maximum, and 0
when no :math:`H'_i` exceeds :math:`\varepsilon`.

**Possible issues.** When no bin's gradient is above 0, the search never replaces its starting
value: ``IH_MAX_GRADIENT`` is :math:`2.2251 \times 10^{-308}` and ``IH_MAX_GRADIENT_IDX`` is 0,
which is not a bin. IBSI gives the largest :math:`H'_i`, which is then 0 or negative, and its bin.
A ROI of two pixels, 1 and 2, with 2 bins has :math:`H' = (0, 0)`: IBSI gives 0 at bin 1. The
pixels 1, 1 and 2 give :math:`H' = (-1, -1)`: IBSI gives :math:`-1` at bin 1.


.. _feature-ih-mean-val:
.. _feature-ih-mean-idx:

Mean
^^^^

:Codes: ``IH_MEAN_VAL``, ``IH_MEAN_IDX``
:Range: :math:`[X_{\min}, X_{\max}]`, intensity units; :math:`[1, N]`, bins

The probability-weighted mean of the bin values. ``IH_MEAN_IDX`` is the IBSI mean discretised
intensity.

.. math::

   \mu = \sum_i p_i v_i

**Relations.** :math:`\mathrm{IH\_MEAN\_VAL} = X_{\min} + (\mathrm{IH\_MEAN\_IDX} - \tfrac{1}{2})\,w`.
It is the mean of the bin centres, not of the intensities: 2.84 on the worked example with 5 bins
and 2.824 with 64, where ``MEAN`` is 2.8.


.. _feature-ih-mean-absolute-deviation-val:
.. _feature-ih-mean-absolute-deviation-idx:

Mean absolute deviation
^^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``IH_MEAN_ABSOLUTE_DEVIATION_VAL``, ``IH_MEAN_ABSOLUTE_DEVIATION_IDX``
:Range: :math:`[0, \infty)`, intensity units; bins

The mean distance of the pixels' bin values from their mean.

.. math::

   \mathrm{MAD} = \sum_i p_i \left| v_i - \mu \right|

**Relations.** The ``_VAL`` value is :math:`w` times the ``_IDX`` value.


.. _feature-ih-median-val:
.. _feature-ih-median-idx:

Median
^^^^^^

:Codes: ``IH_MEDIAN_VAL``, ``IH_MEDIAN_IDX``
:Range: intensity units; bins

The bin of the middle pixel, and its centre. The median bin :math:`m` is the first bin at which
the running count exceeds half the pixels, rounded down:

.. math::

   m = \min\left\{\, i : \sum_{k \le i} h_k > \lfloor N_v / 2 \rfloor \right\}, \qquad
   \mathrm{IH\_MEDIAN\_IDX} = m, \qquad \mathrm{IH\_MEDIAN\_VAL} = c_m

**Possible issues.** For an even :math:`N_v` this is the upper of the two middle pixels, with no
averaging. IBSI uses the sample median, which averages them. For the pixels 1, 1, 2 and 2 with 2
bins, ``IH_MEDIAN_IDX`` is 2 and ``IH_MEDIAN_VAL`` 1.75, where IBSI gives 1.5. The two agree when
the two middle pixels share a bin.


.. _feature-ih-median-absolute-deviation-val:
.. _feature-ih-median-absolute-deviation-idx:

Median absolute deviation
^^^^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``IH_MEDIAN_ABSOLUTE_DEVIATION_VAL``, ``IH_MEDIAN_ABSOLUTE_DEVIATION_IDX``
:Range: :math:`[0, \infty)`, intensity units; bins

The mean distance of the pixels' bin values from the median.

.. math::

   \mathrm{MedAD} = \sum_i p_i \left| v_i - v_m \right|

where :math:`v_m` is ``IH_MEDIAN_VAL`` or ``IH_MEDIAN_IDX``. It inherits the median's convention.


.. _feature-ih-minimum-val:
.. _feature-ih-minimum-idx:

Minimum
^^^^^^^

:Codes: ``IH_MINIMUM_VAL``, ``IH_MINIMUM_IDX``
:Range: intensity units; bins

The smallest intensity of the ROI, and its bin.

.. math::

   \mathrm{IH\_MINIMUM\_VAL} = X_{\min}, \qquad \mathrm{IH\_MINIMUM\_IDX} = b(X_{\min}) = 1

``IH_MINIMUM_VAL`` is the intensity itself, not the centre of the first bin.


.. _feature-ih-min-gradient:
.. _feature-ih-min-gradient-idx:

Minimum histogram gradient
^^^^^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``IH_MIN_GRADIENT``, ``IH_MIN_GRADIENT_IDX``
:Range: pixels per bin; :math:`[1, N]`, bins

The steepest fall of the histogram, and the bin where it occurs. :math:`H'_i` is defined under
:ref:`Maximum histogram gradient <feature-ih-max-gradient>`.

.. math::

   \mathrm{IH\_MIN\_GRADIENT} = \min_i H'_i

``IH_MIN_GRADIENT_IDX`` is the lowest :math:`i` at which :math:`H'_i` reaches the minimum.


.. _feature-ih-mode-val:
.. _feature-ih-mode-idx:

Mode
^^^^

:Codes: ``IH_MODE_VAL``, ``IH_MODE_IDX``
:Range: intensity units; bins

The most populated bin, and its centre.

.. math::

   \mathrm{IH\_MODE\_IDX} = \min\left\{\, i : h_i = \max_k h_k \right\}, \qquad
   \mathrm{IH\_MODE\_VAL} = c_{\mathrm{IH\_MODE\_IDX}}

**Possible issues.** Ties go to the lowest bin. IBSI picks the tied bin closest to the mean. For
the pixels 1, 1, 3, 4, 4, 5 and 5 with 5 bins, bins 1, 4 and 5 each hold 2 pixels and the mean
is 3.29: ``IH_MODE_IDX`` is 1, where IBSI gives 4.


.. _feature-ih-p10-val:
.. _feature-ih-p10-idx:
.. _feature-ih-p90-val:
.. _feature-ih-p90-idx:

Percentiles
^^^^^^^^^^^

:Codes: ``IH_P10_VAL``, ``IH_P10_IDX``, ``IH_P90_VAL``, ``IH_P90_IDX``
:Range: :math:`[X_{\min}, X_{\max}]`, intensity units; :math:`[1, N]`, bins

The 10th and 90th percentiles, read off the histogram as grouped data, and the bins that hold
them. The quantile :math:`Q(q)` lies in bin :math:`k`, chosen from the bottom for :math:`q < 0.5`
and from the top otherwise, and is interpolated linearly inside it:

.. math::

   k = \begin{cases}
   \min\{\, i : F_i \ge q \} & q < 0.5\\
   \max\{\, i : F_{i-1} \le q \} & q \ge 0.5
   \end{cases}
   \qquad
   Q(q) = X_{\min} + (k - 1)\,w + w\,\frac{q - F_{k-1}}{p_k}

.. math::

   \mathrm{IH\_P10\_VAL} = Q(0.1), \quad \mathrm{IH\_P10\_IDX} = b\left(Q(0.1)\right), \quad
   \mathrm{IH\_P90\_VAL} = Q(0.9), \quad \mathrm{IH\_P90\_IDX} = b\left(Q(0.9)\right)

The quartiles :math:`Q(0.25)` and :math:`Q(0.75)` used by the interquartile range and the
quartile coefficient of dispersion are computed the same way.

**Possible issues.** IBSI defines the percentiles as sample percentiles of the bin indices and
does not fix the interpolation. ``IH_P10_IDX`` and ``IH_P90_IDX`` are the bins of the
interpolated percentiles instead. On the worked example ``IH_P90_IDX`` is 5, where
``numpy.percentile`` gives 4.6 with linear interpolation, 4 with ``method="lower"`` and 5 with
``method="higher"``. The robust features take their bin range from these two codes.

When :math:`q` equals a cumulative fraction exactly, :math:`Q(q)` is the upper edge of its bin, and
the ``_IDX`` code names the next bin, which can be empty. For the pixels 1, eight 3s and 5 with 4
bins, :math:`F_1 = 0.1`, so ``IH_P10_VAL`` is 2 and ``IH_P10_IDX`` is 2, a bin that holds no pixel.
The pixel of intensity 1 is then left out of the robust features: ``IH_ROBUST_MEAN_IDX`` is
3.111.


.. _feature-ih-quantile-coefficient-of-dispersion-val:
.. _feature-ih-quantile-coefficient-of-dispersion-idx:

Quartile coefficient of dispersion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL``, ``IH_QUANTILE_COEFFICIENT_OF_DISPERSION_IDX``
:Range: :math:`[0, 1]` for positive intensities, dimensionless

The interquartile range over the sum of the quartiles. ``IH_QUANTILE_COEFFICIENT_OF_DISPERSION_IDX``
is the IBSI feature.

.. math::

   \mathrm{IH\_QUANTILE\_COEFFICIENT\_OF\_DISPERSION\_VAL} = \frac{Q(0.75) - Q(0.25)}{Q(0.75) + Q(0.25)}

.. math::

   \mathrm{IH\_QUANTILE\_COEFFICIENT\_OF\_DISPERSION\_IDX} =
   \frac{b\left(Q(0.75)\right) - b\left(Q(0.25)\right)}{b\left(Q(0.75)\right) + b\left(Q(0.25)\right)}

**Possible issues.** With negative intensities the ``_VAL`` form can be negative or exceed 1, and
it divides by zero when the quartiles cancel.


.. _feature-ih-range-val:
.. _feature-ih-range-idx:

Range
^^^^^

:Codes: ``IH_RANGE_VAL``, ``IH_RANGE_IDX``
:Range: :math:`(0, \infty)`, intensity units; bins

The distance between the ROI's extremes.

.. math::

   \mathrm{IH\_RANGE\_VAL} = X_{\max} - X_{\min}, \qquad
   \mathrm{IH\_RANGE\_IDX} = b(X_{\max}) - b(X_{\min}) = N - 1

``IH_RANGE_IDX`` therefore depends only on the setting, as in IBSI fixed bin number
discretisation.


.. _feature-ih-robust-mean-val:
.. _feature-ih-robust-mean-idx:

Robust mean
^^^^^^^^^^^

:Codes: ``IH_ROBUST_MEAN_VAL``, ``IH_ROBUST_MEAN_IDX``
:Range: intensity units; bins

The mean bin value of the pixels whose bins lie between the 10th-percentile bin and the
90th-percentile bin, inclusive. IBSI defines no histogram robust mean; it uses this mean inside
the robust mean absolute deviation.

.. math::

   \mu_R = \frac{\sum_{i=a}^{b} h_i v_i}{\sum_{i=a}^{b} h_i}, \qquad
   a = \mathrm{IH\_P10\_IDX}, \quad b = \mathrm{IH\_P90\_IDX}


.. _feature-ih-robust-mean-absolute-deviation-val:
.. _feature-ih-robust-mean-absolute-deviation-idx:

Robust mean absolute deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_VAL``, ``IH_ROBUST_MEAN_ABSOLUTE_DEVIATION_IDX``
:Range: :math:`[0, \infty)`, intensity units; bins

The mean distance from :math:`\mu_R` of the same pixels the robust mean uses.

.. math::

   \mathrm{RMAD} = \frac{\sum_{i=a}^{b} h_i \left| v_i - \mu_R \right|}{\sum_{i=a}^{b} h_i}

**Possible issues.** The bin range comes from ``IH_P10_IDX`` and ``IH_P90_IDX`` (see
:ref:`Percentiles <feature-ih-p10-val>`). On the worked example it covers every bin, so the value
equals the mean absolute deviation, 0.96. With IBSI's set taken between ``numpy.percentile``
linear percentiles, 1 and 4.6, the 5s are left out and the value is 0.8182.


.. _feature-ih-skewness-val:
.. _feature-ih-skewness-idx:

Skewness
^^^^^^^^

:Codes: ``IH_SKEWNESS_VAL``, ``IH_SKEWNESS_IDX``
:Range: :math:`(-\infty, \infty)`, dimensionless

The asymmetry of the binned distribution. Positive values mean a longer tail towards high
intensities.

.. math::

   \mathrm{Skewness} = \frac{m_3}{m_2^{3/2}}

**Relations.** Both codes return the same value.


.. _feature-ih-uniformity-val:
.. _feature-ih-uniformity-idx:

Uniformity
^^^^^^^^^^

:Codes: ``IH_UNIFORMITY_VAL``, ``IH_UNIFORMITY_IDX``
:Range: :math:`[1/N, 1]`, dimensionless

The sum of the squared bin probabilities. High values mean most pixels share few bins. Both codes
return the same value; ``IH_UNIFORMITY_IDX`` is the IBSI feature.

.. math::

   \mathrm{Uniformity} = \sum_i p_i^2


.. _feature-ih-variance-val:
.. _feature-ih-variance-idx:

Variance
^^^^^^^^

:Codes: ``IH_VARIANCE_VAL``, ``IH_VARIANCE_IDX``
:Range: :math:`[0, \infty)`, squared intensity units; squared bins

The population variance of the bin values, with :math:`N_v`, as IBSI defines it.

.. math::

   \sigma^2 = m_2 = \sum_i p_i (v_i - \mu)^2

**Relations.** The ``_VAL`` value is :math:`w^2` times the ``_IDX`` value.


.. _intensity-histogram-possible-issues:

Possible issues
---------------

IBSI mode only
   Without ``--ibsi=true`` the ``IH_*`` features are removed from the request, whether they were
   named one by one, through ``*ALL_IH*`` or through ``*ALL*``. No column is written, rather than a
   column of substitutes.

Values depend on the bin count
   Every feature except ``IH_MINIMUM_VAL``, ``IH_MINIMUM_IDX``, ``IH_MAXIMUM_VAL`` and
   ``IH_RANGE_VAL`` changes with
   :math:`N`. On the worked example ``IH_MEAN_IDX`` is 2.8 with 5 bins and 29.68 with the default
   64; ``IH_MEDIAN_VAL`` is 3 and 3.031, ``IH_P90_VAL`` 4.333 and 4.948. Compare values only
   between runs with the same ``--coarseGrayDepth``.

``_VAL`` features are not the intensity statistics
   Except for the extremes and the percentiles, the ``_VAL`` features use bin centres, not the
   pixels' own intensities. ``IH_MEAN_VAL`` is 2.84 on the worked example where ``MEAN`` is 2.8.
   For statistics of the intensities themselves use the :doc:`pixel intensity features
   <f_pixel_intensity>`.

Differences from IBSI conventions
   The median takes the upper middle pixel for an even pixel count, mode ties go to the lowest
   bin, and the percentile codes are the bins of interpolated percentiles. Each entry above gives
   an example. When no bin's gradient is above 0, ``IH_MAX_GRADIENT`` is
   :math:`2.2251 \times 10^{-308}` and ``IH_MAX_GRADIENT_IDX`` is 0.

Values on a bin edge
   The bin of a value is computed in floating point as :math:`\lfloor (x - X_{\min}) / w \rfloor`.
   A value that lies exactly on an inner edge can fall one bin low. For the pixels 1, 10 and 19
   with 14 bins, 10 lies on the lower edge of bin 8, but ``IH_MEDIAN_IDX`` is 7.

No histogram
   A constant ROI, or a bin count below 2, makes every ``IH_*`` feature return the ``--noval``
   substitute, 0 by default. ``--coarseGrayDepth=0`` on the command line gives this result too.
   IBSI puts a constant ROI in a single bin and defines its features.

Pixels of intensity 0 inside the mask
   They are ROI pixels and take part in every feature. Setting one pixel of the worked example to
   0 lowers ``IH_MINIMUM_VAL`` to 0 and widens the 5 bins to 1.

In-memory arrays are not in the input's units
   ``Nyxus.featurize`` shifts an array with negative values up by its minimum and does not map the
   features back. For the worked example minus 10, ``IH_MINIMUM_VAL`` is 0 and ``IH_MEAN_VAL``
   1.84. Read from a file, the same image gives :math:`-9` and :math:`-7.16`. The ``_IDX`` features
   do not change.

Out-of-core ROIs of large striped TIFF files
   A ROI processed out of core is read from the file again. For a striped (not tiled) TIFF more
   than 1024 pixels high or wide, the pixels read back do not match the image, and most ``IH_*``
   features differ from the in-memory values; the minimum, maximum, range, bin size and number of
   bins come from the first pass over the image and do not change. On a 1200 × 1200 image of four
   quadrants of 100, 1100, 2100 and 3100, ``IH_MAX_GRADIENT`` was 1123200 in place of 360000 and
   ``IH_MEAN_VAL`` 1635.0 in place of 1600. The intensity features are affected too; see
   :ref:`intensity-possible-issues`. Tiled TIFF files of the same size, and striped files of
   1024 × 1024 pixels, give the in-memory values.


Validation
----------

The test suite checks the family against two references, in IBSI mode:

- **MIRP** 2.6.0 for the 19 ``_IDX`` statistics other than ``IH_ROBUST_MEAN_IDX`` and the 4
  gradient features, on the IBSI digital phantom taken as one ROI with 6 bins, to a relative
  10\ :sup:`-9`; the largest difference measured is 1.3 × 10\ :sup:`-15`.
- **Analytic** closed forms, evaluated from the published definitions on the phantom and on small
  test ROIs, for the ``_VAL`` features except the four below, and for ``IH_ROBUST_MEAN_IDX``,
  ``IH_NUM_BINS`` and ``IH_BIN_SIZE``, to a relative 10\ :sup:`-9`.

``IH_P10_VAL``, ``IH_P90_VAL``, ``IH_INTERQUANTILE_RANGE_VAL`` and
``IH_QUANTILE_COEFFICIENT_OF_DISPERSION_VAL`` are regression-only: no reference tool reports a
percentile interpolated inside a histogram bin. The conventions listed under Possible issues do
not show on the phantom. The worked-example values quoted there, the in-memory and file values of
the worked example minus 10 among them, are pinned by the worked-example script; the out-of-core
example was measured once and is not pinned. None is compared with a reference tool.


References
----------

1. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io, section *Intensity histogram features*.
2. Zwanenburg A, Leger S, Agolli L, et al. *Assessing robustness of radiomic features by image
   perturbation.* Scientific Reports. 2019;9:614. (MIRP)
