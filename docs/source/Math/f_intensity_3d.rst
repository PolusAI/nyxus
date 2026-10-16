3D intensity features
=====================

3D intensity features are first-order statistics of the intensities of a ROI's voxels. They are
the 3D counterparts of the :doc:`2D intensity features <f_pixel_intensity>`: the same 36
statistics, computed by the same formulas over every voxel of a volumetric ROI, whatever slice it
lies in. Voxel positions and neighbours play no part, and anisotropy settings do not change
them.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``3COV`` to ``3VARIANCE_BIASED``: 36 features, one value per ROI
   * - **Group**
     - ``*3D_ALL_INTENSITY*``; also part of ``*3D_ALL*``
   * - **Dimensions**
     - 3D (``--dim=3`` or the ``Nyxus3D`` class). The 2D family, ``MEAN``, ``MEDIAN`` and so on,
       is defined on :doc:`f_pixel_intensity`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory, for inputs Nyxus can stream. When one read of the
       file returns the whole volume (NIfTI, or OME-Zarr chunked over all of Z), an oversized ROI
       stops the run with an error (see :doc:`/large_images`).
   * - **IBSI**
     - 18 features have an identifier from the IBSI intensity-based statistics, which list 18
       features. ``3COV`` and the percentiles follow a different convention from IBSI; their 2D
       entries say how.

.. tip::

   Every 3D code computes exactly the formula of the 2D code without the leading ``3``, over
   voxels instead of pixels. The entries below link the 2D definition and give the value on a
   3D example. ``3COVERED_IMAGE_INTENSITY_RANGE`` is written out in full because its denominator
   spans the voxels of every ROI in the volume. The 3D family has no ``HISTOGRAM`` code.


How the features are computed
-----------------------------

Every voxel of the mask that carries the ROI's label is used once, with its intensity. Nothing
else is used: not the voxel's position or slice, not its neighbours, not the voxels of the
bounding box outside the mask.

- **Which voxels:** all voxels of the ROI in all slices, including voxels of intensity 0 inside
  the mask. :math:`N_v` is the number of ROI voxels.
- **Values:** the intensities themselves. Nothing is binned except for ``3ENTROPY`` and
  ``3UNIFORMITY``, which use a histogram of ``--coarseGrayDepth`` bins, and the percentiles, which
  are read off a histogram of 100 bins. Both histograms have equal-width bins over the ROI's own
  range, from :math:`X_{\min}` to :math:`X_{\max}`; the maximum goes into the last bin, and a
  constant ROI puts every voxel into the first bin.
- **Units:** the features are reported in the volume's own units; see :ref:`intensity-units`.
  Location features (minimum, maximum, mean, median, mode, percentiles, robust mean, integrated
  intensity, energy, root mean square) go through the whole intensity map, spread features and
  ``3COVERED_IMAGE_INTENSITY_RANGE`` through its slope, variances through the square of its slope,
  and the ratios ``3COV``, ``3QCOD`` and ``3UNIFORMITY_PIU`` are recomputed from the mapped parts.
  A signed integer volume is therefore reported in its own signed values. A floating-point volume
  is reported in its own range, to the resolution of its grey levels (``--fpimgdr``, 10000 steps
  over the volume's range by default).
- **Files only:** ``Nyxus3D`` reads volumes from files (``featurize_files`` or
  ``featurize_directory``); there is no in-memory path.
- **IBSI mode:** ``--ibsi=true`` does not change any 3D intensity feature.


Worked example
^^^^^^^^^^^^^^

Take a 3 × 4 × 4 volume and a ROI of 22 voxels over its three slices. The ROI voxels are shown
with their intensities; a dot is a voxel outside the mask:

.. math::

   z=0:\ \begin{bmatrix}
   \cdot & \cdot & \cdot & \cdot\\
   \cdot & 4 & 5 & \cdot\\
   \cdot & 5 & 6 & \cdot\\
   \cdot & \cdot & \cdot & \cdot \end{bmatrix}
   \qquad
   z=1:\ \begin{bmatrix}
   \cdot & 3 & 4 & \cdot\\
   3 & 6 & 6 & 5\\
   4 & 6 & 5 & 3\\
   \cdot & 4 & 3 & \cdot \end{bmatrix}
   \qquad
   z=2:\ \begin{bmatrix}
   \cdot & \cdot & \cdot & \cdot\\
   \cdot & 3 & 4 & 2\\
   \cdot & 4 & 5 & 3\\
   \cdot & \cdot & \cdot & \cdot \end{bmatrix}

The voxels outside the mask hold intensities from 0 to 4 and are not used, so ``3MIN`` is 2. The
ROI holds one 2, six 3s, six 4s, five 5s and four 6s, so :math:`N_v = 22`, :math:`\sum x = 93`
and :math:`\mu = 4.2273`. The squared deviations sum to 29.864, so ``3VARIANCE`` is
:math:`29.864/21 = 1.4221`. :math:`N_v` is even, so the median is the mean of the 11th and 12th
values, both 4. The 3s and the 4s share the highest count; ``3MODE`` is the smaller, 3.

The 100-bin percentile histogram has bins of width :math:`w = (6-2)/100 = 0.04`. The 2 falls in
bin 0, the 3s in bin 25, the 4s in bin 50, the 5s in bin 75 and the 6s in bin 99. For ``3P90``
the target count is :math:`0.9 \times 22 = 19.8`. The first 18 voxels lie below bin 99, which
holds 4 voxels, so

.. math::

   P_{90} = 2 + 99 \times 0.04 + 0.04 \times \frac{19.8 - 18}{4} = 5.978

where ``numpy.percentile`` gives 6. In the same way :math:`P_{10} = 3.008`, so ``3ROBUST_MEAN`` is
the mean of the 4s and 5s, 4.4545. With the default 64 coarse bins the five intensities fall in
bins 0, 16, 32, 48 and 63, and ``3ENTROPY`` is the entropy of the probabilities 1/22, 6/22, 6/22,
5/22 and 4/22, 2.1581 bits. The ROI is the only one in the volume, so
``3COVERED_IMAGE_INTENSITY_RANGE`` is 1.

The *Example* value of each feature in the :ref:`index <intensity-3d-index>` is what Nyxus
returns for this ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/intensity_3d_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --dim=3 --features="*3D_ALL_INTENSITY*" \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`x_k`
     - Intensity of the :math:`k`-th voxel of the ROI.
   * - :math:`N_v`
     - Number of ROI voxels.
   * - :math:`X_{\min},\; X_{\max}`
     - Smallest and largest intensity in the ROI.
   * - :math:`V_{\min},\; V_{\max}`
     - Smallest and largest intensity of the voxels that belong to any ROI of the volume.
   * - :math:`\mu`
     - Mean intensity, :math:`\mu = \frac{1}{N_v}\sum_k x_k`.
   * - :math:`m_r`
     - Central moment of order :math:`r`, :math:`m_r = \frac{1}{N_v}\sum_k (x_k - \mu)^r`.
   * - :math:`\sigma,\; \sigma_b`
     - Standard deviation with :math:`N_v - 1` and with :math:`N_v`; both 0 when
       :math:`N_v = 1`.
   * - :math:`M`
     - Median intensity.
   * - :math:`P_q`
     - The ``3Pqq`` feature: the :math:`q`-th percentile read off the 100-bin histogram.
   * - :math:`N_b`
     - Number of bins of the coarse histogram, the absolute value of ``--coarseGrayDepth``.
   * - :math:`p_j`
     - Fraction of ROI voxels in bin :math:`j` of the coarse histogram.

The symbols are those of the 2D page, with voxels in place of pixels. Sums over :math:`k` run over
the ROI voxels of all slices.


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
     - ``Nyxus3D(..., coarse_gray_depth=64)``
     - :math:`N_b`, the number of bins of ``3ENTROPY`` and ``3UNIFORMITY``. The sign is ignored.
       No other 3D intensity feature depends on it. Python accepts only positive values.
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus3D(..., ibsi=True)``
     - No effect on these features.
   * - Image intensity map
     - ``--fpimgdr``, ``--fpimgmin``, ``--fpimgmax``, ``--preserve-hu``
     - ``Nyxus3D(..., dynamic_range=..., min_intensity=..., max_intensity=..., preserve_hu=True)``
     - Decide how a file's intensities become grey levels, and so the units the features are
       reported in (see :ref:`intensity-units`).
   * - Anisotropy
     - ``--anisox``, ``--anisoy``, ``--anisoz``, or ``--use-physical-spacing=true``
     - ``Nyxus3D(..., anisotropy_x=..., anisotropy_y=..., anisotropy_z=...)``, or
       ``use_physical_spacing=True``
     - No effect on these features: the statistics run over the voxels as acquired.
   * - RAM limit
     - ``--ramLimit``
     - ``Nyxus3D(..., ram_limit=...)``, in megabytes
     - A ROI larger than the limit is streamed from disk slice by slice, and the values do not
       change. When one read of the file returns the whole volume (NIfTI, or OME-Zarr chunked over
       all of Z), such a ROI stops the run with an error instead.

The 3D intensity family has no ``3...`` metaparameter of its own, and the percentile histogram
always has 100 bins. All options are listed in the :ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named with its code: ``3MEAN``, ``3P25`` and so on.


.. _intensity-3d-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example.

.. include:: intensity_3d_index.inc


Features
--------

Each entry gives the value on the worked example. Unless the entry says otherwise, the formula,
its conventions and its zero rules are those of the linked 2D definition, with voxels in place of
pixels.


.. _feature-3cov:

3D coefficient of variation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3COV``
:Range: :math:`[0, \infty)` for non-negative intensities, dimensionless

:math:`\sigma / \mu`, 0 when :math:`\mu = 0`; see :ref:`COV <feature-cov>`. 0.2821 on the worked
example; IBSI's :math:`\sigma_b / \mu` gives 0.2756.


.. _feature-3covered-image-intensity-range:

3D covered image intensity range
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3COVERED_IMAGE_INTENSITY_RANGE``
:Range: :math:`[0, 1]`, dimensionless

The ROI's intensity range as a fraction of the intensity range of all ROIs in the volume.

.. math::

   \mathrm{3COVERED\_IMAGE\_INTENSITY\_RANGE} = \frac{X_{\max} - X_{\min}}{V_{\max} - V_{\min}}

:math:`V_{\min}` and :math:`V_{\max}` come from the voxels that belong to any ROI of the volume, in
every slice. Voxels outside every ROI are not used.

**Possible issues.** In a volume with one ROI the value is always 1, as on the worked example, or
the ``--noval`` substitute for a constant ROI (see below); it carries information only when a
volume holds several ROIs. Adding to the worked example a voxel of 40 outside the mask and a
second ROI of intensities 0 and 3 gives 0.6667 for the first ROI (4/6) and 0.5 for the second
(3/6); the 40 plays no part. When every ROI voxel of the volume has one intensity, the division is
0/0 and the output is the ``--noval`` substitute, 0 by default (``-0.0`` in Arrow and Parquet
files; see :ref:`conventions-missing`). For a floating-point volume the value moves in steps of
grey levels: on random float volumes, ROIs that held the volume's extremes gave 0.9999 or 1.


.. _feature-3energy:

3D energy
^^^^^^^^^

:Code: ``3ENERGY``
:Range: :math:`[0, \infty)`, squared intensity units

:math:`\sum_k x_k^2`; see :ref:`ENERGY <feature-energy>`. 423 on the worked example.

**Possible issues.** Each voxel's square is formed as a 32-bit unsigned integer before it is
summed, a known defect (`#490 <https://github.com/PolusAI/nyxus/issues/490>`__), so a stored grey level above 65535 wraps modulo :math:`2^{32}`. Multiplying the worked
example by 20000 gives 4.894 × 10\ :sup:`10` where the sum of squares is 1.692 × 10\ :sup:`11`.
Volumes of 8- and 16-bit integers are not affected.


.. _feature-3entropy:

3D entropy
^^^^^^^^^^

:Code: ``3ENTROPY``
:Range: :math:`[0, \log_2 N_b]`, bits

:math:`-\sum_j p_j \log_2(p_j + 2.2 \times 10^{-16})` over :math:`N_b` bins; see
:ref:`ENTROPY <feature-entropy>`. 2.158 on the worked example with 64 bins, 1.753 with 4.


.. _feature-3excess-kurtosis:

3D excess kurtosis
^^^^^^^^^^^^^^^^^^

:Code: ``3EXCESS_KURTOSIS``
:Range: :math:`[-2, \infty)`, dimensionless

:math:`m_4 / m_2^2 - 3`, 0 when :math:`N_v < 5` or :math:`m_2 = 0`; see
:ref:`EXCESS_KURTOSIS <feature-excess-kurtosis>`. −1.038 on the worked example.


.. _feature-3hyperflatness:

3D hyperflatness
^^^^^^^^^^^^^^^^

:Code: ``3HYPERFLATNESS``
:Range: :math:`[0, \infty)`, dimensionless

:math:`m_6 / \sigma^6`, 0 when :math:`\sigma = 0`; see :ref:`HYPERFLATNESS <feature-hyperflatness>`
for the mixed :math:`N_v` and :math:`N_v - 1` standardization. 4.232 on the worked example.


.. _feature-3hyperskewness:

3D hyperskewness
^^^^^^^^^^^^^^^^

:Code: ``3HYPERSKEWNESS``
:Range: :math:`(-\infty, \infty)`, dimensionless

:math:`m_5 / \sigma^5`, 0 when :math:`\sigma = 0`; see
:ref:`HYPERSKEWNESS <feature-hyperskewness>`. −0.002163 on the worked example.


.. _feature-3integrated-intensity:

3D integrated intensity
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3INTEGRATED_INTENSITY``
:Range: :math:`[0, \infty)` for non-negative intensities, intensity units

:math:`\sum_k x_k`; see :ref:`INTEGRATED_INTENSITY <feature-integrated-intensity>`. 93 on the
worked example.


.. _feature-3interquartile-range:

3D interquartile range
^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3INTERQUARTILE_RANGE``
:Range: :math:`[0, X_{\max} - X_{\min}]`, intensity units

:math:`P_{75} - P_{25}` from the 100-bin histogram; see
:ref:`INTERQUARTILE_RANGE <feature-interquartile-range>`. 1.998 on the worked example; exact
quartiles give 2.


.. _feature-3kurtosis:

3D kurtosis
^^^^^^^^^^^

:Code: ``3KURTOSIS``
:Range: :math:`[1, \infty)`, dimensionless

:math:`m_4 / m_2^2`, 0 when :math:`N_v < 5` or :math:`m_2 = 0`; see
:ref:`KURTOSIS <feature-kurtosis>`. 1.962 on the worked example.


.. _feature-3max:

3D maximum
^^^^^^^^^^

:Code: ``3MAX``
:Range: intensity units

:math:`X_{\max}`; see :ref:`MAX <feature-max>`. 6 on the worked example.


.. _feature-3mean:

3D mean
^^^^^^^

:Code: ``3MEAN``
:Range: :math:`[X_{\min}, X_{\max}]`, intensity units

:math:`\mu`; see :ref:`MEAN <feature-mean>`. 4.227 on the worked example.


.. _feature-3mean-absolute-deviation:

3D mean absolute deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3MEAN_ABSOLUTE_DEVIATION``
:Range: :math:`[0, \infty)`, intensity units

:math:`\frac{1}{N_v}\sum_k |x_k - \mu|`; see
:ref:`MEAN_ABSOLUTE_DEVIATION <feature-mean-absolute-deviation>`. 0.9959 on the worked example.


.. _feature-3median:

3D median
^^^^^^^^^

:Code: ``3MEDIAN``
:Range: :math:`[X_{\min}, X_{\max}]`, intensity units

The exact median, the mean of the two middle values when :math:`N_v` is even; see
:ref:`MEDIAN <feature-median>`. 4 on the worked example.


.. _feature-3median-absolute-deviation:

3D median absolute deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3MEDIAN_ABSOLUTE_DEVIATION``
:Range: :math:`[0, \infty)`, intensity units

:math:`\frac{1}{N_v}\sum_k |x_k - M|`, a mean, not a median; see
:ref:`MEDIAN_ABSOLUTE_DEVIATION <feature-median-absolute-deviation>`. 0.9545 on the worked
example, where the median of the absolute deviations is 1.


.. _feature-3min:

3D minimum
^^^^^^^^^^

:Code: ``3MIN``
:Range: intensity units

:math:`X_{\min}`; see :ref:`MIN <feature-min>`. 2 on the worked example. A voxel of intensity 0
inside the mask makes it 0.


.. _feature-3mode:

3D mode
^^^^^^^

:Code: ``3MODE``
:Range: :math:`[X_{\min}, X_{\max}]`, intensity units

The most frequent intensity, the smallest of them on a tie; see :ref:`MODE <feature-mode>`. 3 on
the worked example, where the 3s and the 4s both occur six times.


.. _feature-3p01:
.. _feature-3p10:
.. _feature-3p25:
.. _feature-3p75:
.. _feature-3p90:
.. _feature-3p99:

3D percentiles
^^^^^^^^^^^^^^

:Codes: ``3P01``, ``3P10``, ``3P25``, ``3P75``, ``3P90``, ``3P99``
:Range: :math:`[X_{\min}, X_{\max}]`, intensity units

The 1st, 10th, 25th, 75th, 90th and 99th percentiles, interpolated within a histogram of 100
equal-width bins over the ROI's range; see :ref:`Percentiles <feature-p25>` for the formula, the
bin-boundary rule and the error against exact percentiles. On the worked example they are 2.009,
3.008, 3.03, 5.028, 5.978 and 5.998; ``numpy.percentile`` with linear interpolation gives 2.21, 3,
3, 5, 6 and 6.


.. _feature-3qcod:

3D quartile coefficient of dispersion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3QCOD``
:Range: :math:`[0, 1]` for non-negative intensities, dimensionless

:math:`(P_{75} - P_{25}) / (P_{75} + P_{25})`, 0 when the denominator is 0; see
:ref:`QCOD <feature-qcod>`. 0.248 on the worked example.


.. _feature-3range:

3D range
^^^^^^^^

:Code: ``3RANGE``
:Range: :math:`[0, \infty)`, intensity units

:math:`X_{\max} - X_{\min}`; see :ref:`RANGE <feature-range>`. 4 on the worked example.


.. _feature-3robust-mean:

3D robust mean
^^^^^^^^^^^^^^

:Code: ``3ROBUST_MEAN``
:Range: :math:`[P_{10}, P_{90}]`, intensity units

The mean of the voxel intensities between the histogram percentiles :math:`P_{10}` and
:math:`P_{90}`, both included, 0 when the interval holds no voxel; see
:ref:`ROBUST_MEAN <feature-robust-mean>`. On the worked example :math:`P_{10} = 3.008` and
:math:`P_{90} = 5.978` exclude the 2, the 3s and the 6s, and the value is 4.4545; with exact
percentiles (3 and 6) the 3s and 6s are kept and the mean is 4.3333.


.. _feature-3robust-mean-absolute-deviation:

3D robust mean absolute deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3ROBUST_MEAN_ABSOLUTE_DEVIATION``
:Range: :math:`[0, \infty)`, intensity units

The mean absolute deviation of the voxel intensities between :math:`P_{10}` and :math:`P_{90}`
from their own mean; see :ref:`ROBUST_MEAN_ABSOLUTE_DEVIATION <feature-robust-mean-absolute-deviation>`.
0.4959 on the worked example.


.. _feature-3root-mean-squared:

3D root mean square
^^^^^^^^^^^^^^^^^^^

:Code: ``3ROOT_MEAN_SQUARED``
:Range: :math:`[0, \infty)`, intensity units

:math:`\sqrt{\mathrm{3ENERGY}/N_v}`; see :ref:`ROOT_MEAN_SQUARED <feature-root-mean-squared>`.
4.385 on the worked example. It inherits the wrap-around of ``3ENERGY`` for grey levels above
65535, a known defect (`#490 <https://github.com/PolusAI/nyxus/issues/490>`__): 47166 instead of 87698 for the worked example times 20000.


.. _feature-3skewness:

3D skewness
^^^^^^^^^^^

:Code: ``3SKEWNESS``
:Range: :math:`(-\infty, \infty)`, dimensionless

:math:`m_3 / m_2^{3/2}`, 0 when :math:`N_v < 4` or :math:`m_2 = 0`; see
:ref:`SKEWNESS <feature-skewness>`. 0.06841 on the worked example.


.. _feature-3standard-deviation:

3D standard deviation
^^^^^^^^^^^^^^^^^^^^^

:Code: ``3STANDARD_DEVIATION``
:Range: :math:`[0, \infty)`, intensity units

:math:`\sigma`, with :math:`N_v - 1`; see :ref:`STANDARD_DEVIATION <feature-standard-deviation>`.
1.193 on the worked example.


.. _feature-3standard-deviation-biased:

3D biased standard deviation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3STANDARD_DEVIATION_BIASED``
:Range: :math:`[0, \infty)`, intensity units

:math:`\sigma_b`, with :math:`N_v`; see
:ref:`STANDARD_DEVIATION_BIASED <feature-standard-deviation-biased>`. 1.165 on the worked
example.


.. _feature-3standard-error:

3D standard error
^^^^^^^^^^^^^^^^^

:Code: ``3STANDARD_ERROR``
:Range: :math:`[0, \infty)`, intensity units

:math:`\sigma / \sqrt{N_v}`; see :ref:`STANDARD_ERROR <feature-standard-error>`. 0.2542 on the
worked example.


.. _feature-3uniformity:

3D uniformity
^^^^^^^^^^^^^

:Code: ``3UNIFORMITY``
:Range: :math:`[1/N_b, 1]`, dimensionless

:math:`\sum_j p_j^2` over :math:`N_b` bins; see :ref:`UNIFORMITY <feature-uniformity>`. 0.2355 on
the worked example with 64 bins, 0.3182 with 4.


.. _feature-3uniformity-piu:

3D percent image uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``3UNIFORMITY_PIU``
:Range: :math:`[0, 100]` for non-negative intensities, percent

:math:`100\,(1 - (X_{\max} - X_{\min}) / (X_{\max} + X_{\min}))`, 0 when
:math:`X_{\max} + X_{\min} = 0`; see :ref:`UNIFORMITY_PIU <feature-uniformity-piu>`. 50 on the
worked example; 133.3 for the worked example minus 10 read as a signed volume.


.. _feature-3variance:

3D variance
^^^^^^^^^^^

:Code: ``3VARIANCE``
:Range: :math:`[0, \infty)`, squared intensity units

:math:`\sigma^2`, with :math:`N_v - 1`; see :ref:`VARIANCE <feature-variance>`. 1.422 on the
worked example.


.. _feature-3variance-biased:

3D biased variance
^^^^^^^^^^^^^^^^^^

:Code: ``3VARIANCE_BIASED``
:Range: :math:`[0, \infty)`, squared intensity units

:math:`\sigma_b^2 = m_2`, with :math:`N_v`, the IBSI intensity variance; see
:ref:`VARIANCE_BIASED <feature-variance-biased>`. 1.357 on the worked example.


.. _intensity-3d-possible-issues:

Possible issues
---------------

Histogram percentiles
   The six percentiles, and ``3INTERQUARTILE_RANGE``, ``3QCOD``, ``3ROBUST_MEAN`` and
   ``3ROBUST_MEAN_ABSOLUTE_DEVIATION`` through them, come from a 100-bin histogram rather than from
   the sorted intensities, exactly as in 2D. On the worked example ``3P90`` is 5.978 where the
   exact value is 6, and ``3ROBUST_MEAN`` 4.4545 where exact bounds give 4.3333. ``3MEDIAN`` is
   exact.

Covered range with one ROI
   ``3COVERED_IMAGE_INTENSITY_RANGE`` divides by the range of all ROI voxels of the volume, so in
   a volume with a single ROI it is 1, or the ``--noval`` substitute when that ROI is constant.
   Compare ROIs only within one volume.

Grey levels above 65535
   ``3ENERGY`` and ``3ROOT_MEAN_SQUARED`` square each voxel as a 32-bit unsigned integer, a known
   defect (`#490 <https://github.com/PolusAI/nyxus/issues/490>`__), which
   wraps for grey levels above 65535 (32-bit integer volumes, or floating-point volumes read with
   ``--fpimgdr`` above 65535). The other features are computed in floating point and are not
   affected.

Mixed standardizations
   ``3SKEWNESS``, ``3KURTOSIS`` and ``3EXCESS_KURTOSIS`` standardize with :math:`\sigma_b`;
   ``3HYPERSKEWNESS``, ``3HYPERFLATNESS`` and ``3COV`` use :math:`\sigma`, as in 2D.

Small ROIs
   ``3SKEWNESS`` is 0 for ROIs of fewer than 4 voxels, and ``3KURTOSIS`` and ``3EXCESS_KURTOSIS``
   for fewer than 5. ``3HYPERSKEWNESS`` and ``3HYPERFLATNESS`` have no minimum: one voxel of 1, 2
   and 4 in three slices gives 0.3464 and 0.7099.

Constant ROI
   When all of a ROI's voxels have one intensity, the location features, the percentiles and the
   robust mean return that intensity, ``3UNIFORMITY`` is 1 and ``3UNIFORMITY_PIU`` is 100. The
   dispersions, ``3COV``, ``3QCOD`` and the moments are 0; ``3ENTROPY`` prints as about
   :math:`-3.2 \times 10^{-16}`.

Voxels of intensity 0 inside the mask
   They are ROI voxels and take part in every feature. Setting one 6 of the worked example to 0
   lowers ``3MIN`` to 0 and ``3MEAN`` to 3.955.

Negative intensities
   A signed volume read from a file is reported in its own units: the worked example minus 10
   gives ``3MIN`` −8 and ``3MEAN`` −5.773. ``3COV``, ``3QCOD`` and ``3UNIFORMITY_PIU`` are ratios
   meant for non-negative intensities; with negative values they can be negative or leave their
   usual range.


Validation
----------

The test suite checks the 3D intensity features against two references:

- **MATLAB** (R2026a), for every feature except ``3COVERED_IMAGE_INTENSITY_RANGE``,
  ``3ENERGY``, ``3ENTROPY``, ``3MEDIAN_ABSOLUTE_DEVIATION``, ``3ROBUST_MEAN``,
  ``3ROBUST_MEAN_ABSOLUTE_DEVIATION`` and ``3UNIFORMITY``: to a relative 10\ :sup:`-3`, and to
  10\ :sup:`-2` for the six percentiles, ``3INTERQUARTILE_RANGE`` and ``3QCOD``.
- **PyRadiomics** 3.0.1, for ``3ENERGY``, ``3ENTROPY``, ``3KURTOSIS``, ``3MAX``, ``3MEAN``,
  ``3MEAN_ABSOLUTE_DEVIATION``, ``3MEDIAN``, ``3MIN``, ``3RANGE``, ``3ROOT_MEAN_SQUARED``,
  ``3SKEWNESS`` and ``3UNIFORMITY`` to a relative 10\ :sup:`-9`; ``3VARIANCE`` to
  10\ :sup:`-3`, because PyRadiomics divides by :math:`N_v`; ``3P10``, ``3P90``,
  ``3INTERQUARTILE_RANGE`` and ``3ROBUST_MEAN_ABSOLUTE_DEVIATION`` to 10\ :sup:`-2`.

``3COVERED_IMAGE_INTENSITY_RANGE``, ``3MEDIAN_ABSOLUTE_DEVIATION`` and ``3ROBUST_MEAN`` are
regression-only: no reference tool computes the same quantity (MATLAB's ``mad`` takes a median of
the deviations and ``trimmean`` trims sample tails). The worked-example values quoted on this page
are pinned by the worked-example script; the floating-point covered-range observation is not, and
the constant-ROI ``3ENTROPY`` is pinned only to below 10\ :sup:`-15` in magnitude. None is
compared with a reference tool.


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
