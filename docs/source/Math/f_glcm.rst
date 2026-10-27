Grey-level co-occurrence matrix (GLCM)
======================================

GLCM features describe how often pairs of grey levels occur next to each other at a fixed distance
and direction inside an ROI. They capture texture properties such as contrast, homogeneity and
regularity that first-order intensity statistics cannot see.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``GLCM_*``: 30 features, each with a column per direction, and 29 ``GLCM_*_AVE`` direction
       averages (``GLCM_HOM2`` has none)
   * - **Group**
     - ``*ALL_GLCM*``; also part of ``*ALL*`` and ``*WHOLESLIDE*``
   * - **Dimensions**
     - 2D. The 3D family, ``3GLCM_*``, has its own page, :doc:`f_glcm_3d`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory on an image that Nyxus reads as one tile. On other
       images out-of-core processing of this family raises an error or returns wrong
       values, depending on the ROI (see :ref:`large-images-tiles`).
   * - **IBSI**
     - 25 of the 30 features have an IBSI identifier. Four of the other 5 return the same value as
       one that does; ``GLCM_VARIANCE`` equals IBSI joint variance, which ``GLCM_JVAR`` matches only
       on a symmetric matrix.

.. tip::

   To reproduce IBSI, PyRadiomics or MIRP values, run in IBSI mode: ``--ibsi=true`` on the command
   line, ``ibsi=True`` in Python. With the default settings each neighbour pair is counted in one
   order only, and several features then differ from those tools (see
   :ref:`glcm-possible-issues`).


How the matrix is built
-----------------------

For a direction :math:`\theta` and a distance :math:`\delta`, Nyxus visits every pixel of the ROI
and looks at the pixel :math:`\delta` pixels away along :math:`\theta`. If both pixels are inside
the image and neither has intensity 0, the pair of their grey levels is counted.

- **Directions:** :math:`\theta \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}`. Each direction
  gets its own matrix and its own feature values. With rows numbered from the top, 0° pairs a
  pixel with the one to its right, 45° with the one below and to the right, 90° with the one below,
  and 135° with the one below and to the left. MATLAB ``graycomatrix``, PyRadiomics and IBSI
  measure angles with the row axis pointing up: their 0° is the same pixel to the right, their 90°
  is the pixel above, and they call the two diagonals 135° and 45°. With the default one-order
  counting the Nyxus 90°, 45° and 135° matrices are therefore the transposes of their 90°, 135°
  and 45° matrices, and only 0° matches as it is.
- **Distance:** :math:`\delta = 1` pixel.
- **Grey levels:** intensities are first binned into :math:`N_g` grey levels, as set by
  ``--coarseGrayDepth`` (64 by default) or, from Python, by ``glcm/greydepth``, which overrides it
  for this family. In IBSI mode they are not binned: each integer intensity is a grey level.
- **Symmetry:** in IBSI mode, and with a negative or zero ``--coarseGrayDepth`` (PyRadiomics-style
  bins or no binning), each pair is counted in both orders, so :math:`\textbf{P}` is symmetric.
  With a positive grey depth, as by default, each ordered pair is counted once.
- **Background:** pairs that include a pixel of intensity 0 are not counted, even inside the mask.


Worked example
^^^^^^^^^^^^^^

Take a 5 × 5 ROI that already has five grey levels:

.. math::

   \textbf{G} = \begin{bmatrix}
   1 & 2 & 5 & 2 & 3\\
   3 & 2 & 1 & 3 & 1\\
   1 & 3 & 5 & 5 & 2\\
   1 & 1 & 1 & 1 & 2\\
   1 & 2 & 4 & 3 & 5 \end{bmatrix}

With :math:`\delta = 1`, :math:`\theta = 0^\circ` and IBSI mode (no binning, both orders counted),
the matrix is below. Row 1 of the image contains the pairs (1,2), (2,5), (5,2) and (2,3), and each
is also counted reversed.

.. math::

   \textbf{P} = \begin{bmatrix}
   6 & 4 & 3 & 0 & 0\\
   4 & 0 & 2 & 1 & 3\\
   3 & 2 & 0 & 1 & 2\\
   0 & 1 & 1 & 0 & 0\\
   0 & 3 & 2 & 0 & 2 \end{bmatrix},
   \qquad \sum_{i,j} \textbf{P}(i,j) = 40

The *Example* value of each feature in the :ref:`index <glcm-index>` is what Nyxus returns for this
ROI in the ``_0`` (0°) column. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/glcm_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features="*ALL_GLCM*" --ibsi=true --glcmAngles=0 \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N_g`
     - Number of grey levels.
   * - :math:`i,\; j`
     - Grey level of the reference pixel (:math:`i`) and of its neighbour :math:`\delta` pixels
       along :math:`\theta` (:math:`j`).
   * - :math:`\textbf{P}(i,j)`
     - Co-occurrence counts for one :math:`(\delta, \theta)`.
   * - :math:`p_{ij}`
     - Joint probability, :math:`p_{ij} = \textbf{P}(i,j) / \sum_{i,j}\textbf{P}(i,j)`.
   * - :math:`p_x(i),\; p_y(j)`
     - Marginal probabilities of the reference and neighbour levels,
       :math:`p_x(i) = \sum_j p_{ij}` and :math:`p_y(j) = \sum_i p_{ij}`.
   * - :math:`\mu_x,\; \mu_y`
     - Marginal means, :math:`\mu_x = \sum_i i\,p_x(i)` and :math:`\mu_y = \sum_j j\,p_y(j)`.
   * - :math:`\sigma_x,\; \sigma_y`
     - Marginal standard deviations.
   * - :math:`p_{x+y}(k)`
     - :math:`\sum_{i+j=k} p_{ij}`, for :math:`k = 2, \dots, 2N_g`.
   * - :math:`p_{x-y}(k)`
     - :math:`\sum_{|i-j|=k} p_{ij}`, for :math:`k = 0, \dots, N_g - 1`.
   * - :math:`\varepsilon`
     - :math:`10^{-9}`, added inside logarithms to avoid :math:`\log 0`.
   * - :math:`HX,\; HY`
     - Marginal entropies, :math:`HX = -\sum_i p_x(i)\log_2\big(p_x(i)+\varepsilon\big)` and
       likewise :math:`HY` for :math:`p_y`.
   * - :math:`HXY`
     - :math:`-\sum_{i,j} p_{ij}\log_2\big(p_{ij}+\varepsilon\big)`
   * - :math:`HXY1`
     - :math:`-\sum_{i,j} p_{ij}\log_2\big(p_x(i)\,p_y(j)+\varepsilon\big)`
   * - :math:`HXY2`
     - :math:`-\sum_{i,j} p_x(i)\,p_y(j)\log_2\big(p_x(i)\,p_y(j)+\varepsilon\big)`

Sums over :math:`i` and :math:`j` run from 1 to :math:`N_g`. On a symmetric matrix
:math:`p_x = p_y`, :math:`\mu_x = \mu_y` and :math:`HX = HY`.

Logarithms are computed with a fast base-2 approximation. On the worked example above, over the
four directions in IBSI mode, ``GLCM_JE``, ``GLCM_SUMENTROPY``, ``GLCM_DIFENTRO`` and
``GLCM_INFOMEAS2`` differ from an exact logarithm by a relative 3.1 × 10\ :sup:`-3` at most, and
``GLCM_INFOMEAS1``, a difference of two entropies divided by a third, by 3.8 × 10\ :sup:`-3` to
5.6 × 10\ :sup:`-3`. ``GLCM_JE_0`` is 3.80453 where the exact value is 3.80869, a relative
1.1 × 10\ :sup:`-3`. On the validation phantoms these features agree with the reference tools to a
relative 5 × 10\ :sup:`-3`. See :ref:`conventions-entropy`.


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
     - Number of grey levels. Positive: equal-width bins; 0: no binning; negative: PyRadiomics-style
       bins.
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus(..., ibsi=True)``
     - No binning, and pairs counted in both orders.
   * - Directions
     - ``--glcmAngles=0,45,90,135``
     - not available
     - Which :math:`\theta` to compute; each enabled angle adds one output column per feature.
   * - Distance
     - ``--glcmOff``
     - ``set_metaparam("glcm/offset=N")``
     - :math:`\delta` in pixels. The command-line option is accepted but has no effect: the
       distance is 1 unless set from Python.

``set_metaparam("glcm/greydepth=N")`` sets the grey depth of the 2D matrix and of its constant-ROI
test (see :ref:`glcm-possible-issues`), in place of ``coarse_gray_depth``.
All options are listed in the :ref:`cli-reference`.


Output columns
--------------

Each feature produces one column per enabled direction and one average:

- ``GLCM_CONTRAST_0``, ``GLCM_CONTRAST_45``, ``GLCM_CONTRAST_90``, ``GLCM_CONTRAST_135``: the value
  for each direction.
- ``GLCM_CONTRAST_AVE``: the mean of the per-direction values.

Each direction has its own matrix; the matrices are not merged. ``GLCM_HOM2`` has no ``_AVE``
column; ``GLCM_IDM_AVE`` has the same value.


.. _glcm-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the 0° value on the worked example.

.. include:: glcm_index.inc


Features
--------

.. _feature-glcm-acor:

Autocorrelation
^^^^^^^^^^^^^^^

:Code: ``GLCM_ACOR``
:Range: :math:`[1, N_g^2]`, squared grey levels

Measures the magnitude of the fineness and coarseness of texture. Pairs of bright pixels contribute
most.

.. math::

   \mathrm{ACOR} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} i\,j\; p_{ij}


.. _feature-glcm-asm:

Angular second moment
^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLCM_ASM``
:Range: :math:`(0, 1]`, dimensionless

Measures how concentrated the co-occurrence probabilities are. A uniform texture with few, repeated
grey-level transitions gives values near 1.

.. math::

   \mathrm{ASM} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} p_{ij}^{\,2}

**Relations.** ``GLCM_ENERGY`` returns the same value. IBSI calls this feature both *angular second
moment* and *joint energy*.


.. _feature-glcm-cluprom:

Cluster prominence
^^^^^^^^^^^^^^^^^^

:Code: ``GLCM_CLUPROM``
:Range: :math:`[0, \infty)`, grey levels to the fourth power

Measures the skewness and peakedness of the sum of the two grey levels in a pair. High values mean
the sums are spread far from their mean.

.. math::

   \mathrm{CLUPROM} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} \big(i + j - 2\mu_y\big)^4\, p_{ij}

**Possible issues.** IBSI and PyRadiomics centre on :math:`\mu_x + \mu_y`; Nyxus centres on
:math:`2\mu_y`, twice the neighbour mean. The two are equal on a symmetric matrix. With the default
counting they differ, and the value changes when the image is mirrored.


.. _feature-glcm-clushade:

Cluster shade
^^^^^^^^^^^^^

:Code: ``GLCM_CLUSHADE``
:Range: :math:`(-\infty, \infty)`, grey levels cubed

Measures the skewness of the sum of the two grey levels in a pair. Its sign shows on which side of
the mean the sums are spread.

.. math::

   \mathrm{CLUSHADE} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} \big(i + j - 2\mu_y\big)^3\, p_{ij}

**Possible issues.** Centred on :math:`2\mu_y` like cluster prominence. The power is odd, so the
difference from IBSI can change the sign: with the default settings the worked example gives
−585.66, while the IBSI definition on the same matrix gives 9490.19.


.. _feature-glcm-clutend:

Cluster tendency
^^^^^^^^^^^^^^^^

:Code: ``GLCM_CLUTEND``
:Range: :math:`[0, \infty)`, squared grey levels

Measures how grey levels group into clusters of similar values.

.. math::

   \mathrm{CLUTEND} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} \big(i + j - 2\mu_y\big)^2\, p_{ij}

**Relations.** ``GLCM_SUMVARIANCE`` returns the same value.

**Possible issues.** Centred on :math:`2\mu_y` like cluster prominence.


.. _feature-glcm-contrast:

Contrast
^^^^^^^^

:Code: ``GLCM_CONTRAST``
:Range: :math:`[0, (N_g-1)^2]`, squared grey levels

Measures local intensity variation. Each pair is weighted by the square of its grey-level
difference, so large jumps between neighbours dominate.

.. math::

   \mathrm{Contrast} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} (i-j)^2\, p_{ij}

**Possible issues.** Contrast scales with the square of the grey-level spacing, so it depends
strongly on binning. On the worked example it is 2.9 in IBSI mode and 473.45 with the default
64-level binning, which spreads the 5 levels over the range 1 to 64. Compare contrast only between
runs with the same ``--coarseGrayDepth``.


.. _feature-glcm-correlation:

Correlation
^^^^^^^^^^^

:Code: ``GLCM_CORRELATION``
:Range: :math:`[-1, 1]`, dimensionless

Measures the linear dependence between the grey levels of neighbouring pixels.

.. math::

   \mathrm{Correlation} =
   \frac{\sum_{i,j} (i-\mu_x)(j-\mu_y)\, p_{ij}}{\sigma_x\,\sigma_y}

**Possible issues.** When either marginal has zero variance, the denominator is 0 and Nyxus returns
the ``--noval`` substitute (0 by default) instead of NaN. This can happen on very small ROIs.


.. _feature-glcm-difave:

Difference average
^^^^^^^^^^^^^^^^^^

:Code: ``GLCM_DIFAVE``
:Range: :math:`[0, N_g-1]`, grey levels

The mean absolute grey-level difference between neighbours.

.. math::

   \mathrm{DIFAVE} = \sum_{k=0}^{N_g-1} k\; p_{x-y}(k)

**Relations.** Equal to ``GLCM_DIS`` with the default binning and in IBSI mode; see
:ref:`glcm-possible-issues` for PyRadiomics-style binning.


.. _feature-glcm-difentro:

Difference entropy
^^^^^^^^^^^^^^^^^^

:Code: ``GLCM_DIFENTRO``
:Range: :math:`[0, \log_2 N_g]` bits

Measures the randomness of the grey-level differences between neighbours.

.. math::

   \mathrm{DIFENTRO} = -\sum_{k=0}^{N_g-1} p_{x-y}(k)\,\log_2\!\big(p_{x-y}(k)+\varepsilon\big)


.. _feature-glcm-difvar:

Difference variance
^^^^^^^^^^^^^^^^^^^

:Code: ``GLCM_DIFVAR``
:Range: :math:`[0, \infty)`, squared grey levels

Measures how much the grey-level differences between neighbours vary around their mean.

.. math::

   \mathrm{DIFVAR} = \sum_{k=0}^{N_g-1} \big(k - \mathrm{DIFAVE}\big)^2\, p_{x-y}(k)


.. _feature-glcm-dis:

Dissimilarity
^^^^^^^^^^^^^

:Code: ``GLCM_DIS``
:Range: :math:`[0, N_g-1]`, grey levels

The mean absolute grey-level difference between neighbours, computed from the matrix directly.

.. math::

   \mathrm{DIS} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} |i-j|\; p_{ij}

**Relations.** Equal to ``GLCM_DIFAVE`` with the default binning and in IBSI mode.


.. _feature-glcm-energy:

Energy
^^^^^^

:Code: ``GLCM_ENERGY``
:Range: :math:`(0, 1]`, dimensionless

The same value as ``GLCM_ASM``, under the name other tools use.

.. math::

   \mathrm{Energy} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} p_{ij}^{\,2}


.. _feature-glcm-entropy:

Entropy
^^^^^^^

:Code: ``GLCM_ENTROPY``
:Range: :math:`[0, 2\log_2 N_g]` bits

The same value as ``GLCM_JE``, under the name other tools use.

.. math::

   \mathrm{Entropy} = -\sum_{i=1}^{N_g}\sum_{j=1}^{N_g} p_{ij}\,\log_2\!\big(p_{ij}+\varepsilon\big)

.. versionchanged:: 0.11.0+

   The sum is taken over the normalized matrix :math:`p_{ij}`. Version 0.11.0 and earlier summed
   raw counts and returned negative values.


.. _feature-glcm-hom1:

Homogeneity 1
^^^^^^^^^^^^^

:Code: ``GLCM_HOM1``
:Range: :math:`(0, 1]`, dimensionless

Measures how close the pairs are to the matrix diagonal, with a weight that falls off linearly with
the grey-level difference.

.. math::

   \mathrm{HOM1} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} \frac{p_{ij}}{1+|i-j|}

**Relations.** Equal to ``GLCM_ID``.


.. _feature-glcm-hom2:

Homogeneity 2
^^^^^^^^^^^^^

:Code: ``GLCM_HOM2`` (no ``_AVE`` column)
:Range: :math:`(0, 1]`, dimensionless

Measures how close the pairs are to the matrix diagonal, with a weight that falls off with the
square of the grey-level difference.

.. math::

   \mathrm{HOM2} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} \frac{p_{ij}}{1+(i-j)^2}

**Relations.** Equal to ``GLCM_IDM``; for the direction average use ``GLCM_IDM_AVE``.

.. versionchanged:: 0.11.0+

   The sum is taken over the normalized matrix :math:`p_{ij}`. Version 0.11.0 and earlier summed
   raw counts and returned values above 1.


.. _feature-glcm-id:

Inverse difference
^^^^^^^^^^^^^^^^^^

:Code: ``GLCM_ID``
:Range: :math:`(0, 1]`, dimensionless

Measures local homogeneity. Pairs with equal grey levels contribute fully, and the weight falls off
linearly with their difference.

.. math::

   \mathrm{ID} = \sum_{k=0}^{N_g-1} \frac{p_{x-y}(k)}{1+k}

**Relations.** Equal to ``GLCM_HOM1``.


.. _feature-glcm-idm:

Inverse difference moment
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLCM_IDM``
:Range: :math:`(0, 1]`, dimensionless

Measures local homogeneity, penalizing large grey-level differences more than the inverse
difference. Also known as *homogeneity*.

.. math::

   \mathrm{IDM} = \sum_{k=0}^{N_g-1} \frac{p_{x-y}(k)}{1+k^2}

**Relations.** Equal to ``GLCM_HOM2``.


.. _feature-glcm-idmn:

Inverse difference moment normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLCM_IDMN``
:Range: :math:`(0, 1]`, dimensionless

The inverse difference moment with the difference scaled by the number of grey levels, so values
are comparable across binnings.

.. math::

   \mathrm{IDMN} = \sum_{k=0}^{N_g-1} \frac{p_{x-y}(k)}{1+k^2/N_g^2}


.. _feature-glcm-idn:

Inverse difference normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLCM_IDN``
:Range: :math:`(0, 1]`, dimensionless

The inverse difference with the difference scaled by the number of grey levels.

.. math::

   \mathrm{IDN} = \sum_{k=0}^{N_g-1} \frac{p_{x-y}(k)}{1+k/N_g}


.. _feature-glcm-infomeas1:

Information measure of correlation 1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLCM_INFOMEAS1``
:Range: :math:`[-1, 0]`, dimensionless

Compares the joint entropy with the entropy expected if neighbouring grey levels were independent.
Values closer to −1 mean stronger dependence.

.. math::

   \mathrm{IMC1} = \frac{HXY - HXY1}{HY}

**Possible issues.** IBSI and PyRadiomics divide by :math:`\max(HX, HY)`; Nyxus divides by
:math:`HY`, the entropy of the neighbour marginal. On a symmetric matrix :math:`HX = HY`, so the
difference shows only with the default counting, where the value can change when the image is
mirrored. When :math:`HY = 0` the result is the ``--noval`` substitute.


.. _feature-glcm-infomeas2:

Information measure of correlation 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLCM_INFOMEAS2``
:Range: :math:`[0, 1)`, dimensionless

A second measure of the dependence between neighbouring grey levels; values near 1 mean strong
dependence.

.. math::

   \mathrm{IMC2} = \sqrt{\,\big|\,1 - e^{-2\,(HXY2 - HXY)}\big|\,}

The absolute value guards against a small negative argument caused by rounding.


.. _feature-glcm-iv:

Inverse variance
^^^^^^^^^^^^^^^^

:Code: ``GLCM_IV``
:Range: :math:`[0, 1]`, dimensionless

Measures homogeneity among pairs with different grey levels; pairs with equal levels are excluded.

.. math::

   \mathrm{IV} = \sum_{k=1}^{N_g-1} \frac{p_{x-y}(k)}{k^2}


.. _feature-glcm-jave:

Joint average
^^^^^^^^^^^^^

:Code: ``GLCM_JAVE``
:Range: :math:`[1, N_g]`, grey levels

The mean grey level of the reference pixel over all counted pairs.

.. math::

   \mathrm{JAVE} = \mu_x = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} i\; p_{ij}


.. _feature-glcm-je:

Joint entropy
^^^^^^^^^^^^^

:Code: ``GLCM_JE``
:Range: :math:`[0, 2\log_2 N_g]` bits

Measures the randomness of the co-occurrence distribution; higher values mean more varied
grey-level pairs.

.. math::

   \mathrm{JE} = -\sum_{i=1}^{N_g}\sum_{j=1}^{N_g} p_{ij}\,\log_2\!\big(p_{ij}+\varepsilon\big)

**Relations.** ``GLCM_ENTROPY`` returns the same value.

**Possible issues.** :math:`\varepsilon` is 10\ :sup:`-9`; PyRadiomics uses machine epsilon, a
second, smaller source of difference besides the approximate logarithm.


.. _feature-glcm-jmax:

Joint maximum
^^^^^^^^^^^^^

:Code: ``GLCM_JMAX``
:Range: :math:`(0, 1]`, dimensionless

The probability of the most frequent grey-level pair. PyRadiomics calls it *maximum probability*.

.. math::

   \mathrm{JMAX} = \max_{i,j}\; p_{ij}


.. _feature-glcm-jvar:

Joint variance
^^^^^^^^^^^^^^

:Code: ``GLCM_JVAR``
:Range: :math:`[0, \infty)`, squared grey levels

Measures the spread of grey levels in the counted pairs around the joint average.

.. math::

   \mathrm{JVAR} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} (j-\mu_x)^2\, p_{ij}

**Possible issues.** IBSI defines joint variance as :math:`\sum_{i,j}(i-\mu_x)^2\,p_{ij}`; Nyxus
takes the neighbour's level :math:`j` around the reference mean :math:`\mu_x`. The two agree on a
symmetric matrix. With the default counting they differ: 345.41 instead of 332.44 on the worked
example with 64-level binning. ``GLCM_VARIANCE`` gives the IBSI value in both cases.


.. _feature-glcm-sumaverage:

Sum average
^^^^^^^^^^^

:Code: ``GLCM_SUMAVERAGE``
:Range: :math:`[2, 2N_g]`, grey levels

The mean of the sum of the two grey levels in a pair.

.. math::

   \mathrm{SUMAVERAGE} = \sum_{k=2}^{2N_g} k\; p_{x+y}(k)

**Relations.** Equal to :math:`\mu_x + \mu_y`, which is twice ``GLCM_JAVE`` on a symmetric matrix.


.. _feature-glcm-sumentropy:

Sum entropy
^^^^^^^^^^^

:Code: ``GLCM_SUMENTROPY``
:Range: :math:`[0, \log_2(2N_g-1)]` bits

Measures the randomness of the sums of the two grey levels in a pair.

.. math::

   \mathrm{SUMENTROPY} = -\sum_{k=2}^{2N_g} p_{x+y}(k)\,\log_2\!\big(p_{x+y}(k)+\varepsilon\big)


.. _feature-glcm-sumvariance:

Sum variance
^^^^^^^^^^^^

:Code: ``GLCM_SUMVARIANCE``
:Range: :math:`[0, \infty)`, squared grey levels

Measures the spread of the sums of the two grey levels in a pair.

.. math::

   \mathrm{SUMVARIANCE} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} \big(i + j - 2\mu_y\big)^2\, p_{ij}

**Relations.** Nyxus returns the ``GLCM_CLUTEND`` value. IBSI defines sum variance as
:math:`\sum_k (k - \mathrm{SUMAVERAGE})^2\, p_{x+y}(k)`, the same quantity on a symmetric matrix.

**Possible issues.** With the default counting the two differ: 882.25 instead of 867.43 on the
worked example with 64-level binning.


.. _feature-glcm-variance:

Variance
^^^^^^^^

:Code: ``GLCM_VARIANCE``
:Range: :math:`[0, \infty)`, squared grey levels

Measures the spread of the reference pixel's grey level around its mean. PyRadiomics calls it *sum
of squares*.

.. math::

   \mathrm{Variance} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_g} (i-\mu_x)^2\, p_{ij}

**Relations.** Matches the IBSI definition of joint variance on every matrix; ``GLCM_JVAR`` matches
it only on a symmetric one.


.. _glcm-possible-issues:

Possible issues
---------------

Asymmetric matrix by default
   With the default (positive) ``--coarseGrayDepth`` each ordered pair is counted once, while
   PyRadiomics, MIRP and IBSI count both orders. Features that use only one side of the pair then
   take it from a fixed side, and some mix the two sides:

   .. list-table::
      :widths: 40 60
      :header-rows: 1

      * - Feature
        - Difference from IBSI with the default settings
      * - ``CLUPROM``, ``CLUSHADE``, ``CLUTEND``, ``SUMVARIANCE``
        - Centred on :math:`2\mu_y` instead of :math:`\mu_x + \mu_y`
      * - ``JVAR``
        - Neighbour level around the reference mean
      * - ``INFOMEAS1``
        - Divides by :math:`HY` instead of :math:`\max(HX, HY)`

   These values can change when the image is mirrored. Run in IBSI mode to match IBSI,
   PyRadiomics or MIRP.

Constant ROI
   Before building any matrix, Nyxus bins the ROI's minimum and maximum intensities to
   ``--coarseGrayDepth`` levels counted from 0 (``glcm/greydepth`` from Python), in IBSI mode too.
   When the two fall into the same level, every GLCM feature returns 0, whatever ``--noval`` is,
   a known defect (`#511 <https://github.com/PolusAI/nyxus/issues/511>`__): the substitute is written and then overwritten by the empty
   per-angle results, which the output pads with 0.
   This covers a constant ROI and, with a positive grey depth :math:`N` (64 by default), any ROI
   whose minimum is at least :math:`(N-1)/N` of its maximum: in IBSI mode with the default
   :math:`N`, a ROI of intensities 1000 and 1001 returns 0 in every column, while one of 10 and
   11 does not. With a zero or negative ``--coarseGrayDepth`` only a constant ROI does.

Grey-level values and indices
   With PyRadiomics-style binning (negative ``--coarseGrayDepth``) the grey levels are the distinct
   binned values present in the ROI. Most features use those values, but ``DIS``, ``HOM1``,
   ``HOM2``, ``DIFENTRO``, ``SUMENTROPY`` and the inverse-difference features use positions 1 to
   :math:`N_g` in that list, and ``IDN`` and ``IDMN`` divide by :math:`N_g`, the number of levels
   present. ``DIFAVE``, ``SUMAVERAGE``, ``IV``, ``JVAR`` and ``DIFVAR`` are a known defect
   (`#510 <https://github.com/PolusAI/nyxus/issues/510>`__) and, when the levels present are not
   consecutive, match neither convention. ``DIFAVE``, ``SUMAVERAGE`` and ``IV`` group pairs by the
   difference or sum of their positions, but weight each group by the value difference or sum of
   one pair in it. ``JVAR`` takes positions minus ``JAVE``, a mean of values, and ``DIFVAR`` takes
   position differences minus ``DIFAVE``. Values and positions coincide with the default binning
   and in IBSI mode.


Validation
----------

The test suite checks every 2D GLCM feature against an independent implementation, named in the
:ref:`index <glcm-index>`. PyRadiomics and MIRP are compared on matched settings (symmetric matrix,
identical grey levels), and IBSI mode reproduces the IBSI consensus values on the IBSI digital
phantom. Because the reference tools count pairs symmetrically, the default asymmetric counting is
covered by regression tests only.


References
----------

1. Haralick RM, Shanmugam K, Dinstein I. *Textural features for image classification.* IEEE
   Transactions on Systems, Man, and Cybernetics. 1973;SMC-3(6):610–621.
2. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
3. van Griethuysen JJM, et al. *Computational radiomics system to decode the radiographic
   phenotype.* Cancer Research. 2017;77(21):e104–e107.
