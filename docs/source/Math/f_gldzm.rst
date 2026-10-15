Grey-level distance-zone matrix (GLDZM)
=======================================

GLDZM features describe the zones of an ROI, groups of connected pixels that share one grey level,
by how far each zone lies from the edge of the ROI. The matrix counts zones by grey level and by
distance, regardless of direction. Zones close to the edge give small distances; large uniform
areas deep inside the ROI give large ones.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``GLDZM_*``: 18 features, one value per ROI
   * - **Group**
     - ``*ALL_GLDZM*``; also part of ``*ALL*`` (not of ``*WHOLESLIDE*``)
   * - **Dimensions**
     - 2D. The 3D family, ``3GLDZM_*``, has its own page, :doc:`f_gldzm_3d`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory on an image that Nyxus reads as one tile. On other
       images out-of-core processing of this family raises an error or returns wrong
       values, depending on the ROI (see :ref:`large-images-tiles`).
   * - **IBSI**
     - 16 of the 18 features have an IBSI identifier. Grey level mean and zone distance mean have
       none.

.. tip::

   To reproduce IBSI or MIRP values, run in IBSI mode: ``--ibsi=true`` on the command line,
   ``ibsi=True`` in Python. With the default settings the pixels of the bounding box that lie
   outside the mask become grey level 1 and form zones, which ROI pixels of level 1 can join, and
   only the edges of the bounding box count as the ROI edge (see :ref:`gldzm-possible-issues`). In
   IBSI mode a pixel next to background inside the bounding box gets distance 2 where IBSI gives 1.


How the matrix is built
-----------------------

Nyxus scans the ROI's bounding box row by row, from the top left. Each pixel that is not yet in a
zone starts a new zone, which grows through every pixel of the same grey level that touches it at
an edge or a corner. Each pixel of the zone has a distance to the edge, and the zone's distance is
the smallest of them. :math:`\textbf{P}(i,j)` counts the zones of grey level :math:`g_i` and
distance :math:`j`.

- **Connectivity:** 8-connected, as IBSI defines for 2D. Two pixels of one grey level that touch
  only at a corner belong to the same zone.
- **Distance:** from a pixel, Nyxus walks left, right, up and down along its row and column. Each
  walk stops at the first pixel of grey level 0 or at the last pixel of the bounding box. The
  pixel's distance is the shortest walk plus 1, and at least 1. A pixel on the edge of the bounding
  box has distance 1, the next ring inwards distance 2, and so on. A pixel next to a pixel of grey
  level 0 has distance 2 (see :ref:`gldzm-possible-issues`).
- **Directions:** none. One matrix is built per ROI, and each feature has one value.
- **Grey levels:** intensities are first binned into :math:`N_g` grey levels, as set by
  ``--coarseGrayDepth`` (64 by default). A pixel of intensity :math:`x` gets level
  :math:`\lfloor N_g\,x / x_{\max} \rfloor + 1`, capped at :math:`N_g`, where :math:`x_{\max}` is
  the ROI's largest intensity. In IBSI mode they are not binned: each integer intensity is a grey
  level.
- **Background:** in IBSI mode, pixels of intensity 0 belong to no zone, whether they lie outside
  the mask or inside it, and the distance walks stop at them. With the default binning, intensity
  0 becomes grey level 1, so the bounding-box pixels outside the mask form zones, and the walks run
  through them to the edges of the bounding box.


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

The ROI fills its bounding box, so the outer ring of pixels has distance 1, the next ring distance 2
and the centre pixel distance 3. In IBSI mode (no binning) the matrix is below; rows are grey
levels 1 to 5 and columns are distances 1 and 2. The 1 in the middle of row 2 touches no other 1,
so it is a zone of its own at distance 2. The two 5s in row 3 form one zone; the centre pixel has
distance 3 and its neighbour distance 2, so the zone has distance 2.

.. math::

   \textbf{P} = \begin{bmatrix}
   3 & 1\\
   4 & 0\\
   3 & 0\\
   1 & 0\\
   2 & 1 \end{bmatrix},
   \qquad N_s = \sum_{i,j} \textbf{P}(i,j) = 15, \qquad N_v = 25

The *Example* value of each feature in the :ref:`index <gldzm-index>` is what Nyxus returns for this
ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/gldzm_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features="*ALL_GLDZM*" --ibsi=true \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N_g`
     - Number of rows of :math:`\textbf{P}`. With binning, the number of grey levels present in the
       bounding box; in IBSI mode, the ROI's largest intensity.
   * - :math:`g_i`
     - Grey level of row :math:`i`. With binning, the :math:`i`-th smallest level present; in IBSI
       mode, :math:`g_i = i`.
   * - :math:`N_d`
     - Number of columns of :math:`\textbf{P}`: the largest zone distance.
   * - :math:`\textbf{P}(i,j)`
     - Number of zones of grey level :math:`g_i` and distance :math:`j`.
   * - :math:`N_s`
     - Number of zones, :math:`N_s = \sum_{i,j}\textbf{P}(i,j)`.
   * - :math:`N_v`
     - Number of ROI pixels: the pixels inside the mask, including those of intensity 0.
   * - :math:`p(i,j)`
     - Zone probability, :math:`p(i,j) = \textbf{P}(i,j) / N_s`.
   * - :math:`m_i,\; m_j`
     - Zones per grey level, :math:`m_i = \sum_j \textbf{P}(i,j)`, and per distance,
       :math:`m_j = \sum_i \textbf{P}(i,j)`.
   * - :math:`\mu_g,\; \mu_d`
     - Mean grey level and mean distance over zones, :math:`\mu_g = \sum_{i,j} g_i\,p(i,j)` and
       :math:`\mu_d = \sum_{i,j} j\,p(i,j)`.
   * - :math:`\varepsilon`
     - :math:`2.2 \times 10^{-16}`, added inside the logarithm.

Sums over :math:`i` run from 1 to :math:`N_g` and over :math:`j` from 1 to :math:`N_d`. Empty rows
and columns add nothing. Every feature uses the grey level :math:`g_i`, never the row position.

Zone distance entropy uses the exact base-2 logarithm in double precision, not the fast
approximation of the GLCM, GLRLM, GLSZM and GLDM entropies (see :ref:`conventions-entropy`).


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
       as level 1; negative: PyRadiomics-style bins over the ROI range (command line only, and only
       for ROIs with no pixel of intensity 0 in the bounding box; see
       :ref:`gldzm-possible-issues`).
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus(..., ibsi=True)``
     - No binning; pixels of intensity 0 are excluded from zones and stop the distance walks.

The 2D family has no direction, distance or metaparameter settings, and zone connectivity is fixed.
No ``set_metaparam`` parameter applies to ``GLDZM_*`` or ``3GLDZM_*``. All options are listed in the
:ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named with its code: ``GLDZM_SDE``, ``GLDZM_ZP`` and so on. There
are no per-direction columns and no ``_AVE`` columns.


.. _gldzm-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with the relative tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example.

.. include:: gldzm_index.inc


Features
--------

.. _feature-gldzm-glm:

Grey level mean
^^^^^^^^^^^^^^^

:Code: ``GLDZM_GLM``
:Range: :math:`[g_1, g_{N_g}]`, grey levels

The mean grey level over all zones, each zone counted once whatever its size. IBSI defines it only
as the mean :math:`\mu` inside grey level variance.

.. math::

   \mathrm{GLM} = \mu_g = \sum_{i=1}^{N_g}\sum_{j=1}^{N_d} g_i\, p(i,j)

**Possible issues.** Depends on the binning: 2.667 in IBSI mode and 34.47 with the default
64-level binning on the worked example.


.. _feature-gldzm-glnu:

Grey level non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_GLNU``
:Range: :math:`[1, N_s]`, zones

Measures how evenly the zones are spread over the grey levels. Low values mean the zones are spread
evenly.

.. math::

   \mathrm{GLNU} = \frac{1}{N_s}\sum_{i=1}^{N_g} m_i^{\,2}

**Relations.** It uses only the zones per grey level, not their distances, so it equals GLSZM's
:ref:`grey level non-uniformity <feature-glszm-gln>` whenever both families find the same zones:
3.4 for both on the worked example.


.. _feature-gldzm-glnun:

Grey level non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_GLNUN``
:Range: :math:`(0, 1]`, dimensionless

Grey level non-uniformity divided by the number of zones once more, so values are comparable across
ROIs.

.. math::

   \mathrm{GLNUN} = \frac{1}{N_s^{\,2}}\sum_{i=1}^{N_g} m_i^{\,2}


.. _feature-gldzm-glv:

Grey level variance
^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_GLV``
:Range: :math:`[0, \infty)`, squared grey levels

The variance of the grey level over all zones, each zone counted once whatever its size.

.. math::

   \mathrm{GLV} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_d} \big(g_i - \mu_g\big)^2\, p(i,j)

**Possible issues.** The value scales with the square of the grey-level spacing. On the worked
example it is 2.089 in IBSI mode and 341.05 with the default 64-level binning, which maps the 5
levels to 13, 26, 39, 52 and 64. Compare it only between runs with the same ``--coarseGrayDepth``.


.. _feature-gldzm-hglze:

High grey level zone emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_HGLZE``
:Range: :math:`[g_1^{\,2}, g_{N_g}^{\,2}]`, squared grey levels

The mean over all zones of the squared grey level. High when bright zones dominate.

.. math::

   \mathrm{HGLZE} = \frac{1}{N_s}\sum_{i=1}^{N_g} g_i^{\,2}\, m_i

**Possible issues.** Depends on the binning: 9.2 in IBSI mode and 1529 with the default settings on
the worked example.


.. _feature-gldzm-lde:

Large distance emphasis
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_LDE``
:Range: :math:`[1, N_d^{\,2}]`, squared pixels

The mean over all zones of the squared zone distance. High when zones lie deep inside the ROI.

.. math::

   \mathrm{LDE} = \frac{1}{N_s}\sum_{j=1}^{N_d} j^2\, m_j

**Possible issues.** Pixels next to background inside the bounding box get a distance one too large
(see :ref:`gldzm-possible-issues`).


.. _feature-gldzm-ldhgle:

Large distance high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_LDHGLE``
:Range: :math:`[g_1^{\,2}, g_{N_g}^{\,2} N_d^{\,2}]`, squared grey levels times squared pixels

The mean over all zones of the squared grey level times the squared zone distance. High for bright
zones deep inside the ROI.

.. math::

   \mathrm{LDHGLE} = \frac{1}{N_s}\sum_{i=1}^{N_g}\sum_{j=1}^{N_d} g_i^{\,2}\, j^2\, \textbf{P}(i,j)


.. _feature-gldzm-ldlgle:

Large distance low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_LDLGLE``
:Range: :math:`(0, N_d^{\,2}]`, squared pixels over squared grey levels

The mean over all zones of the squared zone distance over the squared grey level. High for dark
zones deep inside the ROI.

.. math::

   \mathrm{LDLGLE} = \frac{1}{N_s}\sum_{i=1}^{N_g}\sum_{j=1}^{N_d} \frac{j^2\, \textbf{P}(i,j)}{g_i^{\,2}}


.. _feature-gldzm-lglze:

Low grey level zone emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_LGLZE``
:Range: :math:`[1/g_{N_g}^{\,2}, 1/g_1^{\,2}]`, inverse squared grey levels

The mean over all zones of one over the squared grey level. High when dark zones dominate.

.. math::

   \mathrm{LGLZE} = \frac{1}{N_s}\sum_{i=1}^{N_g} \frac{m_i}{g_i^{\,2}}

**Possible issues.** Depends on the binning: 0.3677 in IBSI mode and 0.002177 with the default
settings on the worked example. With the default binning, zones of pixels outside the mask have
level 1 and dominate it (see :ref:`gldzm-possible-issues`).


.. _feature-gldzm-sde:

Small distance emphasis
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_SDE``
:Range: :math:`[1/N_d^{\,2}, 1]`, inverse squared pixels

The mean over all zones of one over the squared zone distance. High when zones lie close to the
ROI edge.

.. math::

   \mathrm{SDE} = \frac{1}{N_s}\sum_{j=1}^{N_d} \frac{m_j}{j^2}

**Possible issues.** Pixels next to background inside the bounding box get a distance one too large,
which lowers this value (see :ref:`gldzm-possible-issues`).


.. _feature-gldzm-sdhgle:

Small distance high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_SDHGLE``
:Range: :math:`(0, g_{N_g}^{\,2}]`, squared grey levels over squared pixels

The mean over all zones of the squared grey level over the squared zone distance. High for bright
zones near the ROI edge.

.. math::

   \mathrm{SDHGLE} = \frac{1}{N_s}\sum_{i=1}^{N_g}\sum_{j=1}^{N_d} \frac{g_i^{\,2}\, \textbf{P}(i,j)}{j^2}


.. _feature-gldzm-sdlgle:

Small distance low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_SDLGLE``
:Range: :math:`(0, 1/g_1^{\,2}]`, inverse squared grey levels times inverse squared pixels

The mean over all zones of one over the squared grey level times the squared zone distance. High
for dark zones near the ROI edge.

.. math::

   \mathrm{SDLGLE} = \frac{1}{N_s}\sum_{i=1}^{N_g}\sum_{j=1}^{N_d} \frac{\textbf{P}(i,j)}{g_i^{\,2}\, j^2}


.. _feature-gldzm-zde:

Zone distance entropy
^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_ZDE``
:Range: :math:`[0, \log_2(N_g N_d)]` bits

Measures the randomness of the joint distribution of zone grey level and zone distance.

.. math::

   \mathrm{ZDE} = -\sum_{i=1}^{N_g}\sum_{j=1}^{N_d} p(i,j)\,\log_2\!\big(p(i,j)+\varepsilon\big)

The logarithm is exact, so the value agrees with IBSI and MIRP to rounding: 2.6062 on the worked
example.


.. _feature-gldzm-zdm:

Zone distance mean
^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_ZDM``
:Range: :math:`[1, N_d]`, pixels

The mean zone distance over all zones. IBSI defines it only as the mean :math:`\mu` inside zone
distance variance.

.. math::

   \mathrm{ZDM} = \mu_d = \sum_{i=1}^{N_g}\sum_{j=1}^{N_d} j\, p(i,j)


.. _feature-gldzm-zdnu:

Zone distance non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_ZDNU``
:Range: :math:`[1, N_s]`, zones

Measures how evenly the zones are spread over the distances. Low values mean the zone distances are
spread evenly.

.. math::

   \mathrm{ZDNU} = \frac{1}{N_s}\sum_{j=1}^{N_d} m_j^{\,2}


.. _feature-gldzm-zdnun:

Zone distance non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_ZDNUN``
:Range: :math:`(0, 1]`, dimensionless

Zone distance non-uniformity divided by the number of zones once more, so values are comparable
across ROIs.

.. math::

   \mathrm{ZDNUN} = \frac{1}{N_s^{\,2}}\sum_{j=1}^{N_d} m_j^{\,2}


.. _feature-gldzm-zdv:

Zone distance variance
^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDZM_ZDV``
:Range: :math:`[0, \infty)`, squared pixels

The variance of the zone distance over all zones.

.. math::

   \mathrm{ZDV} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_d} \big(j - \mu_d\big)^2\, p(i,j)


.. _feature-gldzm-zp:

Zone percentage
^^^^^^^^^^^^^^^

:Code: ``GLDZM_ZP``
:Range: :math:`(0, 1]` in IBSI mode, dimensionless

The number of zones divided by the number of ROI pixels. High for fine texture, where most zones are
small.

.. math::

   \mathrm{ZP} = \frac{N_s}{N_v}

**Possible issues.** With the default binning, :math:`N_s` also counts the zones formed by
bounding-box pixels outside the mask, while :math:`N_v` counts only the pixels inside it, so the
value can exceed 1.


.. _gldzm-possible-issues:

Possible issues
---------------

Distance next to background is one too large
   The distance walk stops on the first pixel of grey level 0 and counts it as a step, so a pixel
   that touches background inside the bounding box gets distance 2. IBSI erodes the ROI with the
   4-connected cross and gives every pixel on the ROI edge distance 1, so its distance is the
   city-block distance to the nearest pixel outside the ROI. Take this 5 × 5 ROI, whose centre pixel
   lies outside the mask:

   .. math::

      \begin{bmatrix}
      1 & 1 & 1 & 1 & 1\\
      1 & 1 & 1 & 1 & 1\\
      1 & 2 & \cdot & 1 & 1\\
      1 & 1 & 1 & 1 & 1\\
      1 & 1 & 1 & 1 & 1 \end{bmatrix}

   In IBSI mode the zone of the single 2 has distance 2 in Nyxus and 1 under IBSI. Small distance
   emphasis is 0.625 instead of 1, large distance emphasis 2.5 instead of 1 and zone distance mean
   1.5 instead of 1. The walk also looks only along the pixel's row and column, so background that
   is nearest along a diagonal is not seen. Over 200 random 9 × 9 masks with a fifth of the
   interior left out, 40 % of the ROI pixels get a different distance, almost all of them one more
   than IBSI's, and none gets less. A zone's distance changes only when its closest pixel is
   affected, so ROIs that fill their bounding box, and many others, are not affected. On the worked
   example with three pixels of the bounding box left out of the mask, small distance emphasis is
   0.8269 instead of 0.8846 and zone distance mean 1.231 instead of 1.154.

Background as grey level 1 with the default binning
   With a positive ``--coarseGrayDepth``, intensity 0 bins to level 1. The bounding-box pixels
   outside the mask then form zones of level 1, and the bounding box contains no pixel of level 0,
   so the distance walks run to the edges of the bounding box and ignore the mask. ROI pixels whose
   intensity is below :math:`x_{\max} / N_g` also get level 1 and can join those zones. On the
   worked example with three pixels of the bounding box left out of the mask, zone percentage is
   0.6818 instead of 0.5909 in IBSI mode, and low grey level zone emphasis 0.1351 instead of
   0.002038, the value with those pixels excluded. On the ROI with a hole above, the hole becomes a
   zone at distance 3: zone percentage is 0.125 instead of 0.08333 in IBSI mode, and zone distance
   mean 2.
   Rectangular ROIs with no pixel of intensity 0 are not affected.

PyRadiomics-style binning with background in the bounding box
   With a negative ``--coarseGrayDepth``, pixels of intensity 0 keep grey level 0 but still start
   zones, and a zone of level 0 has no row in the matrix. The calculation stops with an error that
   the command line does not report: every ``GLDZM_*`` column of that ROI is 0, and so is every
   ``GLDZM_*`` column of the ROIs that the same thread would have computed after it
   (``--reduceThreads`` splits the ROIs into that many consecutive shares). ROIs computed before
   it, or by another thread, keep their values. ROIs that fill their bounding
   box with non-zero intensities are not affected. The Python API accepts only a positive
   ``coarse_gray_depth``.

Pixels of intensity 0 inside the mask
   In IBSI mode a pixel of intensity 0 belongs to no zone and acts as background for the distances
   of its neighbours, but it is counted in :math:`N_v`.

Constant ROI
   When all of a ROI's pixels have the same intensity, every GLDZM feature returns the ``--noval``
   substitute (0 by default).


Validation
----------

The test suite checks the 16 IBSI features against MIRP on the four slices of the IBSI digital
phantom in IBSI mode, to an absolute 10\ :sup:`-9`, and the four-slice means of 14 of them against
IBSI's published consensus values at a relative 10\ :sup:`-2`, the precision those values are
published to; ``GLDZM_SDLGLE`` and ``GLDZM_ZDV`` are checked against MIRP only. Grey level mean and zone distance mean have no independent implementation and are covered by
regression tests.
Three of the phantom slices have pixels outside the mask inside the bounding box, and 6 of their
pixels get a larger distance from Nyxus than from IBSI, but no zone's distance changes, so these
comparisons do not cover the distance issue described above. In the default mode only the values
quoted on this page are pinned, by the worked-example script; no other test checks the default mode
against recorded or reference values.


References
----------

1. Thibault G, Angulo J, Meyer F. *Advanced statistical matrices for texture characterization:
   application to cell classification.* IEEE Transactions on Biomedical Engineering.
   2014;61(3):630–637.
2. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
