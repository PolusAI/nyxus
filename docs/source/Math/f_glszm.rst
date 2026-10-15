Grey-level size-zone matrix (GLSZM)
===================================

GLSZM features describe the zones of an ROI: groups of connected pixels that share one grey level.
The matrix counts zones by grey level and by size, regardless of direction. Fine textures give many
small zones; coarse or homogeneous textures give few large ones.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``GLSZM_*``: 16 features, one value per ROI
   * - **Group**
     - ``*ALL_GLSZM*``; also part of ``*ALL*`` and ``*WHOLESLIDE*``
   * - **Dimensions**
     - 2D. The 3D family, ``3GLSZM_*``, has its own page, :doc:`f_glszm_3d`.
   * - **GPU**
     - With ``--useGpu=true`` in GPU builds the features are summed on the GPU in single
       precision, with an exact logarithm for zone entropy; zones are found on the CPU. The GPU
       path is not validated by the test suite, and its grey level non-uniformity, normalized grey
       level non-uniformity, grey level variance and zone variance do not follow the formulas on
       this page.
   * - **Out-of-core**
     - Yes, with the same values as in memory on an image that Nyxus reads as one tile. On other
       images out-of-core processing of this family raises an error or returns wrong
       values, depending on the ROI (see :ref:`large-images-tiles`).
   * - **IBSI**
     - All 16 features have an IBSI identifier.

.. tip::

   To reproduce IBSI, PyRadiomics or MIRP values, run in IBSI mode: ``--ibsi=true`` on the command
   line, ``ibsi=True`` in Python. With the default settings the pixels of the bounding box that lie
   outside the mask become grey level 1 and form zones of their own, which changes every feature
   (see :ref:`glszm-possible-issues`).


How the matrix is built
-----------------------

Nyxus scans the ROI's bounding box row by row, from the top left. Each pixel that is not yet in a
zone and has a non-zero grey level starts a new zone. The zone grows through pixels of the same grey
level, one step at a time, to the east, south-east, south or south-west of a pixel already in the
zone. :math:`\textbf{P}(i,j)` counts the zones of grey level :math:`i` and size :math:`j` pixels.

- **Connectivity:** the steps cover half of the 8-neighbourhood. A zone whose pixels can all be
  reached this way from its first pixel is the 8-connected zone that IBSI defines. A zone that needs
  a step to the west, north-west, north or north-east is split into several zones (see
  :ref:`glszm-possible-issues`).
- **Directions:** none. One matrix is built per ROI, and each feature has one value.
- **Grey levels:** intensities are first binned into :math:`N_g` grey levels, as set by
  ``--coarseGrayDepth`` (64 by default). A pixel of intensity :math:`x` gets level
  :math:`\lfloor N_g\,x / x_{\max} \rfloor + 1`, capped at :math:`N_g`, where :math:`x_{\max}` is
  the ROI's largest intensity. In IBSI mode they are not binned: each integer intensity is a grey
  level.
- **Background:** in IBSI mode, and with PyRadiomics-style binning (a negative
  ``--coarseGrayDepth``), pixels of intensity 0 belong to no zone, whether they lie outside the mask
  or inside it. With the default binning, intensity 0 becomes grey level 1, so the bounding-box
  pixels outside the mask form zones and are counted in :math:`N_p`.


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

In IBSI mode (no binning) the matrix is below; rows are grey levels 1 to 5 and columns are zone
sizes 1 to 6. The six pixels of level 1 in the lower left form one zone; the 3 at the start of row
2 joins the 3 below it to the south-east, a zone of size 2. Every zone of this image is the same
under Nyxus' scan and under full 8-connectivity.

.. math::

   \textbf{P} = \begin{bmatrix}
   3 & 0 & 0 & 0 & 0 & 1\\
   2 & 2 & 0 & 0 & 0 & 0\\
   1 & 2 & 0 & 0 & 0 & 0\\
   1 & 0 & 0 & 0 & 0 & 0\\
   2 & 1 & 0 & 0 & 0 & 0 \end{bmatrix},
   \qquad N_z = \sum_{i,j} \textbf{P}(i,j) = 15, \qquad N_p = 25

The *Example* value of each feature in the :ref:`index <glszm-index>` is what Nyxus returns for this
ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/glszm_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features="*ALL_GLSZM*" --ibsi=true \
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
   * - :math:`N_s`
     - Number of columns of :math:`\textbf{P}`: the number of pixels in the bounding box, the
       largest possible zone.
   * - :math:`\textbf{P}(i,j)`
     - Number of zones of grey level :math:`g_i` and size :math:`j` pixels.
   * - :math:`N_z`
     - Number of zones, :math:`N_z = \sum_{i,j}\textbf{P}(i,j)`.
   * - :math:`N_p`
     - Number of pixels in zones: the bounding-box pixels with a non-zero grey level.
   * - :math:`p(i,j)`
     - Zone probability, :math:`p(i,j) = \textbf{P}(i,j) / N_z`.
   * - :math:`s_i,\; s_j`
     - Zones per grey level, :math:`s_i = \sum_j \textbf{P}(i,j)`, and per size,
       :math:`s_j = \sum_i \textbf{P}(i,j)`.
   * - :math:`\mu_g,\; \mu_s`
     - Mean grey level and mean size over zones, :math:`\mu_g = \sum_{i,j} g_i\,p(i,j)` and
       :math:`\mu_s = \sum_{i,j} j\,p(i,j)`.
   * - :math:`\varepsilon`
     - :math:`2.2 \times 10^{-16}`, added inside the logarithm.

Sums over :math:`i` run from 1 to :math:`N_g` and over :math:`j` from 1 to :math:`N_s`. Empty rows
and columns add nothing. Every feature uses the grey level :math:`g_i`, never the row position.

The logarithm in zone entropy is computed with a fast single-precision base-2 approximation (on
the CPU; see :ref:`conventions-entropy` for the GPU path). Zone entropy therefore differs from an
exact logarithm by at most 0.009 bits: a relative 7.1 × 10\ :sup:`-4` on the worked example, and
under 4 × 10\ :sup:`-3` on the IBSI phantom, the tolerance it is validated at.


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
     - No binning; pixels of intensity 0 are excluded.

The 2D family has no direction, distance or metaparameter settings. ``3glszm/greydepth`` applies to
``3GLSZM_*`` only. All options are listed in the :ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named with its code: ``GLSZM_SAE``, ``GLSZM_ZP`` and so on. There
are no per-direction columns and no ``_AVE`` columns.


.. _glszm-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with the relative tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example.

.. include:: glszm_index.inc


Features
--------

.. _feature-glszm-gln:

Grey level non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLSZM_GLN``
:Range: :math:`[1, N_z]`, zones

Measures how evenly the zones are spread over the grey levels. Low values mean the zones are spread
evenly.

.. math::

   \mathrm{GLN} = \frac{1}{N_z}\sum_{i=1}^{N_g} s_i^{\,2}


.. _feature-glszm-glnn:

Grey level non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLSZM_GLNN``
:Range: :math:`(0, 1]`, dimensionless

Grey level non-uniformity divided by the number of zones once more, so values are comparable across
ROIs.

.. math::

   \mathrm{GLNN} = \frac{1}{N_z^{\,2}}\sum_{i=1}^{N_g} s_i^{\,2}


.. _feature-glszm-glv:

Grey level variance
^^^^^^^^^^^^^^^^^^^

:Code: ``GLSZM_GLV``
:Range: :math:`[0, \infty)`, squared grey levels

The variance of the grey level over all zones, each zone counted once whatever its size.

.. math::

   \mathrm{GLV} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_s} \big(g_i - \mu_g\big)^2\, p(i,j)

**Possible issues.** The value scales with the square of the grey-level spacing, so it depends
strongly on binning. On the worked example it is 2.089 in IBSI mode and 341.05 with the default
64-level binning, which maps the 5 levels to 13, 26, 39, 52 and 64. Compare it only between runs
with the same ``--coarseGrayDepth``.


.. _feature-glszm-hglze:

High grey level zone emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLSZM_HGLZE``
:Range: :math:`[g_1^{\,2}, g_{N_g}^{\,2}]`, squared grey levels

The mean over all zones of the squared grey level. High when bright zones dominate.

.. math::

   \mathrm{HGLZE} = \frac{1}{N_z}\sum_{i=1}^{N_g} s_i\, g_i^{\,2}


.. _feature-glszm-lae:

Large area emphasis
^^^^^^^^^^^^^^^^^^^

:Code: ``GLSZM_LAE``
:Range: :math:`[1, N_p^{\,2}]`, squared pixels

The mean over all zones of the squared zone size. High when large zones dominate, as in coarse
texture. IBSI calls it *large zone emphasis*.

.. math::

   \mathrm{LAE} = \frac{1}{N_z}\sum_{j=1}^{N_s} s_j\, j^2

**Possible issues.** Wrong, and often negative, when a zone has more than 46 340 pixels (see
:ref:`glszm-possible-issues`).


.. _feature-glszm-lahgle:

Large area high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLSZM_LAHGLE``
:Range: :math:`(0, \infty)`, squared pixels times squared grey levels

The mean over all zones of the squared zone size times the squared grey level. High for large
bright zones.

.. math::

   \mathrm{LAHGLE} = \frac{1}{N_z}\sum_{i=1}^{N_g}\sum_{j=1}^{N_s} \textbf{P}(i,j)\, g_i^{\,2}\, j^2


.. _feature-glszm-lalgle:

Large area low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLSZM_LALGLE``
:Range: :math:`(0, \infty)`, squared pixels over squared grey levels

The mean over all zones of the squared zone size over the squared grey level. High for large dark
zones.

.. math::

   \mathrm{LALGLE} = \frac{1}{N_z}\sum_{i=1}^{N_g}\sum_{j=1}^{N_s} \frac{\textbf{P}(i,j)\, j^2}{g_i^{\,2}}

**Possible issues.** With the default binning the background zones have level 1, the largest
weight, and dominate the value (see :ref:`glszm-possible-issues`).


.. _feature-glszm-lglze:

Low grey level zone emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLSZM_LGLZE``
:Range: :math:`(0, 1]`, inverse squared grey levels

The mean over all zones of one over the squared grey level. High when dark zones dominate.

.. math::

   \mathrm{LGLZE} = \frac{1}{N_z}\sum_{i=1}^{N_g} \frac{s_i}{g_i^{\,2}}

**Possible issues.** With the default binning the background zones have level 1 and weight 1, far
above any binned ROI level. On the ROI in :ref:`glszm-possible-issues` the value is 0.3342 with
them and 0.001356 without them.


.. _feature-glszm-sae:

Small area emphasis
^^^^^^^^^^^^^^^^^^^

:Code: ``GLSZM_SAE``
:Range: :math:`(0, 1]`, inverse squared pixels

The mean over all zones of one over the squared zone size. High when small zones dominate, as in
fine texture. IBSI calls it *small zone emphasis*.

.. math::

   \mathrm{SAE} = \frac{1}{N_z}\sum_{j=1}^{N_s} \frac{s_j}{j^2}

**Possible issues.** A zone split by the scan adds smaller zones and changes the value: 0.4537
instead of 0.1563 on the 2 × 3 image in :ref:`glszm-possible-issues`.
When the bounding box has 65 536 pixels or more, the value is the ``--noval`` substitute, 0 by
default (``-0.0`` in Arrow and Parquet files; see :ref:`conventions-missing`).


.. _feature-glszm-sahgle:

Small area high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLSZM_SAHGLE``
:Range: :math:`(0, g_{N_g}^{\,2}]`, squared grey levels over squared pixels

The mean over all zones of the squared grey level over the squared zone size. High for small bright
zones.

.. math::

   \mathrm{SAHGLE} = \frac{1}{N_z}\sum_{i=1}^{N_g}\sum_{j=1}^{N_s} \frac{\textbf{P}(i,j)\, g_i^{\,2}}{j^2}


.. _feature-glszm-salgle:

Small area low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLSZM_SALGLE``
:Range: :math:`(0, 1]`, inverse squared pixels and grey levels

The mean over all zones of one over the squared zone size times the squared grey level. High for
small dark zones.

.. math::

   \mathrm{SALGLE} = \frac{1}{N_z}\sum_{i=1}^{N_g}\sum_{j=1}^{N_s} \frac{\textbf{P}(i,j)}{g_i^{\,2}\, j^2}


.. _feature-glszm-szn:

Size zone non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLSZM_SZN``
:Range: :math:`[1, N_z]`, zones

Measures how evenly the zones are spread over the sizes. Low values mean the sizes are spread
evenly.

.. math::

   \mathrm{SZN} = \frac{1}{N_z}\sum_{j=1}^{N_s} s_j^{\,2}


.. _feature-glszm-sznn:

Size zone non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLSZM_SZNN``
:Range: :math:`(0, 1]`, dimensionless

Size zone non-uniformity divided by the number of zones once more, so values are comparable across
ROIs.

.. math::

   \mathrm{SZNN} = \frac{1}{N_z^{\,2}}\sum_{j=1}^{N_s} s_j^{\,2}


.. _feature-glszm-ze:

Zone entropy
^^^^^^^^^^^^

:Code: ``GLSZM_ZE``
:Range: :math:`[0, \log_2(N_g N_s)]` bits

Measures the randomness of the joint distribution of zone grey level and zone size. Higher values
mean more varied zones.

.. math::

   \mathrm{ZE} = -\sum_{i=1}^{N_g}\sum_{j=1}^{N_s} p(i,j)\,\log_2\!\big(p(i,j)+\varepsilon\big)

**Possible issues.** The logarithm is approximate. On the worked example Nyxus returns 3.0544, while
an exact logarithm gives 3.0566, a relative difference of 7.1 × 10\ :sup:`-4`. The approximation
works in single precision, so :math:`\varepsilon` has no effect on non-zero probabilities.


.. _feature-glszm-zp:

Zone percentage
^^^^^^^^^^^^^^^

:Code: ``GLSZM_ZP``
:Range: :math:`[1/N_p, 1]`, dimensionless

The number of zones divided by the number of pixels. High for fine texture, where most zones are
small.

.. math::

   \mathrm{ZP} = \frac{N_z}{N_p}

**Possible issues.** :math:`N_p` counts the bounding-box pixels with a non-zero grey level. With the
default binning that is every pixel of the bounding box, inside the mask or not.


.. _feature-glszm-zv:

Zone variance
^^^^^^^^^^^^^

:Code: ``GLSZM_ZV``
:Range: :math:`[0, \infty)`, squared pixels

The variance of the zone size over all zones. IBSI calls it *zone size variance*.

.. math::

   \mathrm{ZV} = \sum_{i=1}^{N_g}\sum_{j=1}^{N_s} \big(j - \mu_s\big)^2\, p(i,j)


.. _glszm-possible-issues:

Possible issues
---------------

Zones are not fully 8-connected
   A zone grows only by steps to the east, south-east, south and south-west. A same-level pixel that
   can be reached only by a step to the west, north-west, north or north-east starts a zone of its
   own. In the image below, the 1 at the lower left touches the 1 to its right, but the zone starts
   at the top-right 1 and never steps west, so Nyxus finds zones of size 3 and 1 where IBSI has one
   zone of size 4.

   .. math::

      \begin{bmatrix} 2 & 2 & 1\\ 1 & 1 & 1 \end{bmatrix}

   Every feature can change. On this image zone percentage is 0.5 instead of 0.3333, and small area
   emphasis 0.4537 instead of 0.1563. On the 5 × 5 example of the PyRadiomics GLSZM documentation
   Nyxus finds 13 zones instead of 11, zone percentage is 0.52 instead of 0.44 and small area
   emphasis 0.6596 instead of 0.5912. Of the 512 ways to fill a 3 × 3 image with two grey levels,
   152 give a different set of zones. IBSI mode does not change the scan.

Background as grey level 1 with the default binning
   With a positive ``--coarseGrayDepth``, intensity 0 bins to level 1. The bounding-box pixels
   outside the mask then form zones of level 1, and :math:`N_p` is the size of the bounding box. ROI
   pixels whose intensity is below :math:`x_{\max} / N_g` also get level 1 and can join those zones.
   Take this 6 × 5 ROI, whose four corners lie outside the mask:

   .. math::

      \begin{bmatrix}
      \cdot & 3 & 7 & 7 & 7 & \cdot\\
      3 & 5 & 5 & 7 & 2 & 2\\
      6 & 5 & 5 & 9 & 9 & 2\\
      6 & 6 & 4 & 9 & 4 & 4\\
      \cdot & 6 & 4 & 4 & 4 & \cdot \end{bmatrix}

   With the default settings the four corners add four zones of size 1. Nyxus counts 12 zones over
   30 pixels, so zone percentage is 0.4; without the corners the same binning gives 8 zones over 26
   pixels, 0.3077. Small area emphasis is 0.4144 instead of 0.1215, and low grey level zone
   emphasis 0.3342 instead of 0.001356. IBSI mode excludes the corners and gives 8 zones over 26
   pixels.

Pixels of intensity 0 inside the mask
   In IBSI mode a pixel of intensity 0 belongs to no zone and is not counted in :math:`N_p`, even
   inside the mask.

Constant ROI
   When all of a ROI's pixels have the same intensity, every GLSZM feature returns the ``--noval``
   substitute (0 by default).

Very large zones and bounding boxes
   Large area emphasis and small area emphasis square the zone size in 32-bit integer arithmetic.
   Large area emphasis is wrong as soon as a zone has more than 46 340 pixels. On a 250 × 250 ROI
   holding one zone of 62 499 pixels and one of 1 pixel, Nyxus returns -194 421 147 instead of
   1 953 062 501, and on a 300 × 300 ROI with a zone of 89 999 pixels, -245 057 295 instead of
   4 049 910 001. The overflow is undefined behaviour, so a build can also return the true value;
   one macOS ARM build did. Small area emphasis sums over every zone size up to the size of the bounding box.
   When the bounding box has 65 536 pixels or more, one of those terms is undefined and the value
   is replaced by the ``--noval`` substitute (0 by default; ``-0.0`` in Arrow and Parquet files),
   whatever the zones: a random texture of four grey levels gives about 0.5 in a 255 × 256 box and
   0 in a 300 × 300 box. The other 14
   features are computed in double precision and are exact at these sizes, including the two
   large area grey level emphases. IBSI mode does not change this.

Memory use
   :math:`\textbf{P}` has :math:`N_g` rows and one column per bounding-box pixel. In IBSI mode
   :math:`N_g` is the largest intensity, so a 200 × 200 ROI with intensities up to 4095 needs a
   655 MB matrix. Binning to a small :math:`N_g` keeps it small: 23 MB for a 300 × 300 ROI at 64
   levels.


Validation
----------

The test suite checks every 2D GLSZM feature against MIRP on the four slices of the IBSI digital
phantom in IBSI mode. Fifteen features agree to an absolute 10\ :sup:`-9`; zone entropy agrees to a
relative 4 × 10\ :sup:`-3`, which allows for the approximate logarithm. The four-slice means are
also checked against IBSI's published consensus values at a relative 10\ :sup:`-2`, the precision
those values are published to. The phantom's zones are the same under Nyxus' scan and under
8-connectivity, so these comparisons do not cover the split zones described above. The default
mode is covered by regression tests only.


References
----------

1. Thibault G, Fertil B, Navarro C, et al. *Texture indexes and gray level size zone matrix:
   application to cell nuclei classification.* Pattern Recognition and Information Processing
   (PRIP). 2009:140–145.
2. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
3. van Griethuysen JJM, et al. *Computational radiomics system to decode the radiographic
   phenotype.* Cancer Research. 2017;77(21):e104–e107.
