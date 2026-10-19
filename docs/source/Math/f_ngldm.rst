Neighbouring grey-level dependence matrix (NGLDM)
=================================================

NGLDM features describe how many of each pixel's neighbours share its grey level. Every ROI pixel
is the centre of a 3 × 3 neighbourhood, and the matrix counts the pixels by grey level and by
dependence count, regardless of direction. Coarse or homogeneous textures give high dependence
counts; fine textures give low ones.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``NGLDM_*``: 19 features, one value per ROI
   * - **Group**
     - ``*ALL_NGLDM*``; also part of ``*ALL*`` and ``*WHOLESLIDE*``
   * - **Dimensions**
     - 2D. The 3D family, ``3NGLDM_*``, has its own page, :doc:`f_ngldm_3d`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory on an image that Nyxus reads as one tile (see
       :ref:`large-images-tiles` for other images), except on a constant ROI (see
       :ref:`ngldm-possible-issues`).
   * - **IBSI**
     - 17 of the 19 features have an IBSI identifier. Grey level mean and dependence count mean
       have none.

.. tip::

   To reproduce IBSI or MIRP values, run in IBSI mode: ``--ibsi=true`` on the command line,
   ``ibsi=True`` in Python. NGLDM does not use the grey-level binning of the other texture families:
   with the default settings a pixel's grey level is :math:`\lfloor 64\,x / x_{\max} \rfloor`, so the
   eight features that use the grey level's value differ from those tools (see
   :ref:`ngldm-possible-issues`). Pixels outside the mask are left out in every mode.


How the matrix is built
-----------------------

Nyxus visits every ROI pixel. It counts the pixel's 8 neighbours that lie inside the ROI and have
exactly the same grey level, and adds 1 for the pixel itself: that is the pixel's dependence count
:math:`j`, from 1 to 9. A pixel with grey level :math:`g_i` and dependence count :math:`j` adds 1 to
:math:`\textbf{S}(i,j)`.

- **Neighbourhood:** the 8 pixels at Chebyshev distance 1. Two pixels are dependent only when their
  grey levels are equal (IBSI's coarseness parameter :math:`\alpha = 0`). Neither the distance nor
  :math:`\alpha` can be changed.
- **Directions:** none. One matrix is built per ROI, and each feature has one value.
- **Grey levels:** NGLDM converts intensities with its own rule, not with the binning that GLCM,
  GLRLM and GLSZM share. With ``--coarseGrayDepth`` :math:`N` (64 by default), intensity :math:`x`
  becomes grey level :math:`\lfloor N x / x_{\max} \rfloor`, where :math:`x_{\max}` is the ROI's
  largest intensity. Levels run from 0 to :math:`N`: the ROI maximum gets level :math:`N`, and every
  intensity below :math:`x_{\max}/N`, including 0, gets level 0. In IBSI mode the intensities are not
  converted: each integer intensity is a grey level. Nyxus computes :math:`(x / x_{\max})\,N` in
  floating point, so when :math:`N` is not a power of two an intensity whose :math:`N x / x_{\max}`
  is a whole number can get the level below.
- **Rows:** :math:`\textbf{S}` has one row per grey level present in the ROI, in increasing order, in
  both modes. Levels with no pixel have no row.
- **Background:** pixels outside the mask are neither centres nor neighbours, in every mode. ROI
  pixels of intensity 0 are centres with grey level 0 (see :ref:`ngldm-possible-issues`).


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

In IBSI mode the matrix is below; rows are grey levels 1 to 5 and columns are dependence counts 1
to 4. The 1 in row 2 has two neighbouring 1s, in row 3, so its dependence count is 3. None of the
three 5s touches another 5, so all three have dependence count 1.

.. math::

   \textbf{S} = \begin{bmatrix}
   0 & 1 & 2 & 1\\
   1 & 2 & 3 & 0\\
   1 & 4 & 4 & 0\\
   1 & 2 & 0 & 0\\
   3 & 0 & 0 & 0 \end{bmatrix},
   \qquad N_s = \sum_{i,j} \textbf{S}(i,j) = 25

The *Example* value of each feature in the :ref:`index <ngldm-index>` is what Nyxus returns for this
ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/ngldm_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features="*ALL_NGLDM*" --ibsi=true \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N_g`
     - Number of rows of :math:`\textbf{S}`: the number of grey levels present in the ROI.
   * - :math:`g_i`
     - Grey level of row :math:`i`: the :math:`i`-th smallest level present. In IBSI mode it is the
       intensity; otherwise it is :math:`\lfloor N x / x_{\max} \rfloor`.
   * - :math:`j`
     - Dependence count: 1 plus the number of same-level neighbours inside the ROI, from 1 to 9.
   * - :math:`N_r`
     - Largest dependence count in the ROI.
   * - :math:`\textbf{S}(i,j)`
     - Number of ROI pixels with grey level :math:`g_i` and dependence count :math:`j`.
   * - :math:`N_s`
     - Number of neighbourhoods, :math:`N_s = \sum_{i,j}\textbf{S}(i,j)`. Every ROI pixel is one, so
       :math:`N_s` is the number of ROI pixels.
   * - :math:`p_{ij}`
     - Joint probability, :math:`p_{ij} = \textbf{S}(i,j) / N_s`.
   * - :math:`s_{i\cdot},\; s_{\cdot j}`
     - Pixels per grey level, :math:`s_{i\cdot} = \sum_j \textbf{S}(i,j)`, and per dependence count,
       :math:`s_{\cdot j} = \sum_i \textbf{S}(i,j)`.
   * - :math:`\mu_g,\; \mu_j`
     - Grey level mean and dependence count mean, :math:`\mu_g = \sum_{i,j} g_i\,p_{ij}` and
       :math:`\mu_j = \sum_{i,j} j\,p_{ij}`.

Sums over :math:`i` run from 1 to :math:`N_g` and over :math:`j` from 1 to :math:`N_r`. Every
feature uses the grey level :math:`g_i`, never the row position. The three features that divide by
:math:`g_i^2` skip the row of grey level 0, if there is one, but its pixels stay in :math:`N_s`.

Dependence count entropy uses an exact base-2 logarithm.


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
     - :math:`N` in :math:`\lfloor N x / x_{\max} \rfloor`, giving levels 0 to :math:`N`. A negative
       value does not select PyRadiomics-style binning for NGLDM (see
       :ref:`ngldm-possible-issues`).
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus(..., ibsi=True)``
     - No conversion: each integer intensity is a grey level.

The neighbourhood is fixed at the 8 neighbours with :math:`\alpha = 0`. The 2D family has no
direction, distance or metaparameter settings. All options are listed in the :ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named with its code: ``NGLDM_LDE``, ``NGLDM_DCENT`` and so on.
There are no per-direction columns and no ``_AVE`` columns.


.. _ngldm-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example.

.. include:: ngldm_index.inc


Features
--------

.. _feature-ngldm-dcene:

Dependence count energy
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_DCENE``
:Range: :math:`(0, 1]`, dimensionless

The sum of the squared joint probabilities. High values mean a few combinations of grey level and
dependence count dominate.

.. math::

   \mathrm{DCENE} = \sum_{i}\sum_{j} p_{ij}^{\,2}


.. _feature-ngldm-dcent:

Dependence count entropy
^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_DCENT``
:Range: :math:`[0, \log_2(9 N_g)]` bits

Measures the randomness of the joint distribution of grey level and dependence count.

.. math::

   \mathrm{DCENT} = -\sum_{i}\sum_{j} p_{ij}\,\log_2 p_{ij}

Cells with :math:`p_{ij} = 0` add nothing.

**Relations.** In IBSI mode ``GLDM_DE`` is computed on the same matrix, but with an approximate
logarithm: on the worked example it is 3.380, against 3.383 here.


.. _feature-ngldm-dcm:

Dependence count mean
^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_DCM``
:Range: :math:`[1, 9]`, pixels

The mean dependence count over all ROI pixels.

.. math::

   \mathrm{DCM} = \mu_j = \sum_{i}\sum_{j} j\,p_{ij}

**Possible issues.** IBSI defines no such feature; :math:`\mu_j` appears there only inside
dependence count variance.


.. _feature-ngldm-dcnu:

Dependence count non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_DCNU``
:Range: :math:`[N_s/N_r, N_s]`, pixels

Measures how evenly the pixels are spread over the dependence counts. Low values mean the counts
are spread evenly.

.. math::

   \mathrm{DCNU} = \frac{1}{N_s}\sum_{j} s_{\cdot j}^{\,2}


.. _feature-ngldm-dcnun:

Dependence count non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_DCNUN``
:Range: :math:`[1/N_r, 1]`, dimensionless

Dependence count non-uniformity divided by the number of pixels once more, so values are comparable
across ROIs of different size.

.. math::

   \mathrm{DCNUN} = \frac{1}{N_s^{\,2}}\sum_{j} s_{\cdot j}^{\,2}


.. _feature-ngldm-dcp:

Dependence count percentage
^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_DCP``
:Range: 1, dimensionless

The fraction of ROI pixels that are neighbourhood centres. Every ROI pixel is one, so the value is
always 1.

.. math::

   \mathrm{DCP} = \frac{N_s}{N_s} = 1

**Possible issues.** The value carries no information about the ROI. IBSI defines it as the
number of neighbourhoods over the number of voxels, which is also 1 when every voxel counts.


.. _feature-ngldm-dcv:

Dependence count variance
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_DCV``
:Range: :math:`[0, 16]`, squared pixels

The variance of the dependence count over all ROI pixels.

.. math::

   \mathrm{DCV} = \sum_{i}\sum_{j} (j-\mu_j)^2\, p_{ij}


.. _feature-ngldm-glm:

Grey level mean
^^^^^^^^^^^^^^^

:Code: ``NGLDM_GLM``
:Range: :math:`[g_1, g_{N_g}]`, grey levels

The mean grey level over all ROI pixels.

.. math::

   \mathrm{GLM} = \mu_g = \sum_{i}\sum_{j} g_i\,p_{ij}

**Possible issues.** IBSI defines no such feature. With the default settings it is in units of
:math:`x_{\max}/64`: 35.4 on the worked example, against 2.8 in IBSI mode.


.. _feature-ngldm-glnu:

Grey level non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_GLNU``
:Range: :math:`[N_s/N_g, N_s]`, pixels

Measures how evenly the pixels are spread over the grey levels. Low values mean the grey levels are
evenly represented.

.. math::

   \mathrm{GLNU} = \frac{1}{N_s}\sum_{i} s_{i\cdot}^{\,2}


.. _feature-ngldm-glnun:

Grey level non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_GLNUN``
:Range: :math:`[1/N_g, 1]`, dimensionless

Grey level non-uniformity divided by the number of pixels once more, so values are comparable
across ROIs of different size.

.. math::

   \mathrm{GLNUN} = \frac{1}{N_s^{\,2}}\sum_{i} s_{i\cdot}^{\,2}


.. _feature-ngldm-glv:

Grey level variance
^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_GLV``
:Range: :math:`[0, \infty)`, squared grey levels

The variance of the grey level over all ROI pixels.

.. math::

   \mathrm{GLV} = \sum_{i}\sum_{j} (g_i-\mu_g)^2\, p_{ij}

**Possible issues.** Scales with the square of the grey-level spacing: 1.44 in IBSI mode and 243.36
with the default settings on the worked example.


.. _feature-ngldm-hde:

High dependence emphasis
^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_HDE``
:Range: :math:`[1, 81]`, squared pixels

The mean over all ROI pixels of the squared dependence count. High values mean pixels have many
same-level neighbours, as in a coarse texture.

.. math::

   \mathrm{HDE} = \frac{1}{N_s}\sum_{j} j^2\, s_{\cdot j}


.. _feature-ngldm-hdhgle:

High dependence high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_HDHGLE``
:Range: :math:`[0, \infty)`, squared grey levels times squared pixels

The mean over all ROI pixels of the squared dependence count times the squared grey level. High
values mean bright pixels with many same-level neighbours dominate.

.. math::

   \mathrm{HDHGLE} = \frac{1}{N_s}\sum_{i}\sum_{j} g_i^2\, j^2\,\textbf{S}(i,j)

**Possible issues.** Depends on the grey-level conversion: 35.12 in IBSI mode and 5608.0 with the
default settings on the worked example.


.. _feature-ngldm-hdlgle:

High dependence low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_HDLGLE``
:Range: :math:`[0, 81]` when :math:`g_i \ge 1`, squared pixels over squared grey levels

The mean over all ROI pixels of the squared dependence count over the squared grey level. High
values mean dark pixels with many same-level neighbours dominate.

.. math::

   \mathrm{HDLGLE} = \frac{1}{N_s}\sum_{i:\,g_i \neq 0}\;\sum_{j} \frac{j^2\,\textbf{S}(i,j)}{g_i^2}

**Possible issues.** Pixels of grey level 0 add nothing to the sum but count in :math:`N_s` (see
:ref:`ngldm-possible-issues`).


.. _feature-ngldm-hglce:

High grey level count emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_HGLCE``
:Range: :math:`[0, \infty)`, squared grey levels

The mean over all ROI pixels of the squared grey level. High values mean bright pixels dominate.

.. math::

   \mathrm{HGLCE} = \frac{1}{N_s}\sum_{i} g_i^2\, s_{i\cdot}

**Possible issues.** Depends on the grey-level conversion: 9.28 in IBSI mode and 1496.5 with the
default settings on the worked example.


.. _feature-ngldm-lde:

Low dependence emphasis
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_LDE``
:Range: :math:`[1/81, 1]`, inverse squared pixels

The mean over all ROI pixels of one over the squared dependence count. High values mean pixels have
few same-level neighbours, as in a fine texture.

.. math::

   \mathrm{LDE} = \frac{1}{N_s}\sum_{j} \frac{s_{\cdot j}}{j^2}

**Relations.** In IBSI mode ``GLDM_SDE`` is computed on the same matrix and returns the same value.


.. _feature-ngldm-ldhgle:

Low dependence high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_LDHGLE``
:Range: :math:`[0, \infty)`, squared grey levels over squared pixels

The mean over all ROI pixels of the squared grey level over the squared dependence count. High
values mean bright pixels with few same-level neighbours dominate.

.. math::

   \mathrm{LDHGLE} = \frac{1}{N_s}\sum_{i}\sum_{j} \frac{g_i^2\,\textbf{S}(i,j)}{j^2}


.. _feature-ngldm-ldlgle:

Low dependence low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_LDLGLE``
:Range: :math:`[0, 1]`, inverse squared grey levels times inverse squared pixels

The mean over all ROI pixels of one over the squared dependence count times the squared grey level.
High values mean dark pixels with few same-level neighbours dominate.

.. math::

   \mathrm{LDLGLE} = \frac{1}{N_s}\sum_{i:\,g_i \neq 0}\;\sum_{j} \frac{\textbf{S}(i,j)}{g_i^2\, j^2}

**Possible issues.** Pixels of grey level 0 add nothing to the sum but count in :math:`N_s` (see
:ref:`ngldm-possible-issues`).


.. _feature-ngldm-lglce:

Low grey level count emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``NGLDM_LGLCE``
:Range: :math:`[0, 1]`, inverse squared grey levels

The mean over all ROI pixels of one over the squared grey level. High values mean dark pixels
dominate.

.. math::

   \mathrm{LGLCE} = \frac{1}{N_s}\sum_{i:\,g_i \neq 0} \frac{s_{i\cdot}}{g_i^2}

**Possible issues.** Depends on the grey-level conversion: 0.2723 in IBSI mode and 0.00182 with the
default settings on the worked example. Pixels of grey level 0 add nothing to the sum but count in
:math:`N_s` (see :ref:`ngldm-possible-issues`).


.. _ngldm-possible-issues:

Possible issues
---------------

Grey level 0 is counted but not weighted
   ``LGLCE``, ``LDLGLE`` and ``HDLGLE`` divide by :math:`g_i^2`, so they skip the row of grey level
   0. Its pixels still count in :math:`N_s`, which lowers these three features instead of leaving
   them as if those pixels were absent. Grey level 0 comes from ROI pixels of intensity 0, in every
   mode, and with the default conversion from every intensity below :math:`x_{\max}/N`. On the
   worked example with its centre pixel set to 0, IBSI mode gives ``LGLCE`` 0.2323; leaving that
   pixel out of :math:`N_s` too would give 0.2420. With intensities 20 to 100 and one pixel of
   intensity 1, which the default conversion maps to level 0, ``LGLCE`` is 0.00181 instead of
   0.00189. The other features treat level 0 as an ordinary grey level.

NGLDM converts grey levels differently from the other texture families
   NGLDM's :math:`\lfloor N x / x_{\max} \rfloor` gives :math:`N + 1` possible levels, 0 to
   :math:`N`. GLCM, GLRLM and GLSZM use :math:`\lfloor N x / x_{\max} \rfloor + 1`, capped at
   :math:`N`. With :math:`N = 64`, the two formulas give the worked example's intensities 1 to 5
   the levels 12, 25, 38, 51 and 64 for NGLDM and 13, 26, 39, 52 and 64 for GLRLM. Both count from
   intensity 0, not from the ROI minimum. The eight features that use
   :math:`g_i` (``GLM``, ``GLV``, ``LGLCE``, ``HGLCE`` and the four combined emphases) therefore
   depend on ``--coarseGrayDepth``; compare them only between runs with the same setting. The
   features that use only counts (``LDE``, ``HDE``, ``GLNU``, ``GLNUN``, ``DCNU``, ``DCNUN``,
   ``DCM``, ``DCV``, ``DCENT``, ``DCENE``) change only when the conversion merges or splits levels.

Negative or zero ``--coarseGrayDepth``
   A negative value does not select PyRadiomics-style binning for NGLDM. It is read as the unsigned
   number :math:`2^{32} - |N|`, so no two intensities share a level and the grey levels are of the
   order of :math:`10^9`. On the worked example ``--coarseGrayDepth=-64`` gives ``GLM``
   :math:`2.405 \times 10^9` and ``HGLCE`` :math:`6.847 \times 10^{18}`, while the count-only
   features are unchanged. ``--coarseGrayDepth=0`` puts every pixel at level 0, so all 25 pixels of
   the worked example share one level and the eight grey-level features return 0. The ``Nyxus``
   constructor refuses both values.

Constant ROI
   When all of a ROI's pixels have one intensity, every NGLDM feature returns the ``--noval``
   substitute (0 by default). Out of core, they return 0 whatever ``--noval`` is.

The ROI is defined by the mask, not the bounding box
   Pixels outside the mask are neither centres nor neighbours, whatever their intensity. Leaving
   three pixels of the worked example out of the mask gives ``LDE`` 0.3273 and ``HDE`` 5.818 in
   both modes.

No IBSI counterpart for two features
   ``GLM`` and ``DCM`` are not IBSI features. They are the means :math:`\mu_g` and :math:`\mu_j`
   that IBSI uses inside grey level variance and dependence count variance.


Validation
----------

The test suite checks 17 of the 19 2D NGLDM features against MIRP on the IBSI digital phantom, in
IBSI mode, with neighbourhood distance 1 and :math:`\alpha = 0`. Nyxus matches MIRP to a relative
3 × 10\ :sup:`-16`, asserted at 10\ :sup:`-9`. The four-slice means of the same 17 features are
also compared with IBSI's published consensus values at a relative 10\ :sup:`-2`, the precision
those values are published to; that comparison is a consistency check, not one of the validations
in the index. Grey level mean and dependence count mean have no
IBSI or MIRP counterpart and are covered by regression tests only, as is the default grey-level
conversion. The validation of each feature is named in the :ref:`index <ngldm-index>`.


References
----------

1. Sun C, Wee WG. *Neighboring gray level dependence matrix for texture classification.* Computer
   Vision, Graphics, and Image Processing. 1983;23(3):341–352.
2. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
3. MIRP, the medical image radiomics processor: https://github.com/oncoray/mirp
