Grey-level dependence matrix (GLDM)
===================================

GLDM features describe how many pixels of the same grey level surround each pixel of an ROI. They
count pixels by grey level and by dependence, and capture whether a texture is homogeneous (many
pixels with many same-level neighbours) or heterogeneous (few).

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``GLDM_*``: 14 features, one value per ROI
   * - **Group**
     - ``*ALL_GLDM*``; also part of ``*ALL*`` and ``*WHOLESLIDE*``
   * - **Dimensions**
     - 2D. The 3D family, ``3GLDM_*``, has its own page, :doc:`f_gldm_3d`.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory on an image that Nyxus reads as one tile. On other
       images out-of-core processing of this family raises an error or returns wrong
       values, depending on the ROI (see :ref:`large-images-tiles`).
   * - **IBSI**
     - None. IBSI defines no GLDM; the family follows PyRadiomics. IBSI's neighbouring grey level
       dependence matrix with :math:`\alpha = 0` and :math:`\delta = 1` counts the same
       dependences; see also the :doc:`NGLDM family <f_ngldm>`.

.. tip::

   To reproduce PyRadiomics values, run in IBSI mode: ``--ibsi=true`` on the command line,
   ``ibsi=True`` in Python, and use ``binWidth=1`` and ``gldm_a=0`` in PyRadiomics. The values then
   match on ROIs whose intensities start at 1; otherwise PyRadiomics counts levels from the ROI
   minimum and every feature that weights pixels by grey level differs. With the default settings
   the grey levels are binned from 0, so those features differ as well, and levels that binning
   merges change the other features too (see :ref:`gldm-possible-issues`).


How the matrix is built
-----------------------

Nyxus visits every pixel of the ROI and counts its *dependence*: the pixel itself plus each of its
8 neighbours that lies inside the ROI and has exactly the same grey level. A pixel of grey level
:math:`i` and dependence :math:`j` adds 1 to :math:`\textbf{P}(i,j)`.

- **Neighbourhood:** the 8 pixels at Chebyshev distance 1. The family has no directions.
- **Dependence:** a neighbour is dependent only when its grey level equals the centre's
  (PyRadiomics ``gldm_a = 0``). The centre pixel counts, so :math:`j` runs from 1 to 9.
- **Grey levels:** intensities are first binned into grey levels, as set by ``--coarseGrayDepth``
  (64 by default). The default binning maps intensity :math:`x` to
  :math:`\lfloor N x / x_{\max} + 1 \rfloor`, capped at :math:`N`, where :math:`N` is the number of
  levels and :math:`x_{\max}` the ROI's maximum intensity. In IBSI mode they are not binned: each
  integer intensity is a grey level.
- **Background:** pixels of intensity 0 are neither counted nor counted as neighbours, whatever the
  binning. This excludes pixels of the bounding box outside the mask, and also ROI pixels of
  intensity 0.


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

In IBSI mode (no binning), the matrix is below, with grey levels 1 to 5 as rows and dependences 1
to 4 as columns; no pixel has a dependence above 4. The 1 in row 2, column 4 of the image has two
neighbours of level 1, below and below-left, so its dependence is 3.

.. math::

   \textbf{P} = \begin{bmatrix}
   0 & 1 & 2 & 1\\
   1 & 2 & 3 & 0\\
   1 & 4 & 4 & 0\\
   1 & 2 & 0 & 0\\
   3 & 0 & 0 & 0 \end{bmatrix},
   \qquad N_z = \sum_{i,j} \textbf{P}(i,j) = 25

The *Example* value of each feature in the :ref:`index <gldm-index>` is what Nyxus returns for this
ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/gldm_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features="*ALL_GLDM*" --ibsi=true \
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
       binned levels in the bounding box.
   * - :math:`N_d`
     - Largest possible dependence, 9.
   * - :math:`i,\; j`
     - Grey level of a pixel (:math:`i`) and its dependence (:math:`j`). Formulas use the grey
       level's value: the intensity in IBSI mode, the binned level otherwise.
   * - :math:`\textbf{P}(i,j)`
     - Number of ROI pixels with grey level :math:`i` and dependence :math:`j`.
   * - :math:`N_z`
     - Number of ROI pixels with nonzero intensity, :math:`N_z = \sum_{i,j}\textbf{P}(i,j)`.
   * - :math:`p_{ij}`
     - Joint probability, :math:`p_{ij} = \textbf{P}(i,j) / N_z`.
   * - :math:`s_{i\cdot},\; s_{\cdot j}`
     - Pixels per grey level, :math:`s_{i\cdot} = \sum_j \textbf{P}(i,j)`, and pixels per
       dependence, :math:`s_{\cdot j} = \sum_i \textbf{P}(i,j)`.
   * - :math:`\mu_i,\; \mu_j`
     - Mean grey level and mean dependence over all pixels, :math:`\mu_i = \sum_{i,j} i\,p_{ij}` and
       :math:`\mu_j = \sum_{i,j} j\,p_{ij}`.
   * - :math:`\varepsilon`
     - :math:`2.2 \times 10^{-16}`, added inside the logarithm to avoid :math:`\log 0`.

Sums over :math:`i` run over the grey levels and sums over :math:`j` from 1 to :math:`N_d`. Empty
rows and columns of :math:`\textbf{P}` add nothing.

Logarithms are computed with a fast base-2 approximation (see :ref:`conventions-entropy`).
Dependence entropy therefore differs from an exact logarithm by a relative 1.1 × 10\ :sup:`-3` on
the worked example and by up to 1.3 × 10\ :sup:`-3` on the slices of the IBSI digital phantom,
which is why it is validated at a relative tolerance of 2.5 × 10\ :sup:`-3`.


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
     - Number of grey levels. Positive: equal-width bins from 0 to the ROI maximum; negative:
       PyRadiomics-style bins between the ROI minimum and maximum (command line only).
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus(..., ibsi=True)``
     - No binning: each integer intensity is a grey level.

The neighbourhood, the distance and :math:`\alpha` are fixed. No ``set_metaparam`` parameter
changes the 2D matrix; ``3gldm/greydepth`` applies to the 3D family only. All options are listed in
the :ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named by its code, for example ``GLDM_SDE``. There are no
per-direction columns and no ``_AVE`` columns.


.. _gldm-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with the relative tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example.

.. include:: gldm_index.inc


Features
--------

.. _feature-gldm-de:

Dependence entropy
^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_DE``
:Range: :math:`[0, \log_2(N_g N_d)]` bits

Measures the randomness of the joint distribution of grey level and dependence.

.. math::

   \mathrm{DE} = -\sum_{i}\sum_{j=1}^{N_d} p_{ij}\,\log_2\!\big(p_{ij}+\varepsilon\big)

**Possible issues.** The logarithm is approximate: on the worked example Nyxus returns 3.3797,
while an exact logarithm, and PyRadiomics, give 3.3835.


.. _feature-gldm-dn:

Dependence non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_DN``
:Range: :math:`[1, N_z]`, pixels

Measures how evenly the pixels are spread over the dependences. Low values mean the dependences
are spread evenly.

.. math::

   \mathrm{DN} = \frac{1}{N_z}\sum_{j=1}^{N_d} s_{\cdot j}^{\,2}


.. _feature-gldm-dnn:

Dependence non-uniformity normalized
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_DNN``
:Range: :math:`(0, 1]`, dimensionless

Dependence non-uniformity divided by the number of pixels once more, so values are comparable
across ROIs of different size.

.. math::

   \mathrm{DNN} = \frac{1}{N_z^{\,2}}\sum_{j=1}^{N_d} s_{\cdot j}^{\,2}


.. _feature-gldm-dv:

Dependence variance
^^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_DV``
:Range: :math:`[0, 16]`, squared dependences

The variance of the dependence over all pixels.

.. math::

   \mathrm{DV} = \sum_{i}\sum_{j=1}^{N_d} (j-\mu_j)^2\, p_{ij}


.. _feature-gldm-gln:

Grey level non-uniformity
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_GLN``
:Range: :math:`[1, N_z]`, pixels

Measures how evenly the pixels are spread over the grey levels. Low values mean the grey levels are
evenly represented.

.. math::

   \mathrm{GLN} = \frac{1}{N_z}\sum_{i} s_{i\cdot}^{\,2}


.. _feature-gldm-glv:

Grey level variance
^^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_GLV``
:Range: :math:`[0, \infty)`, squared grey levels

The variance of the grey level over all pixels.

.. math::

   \mathrm{GLV} = \sum_{i}\sum_{j=1}^{N_d} (i-\mu_i)^2\, p_{ij}

**Possible issues.** Grey level variance scales with the square of the grey-level spacing. On the
worked example it is 1.44 in IBSI mode and 236.6 with the default 64-level binning, which maps
the levels 1 to 5 to 13, 26, 39, 52 and 64.


.. _feature-gldm-hgle:

High grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_HGLE``
:Range: :math:`[1, \infty)`, squared grey levels

The mean over all pixels of the squared grey level. High values mean bright pixels dominate.

.. math::

   \mathrm{HGLE} = \frac{1}{N_z}\sum_{i} i^2\, s_{i\cdot}

**Possible issues.** Depends on the binning: 9.28 in IBSI mode and 1552.8 with the default
settings on the worked example.


.. _feature-gldm-lde:

Large dependence emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_LDE``
:Range: :math:`[1, 81]`, squared dependences

The mean over all pixels of the squared dependence. High values mean pixels with many same-level
neighbours dominate, as in a homogeneous texture.

.. math::

   \mathrm{LDE} = \frac{1}{N_z}\sum_{i}\sum_{j=1}^{N_d} j^2\,\textbf{P}(i,j)


.. _feature-gldm-ldhgle:

Large dependence high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_LDHGLE``
:Range: :math:`[1, \infty)`, squared grey levels times squared dependences

The mean over all pixels of the squared dependence times the squared grey level. High values mean
bright pixels with many same-level neighbours dominate.

.. math::

   \mathrm{LDHGLE} = \frac{1}{N_z}\sum_{i}\sum_{j=1}^{N_d} i^2 j^2\,\textbf{P}(i,j)


.. _feature-gldm-ldlgle:

Large dependence low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_LDLGLE``
:Range: :math:`(0, 81]`, squared dependences over squared grey levels

The mean over all pixels of the squared dependence over the squared grey level. High values mean
dark pixels with many same-level neighbours dominate.

.. math::

   \mathrm{LDLGLE} = \frac{1}{N_z}\sum_{i}\sum_{j=1}^{N_d} \frac{j^2\,\textbf{P}(i,j)}{i^2}


.. _feature-gldm-lgle:

Low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_LGLE``
:Range: :math:`(0, 1]`, inverse squared grey levels

The mean over all pixels of one over the squared grey level. High values mean dark pixels
dominate.

.. math::

   \mathrm{LGLE} = \frac{1}{N_z}\sum_{i} \frac{s_{i\cdot}}{i^2}

**Possible issues.** Depends on the binning: 0.2723 in IBSI mode and 0.001612 with the default
settings on the worked example.


.. _feature-gldm-sde:

Small dependence emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_SDE``
:Range: :math:`[1/81, 1]`, inverse squared dependences

The mean over all pixels of one over the squared dependence. High values mean pixels with few
same-level neighbours dominate, as in a heterogeneous texture.

.. math::

   \mathrm{SDE} = \frac{1}{N_z}\sum_{j=1}^{N_d} \frac{s_{\cdot j}}{j^2}


.. _feature-gldm-sdhgle:

Small dependence high grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_SDHGLE``
:Range: :math:`(0, \infty)`, squared grey levels over squared dependences

The mean over all pixels of the squared grey level over the squared dependence. High values mean
bright pixels with few same-level neighbours dominate.

.. math::

   \mathrm{SDHGLE} = \frac{1}{N_z}\sum_{i}\sum_{j=1}^{N_d} \frac{i^2\,\textbf{P}(i,j)}{j^2}


.. _feature-gldm-sdlgle:

Small dependence low grey level emphasis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``GLDM_SDLGLE``
:Range: :math:`(0, 1]`, inverse squared grey levels times inverse squared dependences

The mean over all pixels of one over the squared dependence times the squared grey level. High
values mean dark pixels with few same-level neighbours dominate.

.. math::

   \mathrm{SDLGLE} = \frac{1}{N_z}\sum_{i}\sum_{j=1}^{N_d} \frac{\textbf{P}(i,j)}{i^2 j^2}


.. _gldm-possible-issues:

Possible issues
---------------

Grey-level values depend on the binning
   ``GLV`` and the six emphasis features that weight pixels by grey level use the grey level's
   value. The default binning counts levels from intensity 0, not from the ROI minimum, so a ROI
   whose intensities span a narrow range uses only the top levels, and neighbouring intensities
   can merge into one level. Adding 1000 to the worked example (intensities 1001 to 1005) puts
   every pixel in level 64 with the default settings: small dependence emphasis drops from 0.3725
   to 0.02778, large dependence emphasis rises from 5.56 to 49 and grey level non-uniformity is 25,
   the values PyRadiomics gives for a flat ROI. Compare these features only between runs with the same
   ``--coarseGrayDepth``. The features that use only dependences and counts (``SDE``, ``LDE``,
   ``GLN``, ``DN``, ``DNN``, ``DV``, ``DE``) change only when binning merges or splits levels.

Pixels of intensity 0 inside the mask
   A ROI pixel of intensity 0 is treated as background in every mode: it is not counted and does
   not count as a neighbour. PyRadiomics keeps it as a grey level. On the worked example with the
   centre pixel set to 0, Nyxus counts 24 pixels and returns small dependence emphasis 0.4282 and
   large dependence emphasis 4.583, while PyRadiomics returns 0.4511 and 4.44.

Pixels outside the mask
   Pixels of the bounding box outside the mask are excluded in every mode, including the default
   binning, which maps intensity 0 to level 1. On the worked example with three pixels of the
   bounding box left out of the mask, the default settings and IBSI mode give the same matrix:
   small dependence emphasis is 0.3273 and large dependence emphasis 5.818 in both.

Constant ROI
   When all of a ROI's pixels have one intensity, every GLDM feature returns the ``--noval``
   substitute. PyRadiomics returns values for such a ROI; on a constant 5 × 5 ROI large dependence
   emphasis is 49. A ROI whose intensities differ but fall into one binned level is computed, as in
   the narrow-range example above.


Validation
----------

The test suite checks every 2D GLDM feature against PyRadiomics, named in the
:ref:`index <gldm-index>`. In IBSI mode Nyxus matches PyRadiomics (identity binning,
``gldm_a = 0``, distance 1) on each slice of the IBSI digital phantom and on their mean to a
relative 10\ :sup:`-9`; dependence entropy agrees to 2.5 × 10\ :sup:`-3` because of the approximate
logarithm. The four-slice means are also checked against IBSI's published consensus values for the
neighbouring grey level dependence matrix, at a relative 10\ :sup:`-2`, the precision those values
are published to. The default binning is covered by regression tests only.


References
----------

1. Sun C, Wee WG. *Neighboring gray level dependence matrix for texture classification.* Computer
   Vision, Graphics, and Image Processing. 1983;23(3):341–352.
2. van Griethuysen JJM, et al. *Computational radiomics system to decode the radiographic
   phenotype.* Cancer Research. 2017;77(21):e104–e107.
3. Zwanenburg A, et al. *The Image Biomarker Standardization Initiative: standardized quantitative
   radiomics for high-throughput image-based phenotyping.* Radiology. 2020;295(2):328–338.
   Feature definitions: https://ibsi.readthedocs.io
