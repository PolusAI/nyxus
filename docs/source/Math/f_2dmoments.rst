2D geometric moments
====================

Geometric moments sum powers of the pixel coordinates over a ROI. Nyxus computes them twice: the
*shape moments* count every ROI pixel as 1, and the *intensity moments* weight each pixel by its
intensity. Each family has raw, central and normalized moments, Hu's seven invariants, and the same
set again with every pixel also weighted by its distance to the ROI contour.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - Shape moments: ``SPAT_MOMENT_*``, ``CENTRAL_MOMENT_*``, ``NORM_SPAT_MOMENT_*``,
       ``NORM_CENTRAL_MOMENT_*``, ``HU_M1`` to ``HU_M7``, ``WEIGHTED_SPAT_MOMENT_*``,
       ``WEIGHTED_CENTRAL_MOMENT_*``, ``WT_NORM_CTR_MOM_*`` and ``WEIGHTED_HU_M1`` to
       ``WEIGHTED_HU_M7``: 90 features. Intensity moments: ``IMOM_RM_*``, ``IMOM_CM_*``,
       ``IMOM_NRM_*``, ``IMOM_NCM_*``, ``IMOM_HU1`` to ``IMOM_HU7``, ``IMOM_WRM_*``,
       ``IMOM_WCM_*``, ``IMOM_WNCM_*`` and ``IMOM_WHU1`` to ``IMOM_WHU7``: 90 features. One value
       per ROI each.
   * - **Group**
     - ``*GEOMOMS*`` for both families, ``*SGEOMOMS*`` for the shape moments, ``*IGEOMOMS*`` for
       the intensity moments. Both families are also in ``*ALL*``, and the intensity moments in
       ``*WHOLESLIDE*``.
   * - **Dimensions**
     - 2D only.
   * - **GPU**
     - Yes, in GPU builds with ``--useGpu=true`` (Python ``use_gpu_device``), for ROIs computed
       in memory with a label image read from files; out-of-core ROIs and whole slides use the
       CPU, and ``featurize`` on arrays must not be combined with the GPU (see :doc:`/gpu`). The
       GPU values differ from the CPU values: the GPU takes the central moments about the centroid
       rounded down to whole pixels, so the central, normalized central and Hu moments differ, and
       with anisotropy the run can crash. A fix is coming. The GPU's distance search computes
       the step with a single-precision logarithm, and it has no guard for a ROI without a traced
       contour, such as a single pixel.
   * - **Out-of-core**
     - Partly. Without anisotropy: on an image that Nyxus reads as one tile, the 118 unweighted
       moments have the same values as in memory. On an image read as several tiles, out-of-core
       processing of this family raises an error or returns wrong values, depending on the ROI (see
       :ref:`large-images-tiles`). The 62 weighted moments
       differ unless the ROI's bounding box starts at pixel (0, 0), and can differ even then (see
       :ref:`moments-possible-issues` and :doc:`/large_images`). With anisotropy, on an image
       that Nyxus reads as one tile, the weighted intensity moments get their in-memory values and
       only weighted shape moments differ.
   * - **IBSI**
     - None. IBSI defines no geometric moment features.

.. tip::

   The code digits are the powers of :math:`x` and :math:`y`, with :math:`x` the column and
   :math:`y` the row. OpenCV's ``cv2.moments`` uses the same convention, so ``m10`` is
   ``SPAT_MOMENT_10``. scikit-image's ``moments(image)[i, j]`` puts the row first: Nyxus's
   :math:`m_{pq}` is ``moments(image)[q, p]``, or ``moments(image.T)[p, q]``.


How the features are computed
-----------------------------

- **Pixels:** the ROI's pixels, those of the mask with the ROI's label. The other pixels of the
  bounding box take no part.
- **Coordinates:** :math:`x_i` is the column and :math:`y_i` the row of pixel :math:`i`, both
  integers measured from the top-left corner of the ROI's bounding box. A pixel's coordinates are
  those of its centre. Moving the ROI inside the image does not change any value computed in
  memory; out of core the weighted moments depend on the ROI's position (see
  :ref:`moments-possible-issues`). With anisotropy the positions change (see `Settings`_).
- **Values:** the shape moments use :math:`I_i = 1` for every pixel. The intensity moments use the
  pixel's intensity as a grey level, the integer Nyxus holds after loading the image. They are not
  converted back to the image's units (see *Intensity units and CT images* in
  :doc:`/supported_formats`). A ROI pixel of intensity 0 adds nothing to an intensity moment and 1
  to a shape moment.
- **Binning:** none; ``--coarseGrayDepth`` and ``--ibsi`` do not apply.
- **Weighted moments:** each pixel gets the weight :math:`w_i = I_i \ln(d_i + 0.001)`, where
  :math:`d_i` is its distance to the ROI contour. The weight is stored in single precision, then
  summed in double precision. It is negative for :math:`d_i < 0.999`, about 0.001 for
  :math:`d_i = 1` and positive beyond. A pixel at :math:`d_i = 0` gets :math:`I_i \ln 0.001 =
  -6.908\,I_i`.
- **Missing values:** a result that is not a finite number, such as a division by
  :math:`m_{00} = 0`, is output as the ``--noval`` substitute, 0 by default (``-0.0`` in Arrow
  and Parquet files; see :ref:`conventions-missing`).

How the distance to the contour is found
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Contour.** Nyxus traces the border of the ROI in a copy of its bounding box padded by one
   pixel on every side, and concatenates every traced contour into one list. The traced
   coordinates keep the padding offset: each contour pixel lies one pixel to the right of and one
   pixel below the ROI border pixel it stands for. The same contour gives ``PERIMETER``, which the
   offset does not change.
2. **Search.** :math:`d_i^2` is the smallest squared distance from pixel :math:`i` to the contour
   pixels that an approximate search visits. With :math:`n` contour pixels, it first visits every
   :math:`s`-th pixel of the list, :math:`s = \lfloor n / \ln n \rfloor`, starting at the first.
   Around the nearest one it searches a window of :math:`\pm s` list positions, then narrower
   windows (step :math:`\lfloor w / \ln w \rfloor` for a window of :math:`w` positions, 1 once
   :math:`w \le 10`), until the window is 2 positions wide. The window never wraps around the
   ends of the list, so the true nearest contour pixel can be missed, and :math:`d_i` is then too
   large. It is never too small.
3. **Weight.** :math:`d_i` is the square root, and the weight is :math:`I_i \ln(d_i + 0.001)`.

A ROI without a traced contour, such as a single pixel, has :math:`d_i = 0` for every pixel.

In whole-slide mode (``--segDir`` equal to ``--intDir``, or ``single_roi=True`` in Python) the
contour is the four corners of the image, at :math:`(0, 0)`, :math:`(W, 0)`, :math:`(W, H)` and
:math:`(0, H)` for an image of :math:`W \times H` pixels. A pixel in row 0 or column 0 gets the
weight 0.001, whatever its intensity. Every other pixel gets :math:`I_i \ln(d_i + 0.001)` with
:math:`d_i = \min(x_i, y_i, W - x_i, H - y_i)`, the distance to the nearest edge line.


Worked example
^^^^^^^^^^^^^^

Take a ROI of 40 pixels in a 7 × 8 bounding box; the mask is the pixels with a non-zero intensity:

.. math::

   \textbf{I} = \begin{bmatrix}
   \cdot & \cdot & 6 & 2 & 5 & \cdot & \cdot & \cdot\\
   \cdot & 3 & 7 & 4 & 1 & 5 & 2 & \cdot\\
   4 & 1 & 2 & 6 & 3 & 7 & 4 & 1\\
   5 & 7 & 3 & 1 & 6 & 2 & 5 & 3\\
   2 & 4 & 6 & 5 & 2 & 4 & 1 & \cdot\\
   \cdot & 6 & 1 & 3 & 7 & 5 & \cdot & \cdot\\
   \cdot & \cdot & 3 & 2 & 4 & \cdot & \cdot & \cdot \end{bmatrix}

The top-left entry is :math:`x = 0, y = 0`. The shape moments give :math:`m_{00} = 40`,
:math:`m_{10} = 131` and :math:`m_{01} = 117`, so the centroid is :math:`\bar{x} = 3.275`,
:math:`\bar{y} = 2.925`. The intensity moments give :math:`m_{00} = 150`, the sum of the
intensities.

The traced contour has 17 pixels, each one right of and one below a border pixel of the ROI. The
distances :math:`d_i` that Nyxus uses are:

.. math::

   \textbf{d} = \begin{bmatrix}
   \cdot & \cdot & 1.414 & 1 & 1 & \cdot & \cdot & \cdot\\
   \cdot & 2 & 1 & 0 & 0 & 0 & 1 & \cdot\\
   3.162 & 2.236 & 1.414 & 1 & 1 & 1 & 0 & 0\\
   3.606 & 2.828 & 2.236 & 2 & 2 & 1.414 & 1 & 1\\
   1 & 0 & 1 & 2 & 2.828 & 2.236 & 1.414 & \cdot\\
   \cdot & 0 & 1 & 1.414 & 2 & 1.414 & \cdot & \cdot\\
   \cdot & \cdot & 0 & 1 & 1 & \cdot & \cdot & \cdot \end{bmatrix}

The left-hand pixels of rows 2 and 3 are border pixels, but get :math:`d = 3.162` and
:math:`d = 3.606`. For :math:`(x, y) = (0, 3)` the search visits list positions 0, 6 and 12
(:math:`s = \lfloor 17 / \ln 17 \rfloor = 6`), keeps position 0 at :math:`d^2 = 13`, and
searches positions 0 to 5 around it. The traced pixel at distance 1, position 15, lies outside
that window. Eight of the 40 pixels get a larger :math:`d` than their nearest traced pixel, by up
to 2.83 pixels.

The weights sum to :math:`m^w_{00} = -42.77` for the shape and :math:`-144.7` for the intensity.
Both are negative, so every weighted normalized moment of odd order, and weighted Hu invariants 3
to 7, are not real numbers and are output as 0 (see :ref:`moments-possible-issues`).

The *Example* value of each feature in the :ref:`index <moments-index>` is what Nyxus returns for
this ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/moments_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features="*GEOMOMS*" \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`x_i, y_i`
     - Column and row of ROI pixel :math:`i`, from the top-left corner of the bounding box.
   * - :math:`I_i`
     - 1 for the shape moments; the pixel's grey level for the intensity moments.
   * - :math:`m_{pq}`
     - Raw moment, :math:`\sum_i I_i\, x_i^p\, y_i^q`, summed over the ROI's pixels.
   * - :math:`\bar{x}, \bar{y}`
     - Centroid, :math:`\bar{x} = m_{10} / m_{00}` and :math:`\bar{y} = m_{01} / m_{00}`.
   * - :math:`\mu_{pq}`
     - Central moment, :math:`\sum_i I_i\, (x_i - \bar{x})^p\, (y_i - \bar{y})^q`.
   * - :math:`\eta_{pq}`
     - Normalized central moment, :math:`\mu_{pq} / m_{00}^{(p+q)/2+1}`.
   * - :math:`d_i`
     - Distance from pixel :math:`i` to the traced contour, as found by the search above.
   * - :math:`w_i`
     - Weight of pixel :math:`i` in the weighted moments, :math:`I_i \ln(d_i + 0.001)`.
   * - :math:`m^w_{pq}, \mu^w_{pq}, \eta^w_{pq}`
     - The same moments with :math:`w_i` in place of :math:`I_i`, about the weighted centroid
       :math:`\bar{x}_w = m^w_{10} / m^w_{00}`, :math:`\bar{y}_w = m^w_{01} / m^w_{00}`.

In a code, the first digit is :math:`p`, the power of :math:`x`, and the second is :math:`q`, the
power of :math:`y`.


Settings
--------

.. list-table::
   :widths: 16 28 26 30
   :header-rows: 1

   * - Setting
     - Command line
     - Python
     - Effect
   * - Whole-slide mode
     - ``--segDir`` equal to ``--intDir``
     - ``featurize_files(..., single_roi=True)``
     - The image is one ROI, and the weighted moments measure distance to the image's edges (see
       above).
   * - Missing value
     - ``--noval=0``
     - not available
     - Replaces results that are not finite numbers. Python output uses 0; Arrow and Parquet
       files hold ``-0.0``.
   * - Memory limit
     - ``--ramLimit``
     - ``Nyxus(..., ram_limit=...)``
     - ROIs whose footprint reaches the limit are computed out of core, which changes the weighted
       moments; on an image read as several tiles it raises an error or changes the other moments
       too.
   * - GPU
     - ``--useGpu=true``
     - ``Nyxus(..., use_gpu_device=0)``
     - Computes the moments of in-memory ROIs on the GPU, in GPU builds (see :doc:`/gpu`).
   * - Anisotropy
     - ``--anisox``, ``--anisoy``
     - ``Nyxus(..., anisotropy_x=..., anisotropy_y=...)``
     - Files only. The shape moments are computed on the ROI resampled by the factors, in pixels of
       the resampled grid. The intensity moments are computed on the pixels as acquired, each at its
       position scaled by the factors and weighted by its area: with ``anisotropy_x=2``
       ``SPAT_MOMENT_00`` and ``IMOM_RM_00`` both double. Arrays passed to ``featurize`` are not
       resampled.

The moments have no metaparameters. ``--coarseGrayDepth``, ``--ibsi`` and ``--pixelDistance`` do
not change them. All options are listed in the :ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named with its code: ``SPAT_MOMENT_00``, ``IMOM_HU1`` and so
on. There are no per-direction columns and no ``_AVE`` columns.


.. _moments-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example.

**Shape moments**

.. include:: shape_moments_index.inc

**Intensity moments**

.. include:: intensity_moments_index.inc


Features
--------

Each section below covers one kind of moment for both families. The shape code uses
:math:`I_i = 1`; the intensity code uses the pixel's grey level.

.. _feature-spat-moment-00:
.. _feature-spat-moment-01:
.. _feature-spat-moment-02:
.. _feature-spat-moment-03:
.. _feature-spat-moment-10:
.. _feature-spat-moment-11:
.. _feature-spat-moment-12:
.. _feature-spat-moment-13:
.. _feature-spat-moment-20:
.. _feature-spat-moment-21:
.. _feature-spat-moment-22:
.. _feature-spat-moment-23:
.. _feature-spat-moment-30:
.. _feature-imom-rm-00:
.. _feature-imom-rm-01:
.. _feature-imom-rm-02:
.. _feature-imom-rm-03:
.. _feature-imom-rm-10:
.. _feature-imom-rm-11:
.. _feature-imom-rm-12:
.. _feature-imom-rm-13:
.. _feature-imom-rm-20:
.. _feature-imom-rm-21:
.. _feature-imom-rm-22:
.. _feature-imom-rm-23:
.. _feature-imom-rm-30:

Raw moments
^^^^^^^^^^^

:Codes: ``SPAT_MOMENT_pq`` and ``IMOM_RM_pq``, for :math:`pq` = 00, 01, 02, 03, 10, 11, 12, 13,
        20, 21, 22, 23 and 30
:Range: :math:`[0, \infty)`

Sums of the coordinate powers over the ROI. :math:`m_{00}` is the ROI's area in pixels for the
shape, and the sum of its intensities for the intensity moments. :math:`m_{10}/m_{00}` and
:math:`m_{01}/m_{00}` are the centroid.

.. math::

   m_{pq} = \sum_i I_i\, x_i^p\, y_i^q

Because :math:`x_i` and :math:`y_i` are measured from the bounding-box corner, every raw moment is
non-negative.


.. _feature-central-moment-00:
.. _feature-central-moment-01:
.. _feature-central-moment-02:
.. _feature-central-moment-03:
.. _feature-central-moment-10:
.. _feature-central-moment-11:
.. _feature-central-moment-12:
.. _feature-central-moment-13:
.. _feature-central-moment-20:
.. _feature-central-moment-21:
.. _feature-central-moment-22:
.. _feature-central-moment-23:
.. _feature-central-moment-30:
.. _feature-central-moment-31:
.. _feature-central-moment-32:
.. _feature-central-moment-33:
.. _feature-imom-cm-00:
.. _feature-imom-cm-01:
.. _feature-imom-cm-02:
.. _feature-imom-cm-03:
.. _feature-imom-cm-10:
.. _feature-imom-cm-11:
.. _feature-imom-cm-12:
.. _feature-imom-cm-13:
.. _feature-imom-cm-20:
.. _feature-imom-cm-21:
.. _feature-imom-cm-22:
.. _feature-imom-cm-23:
.. _feature-imom-cm-30:
.. _feature-imom-cm-31:
.. _feature-imom-cm-32:
.. _feature-imom-cm-33:

Central moments
^^^^^^^^^^^^^^^

:Codes: ``CENTRAL_MOMENT_pq`` and ``IMOM_CM_pq``, for :math:`p, q` from 0 to 3 (16 codes each)
:Range: :math:`(-\infty, \infty)`; non-negative when :math:`p` and :math:`q` are both even

The raw moments taken about the centroid, so they do not depend on where the ROI lies. The
second-order moments describe the spread, the third-order ones the skew.

.. math::

   \mu_{pq} = \sum_i I_i\, (x_i - \bar{x})^p\, (y_i - \bar{y})^q,
   \qquad \bar{x} = \frac{m_{10}}{m_{00}},\quad \bar{y} = \frac{m_{01}}{m_{00}}

:math:`\mu_{00} = m_{00}`. :math:`\mu_{01}` and :math:`\mu_{10}` are 0 by definition; Nyxus returns
rounding residues of order :math:`10^{-14}` for them.


.. _feature-norm-spat-moment-00:
.. _feature-norm-spat-moment-01:
.. _feature-norm-spat-moment-02:
.. _feature-norm-spat-moment-03:
.. _feature-norm-spat-moment-10:
.. _feature-norm-spat-moment-11:
.. _feature-norm-spat-moment-12:
.. _feature-norm-spat-moment-13:
.. _feature-norm-spat-moment-20:
.. _feature-norm-spat-moment-21:
.. _feature-norm-spat-moment-22:
.. _feature-norm-spat-moment-23:
.. _feature-norm-spat-moment-30:
.. _feature-norm-spat-moment-31:
.. _feature-norm-spat-moment-32:
.. _feature-norm-spat-moment-33:
.. _feature-imom-nrm-00:
.. _feature-imom-nrm-01:
.. _feature-imom-nrm-02:
.. _feature-imom-nrm-03:
.. _feature-imom-nrm-10:
.. _feature-imom-nrm-11:
.. _feature-imom-nrm-12:
.. _feature-imom-nrm-13:
.. _feature-imom-nrm-20:
.. _feature-imom-nrm-21:
.. _feature-imom-nrm-22:
.. _feature-imom-nrm-23:
.. _feature-imom-nrm-30:
.. _feature-imom-nrm-31:
.. _feature-imom-nrm-32:
.. _feature-imom-nrm-33:

Normalized raw moments
^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``NORM_SPAT_MOMENT_pq`` and ``IMOM_NRM_pq``, for :math:`p, q` from 0 to 3 (16 codes each)
:Range: :math:`[0, \infty)`

Each raw moment divided by :math:`m_{00}` to the power used for the normalized central moments.

.. math::

   \frac{m_{pq}}{m_{00}^{(p+q)/2+1}}

This applies the exponent of :math:`\eta_{pq}` to the *raw* moments, which are measured from the
bounding-box corner. It is neither :math:`\eta_{pq}` nor the moment per unit mass
:math:`m_{pq}/m_{00}`: on the worked example ``NORM_SPAT_MOMENT_10`` is
:math:`131 / 40^{1.5} = \bar{x} / \sqrt{40} = 0.5178`. ``NORM_SPAT_MOMENT_00`` is always 1.
The codes 31, 32 and 33 are computed from :math:`m_{31}`, :math:`m_{32}` and :math:`m_{33}`, which
have no code of their own.


.. _feature-norm-central-moment-02:
.. _feature-norm-central-moment-03:
.. _feature-norm-central-moment-11:
.. _feature-norm-central-moment-12:
.. _feature-norm-central-moment-20:
.. _feature-norm-central-moment-21:
.. _feature-norm-central-moment-30:
.. _feature-imom-ncm-02:
.. _feature-imom-ncm-03:
.. _feature-imom-ncm-11:
.. _feature-imom-ncm-12:
.. _feature-imom-ncm-20:
.. _feature-imom-ncm-21:
.. _feature-imom-ncm-30:

Normalized central moments
^^^^^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``NORM_CENTRAL_MOMENT_pq`` and ``IMOM_NCM_pq``, for :math:`pq` = 02, 03, 11, 12, 20, 21
        and 30
:Range: :math:`(-\infty, \infty)`; non-negative for 02 and 20

The central moments made independent of size. For the shape, scaling the ROI leaves them nearly
unchanged; for the intensity, multiplying every intensity by :math:`c` multiplies them by
:math:`c^{-(p+q)/2}`.

.. math::

   \eta_{pq} = \frac{\mu_{pq}}{m_{00}^{(p+q)/2+1}}

This is the normalization of Hu, of scikit-image's ``moments_normalized`` and of OpenCV's
``nu`` moments.


.. _feature-hu-m1:
.. _feature-hu-m2:
.. _feature-hu-m3:
.. _feature-hu-m4:
.. _feature-hu-m5:
.. _feature-hu-m6:
.. _feature-hu-m7:
.. _feature-imom-hu1:
.. _feature-imom-hu2:
.. _feature-imom-hu3:
.. _feature-imom-hu4:
.. _feature-imom-hu5:
.. _feature-imom-hu6:
.. _feature-imom-hu7:

Hu invariants
^^^^^^^^^^^^^

:Codes: ``HU_M1`` to ``HU_M7`` and ``IMOM_HU1`` to ``IMOM_HU7``
:Range: :math:`I_1, I_2, I_3, I_4 \ge 0`; :math:`I_5, I_6, I_7` of either sign

Hu's seven combinations of the normalized central moments, unchanged by translation and rotation.
:math:`I_7` changes sign when the ROI is mirrored.

.. math::

   I_1 &= \eta_{20} + \eta_{02}\\
   I_2 &= (\eta_{20} - \eta_{02})^2 + 4\eta_{11}^2\\
   I_3 &= (\eta_{30} - 3\eta_{12})^2 + (3\eta_{21} - \eta_{03})^2\\
   I_4 &= (\eta_{30} + \eta_{12})^2 + (\eta_{21} + \eta_{03})^2\\
   I_5 &= (\eta_{30} - 3\eta_{12})(\eta_{30} + \eta_{12})
          \left[(\eta_{30} + \eta_{12})^2 - 3(\eta_{21} + \eta_{03})^2\right]\\
       &\quad + (3\eta_{21} - \eta_{03})(\eta_{21} + \eta_{03})
          \left[3(\eta_{30} + \eta_{12})^2 - (\eta_{21} + \eta_{03})^2\right]\\
   I_6 &= (\eta_{20} - \eta_{02})\left[(\eta_{30} + \eta_{12})^2 - (\eta_{21} + \eta_{03})^2\right]
          + 4\eta_{11}(\eta_{30} + \eta_{12})(\eta_{21} + \eta_{03})\\
   I_7 &= (3\eta_{21} - \eta_{03})(\eta_{30} + \eta_{12})
          \left[(\eta_{30} + \eta_{12})^2 - 3(\eta_{21} + \eta_{03})^2\right]\\
       &\quad - (\eta_{30} - 3\eta_{12})(\eta_{21} + \eta_{03})
          \left[3(\eta_{30} + \eta_{12})^2 - (\eta_{21} + \eta_{03})^2\right]

``HU_Mk`` and ``IMOM_HUk`` are :math:`I_k`. The values equal OpenCV's ``cv2.HuMoments``, and
scikit-image's ``moments_hu`` computed on ``image.T``; on ``image`` itself scikit-image's
:math:`I_7` has the opposite sign, because it puts the row first.


.. _feature-weighted-spat-moment-00:
.. _feature-weighted-spat-moment-01:
.. _feature-weighted-spat-moment-02:
.. _feature-weighted-spat-moment-03:
.. _feature-weighted-spat-moment-10:
.. _feature-weighted-spat-moment-11:
.. _feature-weighted-spat-moment-12:
.. _feature-weighted-spat-moment-20:
.. _feature-weighted-spat-moment-21:
.. _feature-weighted-spat-moment-30:
.. _feature-imom-wrm-00:
.. _feature-imom-wrm-01:
.. _feature-imom-wrm-02:
.. _feature-imom-wrm-03:
.. _feature-imom-wrm-10:
.. _feature-imom-wrm-11:
.. _feature-imom-wrm-12:
.. _feature-imom-wrm-20:
.. _feature-imom-wrm-21:
.. _feature-imom-wrm-30:

Weighted raw moments
^^^^^^^^^^^^^^^^^^^^

:Codes: ``WEIGHTED_SPAT_MOMENT_pq`` and ``IMOM_WRM_pq``, for :math:`pq` = 00, 01, 02, 03, 10, 11,
        12, 20, 21 and 30
:Range: :math:`(-\infty, \infty)`

The raw moments with each pixel weighted by :math:`w_i = I_i \ln(d_i + 0.001)`, where :math:`d_i`
is its distance to the traced contour (see `How the distance to the contour is found`_). Pixels
on the traced contour count negatively (:math:`-6.9\,I_i`), pixels one step from it about 0, and
pixels deeper inside positively.

.. math::

   m^w_{pq} = \sum_i w_i\, x_i^p\, y_i^q, \qquad w_i = I_i \ln(d_i + 0.001)

**Possible issues.** :math:`d_i` comes from a contour shifted by one pixel and from an approximate
search, so these values differ from the same formula with the exact distance to the ROI border
(see :ref:`moments-possible-issues`).


.. _feature-weighted-central-moment-02:
.. _feature-weighted-central-moment-03:
.. _feature-weighted-central-moment-11:
.. _feature-weighted-central-moment-12:
.. _feature-weighted-central-moment-20:
.. _feature-weighted-central-moment-21:
.. _feature-weighted-central-moment-30:
.. _feature-imom-wcm-02:
.. _feature-imom-wcm-03:
.. _feature-imom-wcm-11:
.. _feature-imom-wcm-12:
.. _feature-imom-wcm-20:
.. _feature-imom-wcm-21:
.. _feature-imom-wcm-30:

Weighted central moments
^^^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``WEIGHTED_CENTRAL_MOMENT_pq`` and ``IMOM_WCM_pq``, for :math:`pq` = 02, 03, 11, 12, 20,
        21 and 30
:Range: :math:`(-\infty, \infty)`

The weighted moments taken about the weighted centroid.

.. math::

   \mu^w_{pq} = \sum_i w_i\, (x_i - \bar{x}_w)^p\, (y_i - \bar{y}_w)^q,
   \qquad \bar{x}_w = \frac{m^w_{10}}{m^w_{00}},\quad \bar{y}_w = \frac{m^w_{01}}{m^w_{00}}

When :math:`m^w_{00}` is close to 0 the weighted centroid can lie far outside the ROI, and these
moments become very large.


.. _feature-wt-norm-ctr-mom-02:
.. _feature-wt-norm-ctr-mom-03:
.. _feature-wt-norm-ctr-mom-11:
.. _feature-wt-norm-ctr-mom-12:
.. _feature-wt-norm-ctr-mom-20:
.. _feature-wt-norm-ctr-mom-21:
.. _feature-wt-norm-ctr-mom-30:
.. _feature-imom-wncm-02:
.. _feature-imom-wncm-03:
.. _feature-imom-wncm-11:
.. _feature-imom-wncm-12:
.. _feature-imom-wncm-20:
.. _feature-imom-wncm-21:
.. _feature-imom-wncm-30:

Weighted normalized central moments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``WT_NORM_CTR_MOM_pq`` and ``IMOM_WNCM_pq``, for :math:`pq` = 02, 03, 11, 12, 20, 21 and
        30
:Range: :math:`(-\infty, \infty)`

The weighted central moments normalized by the sum of the weights.

.. math::

   \eta^w_{pq} = \frac{\mu^w_{pq}}{\left(m^w_{00}\right)^{(p+q)/2+1}}

**Possible issues.** When :math:`m^w_{00} < 0` the odd orders (03, 12, 21, 30) raise a negative
number to a half-integer power. The result is not a real number and is output as the ``--noval``
substitute, 0 by default. The second-order ones (02, 11, 20) are always computed: their
denominator :math:`(m^w_{00})^2` is positive, so they take the sign of :math:`\mu^w_{pq}`.


.. _feature-weighted-hu-m1:
.. _feature-weighted-hu-m2:
.. _feature-weighted-hu-m3:
.. _feature-weighted-hu-m4:
.. _feature-weighted-hu-m5:
.. _feature-weighted-hu-m6:
.. _feature-weighted-hu-m7:
.. _feature-imom-whu1:
.. _feature-imom-whu2:
.. _feature-imom-whu3:
.. _feature-imom-whu4:
.. _feature-imom-whu5:
.. _feature-imom-whu6:
.. _feature-imom-whu7:

Weighted Hu invariants
^^^^^^^^^^^^^^^^^^^^^^

:Codes: ``WEIGHTED_HU_M1`` to ``WEIGHTED_HU_M7`` and ``IMOM_WHU1`` to ``IMOM_WHU7``
:Range: :math:`(-\infty, \infty)`

Hu's formulas (see `Hu invariants`_) applied to :math:`\eta^w_{pq}` in place of
:math:`\eta_{pq}`. Unlike :math:`I_1`, the weighted :math:`I_1` can be negative: it is
:math:`-0.194` on the worked example.

**Possible issues.** Invariants 3 to 7 use the odd-order :math:`\eta^w_{pq}`, so they are output as
the ``--noval`` substitute whenever :math:`m^w_{00} < 0`. Invariants 1 and 2 use only second-order
moments and are always computed.


.. _moments-possible-issues:

Possible issues
---------------

The contour is shifted by one pixel
   Each traced contour pixel lies one pixel right of and below the border pixel it stands for, so
   the distances :math:`d_i` are measured to the wrong pixels. Border pixels on the top and left
   sides are at distance 1 or more from the traced contour instead of 0. On a 48 × 40 rectangle,
   85 of the 1920 pixels lie on the traced contour, against 172 border pixels. With the exact
   distance to the traced contour ``WEIGHTED_SPAT_MOMENT_00`` would be 2422.7; with the exact
   distance to the ROI's own border it is 1821.3.

The distance search is approximate
   The search can miss the nearest contour pixel, so :math:`d_i` can be too large, never too
   small. On the 48 × 40 rectangle 121 of the 1920 pixels get too large a distance, by up to 12
   pixels, and ``WEIGHTED_SPAT_MOMENT_00`` is 2498.7 instead of 2422.7. Together with the shifted
   contour, Nyxus reports 37 % more than the exact distance to the ROI border gives. On the worked
   example 8 of the 40 pixels get too large a distance.

The sum of the weights is often negative
   Every pixel at :math:`d_i < 0.999` has a negative weight, and a pixel at :math:`d_i = 0` weighs
   :math:`-6.9\,I_i`. On small ROIs the sum :math:`m^w_{00}` is therefore negative, and all odd-order
   weighted normalized moments and weighted Hu invariants 3 to 7 are output as the ``--noval``
   substitute. On digital disks ``WEIGHTED_SPAT_MOMENT_00`` is -68.94 at radius 5, 6.806 at radius 8
   and 1549 at radius 20. Near the sign change the normalization divides by a number close to 0:
   ``WT_NORM_CTR_MOM_30`` is 7881 at radius 8 and 0.007713 at radius 20.

Weighted moments differ out of core
   Without anisotropy, an ROI computed out of core gets its contour from a different routine, which places it
   relative to the ROI's bounding box while the pixels keep their image positions. The distances
   :math:`d_i` are then off by the box's offset, and all 62 weighted moments depend on where the
   ROI lies. On an ellipse of 245 pixels ``WEIGHTED_SPAT_MOMENT_00`` is 49.81 in memory wherever
   it lies, and 7.484, 58.50 or 338.2 out of core depending on where its bounding box starts:
   (2, 2), (3, 5) or (10, 12). A ROI whose bounding box starts at (0, 0) is placed correctly, but
   the out-of-core contour can list its pixels in a different order, which changes the
   approximate distance search: a 48 × 40 rectangle there gets the in-memory values, while the
   ellipse gives 43.75. The 118 unweighted moments are the same on both paths.

Whole-slide mode
   The image corners used as the contour lie one pixel past the last column and row. Row 0 and
   column 0 get the weight 0.001 whatever their intensity, while the last row and column get
   :math:`I_i \ln 1.001`, so the weighting is not symmetric.

Weighted values carry single-precision rounding
   The weights are stored as 32-bit floats. The weighted moments agree with a double-precision
   computation of the same formula to about :math:`10^{-7}` relative.

Intensity moments depend on the intensity scale
   Multiplying every intensity by :math:`c` multiplies ``IMOM_NCM_pq`` by :math:`c^{-(p+q)/2}`,
   ``IMOM_NRM_pq`` by :math:`c^{-(p+q)/2}`, ``IMOM_HU1`` by :math:`c^{-1}` and ``IMOM_HU2`` by
   :math:`c^{-2}`. On the worked example with intensities times 10, ``IMOM_HU1`` drops from
   0.04106 to 0.004106. Floating-point images are rescaled to grey levels when they are loaded (see
   :doc:`/supported_formats`), so compare intensity moments only between images loaded the same
   way. The shape moments do not depend on intensity.

ROI with all intensities 0
   :math:`m_{00} = 0` for the intensity moments. The raw and weighted raw intensity moments are
   0. The others divide by :math:`m_{00}` or use the centroid, which is :math:`0/0`, and are
   output as the ``--noval`` substitute, 0 by default. The shape moments are computed as usual.

Single-pixel ROI
   No contour is traced, so the pixel has :math:`d = 0` and the weight :math:`\ln 0.001`.
   ``WEIGHTED_SPAT_MOMENT_00`` is -6.908, and the central and normalized central moments are 0.


Validation
----------

The test suite checks the 118 unweighted moments of both families (raw, central, normalized raw,
normalized central and Hu) against scikit-image to a relative :math:`10^{-6}`. scikit-image has no
function for the normalized raw moment, so ``NORM_SPAT_MOMENT_pq`` and ``IMOM_NRM_pq`` are
checked against its raw moments divided by the same power of :math:`m_{00}`. The 62 weighted
moments are covered by regression tests only: no reference tool computes the distance weight.
The worked-example script asserts every feature value of the worked example and the feature values
quoted under Possible issues, including the out-of-core ones. It does not assert the distance
matrix of the worked example, the counts of pixels given too large a distance, the
single-precision rounding of the weights or the comparison with OpenCV and scikit-image Hu
invariants; these were measured separately. The script compares no value with a reference tool.


References
----------

1. Hu MK. *Visual pattern recognition by moment invariants.* IRE Transactions on Information
   Theory. 1962;8(2):179–187. doi:10.1109/TIT.1962.1057692
2. Flusser J, Suk T, Zitová B. *2D and 3D Image Analysis by Moments.* Chichester: Wiley; 2016.
