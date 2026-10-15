Image quality features
======================

Image quality features score a whole image, or each ROI, for focus, sharpness and saturation. They
are meant to find blurred, out-of-focus or overexposed fields of view before analysis. Each one is
computed on the ROI's bounding box as a rectangular image.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - 6 features, one value per ROI: ``FOCUS_SCORE``, ``LOCAL_FOCUS_SCORE``,
       ``POWER_SPECTRUM_SLOPE``, ``MAX_SATURATION``, ``MIN_SATURATION`` and ``SHARPNESS``
   * - **Group**
     - ``*ALL_IMQ*``. It is not part of ``*ALL*``, and image quality features cannot be requested
       in the same run as other features.
   * - **Dimensions**
     - 2D only. There is no 3D family.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Partly, with a label image only. On an image that Nyxus reads as one tile (see
       :ref:`large-images-tiles`), ``FOCUS_SCORE`` and ``LOCAL_FOCUS_SCORE`` have the same values
       as in memory, and so do the saturations unless the ROI is constant. On a tiled image read
       as several tiles the call raises an error when a misplaced pixel falls outside the ROI's
       bounding box, and otherwise the focus scores and the saturations differ from the in-memory
       values.
       ``POWER_SPECTRUM_SLOPE`` and ``SHARPNESS`` are not computed out of core (see
       :ref:`imq-possible-issues`). In whole-image mode an image too large for memory is refused.
   * - **IBSI**
     - None. IBSI does not define image quality features.

.. tip::

   To detect blur, use ``FOCUS_SCORE`` or ``LOCAL_FOCUS_SCORE``. On an image blurred step by step
   they fall steadily, while ``POWER_SPECTRUM_SLOPE`` rises and ``SHARPNESS`` goes up and down (see
   :ref:`imq-possible-issues`).


How the features are computed
-----------------------------

Every feature reads one rectangular image, the *image matrix* :math:`I`, of :math:`H` rows and
:math:`W` columns, and nothing else (one exception under anisotropy, see `Settings`_).

- **Which pixels:** with a label image, :math:`I` is the ROI's bounding box. Pixels of the box that
  lie outside the mask are set to 0 and take part like any other pixel. In whole-image mode, where
  the intensity directory is also given as the label directory, :math:`I` is the whole image plus
  one row of 0 below it and one column of 0 to its right, so a 5 × 5 image is scored as 6 × 6.
  ``ImageQuality.featurize`` on in-memory arrays uses the bounding box of the mask, with no added
  row or column.
- **Values:** grey levels, as Nyxus stores them. Image quality features are not converted back to
  the image's units (see :ref:`intensity-units`). An integer image without negative values is used
  as it is; an image with negative values is shifted up by its minimum; a floating-point image is
  rescaled to ``0 .. --fpimgdr`` grey levels. ``ImageQuality.featurize`` on in-memory arrays does
  not do this: it casts the array to unsigned 32-bit integers, so fractions are truncated and
  negative values wrap around.
- **Binning:** none. ``--coarseGrayDepth`` and ``--ibsi`` have no effect.
- **Borders:** each feature has its own rule. The Laplacian and the first and second differences
  treat pixels outside :math:`I` as 0. The median filter of ``SHARPNESS`` repeats the edge pixels.


Worked example
^^^^^^^^^^^^^^

Take the 5 × 5 image of the texture pages, written to a file and read in whole-image mode. Nyxus
adds a row and a column of 0, so the image matrix and its Laplacian (kernel
:math:`[0, 1, 0;\ 1, -4, 1;\ 0, 1, 0]`, 0 outside) are

.. math::

   I = \begin{bmatrix}
   5 & 2 & 5 & 4 & 4 & 0\\
   3 & 3 & 3 & 1 & 3 & 0\\
   2 & 1 & 1 & 1 & 3 & 0\\
   4 & 2 & 2 & 2 & 3 & 0\\
   3 & 5 & 3 & 3 & 2 & 0\\
   0 & 0 & 0 & 0 & 0 & 0 \end{bmatrix},
   \qquad
   \nabla^2 I = \begin{bmatrix}
   -15 & 5 & -11 & -6 & -9 & 4\\
   -2 & -3 & -2 & 7 & -4 & 3\\
   0 & 4 & 3 & 3 & -5 & 3\\
   -9 & 4 & 0 & 1 & -5 & 3\\
   -3 & -12 & -2 & -5 & -2 & 2\\
   3 & 5 & 3 & 3 & 2 & 0 \end{bmatrix}

``FOCUS_SCORE`` is the population variance of the 36 values of :math:`\nabla^2 I`, 27.86.
``LOCAL_FOCUS_SCORE`` cuts :math:`I` into four 3 × 3 tiles, each filtered with 0 outside the tile;
their scores are 42.32, 32.89, 35.06 and 16.77, and their mean is 31.76. The largest value, 5, holds
3 of the 36 pixels, so ``MAX_SATURATION`` is 0.08333; the smallest is one of the 11 added zeros, so
``MIN_SATURATION`` is 0.3056. ``POWER_SPECTRUM_SLOPE`` is 0 because the image is narrower than 24
pixels. ``SHARPNESS`` is 1.903.

The same image as an array, or read with a mask of ones, has no added row and column. It gives
30.12, 22.69, 0.12, 0.16, 0 and 1.384.

The *Example* value of each feature in the :ref:`index <imq-index>` is what Nyxus returns for the
whole-image run. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/imq_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features="*ALL_IMQ*" \
                 --intDir=int/ --segDir=int/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`I(r, c)`
     - Grey level of the image matrix at row :math:`r` and column :math:`c`, counted from 0.
       :math:`I(r, c) = 0` outside the matrix, unless a feature says otherwise.
   * - :math:`H,\; W`
     - Number of rows and columns of the image matrix.
   * - :math:`N`
     - Number of pixels of the image matrix, :math:`N = HW`.
   * - :math:`\nabla^2 I`
     - Laplacian of :math:`I`,
       :math:`I(r-1,c) + I(r+1,c) + I(r,c-1) + I(r,c+1) - 4\,I(r,c)`, with 0 outside the matrix.
   * - :math:`\operatorname{Var}`
     - Population variance, dividing by the number of values.
   * - :math:`I_{\min},\; I_{\max}`
     - Smallest and largest grey level of the image matrix.


Settings
--------

.. list-table::
   :widths: 16 28 26 30
   :header-rows: 1

   * - Setting
     - Command line
     - Python
     - Effect
   * - Undefined value
     - ``--noval=0``
     - none
     - Value of ``LOCAL_FOCUS_SCORE`` when a side of the image matrix is shorter than 2 pixels.
       Python always uses 0.
   * - Image intensity map
     - ``--fpimgdr``, ``--fpimgmin``, ``--fpimgmax``
     - ``ImageQuality(..., dynamic_range=..., min_intensity=..., max_intensity=...)``
     - Decide the grey levels of a floating-point image, and so every feature value. The
       saturations change only where the map merges or splits grey levels at the extremes, for
       example with a small ``--fpimgdr``. ``ImageQuality`` applies ``dynamic_range`` but prints an
       "unexpected keyword argument" warning for it.
   * - Anisotropy
     - ``--anisox``, ``--anisoy``
     - ``ImageQuality(..., anisotropy_x=..., anisotropy_y=...)``
     - With files, ``POWER_SPECTRUM_SLOPE`` is computed on the ROI resampled by the factors: with
       ``anisotropy_x=2`` it has the value of the ROI made twice as wide. A fix is coming. The other
       five features are computed on the pixels as acquired and do not change. In-memory arrays are
       not affected.
   * - Memory limit
     - ``--ramLimit``
     - ``ImageQuality(..., ram_limit=...)``
     - With a label image, a ROI whose footprint reaches the limit is processed out of core. In
       whole-image mode, an image that reaches it is refused with an error; in Python the call
       raises ``RuntimeError``.

The features have no parameters: the Laplacian kernel, the 2 × 2 tile grid, the 24-pixel minimum of
``POWER_SPECTRUM_SLOPE`` and every constant of ``SHARPNESS`` are fixed. ``--coarseGrayDepth``,
``--ibsi`` and ``--pixelDistance`` have no effect. All options are listed in the
:ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named with its code. There are no per-direction columns and no
``_AVE`` columns.

In whole-image mode each image gives one row with ``ROI_label`` 1 and an empty ``mask_image``.
``intensity_image`` holds the path the image was found under, built from the directory as it was
given: relative when the directory is relative, absolute when it is absolute.


.. _imq-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with the tolerance; see the :doc:`/featurelist` for what the categories mean. *Example* is
the value on the worked example.

.. include:: image_quality_index.inc


Features
--------

.. _feature-focus-score:

Focus score
^^^^^^^^^^^

:Code: ``FOCUS_SCORE``
:Range: :math:`[0, \infty)`, squared grey levels

The variance of the Laplacian, the focus measure of Pech-Pacheco et al. Sharp edges give large
Laplacian values of both signs; blur flattens them. Higher values mean a sharper image.

.. math::

   \mathrm{FOCUS\_SCORE} = \operatorname{Var}\left(\nabla^2 I\right)
   = \frac{1}{N}\sum_{r,c}\left(\nabla^2 I(r,c) - \overline{\nabla^2 I}\right)^2

The variance is taken over the signed Laplacian, not over its magnitude. This equals
``cv2.Laplacian(I, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_CONSTANT).var()``.

**Possible issues.** The zero border makes the edge of the image matrix an edge of the image. A
constant 30 × 30 image of grey level 7 scores 6.098, not 0, and the value grows with the square of
the grey level. Pixels outside the mask, and the row and column that whole-image mode adds, are 0
and add edges of their own: the worked example scores 27.86 in whole-image mode and 30.12 as an
array.


.. _feature-local-focus-score:

Local focus score
^^^^^^^^^^^^^^^^^

:Code: ``LOCAL_FOCUS_SCORE``
:Range: :math:`[0, \infty)`, squared grey levels

The mean focus score of a 2 × 2 grid of equal tiles. Each tile is scored as a separate image, with
0 outside the tile, so the tile borders count as edges.

.. math::

   \mathrm{LOCAL\_FOCUS\_SCORE} = \frac{1}{4}\sum_{t_y = 0}^{1}\sum_{t_x = 0}^{1}
   \operatorname{Var}\left(\nabla^2 T_{t_y t_x}\right)

where :math:`T_{t_y t_x}` holds rows :math:`t_y h` to :math:`t_y h + h - 1` and columns
:math:`t_x w` to :math:`t_x w + w - 1` of :math:`I`, with :math:`h = \lfloor H/2 \rfloor` and
:math:`w = \lfloor W/2 \rfloor`. When :math:`H` or :math:`W` is odd, the last row or column belongs
to no tile. When :math:`h = 0` or :math:`w = 0` the value is the ``--noval`` substitute, 0 by
default.

**Relations.** The tile borders add to every tile score, so the value is usually larger than
``FOCUS_SCORE``: 1404 against 124.0 on the 128 × 128 disk blurred with :math:`\sigma = 1` (see
:ref:`imq-possible-issues`).

**Possible issues.** In whole-image mode the added row and column fall inside the tiles when the
image side is odd, and outside them when it is even. On a 128 × 128 image the value is the same as
with a mask of ones; on the 5 × 5 worked example it is 31.76 against 22.69.


.. _feature-max-saturation:

Maximum saturation
^^^^^^^^^^^^^^^^^^

:Code: ``MAX_SATURATION``
:Range: :math:`[0, 1]`, fraction of pixels

The fraction of the image matrix at its own largest grey level. It is the share of saturated
pixels when the image reaches the top of its range, but the maximum is the observed one, not the
largest value of the pixel type.

.. math::

   \mathrm{MAX\_SATURATION} = \frac{\left|\{(r,c) : I(r,c) = I_{\max}\}\right|}{N}

A constant image gives 1.


.. _feature-min-saturation:

Minimum saturation
^^^^^^^^^^^^^^^^^^

:Code: ``MIN_SATURATION``
:Range: :math:`[0, 1)`, fraction of pixels

The fraction of the image matrix at its own smallest grey level.

.. math::

   \mathrm{MIN\_SATURATION} = \frac{\left|\{(r,c) : I(r,c) = I_{\min} \ne I_{\max}\}\right|}{N}

A pixel equal to both extremes counts as maximal only, so a constant image gives 0.

**Possible issues.** Pixels outside the mask are 0 and are counted, so for any ROI that does not
fill its bounding box the minimum is 0 and the value is the share of the box outside the mask: a
plus-shaped ROI of 10 pixels in a 4 × 3 box gives 2/12. In whole-image mode the added row and
column of 0 make the minimum 0 for every image: the worked example gives 0.3056 where the image's
own minimum, 1, holds 0.16 of its pixels.


.. _feature-power-spectrum-slope:

Power spectrum slope
^^^^^^^^^^^^^^^^^^^^

:Code: ``POWER_SPECTRUM_SLOPE``
:Range: :math:`(-\infty, \infty)`, dimensionless

The slope of a straight line fitted to log power against log bin index, computed from the image's
2D Fourier transform. It is ported from CellProfiler's power log-log slope, but bins the power
differently (see *Possible issues*). The steps are:

1. If :math:`\lfloor \min(H, W)/8 \rfloor < 3`, that is if the shorter side is under 24 pixels, the
   value is 0.
2. Read :math:`I` in row-major order. If the last pixel at :math:`I_{\max}` comes after the first
   pixel at :math:`I_{\min}`, divide every pixel by :math:`m`, the value at position
   :math:`\lfloor N/2 \rfloor` of the sorted :math:`|I(r,c) - \bar{I}|`. Otherwise keep :math:`I`.
   Then subtract the mean, giving :math:`x`.
3. Place :math:`x` in an :math:`n \times n` array, :math:`n` the smallest power of 2 not below
   :math:`\max(H, W)`. Rows from :math:`H` on are 0. When :math:`W < n`, the columns from
   :math:`W` on of the first :math:`H` rows hold values of :math:`x` left over from moving the rows
   in place, not 0.
4. Take :math:`A(u,v) = |X(u,v)|/n`, where :math:`X` is the 2D discrete Fourier transform of the
   array.
5. Put each coefficient in bin :math:`b = \lfloor \sqrt{A(u,v)} \rfloor + 1`, and sum per bin
   :math:`P_b = \sum A(u,v)^2` and :math:`M_b = \sum A(u,v)`.
6. Fit :math:`\log P_b = s \log(b + 2) + c` by least squares over the bins with :math:`M_b > 0`.

.. math::

   \mathrm{POWER\_SPECTRUM\_SLOPE} = s

The value is 0 when every :math:`M_b` is 0, or when fewer than two bins are occupied.

**Possible issues.** The bin of a coefficient is set by its own magnitude, not by its frequency
radius :math:`\sqrt{u^2 + v^2}`, so the fit does not describe how power falls with frequency. On
the blur series under :ref:`imq-possible-issues` the slope is positive and rises with blur, where
a spectrum binned by frequency radius has a negative slope. The test in step 2 compares the
positions of the extremes, not their values, so whether the image is normalized depends on where
its brightest and darkest pixels lie. The Fourier transform steps its twiddle angles in single
precision: on the blur series this moves the slope by up to 3.3 × 10\ :sup:`-4` relative, from
2.3245 to 2.3237, against an exact transform.


.. _feature-sharpness:

Sharpness
^^^^^^^^^

:Code: ``SHARPNESS``
:Range: :math:`[0, \infty)`, dimensionless

An edge-sharpness score built from differences of a median-filtered image, adapted from the
difference-of-differences-of-medians (DOM) measure of Kumar et al. It is not that measure, and it
does not fall steadily with blur (see *Possible issues*). The steps are:

1. Median-filter :math:`I` with a 3 × 3 window, repeating the edge pixels, and divide by 255:
   :math:`B`.
2. From :math:`I`, with 0 outside, take the central differences
   :math:`G_x(r,c) = \tfrac{1}{2}\left(I(r,c+1) - I(r,c-1)\right)` and
   :math:`G_y(r,c) = \tfrac{1}{2}\left(I(r+1,c) - I(r-1,c)\right)`. The edge maps are
   :math:`E_x(r,c) = 1` where :math:`|G_x(r,c)| / (\max G_x + 10^{-8}) > 10^{-4}`, and
   :math:`E_y(r,c) = 1` where :math:`|G_y(c,r)| / (\max G_x + 10^{-8}) > 10^{-4}`, 0 elsewhere.
   :math:`\max G_x` is the signed maximum, and :math:`E_y` reads :math:`G_y` transposed. On a
   non-square image :math:`E_y` reads the elements of :math:`G_y` in the order an in-place swap of
   the two layouts leaves them.
3. From :math:`B`, with 0 outside, take
   :math:`D_x(r,c) = |B(r-2,c) - 2B(r,c) + B(r+2,c)|`,
   :math:`D_y(r,c) = |B(r,c-2) - 2B(r,c) + B(r,c+2)|`,
   :math:`C_x(r,c) = E_x(r,c)\,|B(r+1,c) - B(r,c)|` and
   :math:`C_y(r,c) = E_y(r,c)\,|B(r,c+1) - B(r,c)|`.
4. For :math:`2 \le r \le H-3` and :math:`0 \le c \le W-3`, sum both over the four rows
   :math:`k = r-2, \dots, r+1`:

   .. math::

      S_x(r,c) = \frac{\sum_k D_x(k,c)}{\sum_k C_x(k,c)}, \qquad
      S_y(r,c) = \frac{\sum_k D_y(k,c)}{\sum_k C_y(k,c)}

   A ratio whose denominator is :math:`10^{-3}` or less is 0, and :math:`S_x` and :math:`S_y` are
   0 outside these rows and columns.
5. Combine:

.. math::

   R_x = \frac{\sum_{r,c} S_x(r,c)}{\sum_{r,c} E_x(r,c) + 10^{-8}}, \qquad
   R_y = \frac{\sum_{r,c} S_y(r,c)}{\sum_{r,c} E_y(r,c) + 10^{-8}}, \qquad
   \mathrm{SHARPNESS} = \sqrt{R_x^2 + R_y^2}

An image with fewer than 5 rows, and a constant image, give 0.

**Possible issues.** The DOM measure counts the pixels whose ratio exceeds a threshold and divides
by the number of edge pixels; this feature sums the ratios instead. Its edge maps are taken along
the other axis from the differences they mask, :math:`S_y` is summed down the columns rather than
along the rows, and the last two columns are never scored. On the blur series under
:ref:`imq-possible-issues` the value goes 4.748, 11.23, 3.755, 1.881. The denominator threshold is
in grey levels divided by 255, so the value of a dim, low-contrast image depends on its absolute
brightness.


.. _imq-possible-issues:

Possible issues
---------------

Blur series
   A 128 × 128 disk of grey level 255 and radius 40 on 0, blurred with a Gaussian of
   :math:`\sigma` = 0, 1, 2 and 4 pixels and read in whole-image mode, gives:

   .. list-table::
      :widths: 34 16 16 16 16
      :header-rows: 1

      * - Feature
        - :math:`\sigma = 0`
        - 1
        - 2
        - 4
      * - ``FOCUS_SCORE``
        - 3970
        - 124.0
        - 17.04
        - 2.828
      * - ``POWER_SPECTRUM_SLOPE``
        - 1.178
        - 1.653
        - 1.996
        - 2.324
      * - ``SHARPNESS``
        - 4.748
        - 11.23
        - 3.755
        - 1.881

   ``FOCUS_SCORE`` falls with blur. ``POWER_SPECTRUM_SLOPE`` rises, because it bins power by the
   magnitude of each Fourier coefficient rather than by frequency. ``SHARPNESS`` rises and then
   falls. Use the focus scores to rank images by blur.

Whole-image mode adds a row and a column of 0
   In whole-image mode the image matrix is one pixel larger than the image in each direction, and
   the extra pixels are 0. ``MIN_SATURATION`` then counts them, ``MAX_SATURATION`` divides by the
   larger count, the focus scores see an extra edge, and ``POWER_SPECTRUM_SLOPE`` changes too:
   1.178 in whole-image mode against 2.106 with a mask of ones on the unblurred disk of the blur
   series. On the worked example the six values are 27.86, 31.76, 0, 0.08333, 0.3056 and 1.903;
   with a mask of ones they are 30.12, 22.69, 0, 0.12, 0.16 and 1.384.

Pixels outside the mask
   With a label image every feature is computed on the bounding box, with pixels outside the mask
   set to 0. For ROIs that do not fill their box, the mask outline becomes an edge for the focus
   scores and ``SHARPNESS``, and a share of minimal pixels for ``MIN_SATURATION``.

Out-of-core ROIs
   Out of core, ``POWER_SPECTRUM_SLOPE`` and ``SHARPNESS`` are not computed, and their columns
   hold whatever value happens to be in memory: 0, a tiny number such as :math:`10^{-311}`, or
   anything else; Nyxus does not set them to the ``--noval`` substitute. A constant ROI processed
   out of core skips the saturations too, and both of their columns hold such a value. Discard
   these values, or raise ``--ramLimit`` so that the ROIs fit in memory.

Small images
   ``POWER_SPECTRUM_SLOPE`` is 0 when the shorter side of the image matrix is under 24 pixels, and
   ``SHARPNESS`` when it has fewer than 5 rows. ``LOCAL_FOCUS_SCORE`` is the ``--noval`` substitute
   when a side is shorter than 2 pixels.

Constant images
   A constant image, as an array or with a mask that fills its box, gives ``MAX_SATURATION`` 1,
   ``MIN_SATURATION`` 0, ``POWER_SPECTRUM_SLOPE`` 0 and ``SHARPNESS`` 0. The focus scores are not
   0, because of the zero border: 6.098 and 11.32 for a 30 × 30 image of grey level 7.

Grey levels, not image units
   The features are computed on stored grey levels. For a floating-point image the values depend
   on ``--fpimgdr``, ``--fpimgmin`` and ``--fpimgmax``; the focus scores scale with the square of
   the grey-level range.


Validation
----------

- ``FOCUS_SCORE`` is checked against OpenCV 4.13.0,
  ``cv2.Laplacian(roi, cv2.CV_64F, ksize=1, borderType=cv2.BORDER_CONSTANT).var()``, to an absolute
  10\ :sup:`-9`.
- ``LOCAL_FOCUS_SCORE`` is checked against the mean of the same OpenCV score over the 2 × 2 tile
  grid, to an absolute 10\ :sup:`-9`, and against a closed form for a single spike in each tile,
  exactly.
- ``MAX_SATURATION`` and ``MIN_SATURATION`` are checked against CellProfiler 4.2.8
  ``MeasureImageQuality`` (``PercentMaximal`` and ``PercentMinimal`` divided by 100) to an absolute
  10\ :sup:`-9`. The constant-ROI case and masks narrower than their bounding box, where
  CellProfiler counts differently, are covered by regression tests only.
- ``POWER_SPECTRUM_SLOPE`` and ``SHARPNESS`` are regression-only: no independent implementation
  confirms them, and a reference DOM implementation gives a different value for ``SHARPNESS``.

The out-of-core focus scores are checked against the in-memory values of the same build.

Most values quoted on this page are pinned by the worked-example script; the out-of-core value
:math:`10^{-311}` and the slope of 2.3245 from an exact transform are measured only. Apart from
that slope, the values are Nyxus' own; the focus scores also equal OpenCV's, and the saturations
of ROIs that fill their bounding box CellProfiler's.


References
----------

1. Pech-Pacheco JL, Cristóbal G, Chamorro-Martínez J, Fernández-Valdivia J. *Diatom autofocusing in
   brightfield microscopy: a comparative study.* Proceedings of the 15th International Conference
   on Pattern Recognition (ICPR 2000). 2000;3:314–317. doi:10.1109/ICPR.2000.903548
2. Kumar J, Chen F, Doermann D. *Sharpness estimation for document and scene images.* Proceedings
   of the 21st International Conference on Pattern Recognition (ICPR 2012). 2012:3292–3295.
3. Bray MA, Fraser AN, Hasaka TP, Carpenter AE. *Workflow and metrics for image quality control in
   large-scale high-content screens.* Journal of Biomolecular Screening. 2012;17(2):266–274.
   doi:10.1177/1087057111420292 (CellProfiler ``MeasureImageQuality``: saturation and power
   log-log slope)
