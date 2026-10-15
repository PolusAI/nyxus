Image quality features
======================

Image quality features measure focus, sharpness and saturation, for example to find blurred or
overexposed fields of view before analysis. They are 2D features, computed per ROI on the ROI's
bounding box with pixels outside the ROI set to 0. In whole-image mode (the intensity directory
given as the label directory) the ROI is the whole image plus one row and one column of 0 (see
:ref:`imq-possible-issues`).

The group is ``*ALL_IMQ*``. It is not part of ``*ALL*``, and image quality features cannot be
requested in the same run as other features. The ``ImageQuality`` class refuses other feature
names, except after ``*ALL_IMQ*``: names that follow it are dropped without a message.

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Feature code
     - Description
   * - ``FOCUS_SCORE``
     - Variance of the Laplacian of the image (3×3 kernel). Higher values mean a sharper image.
       In squared grey levels, 0 or more.
   * - ``LOCAL_FOCUS_SCORE``
     - Mean of ``FOCUS_SCORE`` over a 2×2 grid of equal, non-overlapping tiles. When a side of the
       bounding box is shorter than 2 pixels the value is the ``--noval`` substitute.
   * - ``POWER_SPECTRUM_SLOPE``
     - Slope of a straight-line fit of log power against log bin index, computed from the image's
       Fourier transform, with each coefficient binned by its own magnitude rather than by its
       frequency. 0 when the shorter side of the bounding box is under 24 pixels.
       The current implementation does not track blur: its value can rise, fall or change sign as
       an image is blurred. Use ``FOCUS_SCORE`` to detect blur.
   * - ``MAX_SATURATION``
     - Fraction of bounding-box pixels at the maximum grey level, from 0 to 1.
   * - ``MIN_SATURATION``
     - Fraction of bounding-box pixels at the minimum grey level, from 0 to 1. Pixels outside the
       ROI count as 0.
   * - ``SHARPNESS``
     - Edge sharpness from the difference of medians of a 3×3 median-filtered image, combined over
       the horizontal and vertical directions. The value has no fixed upper bound, and it does not
       fall steadily as an image is blurred, so compare it only between images of similar content.

:doc:`Math/f_image_quality` gives the formulas, a worked example and the cases that need care,
such as small bounding boxes and ROIs processed out of core.

The features have no settings of their own. Values are computed on grey levels, so for
floating-point images they depend on ``--fpimgdr`` (see :ref:`intensity-units`).

Command line:

.. code-block:: bash

   nyxus --features="*ALL_IMQ*" --intDir=/data/int --segDir=/data/int --outDir=/data/out --outputType=singlecsv

Python, with the ``ImageQuality`` class:

.. literalinclude:: ../examples/image_quality.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end
