Fractal dimension features
==========================

Fractal dimension features describe how irregular a ROI's shape is. Box counting measures how the
number of boxes needed to cover the ROI grows as the boxes get smaller. The divider method measures
how the length of the ROI's contour grows as the ruler used to walk it gets shorter. Smooth shapes
give values near the shape's topological dimension, 2 for a filled region and 1 for its outline;
rough or ragged shapes give larger values.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``FRACT_DIM_BOXCOUNT`` and ``FRACT_DIM_PERIMETER``, one value per ROI
   * - **Group**
     - ``*ALL_MORPHOLOGY*`` (and ``*ALL*``)
   * - **Dimensions**
     - 2D only. There is no 3D fractal dimension.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Partly. Out of core, box counting always uses the unshifted grid, so ROIs whose padded
       side :math:`S` is 32 or less can read differently (the worked example gives 1.3966 instead
       of 1.3732). The divider walks only the first loop of the contour, and walks a one-pixel-wide
       part once instead of along and back, so ROIs with holes, several parts or one-pixel-wide parts
       can differ (a 24 × 24 square with a 10 × 10 hole gives 1.005 instead of 1.045, while identical
       separate parts read the same on both paths); with anisotropy every ROI's divider value can
       differ. These contour differences are a known
       defect (`#509 <https://github.com/PolusAI/nyxus/issues/509>`__). On an image
       that Nyxus reads as several tiles out-of-core processing raises an error or returns wrong
       values, depending on the ROI (see :ref:`large-images-tiles`
       and :doc:`/large_images`). See :ref:`fractal-possible-issues`.
   * - **IBSI**
     - Neither feature has an IBSI identifier; IBSI defines no fractal dimension.

.. tip::

   Both features depend on the ROI's size as well as its shape. A filled square reads exactly 2
   only for some side lengths, and a 20 × 20 square reads 1.593 (see
   :ref:`fractal-possible-issues`). Compare values between ROIs of similar size, and do not read
   them as the dimension of an ideal fractal.


How the features are computed
-----------------------------

Both features use only the ROI's mask. The intensities play no part.

- **Which pixels:** ``FRACT_DIM_BOXCOUNT`` uses every pixel of the ROI, including pixels of
  intensity 0 inside the mask. ``FRACT_DIM_PERIMETER`` uses the ROI's contour, the ordered chain of
  boundary pixels that the :ref:`contour features <features-contour>` trace and measure
  ``PERIMETER`` on.
- **Values:** pixel positions in pixels. Box sizes are powers of two; ruler lengths are Euclidean
  distances between contour pixel centres.
- **Units:** both features are dimensionless. They do not change under an intensity map, so the
  image's units (see :ref:`intensity-units`) do not matter.
- **Log base:** both regressions use natural logarithms. The slope does not depend on the base.
- **IBSI mode:** ``--ibsi=true`` does not change either feature.

**Box counting.** Nyxus takes the larger side :math:`L` of the ROI's bounding box and rounds it up
to a power of two, :math:`S`. The ROI is placed with its bounding-box corner at the origin of an
:math:`S \times S` canvas. For each box size :math:`s = S, S/2, \ldots, 2` it counts the boxes of an
:math:`s \times s` grid that hold at least one ROI pixel. Box size 1 is not used, and the largest
size, which covers the canvas with one box, is. When :math:`S \le 32` the grid is also shifted by
half a box, along :math:`x`, along :math:`y` and along both, and the smallest of the four counts is
kept. When :math:`S > 32` only the unshifted grid is used. The dimension is minus the least-squares
slope of :math:`\ln N(s)` against :math:`\ln s`.

**Divider walk.** The contour is one closed walk per border of the ROI: around the outside of each
separate part and around each hole, as the :ref:`contour features <features-contour>` trace it.
With :math:`n` the length of the longest walk, Nyxus uses strides of
:math:`d = \lfloor n/4 \rfloor, \lfloor d/2 \rfloor, \ldots, 1` pixels. At each stride every
border is walked on its own: it starts at the border's first pixel and jumps :math:`d` pixels at a
time while a full stride remains, adding the straight-line length of each jump, then closes the
loop with one chord from the last pixel reached back to the first; this chord spans the remaining
:math:`n_b - qd` pixels of the border, which can be fewer than :math:`d`. A border of at most
:math:`d` pixels is not walked: it adds its closed length, as measured at stride 1, and no chord.
The summed length :math:`P(d)` of all borders and the ruler :math:`r(d) = W(d)/m(d)`, the mean of
the :math:`m(d)` chords of the walked borders, whose summed length is :math:`W(d)`, give one point. The dimension is 1 minus the
least-squares slope of :math:`\ln P` against :math:`\ln r`. At stride 1 the walks are the
contour's ``PERIMETER``. No chord joins one border to another.


Worked example
^^^^^^^^^^^^^^

Take this 10 × 9 ROI, with 1 marking the mask:

.. math::

   \textbf{M} = \begin{bmatrix}
   0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
   0 & 1 & 1 & 1 & 1 & 0 & 0 & 1 & 1 & 0\\
   1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 0\\
   1 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 0 & 0\\
   0 & 1 & 1 & 1 & 1 & 1 & 1 & 0 & 0 & 0\\
   0 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 0 & 0\\
   0 & 0 & 1 & 1 & 1 & 1 & 1 & 1 & 1 & 0\\
   0 & 0 & 1 & 1 & 0 & 0 & 1 & 1 & 1 & 1\\
   0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 1 & 0 \end{bmatrix}

**Box counting.** The larger side is :math:`L = 10`, so :math:`S = 16` and the box sizes are 16, 8,
4 and 2. Because :math:`S \le 32`, each size is counted on four grids. A shift of :math:`o_x` adds
:math:`o_x` to every column index before it is divided by :math:`s`.

.. list-table::
   :widths: 12 16 16 16 16 12
   :header-rows: 1

   * - :math:`s`
     - :math:`(0, 0)`
     - :math:`(s/2, 0)`
     - :math:`(0, s/2)`
     - :math:`(s/2, s/2)`
     - :math:`N(s)`
   * - 16
     - 1
     - 2
     - 2
     - 4
     - 1
   * - 8
     - 4
     - 3
     - 4
     - 4
     - 3
   * - 4
     - 8
     - 7
     - 9
     - 8
     - 7
   * - 2
     - 20
     - 18
     - 18
     - 21
     - 18

The least-squares slope of :math:`\ln(1, 3, 7, 18)` against :math:`\ln(16, 8, 4, 2)` is
:math:`-1.3732`, so ``FRACT_DIM_BOXCOUNT`` is 1.3732. The unshifted grid alone, with counts 1, 4, 8
and 20, would give 1.3966.

**Divider walk.** The contour is one border, a walk of :math:`n = 26` pixels, so the strides are
6, 3 and 1.

.. list-table::
   :widths: 14 18 34 34
   :header-rows: 1

   * - :math:`d`
     - :math:`m(d)`
     - :math:`P(d)`
     - :math:`r(d)`
   * - 6
     - 5
     - 26.8249
     - 5.3650
   * - 3
     - 9
     - 29.4352
     - 3.2706
   * - 1
     - 26
     - 32.6274
     - 1.2549

With stride 6 the walk visits pixels 0, 6, 12, 18 and 24, and the closing chord runs from pixel 24
back to pixel 0 across two contour steps. The least-squares slope of :math:`\ln P` against
:math:`\ln r` is :math:`-0.1311`, so ``FRACT_DIM_PERIMETER`` is 1.1311. :math:`P(1) = 32.6274` is
the ROI's ``PERIMETER``.

The *Example* value of each feature in the :ref:`index <fractal-index>` is what Nyxus returns for
this ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/fractal_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features=FRACT_DIM_BOXCOUNT,FRACT_DIM_PERIMETER \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`(x_k, y_k)`
     - Column and row of the :math:`k`-th ROI pixel.
   * - :math:`x_{\min},\; y_{\min}`
     - Smallest column and row of the ROI, the corner of its bounding box.
   * - :math:`L`
     - Larger side of the bounding box, in pixels.
   * - :math:`S`
     - :math:`L` rounded up to a power of two; 1 when :math:`L = 1`.
   * - :math:`s`
     - Box size, one of :math:`S, S/2, \ldots, 2`.
   * - :math:`N(s)`
     - Number of boxes of size :math:`s` that hold at least one ROI pixel.
   * - :math:`K^b_0, \ldots, K^b_{n_b-1}`
     - The walk around border :math:`b`, in walking order; :math:`n_b` is its length and
       :math:`n = \max_b n_b`.
   * - :math:`L_b`
     - Closed length of border :math:`b`'s walk at stride 1.
   * - :math:`d`
     - Stride of the divider walk, in contour pixels.
   * - :math:`q_b`
     - Number of full strides along border :math:`b`, :math:`q_b = \lceil n_b/d \rceil - 1`.
   * - :math:`B(d)`
     - The borders walked at stride :math:`d`, those with :math:`n_b > d`.
   * - :math:`P(d),\; m(d),\; r(d)`
     - Summed length of the borders at stride :math:`d`, the number of chords of the walked
       borders :math:`m(d) = \sum_{b \in B(d)} (q_b + 1)`, and the mean chord
       :math:`r(d) = P(d)/m(d)`.
   * - :math:`\beta\{(a_j, b_j)\}`
     - Least-squares slope of :math:`\ln b_j` against :math:`\ln a_j`, defined below.

The slope over :math:`M` points with :math:`u_j = \ln a_j` and :math:`v_j = \ln b_j` is

.. math::

   \beta = \frac{M \sum_j u_j v_j - \sum_j u_j \sum_j v_j}{M \sum_j u_j^2 - \left(\sum_j u_j\right)^2}

Points with :math:`a_j \le 0` or :math:`b_j \le 0` are skipped. :math:`\beta = 0` when fewer than
two points remain or the denominator is 0.


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
     - The ROI is the whole image, but its bounding box is one pixel larger than the image in
       each direction, a known defect (`#493 <https://github.com/PolusAI/nyxus/issues/493>`__), and the canvas side :math:`S` is taken from that box. ``FRACT_DIM_BOXCOUNT``
       is that of a filled rectangle of the image's size on that canvas: 1.8703 for 200 × 200,
       and 1.8667 for 256 × 256, where a 256 × 256 mask of ones reads 2.
       ``FRACT_DIM_PERIMETER`` is exactly 1, because the contour is the image's four corners.
   * - Memory limit
     - ``--ramLimit``
     - ``Nyxus(..., ram_limit=...)``, with ``featurize_files`` or ``featurize_directory``
     - ROIs whose estimated footprint reaches the limit are processed out of core. Their box count
       always uses the unshifted grid, and their divider walk covers only the first loop of the
       contour. Arrays passed to ``Nyxus.featurize`` are never processed out of core: an oversized
       ROI raises ``RuntimeError``.
   * - Anisotropy
     - ``--anisox``, ``--anisoy``
     - ``Nyxus(..., anisotropy_x=..., anisotropy_y=...)``, with ``featurize_files`` or
       ``featurize_directory``; arrays passed to ``Nyxus.featurize`` are not rescaled
     - The image is resampled before the ROIs are measured, which changes both features. The
       worked example gives 1.3732 and 1.1311 without anisotropy, 1.2459 and 1.1566 with
       ``anisotropy_x=2``, and 1.4458 and 1.1635 with ``anisotropy_y=3``.
   * - Merged ROIs
     - ``--mergerois=true``
     - ``Nyxus(..., mergerois=True)``
     - All labels of a mask form one ROI. Two 25 × 25 squares with labels 1 and 2 give 1.8416 and
       1.000 each; merged, they give 1.4231 and 1.000.

No other setting changes these features: not ``--coarseGrayDepth``, not ``--ibsi`` and not the
image intensity map. All options are listed in the :ref:`cli-reference`.


Output columns
--------------

Each feature produces one column, named with its code: ``FRACT_DIM_BOXCOUNT`` and
``FRACT_DIM_PERIMETER``. There are no ``_AVE`` columns.


.. _fractal-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example.

.. include:: fractal_index.inc


Features
--------

.. _feature-fract-dim-boxcount:

Box-counting dimension
^^^^^^^^^^^^^^^^^^^^^^

:Code: ``FRACT_DIM_BOXCOUNT``
:Range: from 0 to about 2, dimensionless

How fast the number of covering boxes grows as the boxes shrink. Values near 2 mean the ROI fills
its area; lower values mean a thin, sparse or ragged ROI.

.. math::

   N(s) = \min_{o_x,\, o_y \in O(s)} \left|\left\{ \left(
   \left\lfloor \tfrac{x_k - x_{\min} + o_x}{s} \right\rfloor,
   \left\lfloor \tfrac{y_k - y_{\min} + o_y}{s} \right\rfloor \right) \right\}\right|,
   \qquad
   O(s) = \begin{cases} \{0,\; s/2\} & S \le 32 \\ \{0\} & S > 32 \end{cases}

.. math::

   \mathrm{FRACT\_DIM\_BOXCOUNT} = -\beta\left\{ \left(s, N(s)\right) :
   s = S, S/2, \ldots, 2 \right\}

The value is 0 for a ROI of fewer than 2 pixels, and for a ROI whose bounding box has no side
longer than 2 pixels, because only one box size remains.

**Relations.** A ROI that fills a square bounding box of side :math:`2^k` or :math:`2^k - 1`
reads 2, and a one-pixel-wide straight line of length :math:`2^k` or :math:`2^k - 1` reads 1, both
up to rounding of order :math:`10^{-15}`. Other sizes read lower.

**Possible issues.** The value depends on the ROI's size relative to the power of two above it, as
shown under :ref:`fractal-possible-issues`.


.. _feature-fract-dim-perimeter:

Perimeter (divider) dimension
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``FRACT_DIM_PERIMETER``
:Range: about :math:`[1, 2)` in practice, dimensionless; smooth outlines can read slightly below 1
   (a 6 × 6 square gives 0.9947); 0 when the longest border walk has fewer than 3 pixels

How fast the measured length of the contour grows as the ruler shrinks, after Richardson. Values
near 1 mean a smooth outline; larger values mean a rough or wiggly one.

.. math::

   P(d) = \sum_{b \in B(d)} \left( \sum_{j=0}^{q_b-1} \left\| K^b_{(j+1)d} - K^b_{jd} \right\|
   + \left\| K^b_0 - K^b_{q_b d} \right\| \right) + \sum_{b \notin B(d)} L_b,
   \qquad
   r(d) = \frac{P(d)}{m(d)}

.. math::

   \mathrm{FRACT\_DIM\_PERIMETER} = 1 - \beta\left\{ \left(r(d), P(d)\right) :
   d = \lfloor n/4 \rfloor, \lfloor d/2 \rfloor, \ldots, 1 \right\}

The value is 0 when the longest border walk has fewer than 3 pixels. A longest walk of 3 to 7
pixels gives at most one point, so :math:`\beta = 0` and the value is exactly 1.

**Relations.** In memory, :math:`P(1)` is ``PERIMETER``; out of core, ``PERIMETER`` is the number
of contour pixels instead (26 on the worked example). The fit runs against the mean chord
:math:`r(d)`, not against the stride :math:`d`. Copies of one shape under one label read the
dimension of a single copy: at every stride both :math:`P(d)` and :math:`m(d)` scale with the
number of copies.

**Possible issues.** A border no longer than the stride adds its full length and no chord, so a
ROI with small holes or small separate parts can read higher than its outer border alone: a
24 × 24 square with a 10 × 10 hole gives 1.045 against 1.005 for the square (see
:ref:`fractal-possible-issues`).


.. _fractal-possible-issues:

Possible issues
---------------

Box counting depends on the ROI's size
   :math:`N(s)` counts whole boxes, so a ROI whose side does not fill the boxes evenly is
   over-covered at the coarse sizes, and the box that covers the whole canvas always counts 1. A
   filled square reads 2 at sides 16, 31 and 32, but 1.593 at side 20, 1.582 at side 33 and 1.851
   at side 100. For the 20 × 20 square the counts are 1, 4, 9, 25 and 100 at sizes 32 to 2. Disks
   read 1.430 at radius 8 and 1.686 at radius 40. A one-pixel-wide line reads 1 at lengths 16 and
   32 but 0.755 at length 10 and 0.926 at length 100, below the dimension of a line.

One-pixel-wide parts in the divider walk
   The walk around a one-pixel-wide line runs along it and back. A chord that spans an end of the
   line folds back on itself and is shorter than its stride, which shortens the walk at the larger
   strides, so a straight line reads above 1: 1.035 at length 16, 1.101 at length 10. Thin
   branches of larger ROIs are affected in the same way.

Shifted grids only for small ROIs
   Grid shifts are tried only when the padded side :math:`S` is 32 or less. Larger ROIs, and every
   out-of-core ROI, use the unshifted grid anchored at the bounding-box corner. The shifts matter
   for irregular shapes: the worked example reads 1.3732 with them and 1.3966 on the unshifted
   grid alone. On a filled square they change nothing; the drop from 2 for a 32 × 32 square to
   1.582 for a 33 × 33 square comes from :math:`S` doubling from 32 to 64, which leaves the coarse
   boxes mostly empty.

Out-of-core ROIs
   An out-of-core ROI uses the unshifted grid whatever its size, so a ROI with :math:`S \le 32`
   can read a different ``FRACT_DIM_BOXCOUNT``: the worked example gives 1.3966 out of core and
   1.3732 in memory. The out-of-core contour holds only the first loop of the ROI's boundary, so
   the divider walk ignores holes and separate parts: a 24 × 24 square with a 10 × 10 hole gives
   1.005 out of core, the value of the square without its hole, and 1.045 in memory. Without
   anisotropy, a ROI with a single contour can differ too: in memory the walk passes a
   one-pixel-wide part along and back, out of core once, so a 1 × 16 line gives 1.035 in memory
   and 1.190 out of core, and an 8 × 8 square with a one-pixel-wide branch 1.190 and 1.102. The
   worked example, whose border has no one-pixel-wide part, gives the same value on both paths.
   With anisotropy the out-of-core contour is wrong, so single-contour ROIs differ
   too: a disk of radius 8 with ``anisotropy_x=2`` gives 1.0917 in memory and 1.0482 out of core.
   The out-of-core contour is a known defect (`#509 <https://github.com/PolusAI/nyxus/issues/509>`__). Images that Nyxus reads as several tiles have further
   out-of-core differences (see :ref:`large-images-tiles`).

Short borders are measured whole
   A ROI with holes, or with parts that do not touch, has several borders, each walked on its own.
   At a stride longer than a border, that border adds its full length and no chord, so it raises
   :math:`P(d)` at the large strides and leaves the ruler :math:`r(d)` to the borders walked. A 24 × 24 square with a 10 × 10
   hole gives 1.045, where the square alone gives 1.005. Two separate 25 × 25 squares in one ROI
   give 1.000, as one such square does.

Small ROIs
   ``FRACT_DIM_BOXCOUNT`` is 0 for fewer than 2 pixels and for bounding boxes no larger than
   2 × 2. ``FRACT_DIM_PERIMETER`` is 0 when the longest border walk has fewer than 3 pixels and
   exactly 1 when it has 3 to 7. A 3 × 3 square gives 2 and 1.

Largest 32-bit intensity
   This is a known defect (`#515 <https://github.com/PolusAI/nyxus/issues/515>`__). In memory, on ROIs whose padded side :math:`S` exceeds 32, box
   counting marks each ROI pixel with its intensity plus 1, converted to a 32-bit unsigned
   integer. For a pixel of intensity 4294967295 that conversion is out of range, so whether the
   pixel is counted depends on the compiler. Out of core the pixel is counted: a 40 × 40 square
   with one separate pixel reads 1.4156 at any intensity.


Validation
----------

The test suite checks both features against ImageJ/FracLac on an irregular ROI in a 512 × 512
image:

- ``FRACT_DIM_BOXCOUNT`` against FracLac's box count of the same ROI, 1.8706, to a relative
  10\ :sup:`-2`. Nyxus gives 1.8608.
- ``FRACT_DIM_PERIMETER`` against FracLac's box count of the ROI's edge, 1.0493, to a relative
  3 × 10\ :sup:`-2`. This compares two methods for the same boundary dimension. Nyxus gives 1.0572.

The tests also check shapes with a known dimension: box counting gives 2 for a 256 × 256 square, 1
for a straight line and :math:`\log_2 3 = 1.585` for a Sierpinski triangle, to an absolute 0.1; the
divider gives 1 for a disk and :math:`\log 4 / \log 3 = 1.262` for a Koch snowflake, to an absolute
0.05, and two identical separate disks under one label give exactly the divider dimension of one.
The feature values quoted above Validation, and the box counts of the worked example and of
the 20 × 20 square, are asserted by the worked-example script; the divider-walk table of the worked
example is not. None is compared with a reference tool.

A standard box-counting implementation, with the grid anchored at the image corner and box sizes
from 1 to the largest power of two that fits in the image, gives different values: 1.251 on the
worked example and 1.676 on the 20 × 20 square. It uses box size 1, does not try shifted grids and
does not use box sizes larger than the image. Neither result is the exact dimension of a finite
pixel set.


References
----------

1. Mandelbrot B. *How long is the coast of Britain? Statistical self-similarity and fractional
   dimension.* Science. 1967;156(3775):636–638. doi:10.1126/science.156.3775.636
2. Richardson LF. *The problem of contiguity: an appendix to Statistics of Deadly Quarrels.*
   General Systems Yearbook. 1961;6:139–187.
3. ISO 9276-6:2008. *Representation of results of particle size analysis — Part 6: Descriptive
   and quantitative representation of particle shape and morphology.*
4. Karperien A. *FracLac for ImageJ.* https://imagej.net/ij/plugins/fraclac/FLHelp/Introduction.htm
