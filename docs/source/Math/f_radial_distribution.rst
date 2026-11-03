Radial distribution features
============================

Radial distribution features describe how a ROI's pixels and intensities are spread from its centre
outward. Nyxus picks a centre pixel, divides the ROI into 8 rings around it and each ring into 8
angular wedges, and reports, per ring, the share of the ROI's pixels, their mean intensity and how
unevenly the intensity is spread over the wedges.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``FRAC_AT_D``, ``MEAN_FRAC`` and ``RADIAL_CV``: three codes, 8 columns each per ROI,
       ``<code>_0`` (innermost ring) to ``<code>_7``. Requesting any one of them computes all three.
   * - **Group**
     - ``*ALL*`` and ``*WHOLESLIDE*``. No other group includes them; request them by name.
   * - **Dimensions**
     - 2D only.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes. Without anisotropy the values differ from the in-memory ones for a ROI whose bounding
       box does not start at the image's first row and column, and can differ even when it does, a
       known defect (`#509 <https://github.com/PolusAI/nyxus/issues/509>`__);
       with anisotropy an out-of-core ROI gets its in-memory values. On an image that Nyxus reads
       as several tiles out-of-core processing raises an error or returns wrong values in both
       cases, depending on the ROI (see
       :ref:`radial-possible-issues`, :ref:`large-images-tiles` and :doc:`/large_images`).
   * - **IBSI**
     - None. IBSI defines no radial distribution features.

.. tip::

   The three codes carry the names of CellProfiler's ``RadialDistribution_FracAtD``, ``MeanFrac``
   and ``RadialCV`` measurements, but compute different quantities: ``FRAC_AT_D`` is a fraction of
   the ROI's *pixels*, not of its intensity, ``MEAN_FRAC`` is a mean intensity in grey levels, not
   a ratio near 1, and ``RADIAL_CV`` is taken over wedge sums. The rings are also built
   differently. The values cannot be compared with CellProfiler's; see :ref:`radial-other-tools`.


How the features are computed
-----------------------------

Nyxus traces the ROI's contour, chooses the ROI pixel whose distances to the contour vary least as
the centre, takes the largest distance from the centre to the contour as the radius :math:`R`, and
sorts every ROI pixel into a ring by its distance from the centre and into a wedge by its angle.

- **Which pixels:** every pixel of the ROI, with its intensity. A pixel of intensity 0 inside the
  mask counts like any other.
- **Coordinates:** pixel :math:`(x, y)` is column :math:`x` and row :math:`y` of the image, counted
  from 0, with :math:`y` growing downward. Distances are in pixels, between pixel centres; the
  physical pixel size is not used. Anisotropy (files only) scales each pixel's position by the
  factors before distances and angles are taken (see :ref:`conventions-units`).
- **Contour:** the ROI pixels with a 4-neighbour outside the ROI, as Nyxus's contour tracer
  returns them: one closed walk around the outside of each separate part of the ROI and one around
  each hole. Each walk is a border, listed in walking order, and every contour pixel is listed
  once.
- **Whole-slide mode:** the contour is not traced. It is the four points :math:`(0, 0)`,
  :math:`(W, 0)`, :math:`(W, H)` and :math:`(0, H)` of a :math:`W \times H` image, one pixel past
  the last column and row, a known defect (`#493 <https://github.com/PolusAI/nyxus/issues/493>`__).
  The centre found from them is the pixel equally far from all four when :math:`W` and :math:`H`
  are even, :math:`(W/2, H/2)`, half a pixel right of and below the middle of the image.
- **Centre:** the ROI pixel :math:`O` with the smallest spread of squared distances to the contour,
  the largest squared distance minus the smallest. The largest and smallest distances are found
  border by border. A border of at most 10 pixels is checked point by point; a longer one by a
  coarse-to-fine search along its walk, not by checking every contour point, and on many ROIs that
  search returns a smaller maximum or a larger minimum than the true ones. When several pixels
  share the smallest spread, the first one in the order the pixels were read wins: column by column
  for an array given to ``Nyxus.featurize``; tile by tile, and row by row within each tile, for an
  image read from a file.
- **Radius:** :math:`R = \sqrt{\max_c |O - c|^2}` over the contour points :math:`c`, with the same
  search.
- **Rings:** pixel :math:`p` at distance :math:`d_p = |p - O|` from the centre goes to ring
  :math:`k_p = \min(\lfloor 7\, d_p / R \rfloor, 7)`. Rings 0 to 6 are 7 rings of width
  :math:`R/7`; ring 7 holds every pixel at distance :math:`R` or more. The number of rings, 8, is
  fixed. The index is computed in floating point, so a pixel exactly on a ring boundary can fall
  into the ring below.
- **Wedges:** pixel :math:`p` goes to wedge :math:`j_p = \lfloor \theta_p / 45^\circ \rfloor`, where
  :math:`\theta_p = \operatorname{atan2}(y_p - y_O,\; x_p - x_O)` taken in :math:`[0^\circ, 360^\circ)`.
  Wedge 0 starts in the direction of growing :math:`x` and the wedges follow each other towards
  growing :math:`y`, clockwise as the image is displayed. The centre pixel itself lies in wedge 0.
- **Units:** ``FRAC_AT_D`` and ``RADIAL_CV`` are dimensionless. ``MEAN_FRAC`` is an intensity in
  grey levels.
- **Intensity domain:** the features are computed on grey levels and not converted back to the
  image's units (see :ref:`intensity-units`). ``FRAC_AT_D`` does not read intensities at all.
- **Small constant:** the three feature divisions, by :math:`N`, by :math:`n_k` and by
  :math:`\mu_k`, add :math:`\varepsilon = 10^{-9}` to their denominator. The constant is fixed in
  the code; ``--tinyval`` does not change it. An empty ring therefore reports 0 in all three
  features.
- **ROIs of one intensity** are computed like any other ROI.
- **One-pixel ROI:** the pixel is its own contour and the centre, and :math:`R = 0`; ring 0 holds
  the pixel, so ``FRAC_AT_D_0`` is 1, ``MEAN_FRAC_0`` is its intensity and ``RADIAL_CV_0`` is
  :math:`\sqrt{7}` when that intensity is not 0. The other 21 columns are 0.
- **IBSI mode:** ``--ibsi=true`` does not change these features.


Worked example
^^^^^^^^^^^^^^

Take a 47-pixel ROI in a 7 × 9 box at the top-left corner of the image. Every non-zero pixel below
belongs to the ROI, and every zero is background:

.. math::

   \textbf{I} = \begin{bmatrix}
   0&0&3&4&4&3&0&0&0\\
   0&3&5&6&6&5&3&2&0\\
   2&4&6&8&9&7&5&3&0\\
   3&5&7&9&9&8&6&4&2\\
   2&4&6&8&8&7&5&3&2\\
   0&3&4&6&6&5&4&2&0\\
   0&0&2&3&3&0&0&0&0 \end{bmatrix}

The intensities sum to 224. The contour is one border of 20 points, the 20 ROI pixels with a
4-neighbour outside the ROI. The centre is :math:`O = (4, 3)`, intensity 9: its squared distances
to the contour run from 5 to 17, a spread of 12, the smallest of all pixels; the next best,
:math:`(4, 2)`, has 16. The radius is :math:`R = \sqrt{17} = 4.123`. The ring and the wedge of each
pixel are (· is background):

.. math::

   k = \begin{bmatrix}
   \cdot&\cdot&6&5&5&5&\cdot&\cdot&\cdot\\
   \cdot&6&4&3&3&3&4&6&\cdot\\
   7&5&3&2&1&2&3&5&\cdot\\
   6&5&3&1&\mathbf{0}&1&3&5&6\\
   7&5&3&2&1&2&3&5&7\\
   \cdot&6&4&3&3&3&4&6&\cdot\\
   \cdot&\cdot&6&5&5&\cdot&\cdot&\cdot&\cdot \end{bmatrix}
   \qquad
   j = \begin{bmatrix}
   \cdot&\cdot&5&5&6&6&\cdot&\cdot&\cdot\\
   \cdot&4&5&5&6&6&7&7&\cdot\\
   4&4&4&5&6&7&7&7&\cdot\\
   4&4&4&4&\mathbf{0}&0&0&0&0\\
   3&3&3&3&2&1&0&0&0\\
   \cdot&3&3&2&2&1&1&0&\cdot\\
   \cdot&\cdot&2&2&2&\cdot&\cdot&\cdot&\cdot \end{bmatrix}

No pixel lies farther than :math:`R` from the centre, and three lie exactly at :math:`R`, so ring 7
holds 3 pixels. Per ring:

.. list-table::
   :widths: 7 7 8 34 14 14 16
   :header-rows: 1

   * - Ring
     - :math:`n_k`
     - :math:`S_k`
     - Wedge sums :math:`W_{k0}, \dots, W_{k7}`
     - ``FRAC_AT_D``
     - ``MEAN_FRAC``
     - ``RADIAL_CV``
   * - 0
     - 1
     - 9
     - 9, 0, 0, 0, 0, 0, 0, 0
     - 0.02128
     - 9
     - 2.646
   * - 1
     - 4
     - 34
     - 8, 0, 8, 0, 9, 0, 9, 0
     - 0.08511
     - 8.5
     - 1.003
   * - 2
     - 4
     - 30
     - 0, 7, 0, 8, 0, 8, 0, 7
     - 0.08511
     - 7.5
     - 1.004
   * - 3
     - 12
     - 69
     - 11, 5, 12, 6, 13, 6, 11, 5
     - 0.2553
     - 5.75
     - 0.3709
   * - 4
     - 4
     - 16
     - 0, 4, 0, 4, 0, 5, 0, 3
     - 0.08511
     - 4
     - 1.031
   * - 5
     - 11
     - 40
     - 7, 0, 6, 4, 9, 4, 7, 3
     - 0.2340
     - 3.636
     - 0.5292
   * - 6
     - 8
     - 20
     - 4, 0, 2, 3, 6, 3, 0, 2
     - 0.1702
     - 2.5
     - 0.7483
   * - 7
     - 3
     - 6
     - 2, 0, 0, 2, 2, 0, 0, 0
     - 0.06383
     - 2
     - 1.291

For ring 5, ``FRAC_AT_D_5`` is :math:`11/47 = 0.2340` and ``MEAN_FRAC_5`` is :math:`40/11 = 3.636`.
The wedge sums have mean :math:`40/8 = 5` and population variance :math:`56/8 = 7`, so
``RADIAL_CV_5`` is :math:`\sqrt{7}/5 = 0.5292`.

The *Example* value of each feature in the :ref:`index <radial-index>` is its first column, ring 0.
The test suite runs this example and checks all 24 columns.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/radial_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features=FRAC_AT_D,MEAN_FRAC,RADIAL_CV \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`N`
     - Number of pixels in the ROI.
   * - :math:`I_p`
     - Intensity of ROI pixel :math:`p`, in grey levels.
   * - :math:`C`
     - The contour points: the ROI pixels with a 4-neighbour outside the ROI, grouped by
       border.
   * - :math:`O = (x_O, y_O)`
     - The centre pixel.
   * - :math:`R`
     - The radius, the largest distance from :math:`O` to a contour point that the search finds.
   * - :math:`d_p,\; \theta_p`
     - Distance and angle of pixel :math:`p` from the centre, :math:`d_p = |p - O|` and
       :math:`\theta_p = \operatorname{atan2}(y_p - y_O, x_p - x_O) \in [0, 2\pi)`.
   * - :math:`k_p,\; j_p`
     - Ring and wedge of pixel :math:`p`: :math:`k_p = \min(\lfloor 7 d_p / R \rfloor, 7)`,
       :math:`j_p = \lfloor 4 \theta_p / \pi \rfloor`.
   * - :math:`n_k`
     - Number of ROI pixels in ring :math:`k`.
   * - :math:`S_k`
     - Sum of the intensities of the pixels in ring :math:`k`.
   * - :math:`W_{kj}`
     - Sum of the intensities of the pixels in ring :math:`k` and wedge :math:`j`.
   * - :math:`\varepsilon`
     - :math:`10^{-9}`, added to the denominator of each of the three features.


Settings
--------

Only anisotropy changes the centre, the radius, the rings or the wedges. The 8 rings and 8 wedges
are fixed in the code.

.. list-table::
   :widths: 16 28 26 30
   :header-rows: 1

   * - Setting
     - Command line
     - Python
     - Effect
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus(..., ibsi=True)``
     - No effect on these features.
   * - Small constant
     - ``--tinyval``
     - not available
     - No effect; the features use their own :math:`\varepsilon = 10^{-9}`.
   * - Image intensity map
     - ``--fpimgdr``, ``--fpimgmin``, ``--fpimgmax``, ``--preserve-hu``
     - ``Nyxus(..., preserve_hu=True)`` and the matching keywords
     - Decide how a file's intensities become grey levels (see :ref:`intensity-units`).
       ``MEAN_FRAC`` follows the grey levels; a map with an offset also changes ``RADIAL_CV``.
       ``FRAC_AT_D`` does not change.
   * - Anisotropy
     - ``--anisox``, ``--anisoy``
     - ``Nyxus(..., anisotropy_x=..., anisotropy_y=...)``
     - Files only. The pixels are placed at their positions scaled by the factors, so all three
       features change. Equal factors on both axes change nothing.
   * - Memory limit
     - ``--ramLimit``
     - ``set_params(ram_limit=...)``
     - Decides whether a ROI is processed out-of-core, which, without anisotropy, changes the values
       of most ROIs (see :ref:`radial-possible-issues`).

All options are listed in the :ref:`cli-reference`.


Output columns
--------------

Each code produces 8 columns, one per ring, named with the code and the ring number:
``FRAC_AT_D_0`` to ``FRAC_AT_D_7``, ``MEAN_FRAC_0`` to ``MEAN_FRAC_7`` and ``RADIAL_CV_0`` to
``RADIAL_CV_7``, 24 in all. Ring 0 is the innermost; ring 7 holds the pixels at distance :math:`R`
or more. The names are the same in CSV, Arrow and Parquet output and in the tables Python returns.
There are no ``_AVE`` columns.


.. _radial-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example, for the first column, ring 0.

.. include:: radial_index.inc


Features
--------

.. _feature-frac-at-d:

Fraction of pixels in a ring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``FRAC_AT_D`` (columns ``FRAC_AT_D_0`` to ``FRAC_AT_D_7``)
:Range: :math:`[0, 1)`, dimensionless; the 8 columns sum to :math:`N/(N+\varepsilon)`, 1 up to
   :math:`10^{-9}/N`

The share of the ROI's pixels that fall in ring :math:`k`. It depends on the mask only: the
intensities are never read.

.. math::

   \mathrm{FRAC\_AT\_D\_}k = \frac{n_k}{N + \varepsilon}

**Relations.** :math:`N \cdot` ``FRAC_AT_D_k`` is the whole number :math:`n_k`, up to rounding.
:math:`\sum_k N \cdot \mathrm{FRAC\_AT\_D\_}k \cdot \mathrm{MEAN\_FRAC\_}k` is the ROI's total
intensity, 224 on the worked example.

**Possible issues.** CellProfiler's ``FracAtD`` is a fraction of the ROI's intensity; see
:ref:`radial-other-tools`.


.. _feature-mean-frac:

Mean intensity in a ring
^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``MEAN_FRAC`` (columns ``MEAN_FRAC_0`` to ``MEAN_FRAC_7``)
:Range: between the smallest and the largest intensity of the ROI for a non-empty ring, 0 for an
   empty ring; grey levels

The mean intensity of the pixels in ring :math:`k`. Despite its name it is not a fraction and is
not normalized by the ROI's mean intensity.

.. math::

   \mathrm{MEAN\_FRAC\_}k = \frac{S_k}{n_k + \varepsilon}

``MEAN_FRAC_0`` is the intensity of the centre pixel whenever ring 0 holds the centre alone, which
is the case for every ROI with :math:`R \le 7` without anisotropy.

**Possible issues.** CellProfiler's ``MeanFrac`` divides the ring's share of the intensity by its
share of the pixels, and is close to 1; see :ref:`radial-other-tools`.


.. _feature-radial-cv:

Coefficient of variation across the wedges of a ring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``RADIAL_CV`` (columns ``RADIAL_CV_0`` to ``RADIAL_CV_7``)
:Range: :math:`[0, \sqrt{7}]`, dimensionless

How unevenly ring :math:`k`'s intensity is spread over the 8 wedges: the population standard
deviation of the 8 wedge sums divided by their mean. All 8 wedges are used, including those that
hold no pixel of the ring.

.. math::

   \mu_k = \frac{1}{8} \sum_{j=0}^{7} W_{kj},
   \qquad
   \sigma_k = \sqrt{\frac{1}{8} \sum_{j=0}^{7} \left(W_{kj} - \mu_k\right)^2},
   \qquad
   \mathrm{RADIAL\_CV\_}k = \frac{\sigma_k}{\mu_k + \varepsilon}

A ring whose intensity lies in one wedge reports :math:`\sqrt{7} = 2.646`, the largest value; a
ring with equal sums in all 8 wedges reports 0. ``RADIAL_CV_0`` is :math:`\sqrt{7}` whenever ring
0 holds the centre alone and the centre's intensity is not 0. A ring whose intensities are all 0,
and an empty ring, report 0.

**Possible issues.** Because the sums are taken over all 8 wedges, a ring that covers only part of
the circle, as rings near the edge of a non-round ROI do, reports a large value even when its
intensity is uniform: the 5 × 5 square of one intensity reports 1 in ring 7, whose 4 pixels, the
corners, fill 4 of the 8 wedges. CellProfiler
averages wedge means over the occupied wedges only; see :ref:`radial-other-tools`.


.. _radial-possible-issues:

Possible issues
---------------

The centre and the radius come from an approximate search
   On a border of more than 10 pixels the largest and smallest distances to the contour are found
   by stepping along the border's walk and refining around the best step, which assumes the
   distance rises and falls only once along it. Even on a convex ROI the distance to the boundary
   has more than one minimum. Against a full scan of the same contour, the search changes the
   centre or the radius on 191 of 300 random ellipses, 75 of 108 random ROIs without holes and
   1232 of 1388 random ROIs with holes. On a 5 × 4 rectangle in the example script, the search
   finds :math:`R^2 = 5` where the farthest contour point is at 8, so 2 pixels lie beyond
   :math:`R`, ring 7 holds 8 pixels instead of 2, and ``FRAC_AT_D_2`` is 0 instead of 0.2. The
   worked example is not affected.

Ring 7 collects everything at or beyond R
   The ring index is scaled by 7, not 8: rings 0 to 6 share the distances below :math:`R`, and
   ring 7 receives the pixels at :math:`R` or farther. No ROI pixel lies farther from the centre
   than the farthest contour pixel, so with the exact radius ring 7 holds only the pixels at
   :math:`R`: 3 of the 47 pixels on the worked example. Where the search stops short, the pixels
   beyond :math:`R` join them.

Ties between centre candidates depend on the pixel order
   When several pixels share the smallest spread, the first one read wins, and an array given to
   ``Nyxus.featurize`` is read in a different order from the same image read from a file. For an
   11-pixel ROI with two tied candidates, ``MEAN_FRAC_0`` is 9 for the array and 2 for the same
   image read from a file.

Out-of-core ROIs
   Without anisotropy, an out-of-core ROI keeps its contour relative to its bounding box but its
   pixels in image coordinates, a known defect (`#509 <https://github.com/PolusAI/nyxus/issues/509>`__). A ROI whose bounding box does not start at the image's first row
   and column gets different values: the worked example moved 2 pixels right and 3 down reports
   ``FRAC_AT_D_7`` 0.5532 out-of-core instead of 0.06383, and ``MEAN_FRAC_0`` 3 instead of 9. A ROI
   whose bounding box starts there is placed correctly, but the out-of-core contour is a single
   loop that can list its points in a different order, which changes the approximate searches:
   the worked example at the origin gets the in-memory values, while 2 of 15 random ellipses at
   the origin did not. With anisotropy an out-of-core ROI is measured as in memory and gets the
   in-memory values.

Small ROIs
   A ROI of 1 pixel has :math:`R = 0` and puts its pixel in ring 0. A ROI of 2 pixels puts one
   pixel in ring 0 and the other, at :math:`R`, in ring 7.

Intensity domain
   ``MEAN_FRAC`` is reported in grey levels, not in the image's units. An array with negative
   values given to ``Nyxus.featurize`` is shifted up by its minimum first, which shifts
   ``MEAN_FRAC`` and changes ``RADIAL_CV`` (see :ref:`intensity-units`).

.. _radial-other-tools:

Other tools
   CellProfiler's ``MeasureObjectIntensityDistribution``, with 8 scaled bins, differs at every
   step. Its centre is the pixel farthest from the background; its distances from the centre are
   geodesic, measured inside the object; each pixel's radial position is
   :math:`d_c / (d_c + d_e + 0.001)`, with :math:`d_e` its own distance to the edge, so the 8 rings
   follow the object's shape; and the ring index is scaled by 8. ``FracAtD`` is the ring's share of
   the object's intensity, ``MeanFrac`` is ``FracAtD`` divided by the ring's share of the pixels,
   and ``RadialCV`` is the coefficient of variation of the mean intensities of the occupied
   octants. On the worked example, CellProfiler 4.2.8 takes :math:`(3, 3)` as the centre, puts 1, 1,
   4, 7, 4, 15, 15 and 0 pixels in its rings, and reports:

   .. list-table::
      :widths: 8 15 15 15 15 16 16
      :header-rows: 1

      * - Ring
        - ``FRAC_AT_D``
        - ``FracAtD``
        - ``MEAN_FRAC``
        - ``MeanFrac``
        - ``RADIAL_CV``
        - ``RadialCV``
      * - 0
        - 0.02128
        - 0.04018
        - 9
        - 1.888
        - 2.646
        - 0
      * - 1
        - 0.08511
        - 0.04018
        - 8.5
        - 1.888
        - 1.003
        - 0
      * - 2
        - 0.08511
        - 0.1429
        - 7.5
        - 1.679
        - 1.004
        - 0.08839
      * - 3
        - 0.2553
        - 0.2009
        - 5.75
        - 1.349
        - 0.3709
        - 0.1545
      * - 4
        - 0.08511
        - 0.1161
        - 4
        - 1.364
        - 1.031
        - 0.06281
      * - 5
        - 0.2340
        - 0.2812
        - 3.636
        - 0.8812
        - 0.5292
        - 0.2079
      * - 6
        - 0.1702
        - 0.1786
        - 2.5
        - 0.5595
        - 0.7483
        - 0.2041
      * - 7
        - 0.06383
        - 0
        - 2
        - 0
        - 1.291
        - 0

   All 24 values differ by more than 1%. No setting makes Nyxus compute CellProfiler's
   quantities.


Validation
----------

The three features are regression-only. The test suite pins all 24 columns at a relative
:math:`10^{-9}` on a 26-pixel concave ROI with a hole, and checks properties that hold for any
definition of the rings: ``FRAC_AT_D`` sums to 1, an empty ring is 0 in all three features, and
``RADIAL_CV`` stays within :math:`\sqrt{7}`. CellProfiler ``MeasureObjectIntensityDistribution``
was run on the same ROI as a candidate reference and disagrees, for the reasons under
:ref:`radial-other-tools`; it validates none of the features. The worked-example script asserts
the values of the worked example and that the page's full-scan formulas reproduce them, the 5 × 5
and 3 × 3 squares, the 5 × 4 rectangle (:math:`R^2 = 5` against 8, 2 pixels beyond :math:`R`), the
ROIs of 1 and 2 pixels, the tie between centre candidates, the out-of-core values of the moved and
unmoved worked example and the whole-slide rings. It does not assert the CellProfiler values or
the counts over random ROIs (search changes, out-of-core differences at the origin); these were
measured separately.


References
----------

1. Carpenter AE, Jones TR, Lamprecht MR, et al. *CellProfiler: image analysis software for
   identifying and quantifying cell phenotypes.* Genome Biology. 2006;7(10):R100.
   doi:10.1186/gb-2006-7-10-r100
2. McQuin C, Goodman A, Chernyshev V, et al. *CellProfiler 3.0: next-generation image processing
   for biology.* PLoS Biology. 2018;16(7):e2005970. doi:10.1371/journal.pbio.2005970
