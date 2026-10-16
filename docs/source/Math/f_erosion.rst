Erosion features
================

Erosion features count how many times a ROI's mask can be eroded with a small cross-shaped
structuring element before nothing of it is left. Thick, compact ROIs survive more erosions than
thin or ragged ones, so the count is a rough measure of the ROI's thickness in pixels.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``EROSIONS_2_VANISH`` and ``EROSIONS_2_VANISH_COMPLEMENT``, one value per ROI.
       ``EROSIONS_2_VANISH_COMPLEMENT`` is never computed and is always 0.
   * - **Group**
     - ``*ALL_MORPHOLOGY*``; ``*ALL*`` includes it.
   * - **Dimensions**
     - 2D only. There is no 3D erosion family.
   * - **GPU**
     - Yes, in GPU builds with ``--useGpu=true``, for ROIs computed in memory with a label image,
       but only when a moment feature is in the same run: without one every ROI reports 0 (see
       :doc:`/gpu`), a known defect (`#520 <https://github.com/PolusAI/nyxus/issues/520>`__). The GPU uses the same structuring element, the same pixel
       range and the same cap.
   * - **Out-of-core**
     - Yes, with the same values as in memory on an image that Nyxus reads as one tile. On other
       images out-of-core erosion raises an error or returns a wrong value, depending on the ROI; see
       :ref:`large-images-tiles`.
   * - **IBSI**
     - Neither feature has an IBSI identifier; IBSI defines no erosion count.

.. tip::

   A ROI of at least 4 × 4 pixels that fills its bounding box, such as a solid rectangle, reports
   1000, the cap, and so does
   any ROI that touches the edges of its box broadly enough. The count is best read as a
   comparison between ROIs that do not fill their boxes; see :ref:`erosion-possible-issues`.


How the features are computed
-----------------------------

Nyxus copies the ROI's mask into a binary image the size of the ROI's tight bounding box: 1 at
every ROI pixel, 0 elsewhere. It then erodes this image repeatedly and stops as soon as an erosion
leaves no pixel set inside the area it updates.

- **Which pixels:** the ROI's pixels, as a set. Their intensities are not used, except to skip
  ROIs of a single intensity (below). Pixels outside the bounding box are not part of the image:
  nothing pads the box with background.
- **Structuring element:** the 3 × 3 cross, the pixel and its 4 neighbours above, below, left and
  right. A pixel stays set only if all five pixels under the cross are set (a 4-connected
  erosion).
- **Updated area:** with :math:`H` rows and :math:`W` columns in the box, numbered from 0, only
  pixels in rows 2 to :math:`H-2` and columns 2 to :math:`W-2` are eroded. The first two rows and
  columns and the last row and column of the box keep their initial values in every iteration,
  and still count as neighbours of the pixels next to them. This is a known defect (`#501 <https://github.com/PolusAI/nyxus/issues/501>`__). A box of 3 or fewer pixels in either
  direction has no updated pixel at all.
- **Count:** the loop starts at :math:`k = 0`. Each pass erodes the updated area once and counts
  the pixels left set in it. If none are left, the loop stops and reports :math:`k`; otherwise
  :math:`k` grows by 1. The value is therefore the number of erosions performed before the one that
  empties the updated area, one less than the number of erosions performed.
- **Cap:** the loop runs at most 1000 times. A ROI whose updated area never empties reports 1000.
- **ROIs of a single intensity:** a ROI whose pixels all have the same intensity is skipped and
  reports 0, whatever its shape, a known defect (`#502 <https://github.com/PolusAI/nyxus/issues/502>`__). A single-pixel ROI is such a ROI.
- **Units:** a count of erosions, dimensionless. It does not depend on the pixel size, the
  intensities or their units, or ``--coarseGrayDepth``.

Outside the frozen border the erosion is the textbook binary erosion with a 4-connected cross.
``scipy.ndimage.binary_erosion`` reproduces the value exactly when it is given the cross as
``structure`` and a ``mask`` that allows only rows and columns 2 to :math:`H-2` and
:math:`W-2` to change, and when the count is taken as above.


Worked example
^^^^^^^^^^^^^^

Take an irregular ROI of 46 pixels in an 8 × 9 box, with one hole:

.. math::

   \textbf{M} = \begin{bmatrix}
   0&0&1&1&1&0&0&0&0\\
   0&1&1&1&1&1&0&0&0\\
   1&1&1&1&1&1&1&0&0\\
   1&1&1&1&1&1&1&1&0\\
   1&1&1&0&1&1&1&1&1\\
   0&1&1&1&1&1&1&1&1\\
   0&0&1&1&1&1&1&0&0\\
   0&0&0&1&1&0&0&0&0 \end{bmatrix}

The updated area is rows 2 to 6 and columns 2 to 7. The first erosion leaves 17 of its pixels set,
the second 4 and the third none, so the loop stops at :math:`k = 2` and ``EROSIONS_2_VANISH`` is 2.
After the third erosion, 19 ROI pixels are still set, all in the rows and columns that are never
eroded.

Textbook erosion with the same cross and the box padded with background leaves 23, then 4, then 0
pixels: 3 erosions to vanish. Here the two differ only by the one-less count. The mirror image of
the ROI, however, reports 3, because its rows and columns that are never eroded hold up an
erosion more (see :ref:`erosion-possible-issues`).

The *Example* value of each feature in the :ref:`index <erosion-index>` is what Nyxus returns for
this ROI. The test suite runs this example and checks every one of them.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/erosion_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features=EROSIONS_2_VANISH,EROSIONS_2_VANISH_COMPLEMENT \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`H,\; W`
     - Number of rows and columns of the ROI's tight bounding box.
   * - :math:`M_k(r, c)`
     - Binary image of the box after :math:`k` erosions; :math:`M_0` is 1 at the ROI's pixels and
       0 elsewhere. Rows :math:`r` and columns :math:`c` are numbered from 0.
   * - :math:`V`
     - The updated area, :math:`\{(r, c) : 2 \le r \le H-2,\; 2 \le c \le W-2\}`.
   * - :math:`B`
     - The cross, :math:`\{(0,0), (-1,0), (1,0), (0,-1), (0,1)\}`.
   * - :math:`K`
     - The cap, 1000.


Settings
--------

The erosion features have no settings of their own. The structuring element and the cap of 1000
are fixed. ``--ibsi``, ``--coarseGrayDepth`` and the intensity map options do not change them.
Anisotropy (``--anisox``, ``--anisoy``; files only) resamples the ROI before it is eroded, so it
changes the value (see :ref:`conventions-units`). ``--useGpu``
moves the computation to the GPU (see :ref:`erosion-possible-issues`). All options are listed in
the :ref:`cli-reference`.

In whole-slide mode (``--segDir`` equal to ``--intDir``, or ``featurize_files(...,
single_roi=True)`` in Python) and without anisotropy, the ROI is the whole image, background included, and its bounding
box is one pixel larger than the image, a known defect (`#493 <https://github.com/PolusAI/nyxus/issues/493>`__; see
:ref:`large-images-whole`). The extra last row and
column are empty, so the image erodes away from the bottom and right, and the value is the
image's smaller side minus 3: 17 for a 20 × 20 image. Images of a single intensity are not
skipped in this mode.


Output columns
--------------

Each feature produces one column, named with its code: ``EROSIONS_2_VANISH`` and
``EROSIONS_2_VANISH_COMPLEMENT``. There are no ``_AVE`` columns.


.. _erosion-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example.

.. include:: erosion_index.inc


Features
--------

.. _feature-erosions-2-vanish:

Erosions to vanish
^^^^^^^^^^^^^^^^^^

:Code: ``EROSIONS_2_VANISH``
:Range: integers 0 to 1000, dimensionless

The number of erosions with the 3 × 3 cross that the ROI's mask survives inside its bounding box,
counted before the erosion that leaves no pixel set in the updated area. Larger values mean a
thicker ROI.

.. math::

   M_{k+1}(p) = \begin{cases}
   \min_{b \in B} M_k(p + b) & p \in V\\
   M_k(p) & p \notin V \end{cases}

.. math::

   \mathrm{EROSIONS\_2\_VANISH} = \min\left(\{\, k \ge 0 : M_{k+1}(p) = 0 \;\; \forall p \in V \,\}
   \cup \{K\}\right)

The value is 0 when the ROI has a single intensity (a known defect, `#502 <https://github.com/PolusAI/nyxus/issues/502>`__), and when the box is 3 or fewer pixels in
either direction.

On three shapes, Nyxus and textbook erosion with the same cross and background padding give:

.. list-table::
   :widths: 46 18 18 18
   :header-rows: 1

   * - ROI
     - Nyxus
     - Textbook, erosions to vanish
     - Textbook minus 1
   * - Solid 7 × 9 rectangle, filling its box
     - 1000
     - 4
     - 3
   * - Disk of radius 6 (13 × 13 box)
     - 6
     - 7
     - 6
   * - Irregular ROI of the worked example
     - 2
     - 3
     - 2

**Relations.** When the frozen rows and columns do not hold up the erosion, the value is the
textbook count of erosions to vanish, minus 1. This holds for disks of radius 3, 6 and 10 (3, 6
and 10).

**Possible issues.** The rows and columns that are never eroded, a known defect (`#501 <https://github.com/PolusAI/nyxus/issues/501>`__),
change the value for many shapes,
and make it depend on the ROI's orientation; see :ref:`erosion-possible-issues`.


.. _feature-erosions-2-vanish-complement:

Erosions to vanish, complement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:Code: ``EROSIONS_2_VANISH_COMPLEMENT``
:Range: always 0

Not computed. The column exists and is requested with ``*ALL_MORPHOLOGY*``, but no code assigns
it, so it keeps its initial value of 0 for every ROI. Requesting it runs the
``EROSIONS_2_VANISH`` computation and the convex hull, and produces nothing else.


.. _erosion-possible-issues:

Possible issues
---------------

``EROSIONS_2_VANISH_COMPLEMENT`` is always 0
   The feature is listed and written to the output, but never computed.

The edges of the bounding box are never eroded
   Rows 0 and 1, columns 0 and 1, the last row and the last column of the box keep their pixels in
   every iteration, and the pixels next to them lean on them as neighbours. This is a known defect
   (`#501 <https://github.com/PolusAI/nyxus/issues/501>`__). A ROI that fills its
   box therefore never vanishes and reports the cap: a solid 7 × 9 rectangle reports 1000, where
   textbook erosion vanishes after 4 erosions. Removing one corner pixel of the rectangle still
   reports 1000. Every filled rectangle of at least 4 × 4 reports 1000. Where the ROI only touches
   the edges of its box, the frozen pixels can still delay the erosion: a disk of radius 20
   reports 21, where the textbook count minus 1 is 20.

The value depends on orientation
   The frozen strip is two pixels wide at the top and left of the box and one pixel wide at the
   bottom and right, a known defect (`#501 <https://github.com/PolusAI/nyxus/issues/501>`__). The worked example reports 2; mirrored left-right, mirrored top-bottom or
   rotated by 180° it reports 3. A 4 × 4 square with one pixel sticking out at the top and one at
   the left reports 1000; rotated by 180°, so that they stick out at the bottom and right, it
   reports 2. Textbook erosion vanishes after 3 erosions in both orientations.

The count is one less than the erosions performed
   The value counts the erosions before the one that empties the updated area. Textbook
   definitions, and scikit-image loops that erode until nothing is left, count that last erosion
   too. A disk of radius 6 reports 6 and vanishes after 7 erosions.

Small boxes
   A box of 3 or fewer pixels in either direction has no pixel to erode, and the value is 0. A
   1 × 1, 2 × 5, 3 × 3 or 3 × 9 ROI reports 0, while textbook erosion needs 1 or 2 erosions; a
   filled 4 × 4 square reports 1000.

ROIs of a single intensity report 0
   The value depends only on the shape, but a ROI whose pixels all have one intensity is skipped
   and reports 0, a known defect (`#502 <https://github.com/PolusAI/nyxus/issues/502>`__). A solid rectangle that reports 1000 with varying intensities reports 0 when all
   its pixels are 5, and so does a disk of radius 6. ``Nyxus.featurize`` truncates fractional
   values, so an in-memory ROI whose values differ only in their fractions also counts as a ROI of
   a single intensity.

Whole-slide mode
   The ROI is the whole image and its bounding box is one pixel larger than the image, with an
   empty last row and column, a known defect (`#493 <https://github.com/PolusAI/nyxus/issues/493>`__). The value is the image's smaller side minus 3, whatever the
   intensities: 7 for a 10 × 10 image, 17 for a 20 × 20 or a 20 × 40 image, and 0 for a 3 × 9
   image. Images of a single intensity are not skipped, so the cap and the 0 of constant ROIs do
   not apply.

Constant ROIs on the GPU
   The GPU path does not skip ROIs of a single intensity; it erodes them like any other ROI.
   This has not been measured on a GPU build.


Validation
----------

``EROSIONS_2_VANISH`` is regression-only. On the test suite's 26-pixel shape, in a 7 × 6 box,
scikit-image also gives 1 when it erodes with a 3 × 3 square and counts every erosion, but Nyxus
erodes with a cross and counts one fewer, so that agreement confirms neither the structuring
element nor the counting convention. ``EROSIONS_2_VANISH_COMPLEMENT`` is regression-only, and its
regression value is 0. Out-of-core
results are checked against in-memory results of the same build. The Nyxus values quoted on this
page, including the whole-slide values, are pinned by the worked-example script; the textbook
counts and the per-iteration pixel counts quoted beside them are not.


References
----------

1. Soille P. *Morphological Image Analysis: Principles and Applications.* 2nd ed. Springer; 2003.
   doi:10.1007/978-3-662-05088-0
2. Virtanen P, et al. *SciPy 1.0: fundamental algorithms for scientific computing in Python.*
   Nature Methods. 2020;17:261–272. (``scipy.ndimage.binary_erosion``)
3. van der Walt S, et al. *scikit-image: image processing in Python.* PeerJ. 2014;2:e453.
