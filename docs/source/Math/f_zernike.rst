Zernike moment features
=======================

Zernike moments describe how a ROI's intensity is spread over a disk laid on the ROI. Each moment
projects the intensity onto one Zernike polynomial, a complex function of radius and angle on the
unit disk, and Nyxus reports the magnitude of each projection. Low orders capture coarse, broad
patterns and high orders fine radial and angular detail.

.. list-table::
   :widths: 26 74

   * - **Feature codes**
     - ``ZERNIKE2D``: one code, 30 columns per ROI, ``ZERNIKE2D_Z0`` to ``ZERNIKE2D_Z29``
   * - **Group**
     - ``*ALL*`` and ``*WHOLESLIDE*``. No other group includes it; request it by name.
   * - **Dimensions**
     - 2D only.
   * - **GPU**
     - No; computed on the CPU in every build.
   * - **Out-of-core**
     - Yes, with the same values as in memory on an image that Nyxus reads as one tile. On other
       images out-of-core processing of this family raises an error or returns wrong values,
       depending on the ROI (see :ref:`large-images-tiles`).
   * - **IBSI**
     - None. IBSI defines no Zernike features.

.. tip::

   The moments are weighted by intensity, not by shape. A ROI of one intensity, such as a binary
   image measured inside its own foreground, returns the ``--noval`` value, 0 by default, in every
   column (see :ref:`zernike-possible-issues`). The values also depend on the disk Nyxus chooses,
   which differs from CellProfiler's, so they cannot be compared with CellProfiler's
   ``ZernikeMagnitude`` features.


How the features are computed
-----------------------------

Nyxus takes the ROI's bounding box as an image in which pixels outside the mask hold 0. It lays a
unit disk on that image and sums the intensity-weighted Zernike polynomials over the pixels inside
the disk.

- **Which pixels:** every pixel of the bounding box whose centre lies inside the disk, at a
  normalized distance :math:`\rho \le 1` from the disk centre. A pixel outside the mask, or inside
  it with intensity 0, has weight 0. A zero pixel inside the mask still counts for the bounding
  box, and so for the radius, and it stops a ROI of otherwise one intensity from being treated as
  constant. A pixel
  whose centre coincides with the disk centre, :math:`\rho < 2.2 \times 10^{-16}`, is left out.
- **The disk:** its centre is the intensity centroid :math:`(\bar{x}, \bar{y})` of the bounding box
  image. Its radius :math:`R` is the smaller side of the bounding box, in pixels: the full side,
  not half of it. A compact ROI therefore fills only the inner part of the disk, and the ends of an
  elongated ROI can fall outside it.
- **Coordinates:** pixel :math:`(i, j)`, column :math:`i` and row :math:`j` counted from 1 within
  the bounding box, maps to :math:`x = (i - \bar{x})/R` and :math:`y = (j - \bar{y})/R`. The
  angle is :math:`\theta = \operatorname{atan2}(y, x)`, with :math:`y` growing downward.
  Anisotropy (files only) multiplies the column and row offsets, and the bounding-box sides that
  set :math:`R`, by the factors (see :ref:`conventions-units`).
- **Values:** each pixel weighs :math:`w = I/\sum I`, its intensity divided by the sum of the
  intensities of all ROI pixels, including those that fall outside the disk. No pixel-area factor
  is applied; the weights take its place.
- **Orders:** all moments with order :math:`n` from 0 to 9 and repetition :math:`m` from 0 to
  :math:`n` with :math:`n - m` even, 30 in all. The order 9 is fixed in the code.
- **Units:** dimensionless. Scaling the intensities by a constant does not change the values.
- **Intensity domain:** the moments are computed on grey levels and not converted back to the
  image's units (see :ref:`intensity-units`). A map that shifts the intensities, such as the
  offset applied to a CT image, changes the weights and so the values.
- **IBSI mode:** ``--ibsi=true`` does not change these features.


Worked example
^^^^^^^^^^^^^^

Take the 5 × 5 ROI of the texture pages, with every pixel in the mask:

.. math::

   \textbf{I} = \begin{bmatrix}
   5 & 2 & 5 & 4 & 4\\
   3 & 3 & 3 & 1 & 3\\
   2 & 1 & 1 & 1 & 3\\
   4 & 2 & 2 & 2 & 3\\
   3 & 5 & 3 & 3 & 2 \end{bmatrix}

The intensities sum to 70. The intensity centroid is :math:`\bar{x} = 204/70 = 2.9143` and
:math:`\bar{y} = 202/70 = 2.8857`, and the radius is :math:`R = 5`. The farthest pixel centre lies
at :math:`\rho = 0.59`, so all 25 pixels are inside the disk and none sits on the centre.

Because every weight is inside the disk, :math:`A_{00} = \frac{1}{\pi}\sum w = 1/\pi = 0.3183`.
Because the disk is centred on the intensity centroid, :math:`A_{11}` is the first moment about
the centroid and vanishes; Nyxus returns a rounding residue of order :math:`10^{-17}`.

.. list-table::
   :widths: 13 9 18 13 9 18 13 9 18
   :header-rows: 1

   * - Column
     - (n, m)
     - Value
     - Column
     - (n, m)
     - Value
     - Column
     - (n, m)
     - Value
   * - ``Z0``
     - (0, 0)
     - 0.3183
     - ``Z10``
     - (5, 3)
     - 0.03889
     - ``Z20``
     - (8, 0)
     - 0.5813
   * - ``Z1``
     - (1, 1)
     - 0
     - ``Z11``
     - (5, 5)
     - 0.0009361
     - ``Z21``
     - (8, 2)
     - 0.1047
   * - ``Z2``
     - (2, 0)
     - 0.6051
     - ``Z12``
     - (6, 0)
     - 0.3727
     - ``Z22``
     - (8, 4)
     - 0.2508
   * - ``Z3``
     - (2, 2)
     - 0.01349
     - ``Z13``
     - (6, 2)
     - 0.08806
     - ``Z23``
     - (8, 6)
     - 0.002560
   * - ``Z4``
     - (3, 1)
     - 0.009830
     - ``Z14``
     - (6, 4)
     - 0.1363
     - ``Z24``
     - (8, 8)
     - 0.004304
   * - ``Z5``
     - (3, 3)
     - 0.009666
     - ``Z15``
     - (6, 6)
     - 0.0003018
     - ``Z25``
     - (9, 1)
     - 0.08618
   * - ``Z6``
     - (4, 0)
     - 0.2363
     - ``Z16``
     - (7, 1)
     - 0.07493
     - ``Z26``
     - (9, 3)
     - 0.05453
   * - ``Z7``
     - (4, 2)
     - 0.04984
     - ``Z17``
     - (7, 3)
     - 0.06683
     - ``Z27``
     - (9, 5)
     - 0.009942
   * - ``Z8``
     - (4, 4)
     - 0.03165
     - ``Z18``
     - (7, 5)
     - 0.001380
     - ``Z28``
     - (9, 7)
     - 0.01636
   * - ``Z9``
     - (5, 1)
     - 0.03977
     - ``Z19``
     - (7, 7)
     - 0.002329
     - ``Z29``
     - (9, 9)
     - 0.0004792

The *Example* value in the :ref:`index <zernike-index>` is ``ZERNIKE2D_Z0``. The test suite runs
this example and checks all 30 columns.

.. tab-set::

   .. tab-item:: Python

      .. literalinclude:: ../../examples/zernike_worked_example.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

   .. tab-item:: Command line

      .. code-block:: console

         $ nyxus --features=ZERNIKE2D \
                 --intDir=int/ --segDir=seg/ --outDir=out/ --outputType=singlecsv


Notation
--------

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`I_{ij}`
     - Intensity of the bounding-box pixel in column :math:`i` and row :math:`j`, both counted
       from 1; 0 outside the mask.
   * - :math:`W,\; H`
     - Width and height of the ROI's bounding box, in pixels.
   * - :math:`\bar{x},\; \bar{y}`
     - Intensity centroid, :math:`\bar{x} = \sum i\,I_{ij} / \sum I_{ij}` and
       :math:`\bar{y} = \sum j\,I_{ij} / \sum I_{ij}`, over all bounding-box pixels.
   * - :math:`R`
     - Disk radius, :math:`R = \min(W, H)`.
   * - :math:`\rho_{ij},\; \theta_{ij}`
     - Polar coordinates of the pixel on the disk, from :math:`x = (i - \bar{x})/R` and
       :math:`y = (j - \bar{y})/R`: :math:`\rho = \sqrt{x^2 + y^2}`,
       :math:`\theta = \operatorname{atan2}(y, x)`.
   * - :math:`w_{ij}`
     - Pixel weight, :math:`w_{ij} = I_{ij} / \sum I_{ij}`.
   * - :math:`D`
     - The pixels used: those with :math:`\varepsilon \le \rho_{ij} \le 1`, where
       :math:`\varepsilon = 2.2 \times 10^{-16}` is the double-precision machine epsilon.
   * - :math:`n,\; m`
     - Order and repetition of a moment: :math:`0 \le m \le n \le 9`, :math:`n - m` even.
   * - :math:`R_{nm}(\rho)`
     - Zernike radial polynomial of order :math:`n` and repetition :math:`m`.
   * - :math:`A_{nm}`
     - Complex Zernike moment.


Settings
--------

Only anisotropy changes the disk. The weights and the order 9, and with it the 30 columns, are
fixed in the code.

.. list-table::
   :widths: 16 28 26 30
   :header-rows: 1

   * - Setting
     - Command line
     - Python
     - Effect
   * - Value for a constant ROI
     - ``--noval=0``
     - not available; always 0
     - Written to all 30 columns when every ROI pixel has the same intensity.
   * - IBSI mode
     - ``--ibsi=true``
     - ``Nyxus(..., ibsi=True)``
     - No effect on these features.
   * - Anisotropy
     - ``--anisox``, ``--anisoy``
     - ``Nyxus(..., anisotropy_x=..., anisotropy_y=...)``
     - Files only. The pixels are used as acquired, each at its position scaled by the factors, and
       :math:`R` is the smaller side of the bounding box in the same units, so the values change.
       Equal factors on both axes change nothing.
   * - Image intensity map
     - ``--fpimgdr``, ``--fpimgmin``, ``--fpimgmax``, ``--preserve-hu``
     - ``Nyxus(..., preserve_hu=True)`` and the matching keywords
     - Decide how a file's intensities become grey levels (see :ref:`intensity-units`). A map with
       an offset changes the weights and so the moments; a pure scaling does not.

All options are listed in the :ref:`cli-reference`.


Output columns
--------------

``ZERNIKE2D`` produces 30 columns, ``ZERNIKE2D_Z0`` to ``ZERNIKE2D_Z29``. Column ``ZERNIKE2D_Zk``
holds :math:`|A_{nm}|` for the :math:`k`-th pair :math:`(n, m)` taken in order of increasing
:math:`n`, then increasing :math:`m`, keeping pairs with :math:`n - m` even:

.. list-table::
   :widths: 12 88
   :header-rows: 1

   * - :math:`n`
     - Columns
   * - 0
     - ``Z0`` (0, 0)
   * - 1
     - ``Z1`` (1, 1)
   * - 2
     - ``Z2`` (2, 0), ``Z3`` (2, 2)
   * - 3
     - ``Z4`` (3, 1), ``Z5`` (3, 3)
   * - 4
     - ``Z6`` (4, 0), ``Z7`` (4, 2), ``Z8`` (4, 4)
   * - 5
     - ``Z9`` (5, 1), ``Z10`` (5, 3), ``Z11`` (5, 5)
   * - 6
     - ``Z12`` (6, 0), ``Z13`` (6, 2), ``Z14`` (6, 4), ``Z15`` (6, 6)
   * - 7
     - ``Z16`` (7, 1), ``Z17`` (7, 3), ``Z18`` (7, 5), ``Z19`` (7, 7)
   * - 8
     - ``Z20`` (8, 0), ``Z21`` (8, 2), ``Z22`` (8, 4), ``Z23`` (8, 6), ``Z24`` (8, 8)
   * - 9
     - ``Z25`` (9, 1), ``Z26`` (9, 3), ``Z27`` (9, 5), ``Z28`` (9, 7), ``Z29`` (9, 9)

Only non-negative repetitions are reported: :math:`|A_{n,-m}| = |A_{nm}|`. The phases are not
written.


.. _zernike-index:

Feature index
-------------

*Validation* names the independent implementation each feature is checked against in the test
suite, with its tolerance; see the :doc:`/featurelist` for what the categories mean.
*Example* is the value on the worked example, for ``ZERNIKE2D_Z0``.

.. include:: zernike_index.inc


Features
--------

.. _feature-zernike2d:

Zernike moments
^^^^^^^^^^^^^^^

:Code: ``ZERNIKE2D`` (columns ``ZERNIKE2D_Z0`` to ``ZERNIKE2D_Z29``)
:Range: :math:`[0, (n+1)/\pi]` for column :math:`(n, m)`, dimensionless; ``Z0`` lies in
   :math:`[0, 1/\pi]`

The magnitude of the projection of the ROI's normalized intensity onto the Zernike polynomial
:math:`V_{nm}(\rho, \theta) = R_{nm}(\rho)\, e^{i m \theta}`, over the unit disk on the intensity
centroid. Moments with :math:`m = 0` measure how the intensity is spread radially; moments with
:math:`m > 0` measure an :math:`m`-fold angular pattern.

.. math::

   R_{nm}(\rho) = \sum_{s=0}^{(n-m)/2} (-1)^s
   \frac{(n-s)!}{s!\,\left(\frac{n+m}{2}-s\right)!\,\left(\frac{n-m}{2}-s\right)!}\,\rho^{\,n-2s}

.. math::

   A_{nm} = \frac{n+1}{\pi} \sum_{(i,j) \in D} w_{ij}\, R_{nm}(\rho_{ij})\, e^{-i m \theta_{ij}},
   \qquad
   \mathrm{ZERNIKE2D\_Z}k = |A_{nm}|

The code evaluates :math:`R_{nm}` with a three-term recurrence in :math:`m` (Singh and Walia)
rather than with the factorial sum; the two agree to about :math:`10^{-13}` relative. The sign of
the imaginary part does not matter, since only the magnitude is reported.

**Relations.** :math:`\pi\,|A_{00}|` is the share of the ROI's intensity that falls inside the
disk, 1 when no pixel is left out. :math:`|A_{11}|` is 0, up to rounding, when no pixel is left
out, because the disk is centred on the intensity centroid. Rotating the ROI by a multiple of
90°, or mirroring it, leaves every value unchanged.

**Possible issues.** The values depend on the disk convention; see
:ref:`zernike-possible-issues`.


.. _zernike-possible-issues:

Possible issues
---------------

Intensity, not shape
   The weights are intensities. A ROI of one intensity returns the ``--noval`` value, 0 by
   default, in all 30 columns, and so does a binary image measured inside its own foreground. The
   same 0/1 image measured with a mask that also covers its background is not constant: a 3 × 5
   block of 1s in a 7 × 7 mask gives ``Z0`` 0.2971 and ``Z2`` 0.7873. Moments of the ROI's shape
   are not available from this feature. A pixel of intensity 0 inside the mask has weight 0, like
   a pixel outside it, but it still extends the bounding box, which sets the radius, and a ROI
   whose other pixels share one intensity is then computed rather than given the ``--noval``
   value. The worked example with its last row set to 0 gives ``Z2`` 0.5122 when that row is
   outside the mask and 0.6716 when it is inside.

The disk is larger than the ROI
   The radius is the full smaller side of the bounding box, not half of it, and not the radius of
   an enclosing circle. On the worked example the farthest pixel sits at :math:`\rho = 0.59`, so
   the outer part of the disk, where high-order polynomials vary most, holds no pixels.

Elongated ROIs lose their ends
   When the bounding box is much longer than wide, pixels farther than the short side from the
   centroid fall outside the disk. They are dropped from the sums but kept in :math:`\sum I`, so
   :math:`\pi\,|A_{00}| < 1` and :math:`|A_{11}| > 0`. For a ROI with a 3 × 9 bounding box in the
   example script, 5 of the 19 ROI pixels, holding 23 of the total intensity of 93, are dropped:
   ``Z0`` is :math:`(70/93)/\pi = 0.2396` and ``Z1`` is 0.09145.

A pixel on the disk centre is dropped
   A pixel whose centre coincides with the intensity centroid is skipped, although it belongs to
   every moment with :math:`m = 0`. For the symmetric 3 × 3 image with rows 1 2 1, 2 4 2, 1 2 1,
   the centre pixel holds a quarter of the intensity: ``Z0`` is :math:`0.75/\pi = 0.2387` instead
   of :math:`1/\pi`, and ``Z2`` is 0.5040 instead of 0.7427. Moments with :math:`m > 0` do not
   change, since :math:`R_{nm}(0) = 0` for them. Only ROIs whose centroid falls exactly on a pixel
   centre are affected.

Rotation by other angles
   The radius comes from the bounding box, which changes when a ROI is rotated by an angle that is
   not a multiple of 90°. The values are then not rotation invariant, even though the magnitudes of
   Zernike moments on a fixed disk are.

Intensity offsets
   The weights :math:`I/\sum I` do not change when the intensities are scaled, but they do when an
   offset is added. Adding 100 to the worked example changes ``Z2`` from 0.6051 to 0.6481. Images
   with negative values are shifted to non-negative grey levels before the moments are computed
   (see :ref:`intensity-units`).

Other tools
   CellProfiler's ``MeasureObjectIntensityDistribution`` centres the disk on the object's minimum
   enclosing circle and uses its radius, sums the raw intensities and divides by the number of
   pixels, without the :math:`(n+1)/\pi` factor. Its ``MeasureObjectSizeShape`` Zernikes use the
   binary mask on the same circle. ``mahotas.features.zernike_moments`` takes the centre and radius
   as arguments, keeps the pixel on the centre and divides by the intensity inside the disk instead
   of the ROI's total.


Validation
----------

The test suite checks all 30 columns against the closed form: the factorial sum for
:math:`R_{nm}` evaluated directly, at Nyxus's disk centre and radius, on a 26-pixel concave ROI with
a hole. They agree to a relative :math:`10^{-12}` plus an absolute :math:`10^{-15}`, which
matters only for ``Z1``, whose true value is 0; the largest relative difference measured is
:math:`3.4 \times 10^{-15}`. The check covers the moments given the disk; the disk convention
itself is pinned by separate tests and has no external reference. CellProfiler's Zernike
features are not comparable, for the reasons under :ref:`zernike-possible-issues`. Out-of-core
results are checked against in-memory results of the same build on an image that Nyxus reads as
one tile. Every value quoted in the worked example and under Possible issues is asserted by the
worked-example script; none is compared with a reference tool.


References
----------

1. Teague MR. *Image analysis via the general theory of moments.* Journal of the Optical Society of
   America. 1980;70(8):920–930. doi:10.1364/JOSA.70.000920
2. Khotanzad A, Hong YH. *Invariant image recognition by Zernike moments.* IEEE Transactions on
   Pattern Analysis and Machine Intelligence. 1990;12(5):489–497. doi:10.1109/34.55109
3. Singh C, Walia E. *Algorithms for fast computation of Zernike moments and their numerical
   stability.* Image and Vision Computing. 2011;29(4):251–259. doi:10.1016/j.imavis.2010.10.003
4. Boland MV, Markey MK, Murphy RF. *Automated recognition of patterns characteristic of
   subcellular structures in fluorescence microscopy images.* Cytometry. 1998;33(3):366–375.
