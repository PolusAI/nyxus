.. _conventions:

Conventions
===========

A few conventions run through every feature family: which pixels make up a ROI, how intensities
become grey levels, which pixels count as background, which units a value is in, what a ROI of one
intensity returns, and how entropies take their logarithm. This page states each convention once
and summarises, in one table per topic, how each family applies it, in 2D and in 3D. The family
pages give the details, worked examples and measured values; each row links to its page.

.. tip::

   Three rules explain most surprises:

   - A pixel of intensity 0 inside the mask is a ROI pixel. The intensity and shape families use
     it; several texture families treat it as background (see `Background`_).
   - The 2D texture families, and three of the 3D ones, bin intensities into 64 grey levels by
     default, counting from intensity 0, not from the ROI's minimum. IBSI mode turns the binning
     off (see `Grey levels`_ and `2D and 3D defaults`_).
   - Only the intensity families report values in the image's own units. Every other family that
     reads intensities works on the grey levels Nyxus stores (see `Intensity domain`_).


Example
-------

The script computes one intensity feature and two texture features of the 5 × 5 ROI of the texture
pages, of the same ROI with one pixel set to 0, and of a ROI of one intensity, with the default
settings and in IBSI mode:

.. literalinclude:: ../examples/conventions_example.py
   :language: python
   :start-after: # example-start
   :end-before: # example-end

Output:

.. code-block:: text

   ibsi=False  example         2.8000     0.3725  1654.7051
   ibsi=False  zero inside     2.7600     0.4282  1590.1715
   ibsi=False  constant        7.0000     0.0000     0.0000
   ibsi=True   example         2.8000     0.3725     9.9006
   ibsi=True   zero inside     2.7600     0.4282     9.9709
   ibsi=True   constant        7.0000     0.0000     0.0000

- The pixel set to 0 is a ROI pixel for ``MEAN``, which drops from 2.8 to 69/25 = 2.76 in both
  modes.
- ``GLDM_SDE`` leaves that pixel out in both modes: the matrix counts 24 pixels and the value rises
  from 0.3725 to 0.4282.
- ``GLRLM_HGLRE_AVE`` weights runs by the square of their grey level. The default binning maps
  intensities 1 to 5 to levels 13, 26, 39, 52 and 64, so the value is 1654.7; IBSI mode keeps the
  intensities as grey levels and gives 9.90.
- The ROI of one intensity has no texture: both texture features return the ``--noval``
  substitute, 0 in Python.


.. _conventions-roi:

The ROI
-------

A ROI is the set of pixels (voxels in 3D) of the label image that carry the ROI's label. Its
bounding box is the smallest rectangle (box) that holds them. The intensity of a pixel does not
decide whether it belongs to the ROI: a pixel of intensity 0 inside the mask is a ROI pixel, and a
pixel of the bounding box outside the mask is not.

Families that see only the ROI's pixels use every one of them, zeros included. Families that work
on the bounding box as an image fill the pixels outside the mask with 0, and then cannot tell them
apart from ROI pixels of intensity 0. The texture families are covered under `Background`_.

.. list-table::
   :header-rows: 1
   :widths: 26 37 37

   * - Family
     - Pixels it reads
     - A ROI pixel of intensity 0
   * - :doc:`Intensity <Math/f_pixel_intensity>`, :doc:`3D intensity <Math/f_intensity_3d>`
     - The ROI's pixels
     - Counted; lowers ``MIN`` and ``MEAN``
   * - :doc:`Intensity histogram <Math/f_intensity_histogram>` (2D only)
     - The ROI's pixels
     - Counted; widens the bins when it is the ROI's minimum
   * - :doc:`Morphology <Math/f_morphology>`, :doc:`3D shape <Math/f_morphology_3d>`,
       :doc:`erosion <Math/f_erosion>`, :doc:`fractal dimension <Math/f_fractal_dimension>`
     - The ROI's mask; intensities are not used, except by the few intensity-weighted morphology
       features
     - Counted as part of the shape. The chords treat it as a gap, and an erosion count is 0 for a
       ROI of one intensity
   * - :doc:`2D moments <Math/f_2dmoments>`
     - The ROI's pixels
     - Adds 1 to a shape moment and nothing to an intensity moment
   * - :doc:`Gabor <Math/f_gabor>`
     - The whole bounding box, with pixels outside the mask set to 0, and 0 beyond the box
     - The same as a pixel outside the mask
   * - :doc:`Zernike <Math/f_zernike>`
     - The bounding-box pixels inside the Zernike disk, weighted by intensity
     - Weight 0, like a pixel outside the mask, but it still extends the bounding box, which sets
       the radius, and a ROI whose other pixels share one intensity is then computed rather than
       given the ``--noval`` substitute
   * - :doc:`Image quality <Math/f_image_quality>`
     - The whole bounding box, with pixels outside the mask set to 0
     - The same as a pixel outside the mask
   * - Texture families
     - The bounding box or the ROI's pixels, by family
     - See `Background`_

In whole-slide mode (whole-volume mode in 3D) the whole image, background included, is one ROI,
but without anisotropy its bounding box is one pixel or voxel larger than the image on every axis;
see
:ref:`large-images-whole`. In 3D, ``Nyxus3D.featurize_files(..., single_roi=True)`` is not
whole-volume mode: it merges every nonzero label of the mask into one ROI, leaves out voxels where
the mask is 0, and uses the box of the remaining voxels; passing the volumes as their own masks
gives the box of the nonzero voxels (see :doc:`Math/f_morphology_3d`).


.. _conventions-grey-levels:

Grey levels
-----------

The texture families count grey levels, so they first turn intensities into a small set of integer
levels. Most of them share one rule, chosen by the sign of a grey-depth setting :math:`N`: the
global ``--coarseGrayDepth`` (Python ``coarse_gray_depth``, 64 by default) or a family's own
``greydepth`` metaparameter. With :math:`x_{\max}` and :math:`x_{\min}` the ROI's largest and
smallest intensity:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - :math:`N`
     - Grey level of intensity :math:`x`
     - Notes
   * - :math:`N > 0`, MATLAB-style (the default, 64)
     - :math:`\lfloor N x / x_{\max} \rfloor + 1`, capped at :math:`N`
     - The bins span 0 to :math:`x_{\max}`, not :math:`x_{\min}` to :math:`x_{\max}`, so a ROI
       whose intensities lie far from 0 uses only the top levels. Intensity 0 becomes level 1, an
       ordinary grey level.
   * - :math:`N < 0`, PyRadiomics-style
     - :math:`\lfloor (x - x_{\min}) / w \rfloor + 1`, capped at :math:`|N|`, with
       :math:`w = (x_{\max} - x_{\min}) / |N|`
     - :math:`|N|` equal-width bins over the ROI's own range. Intensity 0 stays level 0, the
       background level.
   * - :math:`N = 0`, or IBSI mode
     - :math:`x`
     - No binning: each integer intensity is a grey level. IBSI mode overrides every family's
       setting with 0, except the 2D GLCM constant-ROI test (see :ref:`glcm-possible-issues`).

The ``Nyxus`` and ``Nyxus3D`` constructors accept only a positive ``coarse_gray_depth`` and raise
``ValueError`` for 0 or a negative value. 0 and negative values are given on the command line, or,
for the 3D families that have one, through the family's ``greydepth`` metaparameter, set with
``set_metaparam``. IBSI mode is described on its own page, :ref:`ibsi-mode`.

Two families do not use the shared rule:

- **NGLDM**, 2D and 3D, maps intensity :math:`x` to :math:`\lfloor N x / x_{\max} \rfloor`, with
  :math:`N` the global ``--coarseGrayDepth``. Its levels run from 0 to :math:`N`, every intensity
  below :math:`x_{\max}/N` gets level 0, and a negative :math:`N` does not select PyRadiomics-style
  bins.
- The **intensity** and **intensity histogram** families bin only for their histograms, into
  equal-width bins from the ROI's minimum to its maximum: :math:`|N|` bins for ``ENTROPY`` and
  ``UNIFORMITY`` (the sign is ignored), :math:`N` bins for ``HISTOGRAM``, which writes no columns
  when :math:`N` is negative (a fix is coming), and :math:`N` bins for ``IH_*``, which
  return the ``--noval`` substitute when :math:`N < 2`. IBSI mode does not change these bins.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Family
     - 2D setting (default)
     - 3D setting (default)
   * - GLCM (:doc:`2D <Math/f_glcm>`, :doc:`3D <Math/f_glcm_3d>`)
     - ``--coarseGrayDepth`` (64). ``glcm/greydepth`` (Python) does not change the matrix; it
       changes only the constant-ROI test (see :ref:`glcm-possible-issues`).
     - ``--coarseGrayDepth`` or ``3glcm/greydepth`` (64)
   * - GLRLM (:doc:`2D <Math/f_glrlm>`, :doc:`3D <Math/f_glrlm_3d>`)
     - ``--coarseGrayDepth`` (64)
     - ``3glrlm/greydepth`` (0: no binning). ``--coarseGrayDepth`` has no effect.
   * - GLSZM (:doc:`2D <Math/f_glszm>`, :doc:`3D <Math/f_glszm_3d>`)
     - ``--coarseGrayDepth`` (64)
     - ``3glszm/greydepth`` (0: no binning). ``--coarseGrayDepth`` has no effect, and
       whole-volume mode ignores the metaparameter.
   * - GLDM (:doc:`2D <Math/f_gldm>`, :doc:`3D <Math/f_gldm_3d>`)
     - ``--coarseGrayDepth`` (64)
     - ``3gldm/greydepth`` (0: no binning). ``--coarseGrayDepth`` has no effect.
   * - GLDZM (:doc:`2D <Math/f_gldzm>`, :doc:`3D <Math/f_gldzm_3d>`)
     - ``--coarseGrayDepth`` (64). A negative value fails on ROIs with background in the
       bounding box (see :ref:`gldzm-possible-issues`).
     - ``--coarseGrayDepth`` (64); 0 turns binning off. No metaparameter.
   * - NGLDM (:doc:`2D <Math/f_ngldm>`, :doc:`3D <Math/f_ngldm_3d>`)
     - ``--coarseGrayDepth`` (64), NGLDM's own rule
     - ``--coarseGrayDepth`` (64), NGLDM's own rule. No metaparameter.
   * - NGTDM (:doc:`2D <Math/f_ngtdm>`, :doc:`3D <Math/f_ngtdm_3d>`)
     - ``--coarseGrayDepth`` (64)
     - ``3ngtdm/greydepth`` (0: no binning). ``--coarseGrayDepth`` has no effect.
   * - ``ENTROPY``, ``UNIFORMITY``, ``HISTOGRAM`` (:doc:`2D <Math/f_pixel_intensity>`),
       ``3ENTROPY``, ``3UNIFORMITY`` (:doc:`3D <Math/f_intensity_3d>`)
     - ``--coarseGrayDepth`` (64) histogram bins; the sign is ignored, except that ``HISTOGRAM``
       writes no columns for a negative value (a fix is coming)
     - ``--coarseGrayDepth`` (64) histogram bins; the sign is ignored
   * - :doc:`Intensity histogram <Math/f_intensity_histogram>`
     - ``--coarseGrayDepth`` (64) bins; computed only in IBSI mode
     - No 3D family

The other families, morphology, moments, Gabor, Zernike, erosion, fractal dimension and image
quality, do not bin. ``--coarseGrayDepth`` and IBSI mode do not change them.

Grey-level-weighted texture features, such as grey level variance or the high and low grey level
emphases, use the level's value, so they change with the grey-depth setting and with the
intensity scale. Compare them only between runs with the same setting; each texture page shows the
size of the effect under *Possible issues*.


.. _conventions-2d-3d:

2D and 3D defaults
------------------

The 3D texture families are not the 2D families with one more axis. With the default settings
they differ in these ways:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Topic
     - 2D
     - 3D
   * - Default binning
     - Every texture family bins into 64 levels.
     - GLCM, GLDZM and NGLDM bin into 64 levels. GLRLM, GLSZM, GLDM and NGTDM do not bin, and
       ``--coarseGrayDepth`` does not change them. Their default values equal their IBSI-mode
       values, except 3D NGTDM busyness, which IBSI mode computes over a row for every integer
       from 0 to the ROI maximum (see :doc:`Math/f_ngtdm_3d`).
   * - Directions
     - GLCM uses 0°, 45°, 90° and 135° (``--glcmAngles``); GLRLM always uses all four.
     - GLCM and GLRLM use the 13 directions of the 26-voxel neighbourhood; no setting selects
       them.
   * - What a plain code reports
     - ``GLCM_*`` and ``GLRLM_*`` have one column per angle, ``_0`` to ``_135``, plus ``_AVE``.
     - ``3GLCM_*`` and ``3GLRLM_*`` are one column: the direction :math:`(1,1,1)` only. ``_AVE``
       averages the 13 directions.
   * - Zones
     - GLSZM zones grow only east, south-east, south and south-west, so some 8-connected zones
       split (see :ref:`glszm-possible-issues`). GLDZM zones are 8-connected.
     - GLSZM and GLDZM zones are 26-connected.
   * - GLDZM distance
     - A walk along the pixel's row and column to the first pixel of grey level 0 or the box
       edge; a pixel next to background gets distance 2.
     - The city-block distance to the nearest voxel outside the ROI, as IBSI defines it.
   * - Background
     - GLCM skips every pair with a pixel of intensity 0.
     - GLCM counts pairs with voxels outside the mask, at grey level 1, with the default binning.
       NGTDM uses the voxels outside the mask in every mode. See `Background`_.
   * - Constant ROI
     - Every GLCM feature returns 0, whatever ``--noval`` is; a fix is coming.
     - Most 3D GLCM features return their limiting values when the ROI fills its bounding box. See
       `Constant ROIs and missing values`_.
   * - Neighbourhood size
     - Fixed at distance 1, except the GLCM offset, ``glcm/offset`` from Python.
     - Fixed at distance 1, except the NGTDM radius, ``3ngtdm/radius``.
   * - Voxel count
     - GLSZM's :math:`N_p` counts the bounding-box pixels with a non-zero grey level, which with
       the default binning is every pixel of the box.
     - GLSZM's :math:`N_v` counts the voxels of the mask.

The 3D family pages state these differences in their own *Settings* and *Possible issues*
sections, for example :ref:`glcm-3d-possible-issues` and :ref:`ngtdm-3d-possible-issues`.


.. _conventions-background:

Background
----------

The texture families build their matrices on the ROI's bounding box. Most of them decide what is
background by grey level, not by the mask: a pixel whose grey level is the background level, 0, is
left out, whether it lies outside the mask or inside it. The default MATLAB-style binning turns
intensity 0 into level 1, an ordinary level, so the pixels of the bounding box outside the mask
then take part as a level of their own. IBSI mode and PyRadiomics-style binning mostly keep them
at level 0 and leave them out. NGLDM, 2D and 3D, uses the mask and leaves the pixels outside it out
in every mode. GLDM, 2D and 3D, leaves out every pixel of intensity 0, so the pixels outside the
mask are out in every mode too, together with ROI pixels of intensity 0. 3D NGTDM uses the pixels
outside the mask in every mode. ROIs that fill their bounding box with
non-zero intensities are not affected.

.. list-table::
   :header-rows: 1
   :widths: 18 30 26 26

   * - Family
     - Pixels outside the mask, default settings
     - Pixels outside the mask, IBSI mode
     - ROI pixels of intensity 0
   * - :doc:`GLCM <Math/f_glcm>`
     - Pairs with them are not counted
     - Not counted
     - Not counted, in every mode
   * - :doc:`GLRLM <Math/f_glrlm>`
     - Level 1; they form runs
     - Excluded
     - As pixels outside the mask
   * - :doc:`GLSZM <Math/f_glszm>`
     - Level 1; they form zones and count in :math:`N_p`
     - Excluded
     - As pixels outside the mask
   * - :doc:`GLDM <Math/f_gldm>`
     - Excluded
     - Excluded
     - Excluded, in every mode
   * - :doc:`GLDZM <Math/f_gldzm>`
     - Level 1; they form zones, and the distance walks run through them to the box edge
     - Excluded; the distance walks stop at them
     - Default: level 1. IBSI: in no zone, stop the distance walks of their neighbours, and
       counted in :math:`N_v`
   * - :doc:`NGLDM <Math/f_ngldm>`
     - Excluded, as centres and as neighbours
     - Excluded
     - Centres at grey level 0
   * - :doc:`NGTDM <Math/f_ngtdm>`
     - Level 1; counted and averaged into their neighbours
     - Excluded, as pixels and as neighbours
     - As pixels outside the mask
   * - :doc:`3D GLCM <Math/f_glcm_3d>`
     - Level 1; their pairs are counted
     - Not counted
     - As voxels outside the mask
   * - :doc:`3D GLRLM <Math/f_glrlm_3d>`
     - Excluded (no binning). A positive ``3glrlm/greydepth`` makes them level 1, and they form
       runs.
     - Excluded
     - As voxels outside the mask
   * - :doc:`3D GLSZM <Math/f_glszm_3d>`
     - In no zone, and not counted in :math:`N_v`
     - The same
     - In no zone, but counted in :math:`N_v`. A positive ``3glszm/greydepth`` also leaves out
       every voxel of the lowest level.
   * - :doc:`3D GLDM <Math/f_gldm_3d>`
     - Excluded
     - Excluded
     - Excluded, in every mode. A positive ``3gldm/greydepth`` also leaves out every voxel of
       level 1.
   * - :doc:`3D GLDZM <Math/f_gldzm_3d>`
     - In no zone; outside the ROI for the distances
     - The same
     - Default: level 1. IBSI and PyRadiomics-style: every level of the ROI is raised by one, so
       the voxel keeps its zone.
   * - :doc:`3D NGLDM <Math/f_ngldm_3d>`
     - Excluded, as centres and as neighbours
     - Excluded
     - Centres at grey level 0
   * - :doc:`3D NGTDM <Math/f_ngtdm_3d>`
     - Level 1; visited and averaged into their neighbours. A positive ``3ngtdm/greydepth`` stops
       visiting them, but still averages them into their neighbours.
     - Level 1; visited and averaged into their neighbours
     - As voxels outside the mask

"Excluded" means that a pixel is neither counted nor used as a neighbour. Each family page shows
the effect on a mask that leaves pixels of the bounding box out, under *Possible issues*: for
example :ref:`glrlm-possible-issues` and :ref:`gldzm-possible-issues`.


.. _conventions-intensity-domain:

Intensity domain
----------------

Nyxus stores pixels as unsigned integer grey levels. When it reads an image from a file it maps the
stored values to grey levels and records the map: an integer image without negative values is used
as it is, an image with negative values is shifted up by its minimum, and a floating-point image is
rescaled to ``0 .. --fpimgdr`` grey levels. :ref:`intensity-units` gives the full rules. Arrays
passed to ``Nyxus.featurize`` are converted to unsigned integers, shifted up by their minimum when
they hold negative values and truncated, and no map is recorded for them.

Only the intensity families convert their values back through the map:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Family
     - Values are reported in
   * - :doc:`Intensity <Math/f_pixel_intensity>`, :doc:`3D intensity <Math/f_intensity_3d>`
     - The image's own units, for an image read from a file. Location features go through the
       whole map, spreads through its slope, variances through its square. In-memory arrays are
       reported in the stored grey levels.
   * - :doc:`Intensity histogram <Math/f_intensity_histogram>`
     - ``_VAL`` features: the image's own units, for an image read from a file. ``_IDX``
       features: bin numbers.
   * - Texture families
     - Grey levels after binning, or the stored grey levels in IBSI mode
   * - :doc:`2D moments <Math/f_2dmoments>` (intensity moments), the intensity-weighted
       :doc:`morphology <Math/f_morphology>` features, :doc:`Gabor <Math/f_gabor>`,
       :doc:`Zernike <Math/f_zernike>`, :doc:`image quality <Math/f_image_quality>`
     - The stored grey levels, without binning
   * - :doc:`Morphology <Math/f_morphology>`, :doc:`3D shape <Math/f_morphology_3d>`,
       :doc:`erosion <Math/f_erosion>`, :doc:`fractal dimension <Math/f_fractal_dimension>`
     - Do not depend on intensities

A map that shifts the intensities, such as the offset applied to a CT image, therefore changes
features that depend on the grey levels' distance from 0: the default texture binning, the Zernike
weights, the Gabor scores and the intensity moments. Floating-point images give grey-level-based
features that depend on ``--fpimgdr``, ``--fpimgmin`` and ``--fpimgmax``.


.. _conventions-units:

Units, pixel size and anisotropy
--------------------------------

Nyxus measures in pixels and voxels. It applies no physical pixel size to any feature.

- **Lengths** are in pixels (2D) or voxels (3D), **areas** in pixels, and **volumes** in cubic
  voxels. ``3AREA`` counts exposed voxel faces. Angles are in degrees, except the chord angles,
  which are in radians. Run lengths, zone sizes and zone distances count pixels or voxels,
  whatever the spacing.
- **Pixel size.** ``AREA_UM2`` is always 0. ``--pixelsPerCentimeter`` and the Python
  ``pixels_per_micron`` are accepted but change no feature. The ``phys_x``, ``phys_y`` and
  ``phys_z`` output columns report the file's pixel size (see :ref:`output`).
- **Anisotropy.** A scale factor per axis gives the size of a pixel or voxel along that axis,
  relative to the other axes. Geometric features use it: intensity and texture features are
  computed on the pixels or voxels as acquired and report the same values with or without it. A
  factor of 1 counts as unset.

.. list-table::
   :header-rows: 1
   :widths: 16 30 54

   * - Dimension
     - Settings
     - Effect
   * - 2D
     - ``--anisox``, ``--anisoy``; Python ``anisotropy_x``, ``anisotropy_y``
     - Files only: arrays passed to ``Nyxus.featurize`` are featurized without anisotropy. The
       geometric features are computed on the ROI resampled by nearest-neighbour sampling, in
       pixels of the resampled grid: with ``anisotropy_x=2``, ``AREA_PIXELS_COUNT``,
       ``BBOX_WIDTH`` and ``CENTROID_X`` are those of a ROI twice as wide. The intensity-weighted
       geometric features (``EDGE_*``, ``WEIGHTED_CENTROID_X`` and ``_Y``, ``MASS_DISPLACEMENT``,
       ``IMOM_*``, the radial distribution and ``ZERNIKE2D``) are computed on the pixels as
       acquired, each at its position scaled by the factors. The intensity, texture, Gabor and
       image quality features do not change, except ``POWER_SPECTRUM_SLOPE``, which is computed on
       the resampled ROI; a fix is coming.
   * - 3D
     - ``--anisox``, ``--anisoy``, ``--anisoz``; Python ``anisotropy_x``, ``anisotropy_y``,
       ``anisotropy_z``; or ``--use-physical-spacing=true`` (Python ``use_physical_spacing``)
     - The volume (3D or 2.5D) is featurized on its voxels as acquired, and the 3D
       shape features scale their coordinates by the factors: the volumes scale by the factors'
       product and the axis lengths are those of the scaled coordinates. The 3D intensity and
       texture features do not change. ``use_physical_spacing`` takes the factors from the OME
       ``PhysicalSize`` values, scaled so that the smallest is 1. A factor other than 1 in any
       ``anisotropy_*`` / ``--aniso*`` replaces the physical spacing on all three axes; the axes
       not given then use 1.

:ref:`volumetric` describes the 3D spacing options with an example.


.. _conventions-missing:

Constant ROIs and missing values
--------------------------------

A feature that cannot be computed, such as a division by zero or a texture matrix with a single
grey level, is reported as the ``--noval`` substitute, 0 by default. The Python classes have no
``noval`` option and always use 0. Values that are still NaN or infinite when the results are
written become ``--noval`` in CSV files and in pandas, and ``-0.0`` in Arrow and Parquet files, so
the tables hold no NaN (see :ref:`output`). On the command line, a ``--noval`` that no real value
can take, such as ``--noval=-999``, tells these cells apart from real zeros in CSV output. In Arrow
and Parquet files the values that are still NaN or infinite at output time are ``-0.0`` whatever
``--noval`` is, and some real values are also ``-0.0``, so these cells cannot be told apart from
zeros there.

A ROI whose pixels all have one intensity is the most common case:

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Family
     - 2D
     - 3D
   * - Intensity (:doc:`2D <Math/f_pixel_intensity>`, :doc:`3D <Math/f_intensity_3d>`)
     - Defined: the location features return the intensity, the dispersions 0 (``ENTROPY`` about
       :math:`-3 \times 10^{-16}`), ``UNIFORMITY`` 1; ``COVERED_IMAGE_INTENSITY_RANGE`` is the
       substitute when the image holds only this ROI
     - The same, with ``3ENTROPY`` and ``3COVERED_IMAGE_INTENSITY_RANGE``
   * - :doc:`Intensity histogram <Math/f_intensity_histogram>`
     - Every ``IH_*`` feature returns the substitute
     - No 3D family
   * - GLCM (:doc:`2D <Math/f_glcm>`, :doc:`3D <Math/f_glcm_3d>`)
     - Every feature returns 0, whatever ``--noval`` is; a fix is coming
     - Limiting values (``3GLCM_ASM`` 1, ``3GLCM_CONTRAST`` 0); ``3GLCM_CORRELATION`` and
       ``3GLCM_INFOMEAS1`` return the substitute (``-0.0`` in Arrow and Parquet)
   * - GLRLM, GLSZM, GLDM, GLDZM
     - Every feature returns the substitute, GLRLM's ``_AVE`` columns included
     - The same
   * - NGLDM (:doc:`2D <Math/f_ngldm>`, :doc:`3D <Math/f_ngldm_3d>`)
     - Every feature returns the substitute in memory; 0 out of core
     - The substitute, in memory and out of core
   * - NGTDM (:doc:`2D <Math/f_ngtdm>`, :doc:`3D <Math/f_ngtdm_3d>`)
     - Default binning: every feature returns the substitute. IBSI mode: coarseness and strength
       return the substitute (``-0.0`` in Arrow and Parquet), the other three 0
     - Default, MATLAB-style and PyRadiomics-style binning: the substitute. IBSI mode: as in 2D
   * - :doc:`Gabor <Math/f_gabor>`
     - Every column is 0, whatever ``--noval`` is
     - No 3D family
   * - :doc:`Zernike <Math/f_zernike>`
     - All 30 columns return the substitute
     - No 3D family
   * - :doc:`Erosion <Math/f_erosion>`
     - 0, whatever the ROI's shape, on the CPU. The GPU path erodes the ROI as usual
     - No 3D family
   * - :doc:`Image quality <Math/f_image_quality>`
     - ``MAX_SATURATION`` 1, ``MIN_SATURATION`` 0, ``POWER_SPECTRUM_SLOPE`` and ``SHARPNESS`` 0;
       the focus scores are not 0
     - No 3D family

The table is for a ROI that fills its bounding box. A constant ROI that does not fill it can gain
the pixels outside the mask as a second grey level, and its features are then computed: NGTDM in
2D with the default binning and in 3D, and 3D GLCM with the default binning (see `Background`_).
Image quality also reads the zeros outside the mask, so its values differ too. Morphology, 3D
shape, fractal dimension and the shape moments do not depend on intensity, and a constant ROI
changes nothing for them.


.. _conventions-entropy:

Entropy logarithm
-----------------

Every entropy is in bits, a base-2 logarithm. Two kinds of logarithm are in use:

- **A fast approximation** in the GLCM, GLRLM, GLSZM and GLDM families, 2D and 3D. It works in
  single precision and replaces :math:`\log_2` of the significand by a quadratic fit. It is a
  deliberate fast path, kept for speed. A single :math:`\log_2 p` is off by up to 0.009, mostly
  upward. An entropy is a probability-weighted sum, so its error is a probability-weighted average
  of the per-term errors: at most 0.009 bits, and it usually reads slightly low. On the worked
  examples the families' entropies differ from an exact logarithm by a relative
  :math:`3.1 \times 10^{-3}` at most (0 where every probability is a power of 2). On random
  distributions the relative error reaches about :math:`8 \times 10^{-3}`, and it grows without
  bound as the entropy goes to 0. Measures derived from entropies can differ more:
  ``GLCM_INFOMEAS1``, a difference of two entropies divided by a third, by up to
  :math:`7 \times 10^{-3}` on the worked example. Reference tools are therefore compared with
  these features at a looser tolerance.
- **The exact logarithm**, in double precision, everywhere else that an entropy appears.

.. list-table::
   :header-rows: 1
   :widths: 24 34 14 28

   * - Family
     - Features
     - Logarithm
     - Relative difference from exact, worked example
   * - :doc:`GLCM <Math/f_glcm>`
     - ``ENTROPY``, ``JE``, ``DIFENTRO``, ``SUMENTROPY``, ``INFOMEAS1``, ``INFOMEAS2``
     - Fast
     - IBSI mode, over the four angles: entropies up to :math:`3.1 \times 10^{-3}`
       (``DIFENTRO``), exact where every probability is a power of 2 (``SUMENTROPY_45``);
       ``INFOMEAS2`` up to :math:`3.0 \times 10^{-3}`, ``INFOMEAS1`` up to
       :math:`5.6 \times 10^{-3}`. Default settings: ``INFOMEAS1`` up to
       :math:`7.0 \times 10^{-3}`
   * - :doc:`3D GLCM <Math/f_glcm_3d>`
     - The same, as ``3GLCM_*``
     - Fast
     - :math:`2 \times 10^{-4}` (``JE``) to :math:`2 \times 10^{-3}` (``INFOMEAS1``)
   * - :doc:`GLRLM <Math/f_glrlm>`, :doc:`3D GLRLM <Math/f_glrlm_3d>`
     - ``RE``
     - Fast
     - :math:`2 \times 10^{-4}` to :math:`1.2 \times 10^{-3}` (``GLRLM_RE`` over the four
       angles), :math:`1.6 \times 10^{-3}` (``3GLRLM_RE``)
   * - :doc:`GLSZM <Math/f_glszm>`, :doc:`3D GLSZM <Math/f_glszm_3d>`
     - ``ZE``
     - Fast
     - :math:`7.1 \times 10^{-4}`, :math:`2.9 \times 10^{-4}`
   * - :doc:`GLDM <Math/f_gldm>`, :doc:`3D GLDM <Math/f_gldm_3d>`
     - ``DE``
     - Fast
     - :math:`1.1 \times 10^{-3}`, :math:`5.7 \times 10^{-4}`
   * - :doc:`GLDZM <Math/f_gldzm>`, :doc:`3D GLDZM <Math/f_gldzm_3d>`
     - ``ZDE``
     - Exact
     - :math:`10^{-15}` or less
   * - :doc:`NGLDM <Math/f_ngldm>`, :doc:`3D NGLDM <Math/f_ngldm_3d>`
     - ``DCENT``
     - Exact
     - :math:`10^{-15}` or less
   * - :doc:`Intensity <Math/f_pixel_intensity>`, :doc:`3D intensity <Math/f_intensity_3d>`,
       :doc:`intensity histogram <Math/f_intensity_histogram>`
     - ``ENTROPY``, ``3ENTROPY``, ``IH_ENTROPY_VAL``, ``IH_ENTROPY_IDX``
     - Exact
     - —

NGTDM has no entropy feature. In GPU builds run with ``--useGpu=true`` on the command line, 2D
GLSZM zone entropy is summed on the GPU with the exact logarithm in single precision.
