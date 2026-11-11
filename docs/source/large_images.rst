.. _large-images:

Whole images and large images
=============================

This page covers two cases: computing features of a whole image or volume without a label image,
and processing ROIs that do not fit in the memory budget.

.. _large-images-whole:

Whole-slide and whole-volume mode
---------------------------------

In whole-slide mode (whole-volume mode in 3D), Nyxus reads no label image. It treats each intensity
image as one ROI with label 1 that covers every pixel, including zero-intensity background.

.. tab-set::

   .. tab-item:: Python

      Pass ``single_roi=True`` to ``featurize_files``:

      .. literalinclude:: ../examples/whole_slide.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

      Output:

      .. code-block:: text

                  intensity_image mask_image  ROI_label         MEAN  PERIMETER
         0  slides/slide1.ome.tif                     1  2002.230794      224.0
         1  slides/slide2.ome.tif                     1  1981.880208      224.0

      ``featurize_directory`` switches to whole-slide mode when ``label_dir`` is ``None`` or the
      same path string as ``intensity_dir``. For whole volumes, background included, call
      ``Nyxus3D.featurize_directory(volume_dir, None, ".*")`` or
      ``Nyxus3D.featurize_directory(volume_dir, volume_dir, ".*")``.
      ``Nyxus3D.featurize_files(volumes, volumes, single_roi=True)`` is not whole-volume mode: it
      uses each volume as its own mask, so the ROI is the volume's nonzero voxels (see
      :doc:`Math/f_morphology_3d`); its 3D shape features describe the box of those voxels, a
      known defect (`#494 <https://github.com/PolusAI/nyxus/issues/494>`__).

   .. tab-item:: Command line

      Give the intensity directory as the label directory too. The two paths are compared as
      text, ignoring case:

      .. code-block:: bash

         nyxus --features="*WHOLESLIDE*" --intDir=/data/int --segDir=/data/int --outDir=/data/out --outputType=singlecsv

      Add ``--dim=3`` for whole volumes.

Each image gives one row with ``ROI_label`` 1 and an empty ``mask_image``. ``intensity_image``
holds the path as it was given, such as ``slides/slide1.ome.tif``, where a run with label images
holds the bare file name. ``--aggr`` and ``--skiproi`` have no effect in this mode.

Which features to request
^^^^^^^^^^^^^^^^^^^^^^^^^

Nyxus computes every feature you request, but many describe the shape of a ROI, and in this mode
the ROI is the image rectangle. The ``*WHOLESLIDE*`` group selects the families that still
describe the image: intensity, the contour family, the texture families except GLDZM, Gabor,
intensity moments, radial distribution and Zernike. :doc:`featuregroups` lists its members.

Whole-slide features differ from those of an ordinary ROI in these ways:

* **The contour is the bounding box.** Nyxus does not trace a contour. It uses the four corners of
  the image, so ``PERIMETER`` is 2 x (width + height): 224 for a 64 x 48 image. Every feature
  built on the contour describes this rectangle: circularity, the perimeter-equivalent diameter,
  geodetic length and thickness, the ROI radii, fractal dimension, radial distribution and the
  weighted moments. The radial distribution's centre is the pixel nearest the middle of the four
  corners.
* **Edge features are degenerate.** The edge is the four corners, and each corner carries the
  image's maximum grey level. ``EDGE_MIN_INTENSITY`` and ``EDGE_MAX_INTENSITY`` both equal
  ``MAX``, ``EDGE_STDDEV_INTENSITY`` is 0 and ``EDGE_INTEGRATED_INTENSITY`` is 4 x ``MAX``.
* **Some features are not computed** and are reported as 0: the neighbor features,
  ``POLYGONALITY_AVE``, ``HEXAGONALITY_AVE``, ``HEXAGONALITY_STDDEV`` and the minimum enclosing,
  inscribing and circumscribing circle diameters.
* **The bounding box is one pixel larger than the image**, a known defect (`#493 <https://github.com/PolusAI/nyxus/issues/493>`__). On a
  256 x 256 image, ``BBOX_WIDTH``
  and ``BBOX_HEIGHT`` are 257 and ``EXTENT`` is 0.992.

A whole-slide run is therefore not the same as a run with a label image of ones. On a 256 x 256
test image the two agree on the intensity and GLCM features, but differ in 263 of the 747 output
columns of ``*ALL*``,
including the GLRLM, GLSZM, GLDZM, NGTDM and Gabor features. For example, ``GLDZM_SDE`` is 0.0238
in whole-slide mode and 0.0351 with a label image of ones.

A 2D image must fit in the memory budget described below. A larger image is refused with
``slide is non-trivial``, and the Python call raises ``RuntimeError``. A whole volume larger than
the budget is streamed out of core like any oversized 3D ROI. When one read of the file returns
the whole volume, as with a NIfTI file, it cannot be streamed and the run stops with an error
(see the end of :ref:`large-images-tiles`).

In 2D whole-slide mode, ``--reduceThreads`` (Python: ``n_feature_calc_threads``) sets how many
images are processed at the same time.

Memory budget and out-of-core processing
----------------------------------------

Before it computes features, Nyxus estimates the memory footprint of each ROI and compares it with
a RAM limit. A ROI whose estimate reaches the limit is *oversized* and is processed out of core:
Nyxus streams its pixels from the file into a temporary file and computes its features from
there. The other ROIs are computed in memory, in batches that fit the limit.

.. tab-set::

   .. tab-item:: Python

      Pass ``ram_limit``, in megabytes, when you create the object. ``Nyxus.set_params``
      can change it later; ``Nyxus3D`` accepts it only when the object is created.

      .. literalinclude:: ../examples/out_of_core.py
         :language: python
         :start-after: # example-start
         :end-before: # example-end

      Output:

      .. code-block:: text

            ROI_label         MEAN  GLCM_CONTRAST_AVE
         0          1  1984.574661         625.447468
         1          2  2038.488800         711.010030
         2          3  2045.812302         644.997573
            ROI_label         MEAN  GLCM_CONTRAST_AVE
         0          1  1984.574661         625.447468
         1          2  2038.488800         711.010030
         2          3  2045.812302         644.997573

      ``get_params("ram_limit")`` returns the current limit in megabytes.

   .. tab-item:: Command line

      .. code-block:: bash

         nyxus --ramLimit=2048 --intDir=/data/int --segDir=/data/seg --outDir=/data/out --outputType=singlecsv

The default limit is half of the physical memory available when the run starts, or when the
Python object is created. A limit above the available memory is refused: the command line stops
with ``exceeds the actual amount of available RAM``. The Python classes print that error and keep
their previous limit.

What counts as available memory depends on the platform:

* **Windows:** the available physical memory.
* **Linux:** the free memory, ``MemFree`` in ``/proc/meminfo``. It leaves out the page cache,
  which Linux fills with recently read files and releases on demand, so the default limit is
  usually lower than the memory a process can get, and it falls as the cache grows. The same
  figure caps ``--ramLimit``, so a limit can be refused while enough memory is available.
* **macOS:** the total physical memory, however much of it is in use.

Container memory limits are not read. In a Docker container or another cgroup with a memory
limit, the default limit is computed from the host's memory, so a ROI that fits it can need more
memory than the container allows, and the container then stops Nyxus. In a container, set
``--ramLimit`` (Python ``ram_limit``) below the container's memory limit.

``Nyxus.featurize``, which takes numpy arrays, has no out-of-core path. An oversized ROI makes it
raise ``RuntimeError`` with ``the following ROIS are oversized and cannot be processed``.

Temporary files
^^^^^^^^^^^^^^^

Out-of-core ROIs are written to the system temporary directory. ``--tempDir`` is accepted, must
name an existing directory and is shown in the run summary, but the temporary files are written to
the system temporary directory regardless. To move them, set the ``TMPDIR`` environment variable
on Linux and macOS, or ``TMP`` on Windows, before you start Nyxus.

``--onlineStatsThresh`` is a reserved name. The option is not accepted: a run that passes it stops
with ``unrecognized arguments``.

Which features differ out of core
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In 3D, every family gives the same values out of core as in memory, the 3D axis features up to
the last digits (see :doc:`Math/f_morphology_3d`). A test volume with two ROIs, computed with
``ram_limit=0``, matched the in-memory run in the 213 ``*3D_ALL*`` features, and so did a whole
volume. This holds for inputs Nyxus can stream; see the end of
:ref:`large-images-tiles` for those it refuses.

In 2D, every family matches the in-memory values on tiled TIFF files and on striped files of up to
1024 x 1024 pixels (see :ref:`large-images-tiles` for larger striped files), with these
exceptions:

* In IBSI mode 2D NGTDM matches only on the first ROI computed out of core, a known defect
  (`#512 <https://github.com/PolusAI/nyxus/issues/512>`__; see :doc:`Math/f_ngtdm`).
* NGLDM differs on a constant ROI (see :doc:`Math/f_ngldm`).
* ``POWER_SPECTRUM_SLOPE`` and ``SHARPNESS`` are not computed, a known defect
  (`#508 <https://github.com/PolusAI/nyxus/issues/508>`__; see :doc:`Math/f_image_quality`).
* The neighbor features (``NUM_NEIGHBORS``, ``PERCENT_TOUCHING``, ``CLOSEST_NEIGHBOR*``,
  ``ANG_BW_NEIGHBORS_*``), ``POLYGONALITY_AVE`` and ``HEXAGONALITY_*`` are not computed. The
  neighbor features are reported as 0, and ``POLYGONALITY_AVE`` and ``HEXAGONALITY_*`` as -1,
  their value for a ROI with fewer than 3 neighbors. An out-of-core ROI is also not seen as a
  neighbor by the ROIs computed in memory.

The contour is traced out of core by the same border following as in memory, on the ROI's padded
image held in a temporary file: every border of the ROI, the outer border of each fragment and
each hole's border, in image coordinates. So the contour-based features (``PERIMETER`` and the
features built on it, ``EDGE_*``, the circles, ``ROI_RADIUS_*``, the fractal dimensions, the
weighted moments and the radial distribution) have their in-memory values, with or without
``--anisox`` and ``--anisoy`` (Python ``anisotropy_x``, ``anisotropy_y``). If you need the
neighbor features for large ROIs, raise the RAM limit so that the ROIs are computed in memory.

When a feature fails on an out-of-core 2D ROI, Python raises ``RuntimeError`` with the message
``Error while computing feature ... over oversized ROI ...``. The command line prints the same
message and continues: the ROI's row is written with 0 in that feature's columns, and the run
exits with status 0. For example, on a 300 x 300 ROI of a striped 1200 x 1200 image, the contour
failed out of core with ``pixel ... lies outside the bounding box ... that sizes its buffer`` (see
:ref:`large-images-tiles`). After a command-line run with large ROIs, search the console output for
``Error while computing feature``.

.. _large-images-tiles:

Tiled images and ROIs spanning tiles
------------------------------------

Nyxus reads tiled and striped TIFF files tile by tile. A tiled file is read in its own tiles; a
striped file is read in tiles of 1024 x 1024 pixels, so a striped image larger than 1024 pixels in
either direction is read as several tiles. An intensity image and its label image must have the
same tile size; a pair with different tile sizes is refused with an error that starts
``mismatch in tile height, width``.

In memory and in whole-slide mode, tiling does not change the results: a 256 x 256 image with 66
ROIs, 30 of them crossing 64 x 64 tile borders, gave the same features tiled and striped.

Out of core, tiling does not change the results either: a tiled TIFF of several tiles gives the
values of a striped copy of the same image, and the in-memory values, for ROIs inside one tile and
for ROIs that span tiles.

.. warning::

   Out-of-core processing of a 2D striped TIFF larger than 1024 pixels in either direction, which
   Nyxus reads as several 1024 x 1024 tiles, places the pixels of later tiles at wrong positions
   and can read ROI pixels more than once, so the results depend on the family. Measured on a
   300 x 300 square and a 300 x 240 elliptical ROI inside the first 1024 x 1024 region of a
   striped 1200 x 1200 image:

   * Every family that builds the ROI's pixel matrix or contour (the texture families, the
     moments, morphology, the radial distribution, the fractal dimension, Zernike and Gabor) stops
     when a misplaced pixel lies outside the ROI's bounding box: from Python the call raises
     ``RuntimeError`` ("pixel ... lies outside the bounding box ... that sizes its buffer"); the
     command line prints that error and leaves those features uncomputed.
   * When every misplaced pixel stays inside the bounding box, the same families return wrong
     values without an error.
   * ROI pixels were read more than once: the intensity features built from sums, such as
     ``MEAN``, differed from the in-memory values, while ``MIN`` and ``MAX`` kept them.

   Keep large 2D ROIs of such images in memory by raising the RAM limit, store the images as tiled
   TIFF files, or split them so that each is at most 1024 x 1024 pixels.

Out-of-core processing of a volume streams it one read at a time. When a single read of the
intensity or label file returns the whole volume, as with a NIfTI file or an OME-Zarr array
chunked over the whole Z axis, Nyxus cannot stream it, and the run stops with an error that names
the input and the number of planes one read delivers. An array
chunked several planes deep, but not over all of Z, is streamed. Re-chunk the array along
Z, convert it to a Z stack of TIFF planes, or raise the RAM limit.
