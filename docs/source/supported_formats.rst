Supported image formats
=======================

Nyxus reads grayscale images in TIFF, OME-TIFF, OME-Zarr, NIfTI and DICOM files, as 2D images or as
3D volumes. It picks the reader from the file's last extension, ignoring case, and refuses any other
name with an error that lists the accepted extensions.

.. list-table::
   :header-rows: 1
   :widths: 16 22 14 48

   * - Format
     - Extensions
     - Dimensions
     - Pixel types
   * - TIFF, OME-TIFF
     - ``.tif``, ``.tiff``, ``.ome.tif``, ``.ome.tiff``
     - 2D, 3D, with channels and timepoints in OME-TIFF
     - unsigned and signed integers of 8, 16, 32 and 64 bits; floating point of 16, 32 and 64 bits
   * - OME-Zarr
     - ``.zarr``, ``.ome.zarr``
     - 2D, 3D, with channels and timepoints
     - unsigned and signed integers of 8, 16, 32 and 64 bits; floating point of 32 and 64 bits
   * - NIfTI
     - ``.nii``, ``.nii.gz``
     - 2D, 3D, with timepoints
     - unsigned and signed integers of 8, 16, 32 and 64 bits; floating point of 32 and 64 bits
   * - DICOM
     - ``.dcm``, ``.dicom``
     - 2D
     - unsigned and signed integers of 8 and 16 bits; 1-bit segmentation frames

A TIFF stored under another extension, such as ``.btf`` or ``.tf8``, must be renamed first. Images
must have one sample per pixel: RGB images are refused. ``--filePattern`` alone decides which files
in a directory are read, so a file with an unsupported extension that matches the pattern stops the
run with that error.

OME-Zarr and DICOM support are optional parts of a source build (see the CMake options in
:doc:`installation`). The PyPI and conda packages include both, except that the Windows wheels
on PyPI have no OME-Zarr support since 0.9.3, and the first conda-forge Windows builds of
0.11.0 (build number 0) have neither. Reading ``.nii.gz`` needs zlib, which a build includes
together with OME-Zarr support, so those packages do not read ``.nii.gz`` either.

TIFF and OME-TIFF
-----------------

When the first image description holds OME-XML, Nyxus reads the image size along X, Y, Z, C and T,
the dimension order, the ``TiffData`` plane map and the physical pixel size from it, and finds each
plane through that metadata. A plain TIFF is read as one channel and one timepoint, with one Z plane
per page; consecutive pages of the same size form the volume.

* Tiled and striped TIFF are both read, but an intensity image and its label image must both be
  striped, or both be tiled with the same tile size. Decompression is done by libtiff, so the supported codecs
  are those of the libtiff that Nyxus is built against; the packaged builds include Deflate, LZW,
  JPEG and Zstandard.
* Only the full-resolution image is read. Pyramid levels and reduced-resolution pages are skipped.
* An OME-TIFF whose planes are stored in companion files (multi-file OME-TIFF) is refused. Combine
  the planes into one file first.
* Label values are read as integers: a floating-point label image is accepted and its values
  truncated.

OME-Zarr
--------

Nyxus reads OME-Zarr image groups in Zarr format version 2 and version 3, including sharded
version 3 arrays, with OME-NGFF metadata version 0.4 or 0.5. The metadata is read from ``.zattrs``
(version 2) or ``zarr.json`` (version 3).

* The full-resolution level, the first dataset of the first multiscale, is read.
* The axis order comes from the ``axes`` metadata. T and C may stand anywhere; Z, Y and X must
  come in that order, as OME-NGFF requires. An array with X before Y is read with its pixels out
  of place, without an error, a known defect (`#522 <https://github.com/PolusAI/nyxus/issues/522>`__).
* The physical pixel size comes from the level's ``scale`` transformation.
* Arrays may be uncompressed or compressed with Blosc or zlib; Zstandard is read when the build
  includes it.
* Big-endian arrays are refused. bioformats2raw 0.9 writes them and 0.12 writes little-endian
  arrays; convert again with 0.12 or newer.
* The root of a bioformats2raw store is not an image group: open the numbered series inside it,
  for example ``store.zarr/0``, under a name that ends in ``.zarr``.

NIfTI
-----

Nyxus reads single-file NIfTI-1 and NIfTI-2 volumes (``.nii``, or ``.nii.gz`` compressed) with up
to four dimensions: X, Y, Z and time. The ``scl_slope`` and ``scl_inter`` rescale is applied to the
reported intensities (see :ref:`intensity-units`). The voxel spacing in the header is not read, so
NIfTI volumes are treated as having unit spacing.

DICOM
-----

Nyxus reads 2D grayscale DICOM images of 8- or 16-bit integers, and binary (1-bit) segmentation
images as label images. Compressed pixel data is decoded with DCMTK: JPEG, JPEG-LS and RLE, and
JPEG 2000 when the build includes it. ``RescaleSlope`` and ``RescaleIntercept`` are applied to the
reported intensities (see :ref:`intensity-units`). Pixel spacing is not read.

Dimensions, channels and timepoints
-----------------------------------

* **2D** (the default, ``--dim=2``): each file is one image. Nyxus reads its first Z plane; the
  label image must have as many Z planes as the intensity image, or the pair is refused. An
  OME-TIFF or OME-Zarr file with more than one channel or timepoint is refused in 2D.
* **3D** (``--dim=3``, or the ``Nyxus3D`` class): each file is a volume, and every channel and
  timepoint in it is processed as a separate volume. A volume cannot currently be assembled from a
  series of 2D slice files (see :doc:`volumetric`); stack the slices into one file first.

A label image is paired with each channel and timepoint of its intensity image: label frame *n* with
intensity frame *n* when the label image has as many frames, otherwise its first frame.

Every result row starts with these columns, followed by the features:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Content
   * - ``intensity_image``
     - Intensity file name (or the image name for in-memory input).
   * - ``mask_image``
     - Label file name.
   * - ``phys_unit``
     - Unit of the physical pixel size, empty when the file declares none.
   * - ``anno0``, ``anno1``, ...
     - Annotations from the file name, only with ``--annot=true`` and CSV output.
   * - ``ROI_label``
     - The ROI's label value.
   * - ``t_index``, ``c_index``
     - Timepoint and channel of the processed volume, 0 for single-plane files.
   * - ``phys_x``, ``phys_y``, ``phys_z``
     - Physical pixel size along each axis, 1 when the file declares none.

In 3D, ``--use-physical-spacing=true`` (Python: ``use_physical_spacing=True``) scales the shape
features by the OME physical pixel size, normalised so the smallest axis is 1. The voxels are not
resampled, so intensity and texture features do not change. A factor other than 1 in
any of ``--anisox``, ``--anisoy`` and ``--anisoz`` replaces the physical spacing on all three
axes; the axes not given then use 1.

.. _intensity-units:

Intensity units and CT images
-----------------------------

Nyxus stores pixels as unsigned integer grey levels. When it loads an image it maps the stored
values to grey levels, records that map, and converts the intensity features back through it, so
they are reported in the image's own units:

* An image without negative values is used as it is.
* An image with negative values anywhere, such as a CT in Hounsfield units, is shifted up by its
  floored minimum, so one grey level is one intensity unit.
* A floating-point TIFF or OME-Zarr image is rescaled from its own minimum and maximum into
  ``0 .. --fpimgdr`` grey levels (10000 by default). ``--fpimgmin`` and ``--fpimgmax`` replace the
  image's own range with a fixed one; giving any of the three options switches every
  floating-point image to that fixed range, which is 0 to 1 unless set.
* A floating-point NIfTI volume is not rescaled: its values are truncated to whole units (after
  the shift, when it has negative values), so fractional values are lost.
* A DICOM image, or an integer NIfTI volume, keeps its stored integers as grey levels and applies
  its rescale slope and intercept when converting back, so a fractional rescaled value is reported
  exactly.

``--preserve-hu=true`` (Python: ``preserve_hu=True``) changes the map for two cases: a
floating-point image is shifted by its floored minimum instead of rescaled, and a DICOM or integer
NIfTI image is rescaled when it is loaded and then shifted, with each value rounded to the nearest
unit. It is not needed to get Hounsfield units: a CT is reported in Hounsfield units either way.

Only these features are converted back to the image's units:

* the 2D and 3D intensity features. Location features (minimum, maximum, mean, median, mode,
  percentiles, robust mean, integrated intensity, energy, root mean square) go through the whole
  map; spread features (range, interquartile range, absolute deviations, standard deviation,
  standard error, covered intensity range) are scaled by its slope, and variance by the square of
  the slope; ratios (coefficient of variation, quartile coefficient of dispersion) are recomputed;
* the IBSI intensity histogram (``IH_*``) features.

All other features, including texture features, intensity-weighted moments and centroids, edge
intensity features and image quality features, are computed on grey levels.

Python arrays
-------------

``Nyxus.featurize`` takes numpy arrays instead of files. Their values are converted to unsigned
32-bit integers: an array with negative values is first shifted up by its minimum, and fractional
values are truncated. No map is recorded for in-memory images, so their intensity features are
reported in these shifted grey levels, not in the array's original units. To get features in the
original units, write the image to a file and use ``featurize_files`` or ``featurize_directory``.
