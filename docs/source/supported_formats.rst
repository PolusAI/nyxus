
Supported Image Formats
=========================

Currently, Nyxus supports OME-TIFF, OME-Zarr and DICOM 2D Grayscale images.

OME-TIFF images uses the standard TIFF specification to store one or multiple image planes. OME-TIFF images are always structured as 
5D data((T)ime, (C)hannel, Z, Y and X). For 2D single channel image, Z, C and T dimensions are constrained to be one. OME-TIFF images also contain an XML 
document stored under the ``IMAGE_DESCRIPTION`` tag. This XML document contains the metadata to extract Image File Directory (IFD) for a 
given T, C and Z position. Since, Nyxus only processes grayscale 2D images at a fixed time point, the images are 
assumed to have only one IFD, and the internal loader reads the first IFD even if the image file contains multiple IFDs. 
Nyxus can process both uncompressed and compressed TIFF images by internally using ``libdeflate`` and ``zlib`` to decode compressed images.

OME-Zarr images uses Zarr v2 specification to store image data in multiple chunked files instead of a single file.
Similar to OME-TIFF, OME-Zarr files are also structured as 5D data to store 2D snapshot at various Z, C and T value. The root level
*.zattr* file stores an XML document which contains dimensional parameters defining the scope of the image pixels 
(e.g. resolution, number of focal planes, number of time points, number of channels). Similar to OME-TIFF, Nyxus assumes that the 
Z, C and T indices are 0 when retrieving the X and Y pixel values. Nyxus uses ``blosc`` as the compression library which supports all 
the compression schemes that are supported by Zarr v2 specification.

Nyxus can process both Single Frame and Whole Slide Grayscale DICOM images. It also supports binary segmentation images. The following 
Transfer Syntax UID supports are available in Nyxus by utilizing ``dcmtk`` and ``fmjpeg2koj`` library.

* Single Frame Image 

    * Explicit VR Little Endian
    * RLE Lossless
    * JPEG Lossless
    * JPEG Baseline
    * JPEG Extended
    * JPEG-LS Lossless
    * JPEG-LS Lossy
    * JPEG 2000 

* Multi-frame Image

    * Explicit VR Little Endian
    * JPEG 2000

CT / Hounsfield Units
---------------------

Nyxus carries pixels internally as unsigned integers, so a slide holding negative values --
every CT does, air sitting at roughly -1000 HU -- is offset at load time by its own floored
minimum, ``value - floor(slide_min)``, which keeps one grey level equal to one intensity unit
and stops the negatives wrapping on the unsigned cast. For DICOM and NIfTI the
``RescaleSlope`` / ``RescaleIntercept`` tags (``scl_slope`` / ``scl_inter``) are applied first,
so the offset is taken on true Hounsfield values. **That offset is recorded per slide and
undone on the way out**: reported intensity features are in the slide's own domain, so a CT read
in Hounsfield units is reported in Hounsfield units, negative values included. A slide with no
negative pixel takes no offset at all.

.. note::

   **Despite its name, you do not need** ``--preserve-hu`` **to get Hounsfield units.** A CT is
   read and reported in Hounsfield units either way. The flag is named for the case it was first
   written for; what it actually selects is described below, and it applies to floating-point
   slides only.

``--preserve-hu`` (CLI) or ``preserve_hu=True`` (Python) is what a **floating-point** slide needs
to take that same offset map. Left off, a float slide is instead min-max rescaled into
``[0, --fpimgdr]``, which keeps its shape but quantizes it; that rescale is likewise recorded and
undone, so its features come back in the slide's own float range rather than in quantization
steps. Integer slides, DICOM and NIfTI do not need the flag.

Which feature families the load-time map touches:

* All intensity features -- location (mean, median, mode, percentiles, min, max), dispersion,
  and the sum/energy family (integrated intensity, energy, root-mean-squared) -- are reported in
  the slide's own intensity domain, as are the Intensity Histogram (``IH_*``) features.
* Shift-invariant intensity features (variance, standard deviation, skewness, kurtosis, range,
  interquartile range) and all shape/texture features are unaffected by the offset.
* Sub-unit precision is not preserved: grey levels are integers, so a slide whose values are not
  integer-valued is reported rounded down to the grey level it was stored as. Hounsfield units
  are integer-valued, so CT is unaffected.





