
Supported Image Formats
==============

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





