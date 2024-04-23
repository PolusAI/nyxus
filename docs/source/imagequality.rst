Image Quality features
======================

Image quality features are available in Nyxus to determine how blurry an image is. 
These features are available in the `IMAGE_QUALITY` feature group. This group is not included
in the `ALL` group and must be enabled separately. The following features are in the image quality
group.

.. list-table::
   :header-rows: 1

   * - Image Quality feature code
     - Description
   * - FOCUS_SCORE
     - Uses edge detection to highlight regions where intesnity changes rapidly. Higher focus score means lower blurriness
   * - LOCAL_FOCUS_SCORE
     - Tiles image into non-overlapping regions and calculates the FOCUS_SCORE for each region. Higher local focus score means lower blurriness
   * - GLCM_DIS
     - Blurry images low dissimilarity
   * - GLCM_CORRELATION
     - Blurry images have a high correlation
   * - POWER_SPECTRUM
     - The slope of the image log-log power spectrum. A low score means a blurry image
   * - SATURATION
     - Percent of pixels at minimum and maximum pixel values
   * - SHARPNESS
     - Uses median-filtered image as indicator of edge scharpness. Values range from 0 to sqrt(2). Low scores indicate blurriness.
   * - BRISQUE
     - Referenceless Image Spatial Quality Evaluator. Values range from 0 to 100. High scores indicate a blurry image.