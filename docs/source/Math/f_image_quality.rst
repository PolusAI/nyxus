Image Quality
=============

Image quality features provide calculations to determine how blurry or noisy an image is. 


Focus Score
-----------

Focus score first applies the Laplacian operator to the image to detect the edges. The Laplacian 
operator for :math:`\text{ksize}>1` is a filter using the kernel 

.. math::

    \begin{bmatrix}
    2 & 0 & 2\\
    0 & -8 & 0\\
    2 & 0 & 2 \end{bmatrix}.

For the default value of :math:`\text{ksize}=1`, the kernel becomes

.. math::
    
    \begin{bmatrix}
    0 & 1 & 0\\
    1 & -4 & 1\\
    0 & 1 & 0 \end{bmatrix}.

The filtering uses a reflective border condition. For example, given the matrix

.. math::
    
    \begin{bmatrix}
    1 & 2 & 3\\
    4 & 5 & 6\\
    7 & 8 & 9 \end{bmatrix},

the padded matrix becomes

.. math::
    
    \begin{bmatrix}
    9 & 8 & 7 & 8 & 9 & 8 & 7\\
    6 & 5 & 4 & 5 & 6 & 5 & 4\\
    3 & 2 & 1 & 2 & 3 & 2 & 1\\
    6 & 5 & 4 & 5 & 6 & 5 & 4\\
    9 & 8 & 7 & 8 & 9 & 8 & 7\\
    6 & 5 & 4 & 5 & 6 & 5 & 4\\
    3 & 2 & 1 & 2 & 3 & 2 & 1\end{bmatrix}.

After calculating the Laplacian of the image, the focus score is given by the variance of the filtered image :math:`x` is
defined by 

.. math::
    
    \text{variance}(x) = \frac{\sum_i(x_i - \text{mean}(x))}{(\text{length}(x) - 1)}

Local Focus Score
-----------

Local Focus Score is calculated the same way as Focus Score. However, the focus score is calculated for each tile 
of the image, where the number of tiles are determined by the input parameter `scale`. The image is divided into
:math:`\text{scale}^2` non-overlapping tiles and the mean and median values of the tiles are returned.

GLCM Correlation and Dissimilarity
----------------------------------

See the documentation for these features in :doc:`GLCM math documentation <f_glcm>`. 

Power Spectrum Slope
--------------------

This feature calculates the slope of the image's log-log power spectrum. The power spectrum of an :math:`m\times n` image is obtained 
through Fourier transform

.. math::

    F(u,v) = \sum(x,y)l(x,y)-\mu\mu w(x,y)e^{2\pi i(ux/m+vy/n)},

where :math:`u` and :math:`v` are spatial frequency coordinate. The power spectrum :math:`S`can then be found by the 
square amplitude of the Fourier transform

.. math::

    S(u,v) = L|F(u,v)|^2\Gamma(u,v),

where :math:`\Gamma` is the correction factor and :math:`L` is the number of pixels in the image. The resulting power spectrum
of the image is then converted to log-log scale and the slope of log-log power spectrum is obtained through the least squares solution.

Saturation Metrics
------------------

The min saturation and max saturation metrics are obtained by counting the number of pixels at the maximum and minimum intensities of the image and returning 
the percent of pixels found at both intensity levels.

Sharpness
---------

The Sharpness calculation aims to determine the blurriness of edges in an image. To accomplish this estimation,
a difference of differences in gray scale value of the median-filtered image, :math:`\Delta \text{DoM}`, is used[1]. 
To begin, median-filtering is applied to the image to reduce noise while preserving edges. The median-filtered image :math:`I_m` is calculated
using a 3x3 averaging filter, given by,

.. math::

    I_m = \frac{1}{9}\sum_{j=-1}^1\sum_{i=-1}^1 \text{I}(x+i, x+j).

The :math:`\Delta`DoM is calculated separately for horizontal and vertical directions. The horizontal direction is calculated as,

.. math::

    \Delta\text{DoM}_x(i,j) = [I_m(i+2, j) - I_m(i,j)] - [I_m(i, j) - I_m(i-2, j)].


The sharpness at a pixel :math:`S_x(i,j)` is computed as

.. math::

    S_x(i,j) = \frac{\sum_{i-w\leq k \leq i+w} |\Delta \text{DoM}(k,j)|}{\sum_{i-w\leq k \leq i+w} |I(k,j)-I(k-l,j)|}.


This ratio is high at sharp locations and low at blurred locations. The sharpness in the vertical or :math:`y` direction is calculated in a similar way.
The sharpness for the entire image is calculated as,


.. math::

    R_x = \frac{\text{#}SharpPixels_x}{\text{#}EdgePixels_x} \\ \\
    R_y= \frac{\text{#}SharpPixels_y}{\text{#}EdgePixels_y}.


The sharpness of the image is then given by the Frobenius norm,

.. math::

    S_I = \sqrt{R_x^2+R_y^2}.


Brisque
-------

The Brisque calculation is a no-reference image quality metric that predicts the quality of an image[2]. The Brisque calculation in completed in three steps:
extracting the natural scene statistics, creating feature vectors, and then predicting the image quality score using a support vector machine. To begin, the natural 
scene statistics are calculated. In this step, the pixel intensities are normalized. A natural image will follow a Gaussian distribution while unnatural or distorted
images will not follow a normal distribution. The amount of distortion in the image can therefore be measured by its deviation from the Gaussian distribution. In Brisque,
the images are normalized through Mean Subtracted Contrast Normalization (MSCN). To calculate the MSCN coefficients, the image intensity :math:`I` at pixel :math:`(i,j)`
is transformed to luminance :math:`\hat{I}(i,j)`,

.. math::

    \hat{I}(i,j) = \frac{I(i,j) - \mu (i,j)}{\sigma (i,j) + C},


where :math:`\mu (i,j)` is the local mean field and :math:`\sigma (i,j)` is the local variance field. The local mean field is found from the Gaussian blur of the image,

.. math::
    
    \mu = W * I,

where \textbf{W} is the Gaussian blur window function. The local variance field is the Gaussian blur of the square of the difference of the image and :math:`\mu`,

.. math::

    \sigma = \sqrt{W*(I - \mu)^2.

To also capture neighboring pixel relations, pair-wise products of the MSCN image are also calculated for four orientations: horizontal (H), vertical (V), left diagonal (LD),
and right diagonal (RD),

.. math::

    H(i,j) = \hat{I}(i,j)\hat{I}(i,j+1) \\
    V(i,j) = \hat{I}(i,j)\hat{I}(i+1,j) \\
    LD(i,j) = \hat{I}(i,j)\hat{I}(i+1,j+1) \\
    RD(i,j) = \hat{I}(i,j)\hat{I}(i+1,j-1).

Next, the feature vectors are calculated. For each of the 5 derived images from MSCN and the four addition orientations. This feature vector contains 36 columns regardless of 
image size; two columns for the Generlaized Gaussian distribution of the MSCN image and four columns each for the four additional orientations. The columns for each of these orientations 
are found by calculating the shape, mean left variance, and right variance. This gives 18 columns for the feature vector. The remaining 18 columns are found from performing the same calculations 
on the original image that is downsized to half the size, giving the 36 columns. Finally, the image quality score is predicted by feeding the feature vector to a support vector machine (SVM)
to predict the quality of the image. The resulting score ranges from 0 to 100, where a higher score indicates a higher quality image.

[1] J. Kumar, F. Chen and D. Doermann, "Sharpness estimation for document and scene images," Proceedings of the 21st International Conference on Pattern Recognition (ICPR2012), Tsukuba, Japan, 2012, pp. 3292-3295.
[2] https://learnopencv.com/image-quality-assessment-brisque/

