
Fractal dimension features
==========================

The complexity or, informally, roughness of a ROI boundary can be described via its fractal dimension. 

Suppose :math:`A` is a shape's area and :math:`P` is its perimeter, and we are determining :math:`D`.

FRACT_DIM_BOXCOUNT
------------------

The Minkowski–Bouligand box counting method consists in the overlaying a set of boxes of known edge on top of 
the ROI to entirely cover it. The size of the covering box set obviously depends on the box edge length, so 
consecutive covering a ROI with increasingly large boxes can be organized as an iterative procedure. On each 
iteration, the number of cells needed to cover the ROI shape is plotted versus the iteration-specific box edge 
size which is usually varied as an exponent 2 progression i.e. :math:`1 \times 1`, :math:`2 \times 2`, :math:`4 \times 4`, etc. 
The number :math:`N` of boxes of size :math:`r` needed to
cover a ROI follows a power law:

.. math::

    N(r) = N_0 r^{−D}

where :math:`N_0` is a constant and :math:`D` is the dimension of the covering space e.g. 1, 2, 3, etc.

The regression slope :math:`D` of the straight line 

.. math::

    \log N(r)  = −D \log r + \log N0

formed by plotting :math:`\log(N(r))` against :math:`log(r)` indicates the degree of complexity, or fractal dimension, of the ROI. The feature is calculated as FRACT_DIM_BOXCOUNT :math:`=D`.

FRACT_DIM_PERIMETER
-------------------

In Euclidean geometry, the perimeter :math:`P` is related to the diameter :math:`d` or the area :math:`S` as:

.. math::

    P \propto d^D \propto S^{D/2}

The area of the ROI can be expressed as a set of equivalent circles of diameter :math:`d` and consecutive approximations of 
ROI's :math:`S` with a series of :math:`d`. Similar to the boxcount method, by log-log plotting the approximation perimeters versus :math:`d`, 
the fractal dimension FRACT_DIM_PERIMETER :math:`=D` is defined as the slope of a least squares fitted line.
