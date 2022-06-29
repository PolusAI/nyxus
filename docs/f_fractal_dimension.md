# Fractal dimension features

The complexity or, informally, roughness of a ROI boundary can be described via its fractal dimension. 

Suppose $A$ is a shape's area and $P$ is its perimeter, and we are determining $D$.

## FRACT_DIM_BOXCOUNT

The Minkowski–Bouligand box counting method consists in the overlaying a set of boxes of known edge on top of the ROI to entirely cover it. The size of the covering box set obviously depends on the box edge length, so consecutive covering a ROI with increasingly large boxes can be organized as an iterative procedure. On each iteration, the number of cells needed to cover the ROI shape is plotted versus the iteration-specific box edge size which is usually varied as an exponent 2 progression i.e. $1 \times 1$, $2 \times 2$, $4 \times 4$, etc. The number $N$ of boxes of size $r$ needed to
cover a ROI follows a power law:

$$N(r) = N_0 r^{−D}$$

where $N_0$ is a constant and $D$ is the dimension of the covering space e.g. 1, 2, 3, etc.

The regression slope $D$ of the straight line 

$$\log N(r)  = −D \log r + \log N0$$ 

formed by plotting $\log N(r)$ against $\log r$ indicates the degree of complexity, or fractal dimension, of the ROI. The feature is calculated as FRACT_DIM_BOXCOUNT $=D$.


## FRACT_DIM_PERIMETER

In Euclidean geometry, the perimeter $P$ is related to the diameter $d$ or the area $S$ as:

$$P \propto d^D \propto S^{D/2}$$

The area of the ROI can be expressed as a set of equivalent circles of diameter $d$ and consequtive approximations of ROI's $S$ with a series of $d$. Similar to the boxcount method, by log-log plotting the approximation perimeters versus $d$, the fractal dimension FRACT_DIM_PERIMETER $=D$ is defined as the slope of a least squares fitted line.

