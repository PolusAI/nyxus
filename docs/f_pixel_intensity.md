# Pixel intensity features

Given a 2-dimensional greyscale image  $I(x,y)$ of total number of pixels $n$, 

the pixel intensity features are calculated as:

INTEGRATED_INTENSITY $\displaystyle = \sum _x\sum_y I_{x,y}$

MEAN $\displaystyle \gets \mu = \frac{1}{n}\sum _x\sum_y I_{x,y}$

MEDIAN - the value such that an equal number of samples are less than and greater than the value (for an odd sample size), or the average of the two central values (for an even sample size)

MIN $\displaystyle = min \: \textrm I$ treating $I$ as a set $I = \{I_{x,y}\}^n$ of positive values,

MAX $\displaystyle = max \: \textrm I$

RANGE $\displaystyle = max \: \textrm I - min \: \textrm I$

STANDARD_DEVIATION $\displaystyle \gets \sigma = \left[\frac{1}{n}\sum _x\sum_y (I_{x,y}-\mu)^2\right]^{\frac {1}{2}}$

STANDARD_ERROR $\displaystyle = \frac{\sigma}{\sqrt{n}}$ 

SKEWNESS $\displaystyle = \frac{1}{n} \frac {\sum _x \sum _y (I_{x,y}-\mu)^3} {\sigma^3}$, the 3rd central moment

KURTOSIS $\displaystyle = \frac{1}{n} \frac {\sum _x \sum _y (I_{x,y}-\mu)^4} {\sigma^4}$, the 4-th central moment

HYPERSKEWNESS $\displaystyle = \frac{1}{n} \frac {\sum _x \sum _y (I_{x,y}-\mu)^5} {\sigma^5} $, the 5-th central moment

HYPERFLATNESS $\displaystyle = \frac{1}{n} \frac {\sum _x \sum _y (I_{x,y}-\mu)^6} {\sigma^6}$, the 6-th standard moment

MEAN_ABSOLUTE_DEVIATION $\displaystyle = \frac{1}{n} \sum _x\sum_y \left| I_{x,y}-\mu\right| $ where $\mu$ is the mean

ENERGY $\displaystyle \gets E = \sum _x \sum _y I_{x,y}^2$

ROOT_MEAN_SQUARED $\displaystyle = \frac {\sqrt E} {n} $, where $E$ is the energy

ENTROPY $\displaystyle = \sum _x \sum _y I_{Nx,y} \log I_{Nx,y}$ where $I_{Nx,y}$ is a normalized ROI pixel ($\displaystyle I_{Nx,y}=\frac {I_{x,y}}{\max |I|}$),

MODE - the histogram bin value having the highest count,

UNIFORMITY $\displaystyle = \sum _i^k b_{i}^2$ where $b_i$ is a value of the image histogram of size $k$

UNIFORMITY_PIU $\displaystyle = (1 - \frac{max \: \textrm I - min \: \textrm I}{max \: \textrm I + min \: \textrm I}) \times 100$

P01, P10, P25, P75, P90, P99 - the 1%, 10%, 25% (aka $C_1$), 75% (aka $C_3$), 90%, and 99% histogram percentiles

INTERQUARTILE_RANGE $=Q_3 - Q_1$,

ROBUST_MEAN_ABSOLUTE_DEVIATION $\displaystyle = \frac{1}{k} \sum _i^k | b_{Ci} - \mu_b|$ where $b_{Ci}$ is the centered value of bin $i$ and $\mu_b$ is the mean histogram bin value

