# Pixel intensity features

Given a 2-dimensional greyscale image  $I(x,y)$ of total number of pixels $n$, 

the pixel intensity features are calculated as:

INTEGRATED_INTENSITY $= \sum _x\sum_y I_{x,y}$

MEAN  $= \mu = \frac{1}{n}\sum _x\sum_y I_{x,y}$

MEDIAN - the value such that an equal number of samples are less than and greater than the value (for an odd sample size), or the average of the two central values (for an even sample size)

MIN $= min \textrm I$ treating $I$ as a set $I = \{I_{x,y}\}^n$ of positive values,

MAX $= max \textrm I$

RANGE $= max \textrm I - min \textrm I$,

STANDARD_DEVIATION $ = \sigma = \left[\frac{1}{n}\sum _x\sum_y (I_{x,y}-\mu)^2\right]^{\frac {1}{2}}$

STANDARD_ERROR $ = \frac{\sigma}{\sqrt{n}}$, 

SKEWNESS $= \frac {\sqrt n M_3}{M_2^{1.5}}$ if $n>3$ and $M_2 \neq 0$, otherwise $=0$

KURTOSIS $= \frac{n M_4} {M_2^2}$ if $n>4$ and $M_2 \neq 0$, otherwise $=0$

HYPERSKEWNESS $= \frac{n M_4} {M_2^{5/2}}$ if $n>5$ and $M_2 \neq 0$, otherwise $=0$

HYPERFLATNESS $= \frac {n M_5} {M_2^3}$ if $n>6$ and $M_2 \neq 0$, otherwise $=0$

MEAN_ABSOLUTE_DEVIATION $ = \sigma = \frac{1}{n} \sum _x\sum_y \left| I_{x,y}-\mu\right| $ where $\mu$ is the mean

ENERGY $ = E = \sum _x \sum _y I_{x,y}^2$

ROOT_MEAN_SQUARED $= \frac {\sqrt E} {n} $, where $E$ is the energy

ENTROPY $= \sum _i^k - b_{i} \: \textrm log b_{i}$ where $b_i$ is a non-zero value of the image histogram of size $k$,

MODE - the histogram bin value having the highest count,

UNIFORMITY $= \sum _i^k b_{i}^2$ where $b_i$ is a value of the image histogram of size $k$

UNIFORMITY_PIU $= (1 - \frac{max \: \textrm I - min \: \textrm I}{max \: \textrm I + min \: \textrm I}) \times 100$

P01, P10, P25, P75, P90, P99 - the 1%, 10%, 25% (aka $C_1$), 75% (aka $C_3$), 90%, and 99% histogram percentiles

INTERQUARTILE_RANGE $=Q_3 - Q_1$,

ROBUST_MEAN_ABSOLUTE_DEVIATION $= \frac{1}{k} \sum _i^k | b_{Ci} - \mu_b|$ where $b_{Ci}$ is the centered value of bin $i$ and $\mu_b$ is the mean histogram bin value

