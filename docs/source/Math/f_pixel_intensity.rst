
Pixel intensity features
========================

Given a 2-dimensional greyscale image  :math:`I(x,y)` of total number of pixels :math:`n`, 

the pixel intensity features are calculated as:

INTEGRATED_INTENSITY = :math:`\sum _x\sum_y I_{x,y}`

MEAN :math:`\gets \mu = \frac{1}{n}\sum _x\sum_y I_{x,y}`

MEDIAN - the value such that an equal number of samples are less than and greater than the value (for an odd sample size), or the average of the two central values (for an even sample size)

MIN = :math:`\min \: \textrm I` treating :math:`I` as a set :math:`I = {I_{x,y}}^n` of positive values,

MAX = :math:`\max \: \textrm I`

RANGE = :math:`\max \: \textrm I - \min \: \textrm I`,

STANDARD_DEVIATION :math:`\gets \sigma = \left[\frac{1}{n}\sum _x\sum_y (I_{x,y}-\mu)^2\right]^{\frac {1}{2}}`

STANDARD_ERROR = :math:`\frac{\sigma}{\sqrt{n}}`, 

SKEWNESS =  :math:`\frac {\sqrt n M_3}{M_2^{1.5}}` if :math:`n>3` and :math:`M_2 \neq 0`, otherwise :math:`=0`.

KURTOSIS = :math:`\frac{n M_4} {M_2^2}` if :math:`n>4` and :math:`M_2 \neq 0`, otherwise :math:`=0`.

HYPERSKEWNESS = :math:`\frac{n M_4} {M_2^{5/2}}` if :math:`n>5` and :math:`M_2 \neq 0`, otherwise :math:`=0`

HYPERFLATNESS = :math:`\frac {n M_5} {M_2^3}` if :math:`n>6` and :math:`M_2 \neq 0`, otherwise :math:`=0`

MEAN_ABSOLUTE_DEVIATION = :math:`\sigma = \frac{1}{n} \sum _x\sum_y \left| I_{x,y}-\mu\right|` where :math:`\mu` is the mean

ENERGY :math:`\gets E = \sum _x \sum_y I_{x,y}^2`

ROOT_MEAN_SQUARED :math:`= \frac {\sqrt E} {n}`, where :math:`E` is the energy

ENTROPY :math:`= \sum_i^k - b_{i} \: \textrm log b_{i}` where :math:`b_i` is a non-zero value of the image histogram of size :math:`k`,

MODE - the histogram bin value having the highest count,

UNIFORMITY = :math:`\sum_i^k b_{i}^2` where :math:`b_i` is a value of the image histogram of size :math:`k`

UNIFORMITY_PIU = :math:`(1 - \frac{\max \: \textrm I - \min \: \textrm I}{\max \: \textrm I + \min \: \textrm I}) \times 100`

P01, P10, P25, P75, P90, P99 - the 1%, 10%, 25% (aka :math:`C_1`), 75% (aka :math:`C_3`), 90%, and 99% histogram percentiles

INTERQUARTILE_RANGE = :math:`Q_3 - Q_1`,

ROBUST_MEAN_ABSOLUTE_DEVIATION = :math:`\frac{1}{k} \sum_i^k | b_{Ci} - \mu_b|` where :math:`b_{Ci}` is the centered value of bin :math:`i` and :math:`\mu_b` is the mean histogram bin value
