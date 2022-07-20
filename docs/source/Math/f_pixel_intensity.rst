Pixel intensity features
========================

Denote 

- :math:`\mathbb{E}` as the expectation operator, 

- :math:`\mu_k = \mathbb{E}[(X − \mathbb{E}[X])^n]` of a real-valued random variable :math:`X` a central moment of order :math:`k`
- :math:`\min A` and :math:`\max A` the minimum and maximum of set :math:`A`.

Given a 2-dimensional greyscale image  :math:`I(x,y)` of :math:`n` pixels :math:`I_{x,y}`, 
Nyxus pixel intensity features are calculated as:

INTEGRATED_INTENSITY = :math:`\sum _x\sum_y I_{x,y}`

MEAN :math:`\gets \mu = \frac{1}{n} \sum _x\sum_y I_{x,y}`

MEDIAN - the value such that an equal number of samples are less than and greater than the value (for an odd sample size), or the average of the two central values (for an even sample size)

MIN = :math:`\min \: \textrm I` treating :math:`I` as a set :math:`I = \{I_{x,y}\}^n` of positive values,

MAX = :math:`\max \: \textrm I`

RANGE = :math:`\max \: \textrm I - \min \: \textrm I`,

STANDARD_DEVIATION :math:`\gets \sigma = \left[ \frac{1}{n}\sum _x\sum_y (I_{x,y}-\mu)^2 \right ] ^{\frac {1}{2}}`

STANDARD_ERROR = :math:`\frac{\sigma}{\sqrt{n}}`, 

SKEWNESS =  :math:`\frac {\sqrt n \mu_3}{\mu_2^{1.5}}` if :math:`n>3` and :math:`\sigma_2 \neq 0`, otherwise :math:`=0`.

KURTOSIS = :math:`\frac{n \mu_4} {\sigma^4}` if :math:`n>4` and :math:`\mu_2 \neq 0`, otherwise :math:`=0`.

HYPERSKEWNESS = :math:`\frac{n \mu_4} {\mu_2^{5/2}}` if :math:`n>5` and :math:`\mu_2 \neq 0`, otherwise :math:`=0`

HYPERFLATNESS = :math:`\frac {n \mu_5} {\mu_2^3}` if :math:`n>6` and :math:`\mu_2 \neq 0`, otherwise :math:`=0`

MEAN_ABSOLUTE_DEVIATION = :math:`\sigma = \frac{1}{n} \sum _x\sum_y \left| I_{x,y}-\mu\right|` 

ENERGY :math:`\gets E = \sum _x \sum_y I_{x,y}^2`

ROOT_MEAN_SQUARED :math:`= \sqrt {\frac {1} {n} \sum _x \sum_y I_{x,y}^2 }`

ENTROPY :math:`= \sum_i^k (- b_{i} \: \log \: b_{i})` where :math:`b_i` is a non-zero value of the image histogram of size :math:`k = 1 + \log_2 \: n`,

MODE :math:`= x_{uk} + w \frac{f_k - f_{k-1}}{2 f_k - f_{k-1} - f_{k+1}}` where :math:`k` - the index of the histogram bin containing the greatest count, 
:math:`x_{uk}` - lower bound of the histogram bin containing the greatest count, :math:`f_k` - the greatest bin count, :math:`f_{k-1}` and :math:`f_{k+1}` - 
counts of the bins neighboring to the greatest count bin; (informally, the histogram bin value having the highest count)

UNIFORMITY = :math:`\sum_i^k b_{i}^2` where :math:`b_i` is a value of the image histogram of size :math:`k = 256`

UNIFORMITY_PIU = :math:`(1 - \frac{\max \: \textrm I - \min \: \textrm I}{\max \: \textrm I + \min \: \textrm I}) \times 100`

The quantile :math:`q_p` of a random variable (or equivalently of its distribution) is
defined as the smallest number :math:`q`` such that the cumulative distribution function
is greater than or equal to some :math:`p`, where :math:`0<p<1`. This can be calculated
for a continuous random variable with density function :math:`f(x)` by solving

.. math::

    p = \int_{-\inf} {q_p} f(x)dx 

for :math:`q_p`, or by using the inverse of the cumulative distribution function, 

.. math::

    q_p = F^{-1}(p). 
    
The :math:`p`-th quantile of a random variable :math:`X` is the value :math:`q_p` such that 

.. math::
    
    F(q_p) = P(X \leqslant q_p) = p


P01, P10, P25, P75, P90, P99 - are defined as the 1%, 10%, 25%, 75%, 90%, and 99% percentiles. A percentile :math:`q_p` 
where :math:`p=0.01, 0.1, 0.25, etc` is a solution of equation :math:`p = \int _{-\infty} ^{q_p} f(x)dx`, for example 
:math:`0.25 = \int _{-\infty} ^{0.25} f(x)dx`. The 25% and 75% percentiles are called 1st and 3rd quartiles. The 50% 
percentile, or the 50% quartile is the median.

INTERQUARTILE_RANGE = :math:`q_{0.75} - q_{0.25}` - the difference of the 1st and 3rd sample quartiles,

ROBUST_MEAN_ABSOLUTE_DEVIATION = :math:`\frac{1}{k} \sum _x\sum_y |I_{Rx,y} - \mu| \text{ such that } q_{0.1}<I_{Rx,y}<q_{0.9}`  
where :math:`I_R={I_{Rx,y}}^k` - subset of :math:`I` whose elements are in the :math:`(q_{0.1},q{0.9})` value interval;

References
----------

Zwillinger, D. (Ed.). CRC Standard Mathematical Tables and Formulae. Boca Raton, FL: CRC Press, p. 602, 1995.

