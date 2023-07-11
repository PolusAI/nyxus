Pixel intensity features
========================

Denote 

- :math:`f(x,y)` -- real value of a continuous intensity function :math:`f` at real-valued (continuous) Cartesian image location :math:`(x,y)`; 
- :math:`P` -- 2-dimensional point set of greyscale image intensity values at discrete locations; 
- :math:`p(x,y)` -- greyscale value of :math:`f(x,y)` at discrete 2-dimensional image pixel location :math:`(x,y)`, or, simply, pixel :math:`(x,y)` intensity; 
- :math:`G = {p(x,y) | p(x,y)>0}` -- 1-dimensional set of image pixels of non-zero intensity, or so called region of interest (ROI); 
- :math:`g_i` -- element of :math:`G`;
- :math:`n = card(G)` -- number of ROI elements;
- :math:`\mathbb{E}` -- the expectation operator;
- :math:`\min \: X` and :math:`\max \: X` -- the minimum and maximum of random variable :math:`X`;
- :math:`\mu_k` -- central moment of order :math:`k` of a real-valued random variable :math:`X`, :math:`\mu_k = \mathbb{E}[(X − \mathbb{E}[X])^n]`.

Nyxus pixel intensity features are calculated as:

INTEGRATED_INTENSITY = :math:`\sum _i^n g_i`

MEAN :math:`\gets \mu = \frac{1}{n} \sum_i^n g_i`

MIN = :math:`\min \: G` ,

MAX = :math:`\max \: G`,

RANGE = :math:`\max \: G - \min \: G`,

COVERED_IMAGE_INTENSITY_RANGE = :math: `\frac {\max \: G - \min \: G} {\max \: P - \min \: P} `

STANDARD_DEVIATION :math:`\gets \sigma = (\mathbb{E}[(G-\mu)^2]) ^{\frac {1}{2}} = \left[ \frac{1}{n-1} \sum_i^n (g_i-\mu)^2 \right ] ^{\frac {1}{2}}`

STANDARD_DEVIATION_BIASED :math:`\gets \sigma_b = (\mathbb{E}[(G-\mu)^2]) ^{\frac {1}{2}} = \left[ \frac{1}{n} \sum_i^n (g_i-\mu)^2 \right ] ^{\frac {1}{2}}`

COV = :math:`\frac{\sigma}{\mu}`, 

STANDARD_ERROR = :math:`\frac{\sigma}{\sqrt{n}}`, 

SKEWNESS =  :math:`\frac {\sqrt n \mu_3}{\mu_2^{1.5}}` if :math:`n>3` and :math:`\sigma_2 \neq 0`, otherwise :math:`=0`.

KURTOSIS = :math:`\frac{n \mu_4} {\sigma^4}` if :math:`n>4` and :math:`\mu_2 \neq 0`, otherwise :math:`=0`.

EXCESS_KURTOSIS = :math:`\frac{n \mu_4} {\sigma^4} - 3` if :math:`n>4` and :math:`\mu_2 \neq 0`, otherwise :math:`=0`.

HYPERSKEWNESS = :math:`\frac{n \mu_4} {\mu_2^{5/2}}` if :math:`n>5` and :math:`\mu_2 \neq 0`, otherwise :math:`=0`

HYPERFLATNESS = :math:`\frac {n \mu_5} {\mu_2^3}` if :math:`n>6` and :math:`\mu_2 \neq 0`, otherwise :math:`=0`

MEAN_ABSOLUTE_DEVIATION = :math:`\sigma = \frac{1}{n} \sum_i^n \left| g_i-\mu \right|` 

ENERGY :math:`\gets E = \sum _i^n g_i^2`

ROOT_MEAN_SQUARED :math:`= \sqrt {\frac {1} {n} \sum_i^n g_i^2 }`

ENTROPY :math:`= \sum_i^k (- b_{i} \: \log \: b_{i})` where :math:`b_i` is a non-zero value of the image histogram of size :math:`k = 1 + \log_2 \: n`,

MODE :math:`= x_{uk} + w \frac{f_k - f_{k-1}}{2 f_k - f_{k-1} - f_{k+1}}` where :math:`k` - the index of the histogram bin containing the greatest count, 
:math:`x_{uk}` - lower bound of the histogram bin containing the greatest count, :math:`f_k` - the greatest bin count, :math:`f_{k-1}` and :math:`f_{k+1}` - 
counts of the bins neighboring to the greatest count bin; (informally, the histogram bin value having the highest count)

VARIANCE :math:`\gets \sigma = (\mathbb{E}[(G-\mu)^2]) ^{\frac {1}{2}} = \left[ \frac{1}{n-1} \sum_i^n (g_i-\mu)^2 \right ]`

VARIANCE_BIASED :math:`\gets \sigma_b = (\mathbb{E}[(G-\mu)^2]) ^{\frac {1}{2}} = \left[ \frac{1}{n} \sum_i^n (g_i-\mu)^2 \right ]`

UNIFORMITY = :math:`\sum_i^k b_{i}^2` where :math:`b_i` is a value of the image histogram of size :math:`k = 256`

UNIFORMITY_PIU = :math:`(1 - \frac{\max \: G - \min \: G}{\max \: G + \min \: G}) \times 100`

The quantile :math:`q_p` of a random variable (or equivalently of its distribution) is
defined as the smallest number :math:`q`` such that the cumulative distribution function
is greater than or equal to some :math:`p`, where :math:`0<p<1`. This can be calculated
for a continuous random variable with density function :math:`f(x)` by solving

.. math::

    p = \int_{-\inf}^{q_p} f(x)dx 

for :math:`q_p`, or by using the inverse of the cumulative distribution function, 

.. math::

    q_p = F^{-1}(p). 
    
The :math:`p`-th quantile of a random variable :math:`X` is the value :math:`q_p` such that 

.. math::

    F(q_p) = P(X \leqslant q_p) = p


P01, P10, P25, P75, P90, P99 - the 1%, 10%, 25%, 75%, 90%, and 99% percentiles. A percentile :math:`q_p` 
is a solution of equation :math:`p = \int _{-\infty} ^{q_p} f(x)dx` where :math:`p=0.01, 0.1, 0.25, etc`, for example 
:math:`0.25 = \int _{-\infty} ^{0.25} f(x)dx`.  

QCOD = :math:`\frac {P75 - P25} {P75 + P25}`

MEDIAN -- the 50% percentile defined as :math:`0.5 = \int _{-\infty} ^{0.5} f(x)dx`, the value such that an equal number 
of samples are less than and greater than the value (for an odd sample size), or the average of the two central values (for an even sample size).

MEDIAN_ABSOLUTE_DEVIATION = :math:`\sigma = \frac{1}{n} \sum_i^n \left| g_i - MEDIAN \right|` 

INTERQUARTILE_RANGE = :math:`q_{0.75} - q_{0.25}` - the difference of the 1st and 3rd sample quartiles,

ROBUST_MEAN_ABSOLUTE_DEVIATION (RMAD) 

.. math::
    RMAD = \frac{1}{k} \underset{q_{0.1} \leqslant g_i \leqslant q_{0.9}} {\sum_i^n} |g_i - \mu_R| 

where 

.. math::
    \mu_R = \underset{q_{0.1} \leqslant g_i \leqslant q_{0.9}} { \frac{1}{n} \sum_i^n g_i } 
    
or, otherwise, MAD calculated over the subset of :math:`G=\{g_i\}^n` whose elements are in the :math:`[q_{0.1},q_{0.9}]` value interval.

References
----------

Zwillinger, D. (Ed.). CRC Standard Mathematical Tables and Formulae. Boca Raton, FL: CRC Press, p. 602, 1995.

