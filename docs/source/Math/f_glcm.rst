
Texture features / GLCM
=======================

A Gray Level Co-occurrence Matrix (GLCM) of size :math:`N_g \times N_g` describes the joint probability function of an image and is defined as :math:`\textbf{P}(i,j|\delta,\theta)`.
The :math:`(i,j)`-th element of this matrix represents the number of times the combination of
indices :math:`i` and :math:`j` occur in two pixels in the image, that are separated by a distance of :math:`\delta`
pixels along angle :math:`\theta`.
The distance :math:`\delta` from the center pixel is defined as the distance according to the Euclidean norm.
For :math:`\delta=1`, this results in 2 neighbors for each of 13 angles in 3D (26-connectivity) and for
:math:`\delta=2` a 98-connectivity (49 unique angles).

As an example, let the following matrix represent a 5x5 ROI image having 5 discrete
grey levels:

.. math::

  \textbf{G} = \begin{bmatrix}
  1 & 2 & 5 & 2 & 3\\
  3 & 2 & 1 & 3 & 1\\
  1 & 3 & 5 & 5 & 2\\
  1 & 1 & 1 & 1 & 2\\
  1 & 2 & 4 & 3 & 5 \end{bmatrix}


For distance :math:`\delta = 1` (considering pixels with a distance of 1 pixel from each other)
and angle :math:`\theta=0^\circ`` (horizontal plane, i.e. voxels to the left and right of the center voxel),
the following symmetrical GLCM is obtained:

.. math::

    \textbf{P} = \begin{bmatrix}
    6 & 4 & 3 & 0 & 0\\
    4 & 0 & 2 & 1 & 3\\
    3 & 2 & 0 & 1 & 2\\
    0 & 1 & 1 & 0 & 0\\
    0 & 3 & 2 & 0 & 2 \end{bmatrix}


Let:

* :math:`\epsilon` be an arbitrarily small positive number (:math:`\approx 2.2\times10^{-16}`)
* :math:`\textbf{P}(i,j)` be the co-occurence matrix for an arbitrary :math:`\delta` and :math:`\theta``
* :math:`p_{ij}` be the normalized co-occurence matrix and equal to :math:`\frac{\textbf{P}(i,j)}{\sum{\textbf{P}(i,j)}}`
* :math:`N_g`` be the number of discrete intensity levels in the image
* :math:`p_x(i) = \sum^{N_g}_{j=1}{p_{ij}}` be the marginal row probabilities
* :math:`p_y(j) = \sum^{N_g}_{i=1}{p_{ij}}`` be the marginal column probabilities
* :math:`\mu_x`` be the mean gray level intensity of :math:`p_x` and defined as :math:`\mu_x = \sum^{N_g}_{i=1}{p_x(i)i}`
* :math:`\mu_y`` be the mean gray level intensity of :math:`p_y` and defined as :math:`\mu_y = \sum^{N_g}_{j=1}{p_y(j)j}`
* :math:`\sigma_x` be the standard deviation of :math:`p_x``
* :math:`\sigma_y`` be the standard deviation of :math:`p_y`
* :math:`p_{x+y}(k) = \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_{ij}},\text{ where }i+j=k,\text{ and }k=2,3,\dots,2N_g`
* :math:`p_{x-y}(k) = \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_{ij}},\text{ where }|i-j|=k,\text{ and }k=0,1,\dots,N_g-1`
* :math:`HX =  -\sum^{N_g}_{i=1}{p_x(i)\log_2\big(p_x(i)+\epsilon\big)}` be the entropy of :math:`p_x`
* :math:`HY =  -\sum^{N_g}_{j=1}{p_y(j)\log_2\big(p_y(j)+\epsilon\big)}`` be the entropy of :math:`p_y`
* :math:`HXY =  -\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_{ij}\log_2\big(p_{ij}+\epsilon\big)}`` be the entropy of :math:`p_{ij}`
* :math:`HXY1 =  -\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_{ij}\log_2\big(p_x(i)p_y(j)+\epsilon\big)}`
* :math:`HXY2 =  -\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_x(i)p_y(j)\log_2\big(p_x(i)p_y(j)+\epsilon\big)}``

By default, the value of a feature is calculated on the GLCM for each angle separately, after which the mean of these
values is returned. If distance weighting is enabled, GLCM matrices are weighted by weighting factor W and
then summed and normalised. 

The following features are then calculated based on the collocation matrix.

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_ANGULAR2NDMOMENT}} {\textup{angular second moment}} =  \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_{i,j}^2}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_ACOR}} {\textup{autocorrelation}} = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} p(i,j) i j

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_CLUPROM}} {\textup{cluster prominence}} = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} (i + j - \mu_x - \mu_y) ^4 p(i,j)

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_CLUSHADE}} {\textup{cluster shade}} = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} (i + j - \mu_x - \mu_y) ^3 p(i,j)

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_CLUTEND}} {\textup{cluster tendency}} = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} (i + j - \mu_x - \mu_y) ^2 p(i,j)

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_CONTRAST}} {\textup{contrast}} = \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{(i-j)^2p_{ij}}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_CORRELATION}} {\textup{correlation}} = \frac{\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_{ij}ij-\mu_x\mu_y}}{\sigma_x(i)\sigma_y(j)}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_DIFFERENCEAVERAGE}} {\textup{difference average}} =\sum_{k=0}^{N_g-1} k\, p_{i-j,k}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_DIFFERENCEENTROPY}} {\textup{difference entropy}} = \sum^{N_g-1}_{k=0}{p_{x-y}(k)\log_2\big(p_{x-y}(k)+\epsilon\big)}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_DIFFERENCEVARIANCE}} {\textup{difference variance}} = \sum^{N_g-1}_{k=0}{(k-\mu_d)^2p_{x-y}(k)}
	
where the difference average :math:`\mu_d` is 

.. math::
	\mu_d = \sum^{N_g-1}_{k=0} kp_{x-y}(k) .

.. math::  
	\underset{\mathrm{Nyxus \, code: \, GLCM\_DIS}} {\textup{dissimilarity}} = \sum_{i=1}^{N_g} \sum_{j=1}^{N_g} |i-j|\, p_{ij}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_JE}} {\textup{joint entropy}} = - \sum^{N_g}_{i=1} \sum^{N_g}_{j=1} p(i,j) \log_2 ( p(i,j) + \epsilon )

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_HOMOGENEITY1}} {\textup{homogeneity 1}} = \sum^{N_g}_{i=1}\displaystyle\sum^{N_g}_{j=1} \frac{p(i,j)}{1+|i-j|}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_HOM2}} {\textup{homogeneity 2}} = \sum^{N_g}_{i=1} \sum^{N_g}_{j=1}{\frac{p(i,j)}{1+|i-j|^2}}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_INVERSEDIFFERENCEMOMENT}} {\textup{inverse difference moment}} = \sum^{N_g-1}_{k=0}{\frac{p_{x-y}(k)}{1+k^2}}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_IDMN}} {\textup{inverse difference moment normalized}} = \sum^{N_g-1}_{k=0} \frac {p_{x-y}(k)} {1+\frac{k^2}{N_g^2}}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_ID}} {\textup{inverse difference}} =\sum_{i=1}^{N_g} \sum_{j=1}^{N_g} \frac{p_{ij}}{1+|i-j|}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_IDN}} {\textup{inverse difference normalized}} = \sum^{N_g-1}_{k=0} \frac {p_{x-y}(k)} {1+\frac{k}{N_g}}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_INFOMEAS1}} {\textup{informational measure of correlation 1}} = \frac{HXY-HXY1}{\max{HX,HY}}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_INFOMEAS2}} {\textup{information measure of correlation 2}} = \sqrt{1-e^{-2(HXY2-HXY)}}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_VARIANCE}} {\textup{variance}} = \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{(i-\mu_x)^2p_{ij}}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_SUMAVERAGE}} {\textup{sum average}} = \sum^{2N_g}_{k=2} {p_{x+y}(k)k}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_SUMVARIANCE}} {\textup{sum variance}} = \sum^{2N_g}_{k=2} {(k-SA)^2p_{x+y}(k)}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_SUMENTROPY}} {\textup{sum entropy}} = \sum^{2N_g}_{k=2} {p_{x+y}(k)\log_2\big(p_{x+y}(k)+\epsilon\big)}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_JAVE}} {\textup{joint average}} = \sum_{i=1}^{N_g} \sum_{j=1}^{N_g} i\, p_{ij}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_JE}} {\textup{joint entropy}} = -\sum_{i=1}^{N_g} \sum_{j=1}^{N_g} p_{ij} \log_2 p_{ij}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_JMAX}} {\textup{joint maximum}} = \text{max}(p_{ij})

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLCM\_JVAR}} {\textup{joint variance}} = \sum_{i=1}^{N_g} \sum_{j=1}^{N_g} \left(i-\mu\right)^2 p_{ij}

References
----------

Haralick, R., Shanmugan, K., Dinstein, I; Textural features for image classification; IEEE Transactions on Systems, Man and Cybernetics; 1973(3), p610-621
