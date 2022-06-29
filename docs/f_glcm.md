# Texture features / GLCM

 A Gray Level Co-occurrence Matrix (GLCM) of size $N_g \times N_g$ describes the second-order joint probability function of an image and is defined as $\textbf{P}(i,j|\delta,\theta)$.
  The $(i,j)$-th element of this matrix represents the number of times the combination of
  indices $i$ and $j$ occur in two pixels in the image, that are separated by a distance of $\delta$
  pixels along angle $\theta$.
  The distance $\delta$ from the center pixel is defined as the distance according to the Euclidean norm.
  For $\delta=1$, this results in 2 neighbors for each of 13 angles in 3D (26-connectivity) and for
  $\delta=2$ a 98-connectivity (49 unique angles).

  As an example, let the following matrix $\textbf{I}$ represent a 5x5 image, having 5 discrete
  grey levels:

  $$
    \textbf{I} = \begin{bmatrix}
    1 & 2 & 5 & 2 & 3\\
    3 & 2 & 1 & 3 & 1\\
    1 & 3 & 5 & 5 & 2\\
    1 & 1 & 1 & 1 & 2\\
    1 & 2 & 4 & 3 & 5 \end{bmatrix}
$$

  For distance $\delta = 1$ (considering pixels with a distance of 1 pixel from each other)
  and angle $\theta=0^\circ$ (horizontal plane, i.e. voxels to the left and right of the center voxel),
  the following symmetrical GLCM is obtained:

$$
    \textbf{P} = \begin{bmatrix}
    6 & 4 & 3 & 0 & 0\\
    4 & 0 & 2 & 1 & 3\\
    3 & 2 & 0 & 1 & 2\\
    0 & 1 & 1 & 0 & 0\\
    0 & 3 & 2 & 0 & 2 \end{bmatrix}
$$

  Let:

  - $\epsilon$ be an arbitrarily small positive number ($\approx 2.2\times10^{-16}$)
  - $\textbf{P}(i,j)$ be the co-occurence matrix for an arbitrary $\delta$ and $\theta$
  - $p_{ij}$ be the normalized co-occurence matrix and equal to
    $\frac{\textbf{P}(i,j)}{\sum{\textbf{P}(i,j)}}$
  - $N_g$ be the number of discrete intensity levels in the image
  - $p_x(i) = \sum^{N_g}_{j=1}{p_{ij}}$ be the marginal row probabilities
  - $p_y(j) = \sum^{N_g}_{i=1}{p_{ij}}$ be the marginal column probabilities
  - $\mu_x$ be the mean gray level intensity of $p_x$ and defined as
    $\mu_x = \sum^{N_g}_{i=1}{p_x(i)i}$
  - $\mu_y$ be the mean gray level intensity of $p_y$ and defined as
    $\mu_y = \sum^{N_g}_{j=1}{p_y(j)j}$
  - $\sigma_x$ be the standard deviation of $p_x$
  - $\sigma_y$ be the standard deviation of $p_y$
  - $p_{x+y}(k) = \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_{ij}},\text{ where }i+j=k,\text{ and }k=2,3,\dots,2N_g$
  - $p_{x-y}(k) = \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_{ij}},\text{ where }|i-j|=k,\text{ and }k=0,1,\dots,N_g-1$
  - $HX =  -\sum^{N_g}_{i=1}{p_x(i)\log_2\big(p_x(i)+\epsilon\big)}$ be the entropy of $p_x$
  - $HY =  -\sum^{N_g}_{j=1}{p_y(j)\log_2\big(p_y(j)+\epsilon\big)}$ be the entropy of $p_y$
  - $HXY =  -\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_{ij}\log_2\big(p_{ij}+\epsilon\big)}$ be the entropy of
    $p_{ij}$
  - $HXY1 =  -\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_{ij}\log_2\big(p_x(i)p_y(j)+\epsilon\big)}$
  - $HXY2 =  -\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_x(i)p_y(j)\log_2\big(p_x(i)p_y(j)+\epsilon\big)}$

  By default, the value of a feature is calculated on the GLCM for each angle separately, after which the mean of these
  values is returned. If distance weighting is enabled, GLCM matrices are weighted by weighting factor W and
  then summed and normalised. Features are then calculated on the resultant matrix.

## 2nd angular moment 
GLCM_ANGULAR2NDMOMENT $=  \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_{i,j}^2}$


## Contrast 
GLCM_CONTRAST $= \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{(i-j)^2p_{ij}}$


## Correlation 
GLCM_CORRELATION $= \frac{\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_{ij}ij-\mu_x\mu_y}}{\sigma_x(i)\sigma_y(j)}$


## Variance 
GLCM_VARIANCE $= \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{(i-\mu_x)^2p_{ij}}$ 
 

## Inverse difference moment
GLCM_INVERSEDIFFERENCEMOMENT $= \sum^{N_g-1}_{k=0}{\frac{p_{x-y}(k)}{1+k^2}}
$

## Sum average
GLCM_SUMAVERAGE $= \sum^{2N_g}_{k=2}{p_{x+y}(k)k}$

## Sum variance 
GLCM_SUMVARIANCE = $ \sum^{2N_g}_{k=2}{(k-SA)^2p_{x+y}(k)}$

## Sum entropy 
GLCM_SUMENTROPY $= \sum^{2N_g}_{k=2}{p_{x+y}(k)\log_2\big(p_{x+y}(k)+\epsilon\big)} $

## Entropy 
GLCM_ENTROPY $ = - \sum^{N_g}_{i=1}\sum^{N_g}_{j=1}
      {p_{ij}\log_2\big(p_{ij}+\epsilon\big)}$

## Difference variance 
GLCM_DIFFERENCEVARIANCE $= \sum^{N_g-1}_{k=0}{(k-DA)^2p_{x-y}(k)}$

## Difference entropy 
GLCM_DIFFERENCEENTROPY $= \sum^{N_g-1}_{k=0}{p_{x-y}(k)\log_2\big(p_{x-y}(k)+\epsilon\big)} $


## Informational Measure of Correlation 1 
GLCM_INFOMEAS1 $= \frac{HXY-HXY1}{\max\{HX,HY\}}$

## Informational Measure of Correlation 2 
GLCM_INFOMEAS2 $= \sqrt{1-e^{-2(HXY2-HXY)}}$

