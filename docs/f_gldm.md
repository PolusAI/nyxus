# Texture features / GLDM

A Gray Level Dependence Matrix (GLDM) quantifies gray level dependencies in an image.
  A gray level dependency is defined as a the number of connected voxels within distance $\delta$ that are
  dependent on the center voxel.
  A neighbouring voxel with gray level $j$ is considered dependent on center voxel with gray level $i$
  if $|i-j|\le\alpha$. In a gray level dependence matrix $\textbf{P}(i,j)$ the $(i,j)$-th
  element describes the number of times a voxel with gray level $i$ with $j$ dependent voxels
  in its neighbourhood appears in image.

  As an example, consider the following 5x5 image, with 5 gray levels:

 $$
    \textbf{I} = \begin{bmatrix}
    5 & 2 & 5 & 4 & 4\\
    3 & 3 & 3 & 1 & 3\\
    2 & 1 & 1 & 1 & 3\\
    4 & 2 & 2 & 2 & 3\\
    3 & 5 & 3 & 3 & 2 \end{bmatrix}
$$

  For $\alpha=0$ and $\delta = 1$, the GLDM then becomes:

$$
    \textbf{P} = \begin{bmatrix}
    0 & 1 & 2 & 1 \\
    1 & 2 & 3 & 0 \\
    1 & 4 & 4 & 0 \\
    1 & 2 & 0 & 0 \\
    3 & 0 & 0 & 0 \end{bmatrix}
$$

  Let:

  - $N_g$ be the number of discrete intensity values in the image
  - $N_d$ be the number of discrete dependency sizes in the image
  - $N_z$ be the number of dependency zones in the image, which is equal to
    $\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)}$
  - $\textbf{P}(i,j)$ be the dependence matrix
  - $p(i,j)$ be the normalized dependence matrix, defined as $p(i,j) = \frac{\textbf{P}(i,j)}{N_z}$

## Small Dependence Emphasis 
GLDM_SDE $= \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)}{i^2}}}{N_z}$


## Large Dependence Emphasis 
GLDM_LDE $= \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)j^2}}{N_z}$

## Gray Level Non-Uniformity 
GLDM_GLN $= \frac{\sum^{N_g}_{i=1}\left(\sum^{N_d}_{j=1}{\textbf{P}(i,j)}\right)^2}{N_z}$

## Dependence Non-Uniformity 
GLDM_DN $= \frac{\sum^{N_d}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j)}\right)^2}{N_z}$

## Dependence Non-Uniformity Normalized 
GLDM_DNN $= \frac{\sum^{N_d}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j)}\right)^2}{N_z^2}$

## Gray Level Variance 
GLDM_GLV $= \sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{p(i,j)(i - \mu)^2}$ where $\mu = \sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{ip(i,j)}$

## Dependence Variance 
GLDM_DV $= \sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{p(i,j)(j - \mu)^2}$ where $\mu = \sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{jp(i,j)}$

## Dependence Entropy 
GLDM_DE $= -\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{p(i,j)\log_{2}(p(i,j)+\epsilon)}$

## Low Gray Level Emphasis 
GLDM_LGLE $=  \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)}{i^2}}}{N_z}$

## High Gray Level Emphasis 
GLDM_HGLE $=  \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)i^2}}{N_z}$

## Small Dependence Low Gray Level Emphasis 
GLDM_SDLGLE $=  \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)}{i^2j^2}}}{N_z}$

## Small Dependence High Gray Level Emphasis 
GLDM_SDHGLE $=  \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)i^2}{j^2}}}{N_z}$

## Large Dependence Low Gray Level Emphasis 
GLDM_LDLGLE $=  \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)j^2}{i^2}}}{N_z}$

## Large Dependence High Gray Level Emphasis 
GLDM_LDHGLE $=  \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)i^2j^2}}{N_z}$

