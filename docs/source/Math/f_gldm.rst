
Texture features / GLDM
=======================

A Gray Level Dependence Matrix (GLDM) quantifies gray level dependencies in an image.
A gray level dependency is defined as a the number of connected voxels within distance :math:`\delta` that are
dependent on the center voxel.
A neighbouring voxel with gray level :math:`j` is considered dependent on center voxel with gray level :math:`i`
if :math:`|i-j|\le\alpha`. In a gray level dependence matrix :math:`\textbf{P}(i,j)` the :math:`(i,j)`-th
element describes the number of times a voxel with gray level :math:`i` with :math:`j` dependent voxels
in its neighbourhood appears in image.

As an example, consider the following 5x5 ROI image having 5 gray levels:

.. math::

    \textbf{G} = \begin{bmatrix}
    5 & 2 & 5 & 4 & 4\\
    3 & 3 & 3 & 1 & 3\\
    2 & 1 & 1 & 1 & 3\\
    4 & 2 & 2 & 2 & 3\\
    3 & 5 & 3 & 3 & 2 \end{bmatrix}

For :math:`\alpha=0` and :math:`\delta = 1`, the GLDM then becomes:

.. math::

    \textbf{P} = \begin{bmatrix}
    0 & 1 & 2 & 1 \\
    1 & 2 & 3 & 0 \\
    1 & 4 & 4 & 0 \\
    1 & 2 & 0 & 0 \\
    3 & 0 & 0 & 0 \end{bmatrix}


Let:


* :math:`N_g` be the number of discrete intensity values in the image
* :math:`N_d` be the number of discrete dependency sizes in the image
* :math:`N_z` be the number of dependency zones in the image, which is equal to :math:`\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)}`
* :math:`\textbf{P}(i,j)` be the dependence matrix
* :math:`p(i,j)` be the normalized dependence matrix, defined as :math:`p(i,j) = \frac{\textbf{P}(i,j)}{N_z}`

Small Dependence Emphasis
-------------------------

GLDM_SDE :math:`= \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)}{i^2}}}{N_z}`

Large Dependence Emphasis
-------------------------

GLDM_LDE :math:`= \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)j^2}}{N_z}`

Gray Level Non-Uniformity
-------------------------

GLDM_GLN :math:`= \frac{\sum^{N_g}_{i=1}\left(\sum^{N_d}_{j=1}{\textbf{P}(i,j)}\right)^2}{N_z}`

Dependence Non-Uniformity
-------------------------

GLDM_DN :math:`= \frac{\sum^{N_d}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j)}\right)^2}{N_z}`

Dependence Non-Uniformity Normalized
------------------------------------

GLDM_DNN :math:`= \frac{\sum^{N_d}_{j=1}\left(\sum^{N_g}_{i=1}{\textbf{P}(i,j)}\right)^2}{N_z^2}`

Gray Level Variance
-------------------

GLDM_GLV :math:`= \sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{p(i,j)(i - \mu)^2}`

where,

:math:`\mu = \sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{ip(i,j)}`

Dependence Variance
-------------------

GLDM_DV :math:`= \sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{p(i,j)(j - \mu)^2}` where :math:`\mu = \sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{jp(i,j)}`

Dependence Entropy
------------------

GLDM_DE :math:`= -\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{p(i,j)\log_{2}(p(i,j)+\epsilon)}`

Low Gray Level Emphasis
-----------------------

GLDM_LGLE :math:`=  \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)}{i^2}}}{N_z}`

High Gray Level Emphasis
------------------------

GLDM_HGLE :math:`=  \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)i^2}}{N_z}`

Small Dependence Low Gray Level Emphasis
----------------------------------------

GLDM_SDLGLE :math:`=  \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)}{i^2j^2}}}{N_z}`

Small Dependence High Gray Level Emphasis
-----------------------------------------

GLDM_SDHGLE :math:`=  \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)i^2}{j^2}}}{N_z}`

Large Dependence Low Gray Level Emphasis
----------------------------------------

GLDM_LDLGLE :math:`=  \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\frac{\textbf{P}(i,j)j^2}{i^2}}}{N_z}`

Large Dependence High Gray Level Emphasis
-----------------------------------------

GLDM_LDHGLE :math:`=  \frac{\sum^{N_g}_{i=1}\sum^{N_d}_{j=1}{\textbf{P}(i,j)i^2j^2}}{N_z}`
