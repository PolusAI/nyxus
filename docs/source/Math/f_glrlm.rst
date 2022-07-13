
Texture features / GLRLM
========================

A Gray Level Run Length Matrix (GLRLM) quantifies gray level runs, which are defined as the length in number of
pixels, of consecutive pixels that have the same gray level value. In a gray level run length matrix
:math:`\textbf{P}(i,j|\theta)`, the :math:`(i,j)^{\text{th}}` element describes the number of runs with gray level
:math:`i` and length :math:`j` occur in the image (ROI) along angle :math:`\theta`.

As an example, consider the following 5x5 image, with 5 gray levels:

.. math::

  \textbf{I} = \begin{bmatrix}
  5 & 2 & 5 & 4 & 4\\
  3 & 3 & 3 & 1 & 3\\
  2 & 1 & 1 & 1 & 3\\
  4 & 2 & 2 & 2 & 3\\
  3 & 5 & 3 & 3 & 2 \end{bmatrix}


The GLRLM for :math:`\theta = 0`, where 0 degrees is the horizontal direction, then becomes:

.. math::
  \textbf{P} = \begin{bmatrix}
  1 & 0 & 1 & 0 & 0\\
  3 & 0 & 1 & 0 & 0\\
  4 & 1 & 1 & 0 & 0\\
  1 & 1 & 0 & 0 & 0\\
  3 & 0 & 0 & 0 & 0 \end{bmatrix}


Let:

* :math:`N_g` be the number of discrete intensity values in the image
* :math:`N_r` be the number of discrete run lengths in the image
* :math:`N_p` be the number of voxels in the image
* :math:`N_r(\theta)` be the number of runs in the image along angle :math:`\theta`, which is equal to :math:`\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\textbf{P}(i,j|\theta)}` and :math:`1 \leq N_r(\theta) \leq N_p`
* :math:`\textbf{P}(i,j|\theta)` be the run length matrix for an arbitrary direction :math:`\theta`
* :math:`p(i,j|\theta)` be the normalized run length matrix, defined as :math:`p(i,j|\theta) =
\frac{\textbf{P}(i,j|\theta)}{N_r(\theta)}`
