
Texture features / GLSZM
========================

A Gray Level Size Zone (GLSZM) quantifies gray level zones in an image. A gray level zone is defined as a the number
of connected voxels that share the same gray level intensity. A voxel is considered connected if the distance is 1
according to the Euclidean norm (8-connected region in 2D).
In a gray level size zone matrix :math:`P(i,j)` the :math:`(i,j)^{\text{th}}` element equals the number of zones
with gray level :math:`i` and size :math:`j` appear in image. Contrary to GLCM and GLRLM, the GLSZM is rotation
independent, with only one matrix calculated for all directions in the ROI.

As an example, consider the following 5x5 ROI image, with 5 discrete gray levels:

.. math::

    \textbf{G} = \begin{bmatrix}
    5 & 2 & 5 & 4 & 4\\
    3 & 3 & 3 & 1 & 3\\
    2 & 1 & 1 & 1 & 3\\
    4 & 2 & 2 & 2 & 3\\
    3 & 5 & 3 & 3 & 2 \end{bmatrix}

The GLSZM then becomes:

.. math::
    \textbf{P} = \begin{bmatrix}
    0 & 0 & 0 & 1 & 0\\
    1 & 0 & 0 & 0 & 1\\
    1 & 0 & 1 & 0 & 1\\
    1 & 1 & 0 & 0 & 0\\
    3 & 0 & 0 & 0 & 0 \end{bmatrix}


Let:


* :math:`N_g` be the number of discrete intensity values in the image
* :math:`N_s` be the number of discrete zone sizes in the image
* :math:`N_p` be the number of voxels in the image
* :math:`N_z` be the number of zones in the ROI, which is equal to :math:`\sum_{i=1}^{N_g}\sum^{N_s}_{j=1}{\textbf{P}(i,j)}` and :math:`1 \leq N_z \leq N_p`
* :math:`\textbf{P}(i,j)` be the size zone matrix
* :math:`p(i,j)` be the normalized size zone matrix, defined as :math:`p(i,j) = \frac{\textbf{P}(i,j)}{N_z}`

References
----------

Guillaume Thibault; Bernard Fertil; Claire Navarro; Sandrine Pereira; Pierre Cau; Nicolas Levy; Jean Sequeira; Jean-Luc Mari (2009). “Texture Indexes and Gray Level Size Zone Matrix. Application to Cell Nuclei Classification”. Pattern Recognition and Information Processing (PRIP): 140-145.
