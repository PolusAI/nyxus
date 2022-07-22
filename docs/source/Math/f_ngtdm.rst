
Texture features / NGTDM
========================

A Neighbouring Gray Tone Difference Matrix quantifies the difference between a gray value and the average gray value
of its neighbours within distance :math:`\delta`. The sum of absolute differences for gray level :math:`i` is stored in the matrix.
Let :math:`\textbf{X}_{gl}` be a set of segmented voxels and :math:`x_{gl}(j_x,j_y,j_z) \in \textbf{X}_{gl}` be the gray level of a voxel at postion
:math:`(j_x,j_y,j_z)`, then the average gray level of the neigbourhood is:

.. math::

    \bar{A}_i &= \bar{A}(j_x, j_y, j_z) \\
    &= \frac{1}{W} \sum_{k_x=-\delta}^{\delta}\sum_{k_y=-\delta}^{\delta} \sum_{k_z=-\delta}^{\delta}{x_{gl}(j_x+k_x, j_y+k_y, j_z+k_z)},

where

.. math::

    (k_x,k_y,k_z)\neq(0,0,0) 
and 

.. math::

    x_{gl}(j_x+k_x, j_y+k_y, j_z+k_z) \in \textbf{X}_{gl}



Here, :math:`W` is the number of voxels in the neighbourhood that are also in :math:`\textbf{X}_{gl}`.

As a two dimensional example, let the following matrix :math:`\textbf{I}` represent a 4x4 image,
having 5 discrete grey levels, but no voxels with gray level 4:

.. math::
    \textbf{I} = \begin{bmatrix}
    1 & 2 & 5 & 2\\
    3 & 5 & 1 & 3\\
    1 & 3 & 5 & 5\\
    3 & 1 & 1 & 1\end{bmatrix}


The following NGTDM can be obtained:

.. math::
    \begin{array}{cccc}
    i & n_i & p_i & s_i\\
    \hline
    1 & 6 & 0.375 & 13.35\\
    2 & 2 & 0.125 & 2.00\\
    3 & 4 & 0.25  & 2.63\\
    4 & 0 & 0.00  & 0.00\\
    5 & 4 & 0.25  & 10.075\end{array}


6 pixels have gray level 1, therefore:

:math:`s_1 = |1-10/3| + |1-30/8| + |1-15/5| + |1-13/5| + |1-15/5| + |1-11/3| = 13.35`

For gray level 2, there are 2 pixels, therefore:

:math:`s_2 = |2-15/5| + |2-9/3| = 2`

Similar for gray values 3 and 5:

:math:`s_3 = |3-12/5| + |3-18/5| + |3-20/8| + |3-5/3| = 3.03`

:math:`s_5 = |5-14/5| + |5-18/5| + |5-20/8| + |5-11/5| = 10.075`

Let:

:math:`n_i` be the number of voxels in :math:`X_{gl}` with gray level :math:`i`

:math:`N_{v,p}` be the total number of voxels in :math:`X_{gl}` and equal to :math:`\sum{n_i}` (i.e. the number of voxels
with a valid region; at least 1 neighbor). :math:`N_{v,p} \leq N_p`, where :math:`N_p` is the total number of voxels in the ROI.

:math:`p_i` be the gray level probability and equal to :math:`n_i/N_v`

:math:`s_i = \sum^{n_i}{|i-\bar{A}_i|}` when :math:`n_i \neq 0` and :math:`s_i = 0` when :math:`n_i = 0`.

be the sum of absolute differences for gray level :math:`i`.

:math:`N_g` be the number of discrete gray levels

:math:`N_{g,p}` be the number of gray levels where :math:`p_i \neq 0`

Coarseness
----------

NGTDM_COARSENESS :math:`=  \frac{1}{\sum^{N_g}_{i=1}{p_{i}s_{i}}}`

Contrast
--------

Assuming :math:`p_i` and :math:`p_j` are row indices of the NGTDM matrix, 

NGTDM_CONTRAST :math:`= \left(\frac{1}{N_{g,p}(N_{g,p}-1)}\sum^{N_g}_{i=1}\sum^{N_g}_{j=1}{p_{i}p_{j}(i-j)^2}\right) \left(\frac{1}{N_{v,p}}\sum^{N_g}_{i=1}{s_i}\right)` 
where :math:`p_i \neq 0` and :math:`p_j \neq 0`

Busyness
--------

NGTDM_BUSYNESS :math:`= \frac{\sum^{N_g}_{i = 1}{p_{i}s_{i}}}{\sum^{N_g}_{i = 1}\sum^{N_g}_{j = 1}{|ip_i - jp_j|}}` where :math:`p_i \neq 0`, :math:`p_j \neq 0`

Complexity
----------

NGTDM_COMPLEXITY :math:`= \frac{1}{N_{v,p}}\sum^{N_g}_{i = 1}\sum^{N_g}_{j = 1}{|i - j| \frac{p_{i}s_{i} + p_{j}s_{j}}{p_i + p_j}}` 
where :math:`p_i \neq 0, p_j \neq 0`

Strength
--------

NGTDM_STRENGTH :math:`=  \frac{\sum^{N_g}_{i = 1}\sum^{N_g}_{j = 1}{(p_i + p_j)(i-j)^2}}{\sum^{N_g}_{i = 1}{s_i}}` where :math:`p_i \neq 0, p_j \neq 0`

References
----------

Amadasun M, King R; Textural features corresponding to textural properties; Systems, Man and Cybernetics, IEEE Transactions on 19:1264-1274 (1989). doi: 10.1109/21.44046
