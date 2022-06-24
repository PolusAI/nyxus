# Texture features / GLRLM

 A Gray Level Run Length Matrix (GLRLM) quantifies gray level runs, which are defined as the length in number of
  pixels, of consecutive pixels that have the same gray level value. In a gray level run length matrix
  $\textbf{P}(i,j|\theta)$, the $(i,j)^{\text{th}}$ element describes the number of runs with gray level
  $i$ and length $j$ occur in the image (ROI) along angle $\theta$.

  As an example, consider the following 5x5 image, with 5 gray levels:

 $$
    \textbf{I} = \begin{bmatrix}
    5 & 2 & 5 & 4 & 4\\
    3 & 3 & 3 & 1 & 3\\
    2 & 1 & 1 & 1 & 3\\
    4 & 2 & 2 & 2 & 3\\
    3 & 5 & 3 & 3 & 2 \end{bmatrix}
$$

  The GLRLM for $\theta = 0$, where 0 degrees is the horizontal direction, then becomes:

$$
    \textbf{P} = \begin{bmatrix}
    1 & 0 & 1 & 0 & 0\\
    3 & 0 & 1 & 0 & 0\\
    4 & 1 & 1 & 0 & 0\\
    1 & 1 & 0 & 0 & 0\\
    3 & 0 & 0 & 0 & 0 \end{bmatrix}
$$

  Let:

  - $N_g$ be the number of discrete intensity values in the image
  - $N_r$ be the number of discrete run lengths in the image
  - $N_p$ be the number of voxels in the image
  - $N_r(\theta)$ be the number of runs in the image along angle $\theta$, which is equal to
    $\sum^{N_g}_{i=1}\sum^{N_r}_{j=1}{\textbf{P}(i,j|\theta)}$ and $1 \leq N_r(\theta) \leq N_p$
  - $\textbf{P}(i,j|\theta)$ be the run length matrix for an arbitrary direction $\theta$
  - $p(i,j|\theta)$ be the normalized run length matrix, defined as $p(i,j|\theta) =
    \frac{\textbf{P}(i,j|\theta)}{N_r(\theta)}$

