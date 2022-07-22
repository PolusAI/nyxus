
Zernike moment features
=======================

Zernike moments of order n with repetition m for an image function :math:`f(x,y)` defined on a square somain :math:`N \times N` 
are defined as 

.. math::
    A_{nm} = \frac{n+1}{\pi} \underset{x^2+y^2 \le 1} {\int \int} f(x,y) V^*_{nm}(x,y) dxdy


Consider a set of orthogonal functions with simple rotation properties which forms a complete
orthogonal set over the interior of the unit circle. The form of these polynomials is 

.. math:: 
    V_{nm} (x,y) = V_{nm} (\rho \:\text{sin} \theta, \rho \:\text{cos} \theta) = R_{nm} (\rho) e^{j m \theta}

where :math:`V^*_{nm}` is the complex conjugate of the complex polynomials :math:`V_{nm}(x,y)`

.. math::
    V_{nm}(x,y)=R_{nm}(r) e^{j m \theta} 

where :math:`r=\sqrt{x^2+y^2}`, :math:`0 \leqslant r \leqslant 1`, :math:`\sqrt{-1}`, :math:`n \geqslant 0`, 
:math:`|m| \leqslant n`, :math:`n-m=even`, and :math:`\theta = \text{arctg} \frac{y}{x}`.

Zernike real valued radial polynomials :math:`R_{nm}(r)` are given by

.. math::
    R_{nm}(r) = \underset{k=|m| \atop \: n-k=even} {\sum ^n} B_{nmk}r^k

where 

.. math::
    B_{nmk} = \frac{ -1^{\frac{n-k}{2}}(\frac{n+k}{2})! } { (\frac{n-k}{2})! (\frac{k+m}{2})! (\frac{k-m}{2})! }

Approximating the double integration for the discrete image function on the domain of size :math:`N \times N`, we get 

.. math ::
    \hat Z_{nm} = \frac {n+1}{\pi} \sum _{i=0}^{N-1} \underset{x_i^2+y_j^2 \leqslant 1}{\sum _{j=0}^{N-1}} f(x_i,y_j)V_{nm}^* (x_i,y_j) \delta a

where :math:`\delta A = dxdy` is an elemental area of the normalized square image in discrete form when a square image of 
any size is mapped on the unit disk. If the image is square-shaped and :math:`R = \frac {N}{\sqrt{2}}` is the 
enclosing circle radius, then :math:`\delta A = \frac{1}{R^2}`.


References
----------

A. Tahmasbi, F. Saki, S.B. Shokouhi. Classification of benign and malignant masses based on Zernike moments. 
Comput Biol Med. 2011 Aug;41(8): 726-35. doi: 10.1016/j.compbiomed.2011.06.009. Epub 2011 Jul 1. PMID: 21722886.

C. Singh, E. Walia. Algorithms for fast computation of Zernike moments and their numerical stability. 
Image and Vision Computing, Volume 29, Issue 4, 2011: 251-259, ISSN 0262-8856, https://doi.org/10.1016/j.imavis.2010.10.003.