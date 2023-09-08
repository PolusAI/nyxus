
Texture features / GLDZM
========================

The Grey Level Distance Zone Matrix (GLDZM) indicates the number of times each grey level's zones occur within a distance from the zone to the ROI border.

A zone is a continuous set of pixels of same intensity (or "grey level").

The continuity is meant as a 4-connected neighbourhood. For example, the following intensity image matrix :math:`I` of 2 non-zero intensities 1 and 3 contains 4 zones of intensity 3 -- 1 single-pixel, 1 2-pixel, 1 3-pixel, and 2 4-pixel zones.

.. math::

    I = \begin{bmatrix}
    0 & 0 & 0 & 1 & 1 & 1 & 0\\
    0 & 0 & \fbox{3} & 1 & \fbox{3} & 1 & 0\\
    1 & \fbox{3} & 1 & 1 & \fbox{3} & 1 & 0\\
    1 & \fbox{3} & 1 & 1 & \fbox{3} & \fbox{3} & 1\\
    \fbox{3} & 1 & \fbox{3} & \fbox{3} & 1 & 1 & 1\\
    \fbox{3} & 1 & \fbox{3} & \fbox{3} & 1 & 1 & 0\\
    \fbox{3} & 1 & 0 & 0 & 0 & 1 & 0
	\end{bmatrix}

The zone's distance is the minimum of its each pixel's distance to the ROI or image border measured as the number of pixel boundaries to the first off-ROI or off-image pixel.

Considering the following ROI image

.. math::
    R = \begin{bmatrix}
    0 & 0 & 0 & 1 & 1 & 1 & 0\\
    0 & 0 & 1 & 1 & 1 & 1 & 0\\
    1 & 1 & 1 & 1 & 1 & 1 & 0\\
    1 & 1 & 1 & 1 & 1 & 1 & 1\\
    1 & 1 & 1 & 1 & 1 & 1 & 1\\
    1 & 1 & 1 & 1 & 1 & 1 & 0\\
    1 & 1 & 0 & 0 & 0 & 1 & 0
	\end{bmatrix}

the distances of zons of intensity 3, ignoring pixels of other non-zero intensities (shown as :math:`*`), in the masked image (whose off-ROI pixels are shown as :math:`\times`) are

.. math::
    D = \begin{bmatrix}
    \times & \times & \times & * & * & * & \times	\\
    \times & \times & \fbox{2} & * & \fbox{2} & * & \times	\\
    * & \fbox{2} & * & * & \fbox{2} & * & \times	\\
    * & \fbox{2} & * & * & \fbox{2} & \fbox{2} & *	\\
    \fbox{1} & * & \fbox{2} & \fbox{2} & * & * & *	\\
    \fbox{1} & * & \fbox{2} & \fbox{2} & * & * & \times	\\
    \fbox{1} & * & \times & \times & \times & * & \times
	\end{bmatrix}


The following example is an image having 5 discrete grey values masked with the above ROI mask :math:`R` :

.. math::

    I_2 = \begin{bmatrix}
    \times & \times & \times & 4 & 4 & 4 & \times	\\
    \times & \times & 3 & 1 & 3 & 4 & \times	\\
    2 & 1 & 1 & 1 & 3 & 2 & \times	\\
    4 & 4 & 2 & 2 & 3 & 3 & 1	\\
    3 & 5 & 3 & 3 & 2 & 1 & 1	\\
    3 & 5 & 3 & 3 & 2 & 4 & \times	\\
    3 & 1 & \times & \times & \times & 4 & \times
	\end{bmatrix}


Its distance map :math:`D_2` is:

.. math::

    D_2 = \begin{bmatrix}
    \times & \times & \times & 1 & 1 & 1 & \times	\\
    \times & \times & 1 & 2 & 2 & 1 & \times	\\
    1 & 1 & 2 & 3 & 2 & 1 & \times	\\
    1 & 2 & 3 & 3 & 3 & 2 & 1	\\
    1 & 2 & 2 & 2 & 2 & 2 & 1	\\
    1 & 2 & 1 & 1 & 1 & 1 & \times	\\
    1 & 1 & \times & \times & \times & 1 & \times
	\end{bmatrix}

In a grey level distance zone matrix (GLDZM) :math:`M`, the element :math:`(x,d)` describes the number of zones in an image
with grey level :math:`x` located at distance :math:`d` from the edge of the ROI or image border.

Applied to the example, the GLDZM :math:`M(I_2)` of image :math:`I_2` having distance matrix :math:`D_2`, is:

.. math::

    M(I_2)=\begin{bmatrix}
    3 & 0 & 0\\
    3 & 0 & 1\\
    3 & 1 & 0\\
    2 & 0 & 0\\
    1 & 1 & 0\end{bmatrix}

Let
:math:`m(x,d)` be an element of the distance zone matrix corresponding to grey level :math:`x` and zone distance :math:`d` ,

:math:`N_g` -- the number of grey levels ,

:math:`N_d` -- the maximum zone distance, and

:math:`N_s` -- the number of zones of any non-zero intensity.

:math:`p(x,d)` be an element of the normalized distance zone matrix expressing the relative probability of element :math:`(x,d)`, defined as

.. math::
	p_{x,d} = \frac{m_{x,d}}{N_s} .

:math:`N_v` is the number of ROI image pixels.

In addition, the marginal totals

.. math::
	m_{x,\cdot} = m_x = \sum_d m_{x,d}

represent the total of all zones with a given intensity :math:`x`, and

.. math::
	m_{\cdot, d} = m_d = \sum_x m_{x,d}

represent the total of all zones with a given distance :math:`d`.

The following features are then defined:

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_SDE}} {\textup{Small Distance Emphasis}} = \frac{1}{N_s} \sum_d \frac{m_d}{d^2}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_LDE}} {\textup{Large Distance Emphasis}} = \frac{1}{N_s} \sum_d d^2 m_d

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_LGLE}} {\textup{Low Grey Level Emphasis}} = \frac{1}{N_s} \sum_x  \frac{m_x}{x^2}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_HGLE}} {\textup{High Grey Level Emphasis}} = \frac{1}{N_s} \sum_x x^2 m_x

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_SDLGLE}} {\textup{Small Distance Low Grey Level Emphasis}} = \frac{1}{N_s} \sum_x \sum_d \frac{ m_{x,d}}{x^2 d^2}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_SDHGLE}} {\textup{Small Distance High Grey Level Emphasis}} = \frac{1}{N_s} \sum_x \sum_d \frac{x^2  m_{x,d}}{d^2}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_LDLGLE}} {\textup{Large Distance Low Grey Level Emphasis}} = \frac{1}{N_s} \sum_x \sum_d \frac{d^2 m_{x,d}}{x^2}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_LDHGLE}} {\textup{Large Distance High Grey Level Emphasis}} = \frac{1}{N_s} \sum_x \sum_d \x^2 d^2 m_{x,d}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_GLNU}} {\textup{Grey Level Non-Uniformity}} = \frac{1}{N_s} \sum_x m_x^2

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_GLNUN}} {\textup{Grey Level Non-Uniformity Normalized}} = \frac{1}{N_s^2} \sum_x m_x^2

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_ZDNU}} {\textup{Zone Distance Non-Uniformity}} = \frac{1}{N_s} \sum_d m_d^2

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_ZDNUN}} {\textup{Zone Distance Non-Uniformity Normalized}} = \frac{1}{N_s^2} \sum_d m_d^2

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_ZP}} {\textup{Zone Percentage}} = \frac{N_s}{N_v}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_GLM}} {\textup{Grey Level Mean}} = \mu_x = \sum_x \sum_d x p_{x,d}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_GLV}} {\textup{Grey Level Variance}} = \sum_x \sum_d \left(x - \mu_x \right)^2 p_{x,d}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_ZDM}} {\textup{Zone Distance Mean}} = \mu_d = \sum_x \sum_d d p_{x,d}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_ZDV}} {\textup{Zone Distance Variance}} = \sum_x \sum_d \left(d - \mu_d \right)^2 p_{x,d}

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_ZDE}} {\textup{Zone Distance Entropy}} = - \sum_x \sum_d p_{x,d} \textup{log}_2 ( p_{x,d} )

.. math::
	\underset{\mathrm{Nyxus \, code: \, GLDZM\_GLE}} {\textup{Grey Level Entropy}} = - \sum_x \sum_d p_{x,d} \textup{log}_2 ( p_{x,d} )


References
----------

Thibault, G., Angulo, J., and Meyer, F. (2014); Advanced statistical matrices for texture characterization: application to cell classification; IEEE transactions on bio-medical engineering, 61(3):630-7.
