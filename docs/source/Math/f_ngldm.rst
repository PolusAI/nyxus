
Texture features / NGLDM
=============================================================================

The neighbouring grey level dependence matrix (NGLDM) features quantify coarseness of the texture in a ROI in a rotationally invariant way.

NGLDM is based on the concept of a neighbourhood around a pixel determined by position and a concept of dependence between two pixels determined by grey level values. All pixels within Chebyshev distance :math:`\delta`
are considered to belong to the pixel's neighbourhood. All pixels whose grey level differenece is within :math:`\alpha` are considered to be dependent.

Let :math:`I_u` be intensity of a pixel at location :math:`u`,
:math:`d_{cheb}(u,v) = max(|u-v|)` be the Chebyshev distance between locations :math:`u` and :math:`v`,
:math:`[ b ]` be the Iverson bracket over expression :math:`b`.

A pixel at location :math:`m` is said to belong to the neighborhood of
a "central" pixel at location :math:`k` if :math:`d_{cheb} (k-m) \leq \delta`. Additionally, the neighboring pixel is said to be dependent on the central pixel if :math:`|I_k - I_m| \leq \alpha` with respect to
coarseness parameter :math:`\alpha \geq 0`. The number of dependent pixels :math:`j_k` within the neighborhood of pixel at location :math:`k` is defined as

.. math:: j_k = \sum_{m_y{=}-\delta}^\delta \sum_{m_x{=}-\delta}^\delta \big[|X_{d}(\mathbf{k})-X_{d}(\mathbf{k}+\mathbf{m})| \leq \alpha\big]

Let
:math:`N_g` be the number of unique grey level values of the pixels within the ROI,
:math:`N_r=\text{max}(j_k)` be the maximum dependence count across all the neighborhood pixels with respect to chosen :math:`\delta`.

Let
:math:`\mathbf{M}` be the
:math:`N_g \times N_r` neighbouring grey level dependence matrix (NGLDM). Element :math:`s_{ij}` of :math:`\mathbf{M}` is
then the number of neighbourhoods with a center pixel with discretised
grey level :math:`i` and a neighbouring dependence :math:`j`.

Let
:math:`N_v` be the number of pixels in the ROI and
:math:`N_s = \sum_{i=1}^{N_g}\sum_{j=1}^{N_n} s_{ij}` the number of
neighbourhoods.

The following marginal totals can be defined. Let
:math:`s_{i.}=\sum_{j=1}^{N_r}` be the number of neighbourhood pixels having grey level :math:`i`, and let
:math:`s_{j.}=\sum_{i=1}^{N_g}s_{ij}` be the number of neighbourhood pixels having dependence :math:`j`, regardless of grey level.

Note
  that Nyxus presets the coarseness parameter :math:`\alpha=0`
  and the neighbourhood radius :math:`\delta=1`.

The following features are then defined:

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_LDE}} {\textup{Low dependence emphasis}} = \frac{1}{N_s} \sum_{j=1}^{N_r} \frac{s_{.j}}{j^2}

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_HDE}} {\textup{High dependence emphasis}} = \frac{1}{N_s} \sum_{j=1}^{N_r} j^2 s_{.j}

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_LGLCE}} {\textup{Low grey level count emphasis}}= \frac{1}{N_s} \sum_{i=1}^{N_g} \frac{s_{i.}}{i^2}

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_HGLCE}} {\textup{  High grey level count emphasis  }} = \frac{1}{N_s} \sum_{i=1}^{N_g} i^2 s_{i.}

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_LDLGLE  }} {\textup{  Low dependence low grey level emphasis  }} = \frac{1}{N_s} \sum_{i=1}^{N_g} \sum_{j=1}^{N_r} \frac{s_{ij}}{i^2 j^2}

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_LDHGLE  }} {\textup{  Low dependence high grey level emphasis  }} = \frac{1}{N_s} \sum_{i=1}^{N_g} \sum_{j=1}^{N_r} \frac{i^2 s_{ij}}{j^2}

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_HDLGLE  }} {\textup{  High dependence low grey level emphasis  }} = \frac{1}{N_s} \sum_{i=1}^{N_g} \sum_{j=1}^{N_r} \frac{j^2 s_{ij}}{i^2}

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_HDHGLE  }} {\textup{  High dependence high grey level emphasis  }} = \frac{1}{N_s} \sum_{i=1}^{N_g} \sum_{j=1}^{N_r} i^2 j^2 s_{ij}

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_GLNU  }} {\textup{  Grey level non-uniformity  }} = \frac{1}{N_s} \sum_{i=1}^{N_g} s_{i.}^2

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_GLNUN  }} {\textup{  Normalised grey level non-uniformity  }} = \frac{1}{N_s^2} \sum_{i=1}^{N_g} s_{i.}^2

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_DCNU  }} {\textup{  Dependence count non-uniformity  }} = \frac{1}{N_s} \sum_{j=1}^{N_r} s_{.j}^2

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_DCNUN  }} {\textup{  Normalised dependence count non-uniformity  }} = \frac{1}{N_s^2} \sum_{i=1}^{N_r} s_{.j}^2

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_DCP  }} {\textup{  Dependence count percentage  }} = \frac{N_s}{N_v}

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_GLV  }} {\textup{  Grey level variance  }} =  \sum_{i=1}^{N_g} \sum_{j=1}^{N_r} (i-\mu)^2 p_{ij}

where the mean intensity is defined as  :math:`\mu = \sum_{i=1}^{N_g} \sum_{j=1}^{N_r} i\,p_{ij}`.

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_DCV  }} {\textup{  Dependence count variance  }} = \sum_{i=1}^{N_g} \sum_{j=1}^{N_r} (j-\mu)^2 p_{ij}

where the mean dependence count is defined as :math:`\mu = \sum_{i=1}^{N_g} \sum_{j=1}^{N_r} j\,p_{ij}`.

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_DCE  }} {\textup{  Dependence count entropy  }} = - \sum_{i=1}^{N_g} \sum_{j=1}^{N_r} p_{ij} \log_2 p_{ij}

.. math::
	\underset{\mathrm{Nyxus \, code: \, NGLDM\_DCENE  }} {\textup{  Dependence count energy  }} = \sum_{i=1}^{N_g} \sum_{j=1}^{N_r} p_{ij}^2


References
----------

Chengjun Sun; William G Wee (1983). “Neighboring gray level dependence matrix for texture classification”.
Computer Vision, Graphics, and Image Processing, Volume 23, Issue 3, 1983, Pages 341-352, ISSN 0734-189X.
