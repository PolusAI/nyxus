
2D moments
==========

Raw moments
-----------

Raw (spatial) moments :math:`m_{ij}` of a 2-dimensional greyscale image  :math:`I(x,y)`` are calculated as

.. math::
   
   m_{{ij}}=\sum _{x}\sum _{y}x^{i}y^{j}I(x,y)

Spatial moment features are calculated as:

.. math::

   \text{SPAT_MOMENT_00} &=m_{00} \\
   \text{SPAT_MOMENT_01} &=m_{01} \\
   \text{SPAT_MOMENT_02} &=m_{02} \\    
   \text{SPAT_MOMENT_03} &=m_{03} \\  
   \text{SPAT_MOMENT_10} &=m_{10} \\   
   \text{SPAT_MOMENT_11} &=m_{11} \\  
   \text{SPAT_MOMENT_12} &=m_{12} \\   
   \text{SPAT_MOMENT_20} &=m_{20} \\   
   \text{SPAT_MOMENT_21} &=m_{21} \\   
   \text{SPAT_MOMENT_30} &=m_{30} 

Central moments
---------------

A central moment :math:`\mu_{ij}` is defined as 

.. math::

   \mu_{{ij}}=\sum_{{x}}\sum _{{y}}(x-{\bar  {x}})^{i}(y-{\bar  {y}})^{j}I(x,y)

Central moment features are calculated as: 

.. math:: 

   \text{CENTRAL_MOMENT_02} &=\mu_{02} \\
   \text{CENTRAL_MOMENT_03} &=\mu_{03} \\  
   \text{CENTRAL_MOMENT_11} &=\mu_{11} \\  
   \text{CENTRAL_MOMENT_12} &=\mu_{12} \\  
   \text{CENTRAL_MOMENT_20} &=\mu_{20} \\  
   \text{CENTRAL_MOMENT_21} &=\mu_{21} \\  
   \text{CENTRAL_MOMENT_30} &=\mu_{20} \\  

Normalized raw moments
----------------------

Raw (spatial) moments :math:`m_{ij}` of a 2-dimensional greyscale image :math:`I(x,y)` are calculated by

.. math::

   w_{{ij}} = \frac {\mu_{ij}}{\mu_{22}^ {max(i,j)} }

Spatial moment features are calculated as:

.. math::

   \text{NORM_SPAT_MOMENT_00} =w_{00} \\
   \text{NORM_SPAT_MOMENT_01} =w_{01} \\    
   \text{NORM_SPAT_MOMENT_02} =w_{02} \\   
   \text{NORM_SPAT_MOMENT_03} =w_{03} \\  
   \text{NORM_SPAT_MOMENT_10} =w_{10} \\
   \text{NORM_SPAT_MOMENT_20} =w_{20} \\ 
   \text{NORM_SPAT_MOMENT_30} =w_{30} \\  

Normalized central moments
--------------------------

A normalized central moment :math:`\eta_{ij}` is defined as 

.. math::

   \eta_{{ij}}={\frac  {\mu_{{ij}}}{\mu_{{00}}^{{\left(1+{\frac  {i+j}{2}}\right)}}}}\,

where :math:`\mu _{{ij}}` is central moment.

Normalized central moment features are calculated as:

.. math:: 
   \text{NORM_CENTRAL_MOMENT_02} &=\eta_{{02}} \\
   \text{NORM_CENTRAL_MOMENT_03} &=\eta_{{03}} \\
   \text{NORM_CENTRAL_MOMENT_11} &=\eta_{{11}} \\
   \text{NORM_CENTRAL_MOMENT_12} &=\eta_{{12}} \\
   \text{NORM_CENTRAL_MOMENT_20} &=\eta_{{20}} \\
   \text{NORM_CENTRAL_MOMENT_21} &=\eta_{{21}} \\
   \text{NORM_CENTRAL_MOMENT_30} &=\eta_{{30}} 

Hu moments
----------

Hu invariants HU_M1 through HU_M7 are calculated as


.. math::

    \text{HU_M1} =& \eta_{{20}}+\eta _{{02}} \\
    \text{HU_M2} =& (\eta_{{20}}-\eta_{{02}})^{2}+4\eta_{{11}}^{2} \\
    \text{HU_M3} =& (\eta_{{30}}-3\eta_{{12}})^{2}+(3\eta_{{21}}-\eta _{{03}})^{2} \\
    \text{HU_M4} =& (\eta_{{30}}+\eta_{{12}})^{2}+(\eta_{{21}}+\eta _{{03}})^{2} \\
    \text{HU_M5} =& (\eta_{{30}}-3\eta_{{12}})(\eta_{{30}}+\eta_{{12}})[(\eta_{{30}}+\eta_{{12}})^{2}-3(\eta_{{21}}+\eta_{{03}})^{2}]+ \\ 
    &(3\eta_{{21}}-\eta_{{03}})(\eta_{{21}}+\eta_{{03}})[3(\eta_{{30}}+\eta_{{12}})^{2}-(\eta_{{21}}+\eta _{{03}})^{2}] \\
    \text{HU_M6} =& (\eta_{{20}}-\eta_{{02}})[(\eta_{{30}}+\eta_{{12}})^{2}-(\eta_{{21}}+\eta_{{03}})^{2}]+4\eta_{{11}}(\eta_{{30}}+\eta_{{12}})(\eta_{{21}}+\eta_{{03}}) \\
    \text{HU_M7} =& (3\eta_{{21}}-\eta_{{03}})(\eta_{{30}}+\eta_{{12}})[(\eta_{{30}}+\eta_{{12}})^{2}-3(\eta_{{21}}+\eta_{{03}})^{2}]- \\
    &(\eta_{{30}}-3\eta_{{12}})(\eta_{{21}}+\eta_{{03}})[3(\eta_{{30}}+\eta_{{12}})^{2}-(\eta_{{21}}+\eta _{{03}})^{2}]


Weighted raw moments
--------------------

Let :math:`W(x,y)` be a 2-dimensional weighted greyscale image such that each pixel of :math:`I` is weighted with respect to its distance to the nearest contour pixel: :math:`W(x,y) = \frac {I(x,y)} {\min_i d^2(x,y,C_i)}` where C - set of 2-dimensional ROI contour pixels, :math:`d^2(.)` - Euclidean distance norm. Weighted raw moments :math:`w_{Mij}`` are defined as

.. math::
   
   w_{Mij}=\sum_{x}\sum _{y}x^{i}y^{j}W(x,y)

Weighted central moments
------------------------

Weighted central moments :math:`w_{\mu ij}` are defined as 

.. math::

   w_{\mu ij} = \sum_{{x}}\sum_{{y}}(x-{\bar  {x}})^{i}(y-{\bar  {y}})^{j}W(x,y)

Weighted Hu moments
-------------------

A normalized weighted central moment :math:`w_{\eta ij}` is defined as 

.. math::
   
   w_{{\eta ij}}={\frac  {w_{{\mu ij}}}{w_{{\mu 00}}^{{\left(1+{\frac  {i+j}{2}}\right)}}}}\,

where :math:`w _{{\mu ij}}` is weighted central moment.
Weighted Hu moments are defined as

.. math:: 
   \text{WEIGHTED_HU_M1} =& w_{\eta 20}+w_{\eta 02} \\
   \text{WEIGHTED_HU_M2} =& (w_{\eta 20}-w_{\eta 02})^{2}+4w_{\eta 11}^{2} \\
   \text{WEIGHTED_HU_M3} =& (w_{\eta 30}-3w_{\eta 12})^{2}+(3w_{\eta 21}-w _{\eta 03})^{2} \\
   \text{WEIGHTED_HU_M4} =& (w_{\eta 30}+w_{\eta 12})^{2}+(w_{\eta 21}+w _{\eta 03})^{2} \\
   \text{ WEIGHTED_HU_M5} =& (w_{\eta 30}-3w_{\eta 12})(w_{\eta 30}+w_{\eta 12})[(w_{\eta 30}+w_{\eta 12})^{2}-3(w_{\eta 21}+ w_{\eta 03})^{2}]+ \\ 
   &(3w_{\eta 21}-w_{\eta 03})(w_{\eta 21}+w_{\eta 03})[3(w_{\eta 30}+w_{\eta 12})^{2}-(w_{\eta 21}+w _{\eta 03})^{2}] \\
   \text{WEIGHTED_HU_M6} =& (w_{\eta 20}-w_{\eta 02})[(w_{\eta 30}+w_{\eta 12})^{2}-(w_{\eta 21}+w_{\eta 03})^{2}]+ \\
   &4w_{\eta 11}(w_{\eta 30}+w_{\eta 12})(w_{\eta 21}+w_{\eta 03})\\
   \text{WEIGHTED_HU_M7} =& (3w_{\eta 21}-w_{\eta 03})(w_{\eta 30}+w_{\eta 12})[(w_{\eta 30}+w_{\eta 12})^{2}-3(w_{\eta 21}+w_{\eta 03})^{2}]- \\
   &(w_{\eta 30}-3w_{\eta 12})(w_{\eta 21}+w_{\eta 03})[3(w_{\eta 30}+w_{\eta 12})^{2}-(w_{\eta 21}+w _{\eta 03})^{2}] 

