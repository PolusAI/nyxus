# 2D moments

## Raw moments

Raw (spatial) moments $m_{ij}$ of a 2-dimensional greyscale image  $I(x,y)$ are calculated as

$$\displaystyle m_{{ij}}=\sum _{x}\sum _{y}x^{i}y^{j}I(x,y)\$$

Spatial moment features are calculated as:\
SPAT_MOMENT_00 $=m_{00}$    \
SPAT_MOMENT_01 $=m_{01}$    \
SPAT_MOMENT_02 $=m_{02}$    \
SPAT_MOMENT_03 $=m_{03}$    \
SPAT_MOMENT_10 $=m_{10}$    \
SPAT_MOMENT_11 $=m_{11}$    \
SPAT_MOMENT_12 $=m_{12}$    \
SPAT_MOMENT_20 $=m_{20}$    \
SPAT_MOMENT_21 $=m_{21}$    \
SPAT_MOMENT_30 $=m_{30}$    

## Central moments

A central moment $\mu_{ij}$ is defined as 

$$\mu _{{ij}}=\sum _{{x}}\sum _{{y}}(x-{\bar  {x}})^{i}(y-{\bar  {y}})^{j}I(x,y)$$

Central moment features are calculated as:\
CENTRAL_MOMENT_02 $=\mu_{02}$   \
CENTRAL_MOMENT_03 $=\mu_{03}$   \
CENTRAL_MOMENT_11 $=\mu_{11}$   \
CENTRAL_MOMENT_12 $=\mu_{12}$   \
CENTRAL_MOMENT_20 $=\mu_{20}$   \
CENTRAL_MOMENT_21 $=\mu_{21}$   \
CENTRAL_MOMENT_30 $=\mu_{20}$   

## Normalized raw moments
Raw (spatial) moments $m_{ij}$ of a 2-dimensional greyscale image  I(x,y) are calculated by

$$\displaystyle w_{{ij}} = \frac {\mu_{ij}}{\mu_{22}^ {max(i,j)} }$$

Spatial moment features are calculated as:\
NORM_SPAT_MOMENT_00 $=w_{00}$    \
NORM_SPAT_MOMENT_01 $=w_{01}$    \
NORM_SPAT_MOMENT_02 $=w_{02}$    \
NORM_SPAT_MOMENT_03 $=w_{03}$    \
NORM_SPAT_MOMENT_10 $=w_{10}$    \
NORM_SPAT_MOMENT_20 $=w_{20}$    \
NORM_SPAT_MOMENT_30 $=w_{30}$    

## Normalized central moments

A normalized central moment $\eta _{ij}$ is defined as 
$$\eta _{{ij}}={\frac  {\mu _{{ij}}}{\mu _{{00}}^{{\left(1+{\frac  {i+j}{2}}\right)}}}}\,\$$

where $\mu _{{ij}}$ is central moment.

Normalized central moment features are calculated as:\
NORM_CENTRAL_MOMENT_02 $=\eta _{{02}}$\
NORM_CENTRAL_MOMENT_03 $=\eta _{{03}}$\
NORM_CENTRAL_MOMENT_11 $=\eta _{{11}}$\
NORM_CENTRAL_MOMENT_12 $=\eta _{{12}}$\
NORM_CENTRAL_MOMENT_20 $=\eta _{{20}}$\
NORM_CENTRAL_MOMENT_21 $=\eta _{{21}}$\
NORM_CENTRAL_MOMENT_30 $=\eta _{{30}}$

## Hu moments
Hu invariants HU_M1 through HU_M7 are calculated as\

HU_M1 $=\eta _{{20}}+\eta _{{02}}$\
HU_M2 $=(\eta _{{20}}-\eta _{{02}})^{2}+4\eta _{{11}}^{2}$\
HU_M3 $=(\eta _{{30}}-3\eta _{{12}})^{2}+(3\eta _{{21}}-\eta _{{03}})^{2}$\
HU_M4 $=(\eta _{{30}}+\eta _{{12}})^{2}+(\eta _{{21}}+\eta _{{03}})^{2}$\
HU_M5 $=(\eta _{{30}}-3\eta _{{12}})(\eta _{{30}}+\eta _{{12}})[(\eta _{{30}}+\eta _{{12}})^{2}-3(\eta _{{21}}+\eta _{{03}})^{2}]+(3\eta _{{21}}-\eta _{{03}})(\eta _{{21}}+\eta _{{03}})[3(\eta _{{30}}+\eta _{{12}})^{2}-(\eta _{{21}}+\eta _{{03}})^{2}]$\
HU_M6 $=(\eta _{{20}}-\eta _{{02}})[(\eta _{{30}}+\eta _{{12}})^{2}-(\eta _{{21}}+\eta _{{03}})^{2}]+4\eta _{{11}}(\eta _{{30}}+\eta _{{12}})(\eta _{{21}}+\eta _{{03}})$\
HU_M7 $=(3\eta _{{21}}-\eta _{{03}})(\eta _{{30}}+\eta _{{12}})[(\eta _{{30}}+\eta _{{12}})^{2}-3(\eta _{{21}}+\eta _{{03}})^{2}]-(\eta _{{30}}-3\eta _{{12}})(\eta _{{21}}+\eta _{{03}})[3(\eta _{{30}}+\eta _{{12}})^{2}-(\eta _{{21}}+\eta _{{03}})^{2}]$\


## Weighted raw moments
Let $W(x,y)$ be a 2-dimensional weighted greyscale image such that each pixel of $I$ is weighted with respect to its distance to the nearest contour pixel: $W(x,y) = \frac {I(x,y)} {\min_i d^2(x,y,C_i)}$ where C - set of 2-dimensional ROI contour pixels, $d^2(.)$ - Euclidean distance norm. Weighted raw moments $w_{Mij}$ are defined as

$$\displaystyle w_{Mij}=\sum _{x}\sum _{y}x^{i}y^{j}W(x,y)\$$

## Weighted central moments

Weighted central moments $w_{\mu ij}$ are defined as 
$$w_{\mu ij} = \sum _{{x}}\sum _{{y}}(x-{\bar  {x}})^{i}(y-{\bar  {y}})^{j}W(x,y)$$

## Weighted Hu moments

A normalized weighted central moment $w _{\eta ij}$ is defined as 
$$w _{{\eta ij}}={\frac  {w _{{\mu ij}}}{w _{{\mu 00}}^{{\left(1+{\frac  {i+j}{2}}\right)}}}}\,\$$

where $w _{{\mu ij}}$ is weighted central moment.
Weighted Hu moments are defined as

WEIGHTED_HU_M1 $=w _{\eta 20}+w _{\eta 02}$\
WEIGHTED_HU_M2 $=(w _{\eta 20}-w _{\eta 02})^{2}+4w _{\eta 11}^{2}$\
WEIGHTED_HU_M3 $=(w _{\eta 30}-3w _{\eta 12})^{2}+(3w _{\eta 21}-w _{\eta 03})^{2}$\
WEIGHTED_HU_M4 $=(w _{\eta 30}+w _{\eta 12})^{2}+(w _{\eta 21}+w _{\eta 03})^{2}$\
WEIGHTED_HU_M5 $=(w _{\eta 30}-3w _{\eta 12})(w _{\eta 30}+w _{\eta 12})[(w _{\eta 30}+w _{\eta 12})^{2}-3(w _{\eta 21}+w _{\eta 03})^{2}]+(3w _{\eta 21}-w _{\eta 03})(w _{\eta 21}+w _{\eta 03})[3(w _{\eta 30}+w _{\eta 12})^{2}-(w _{\eta 21}+w _{\eta 03})^{2}]$\
WEIGHTED_HU_M6 $=(w _{\eta 20}-w _{\eta 02})[(w _{\eta 30}+w _{\eta 12})^{2}-(w _{\eta 21}+w _{\eta 03})^{2}]+4w _{\eta 11}(w _{\eta 30}+w _{\eta 12})(w _{\eta 21}+w _{\eta 03})$\
WEIGHTED_HU_M7 $=(3w _{\eta 21}-w _{\eta 03})(w _{\eta 30}+w _{\eta 12})[(w _{\eta 30}+w _{\eta 12})^{2}-3(w _{\eta 21}+w _{\eta 03})^{2}]-(w _{\eta 30}-3w _{\eta 12})(w _{\eta 21}+w _{\eta 03})[3(w _{\eta 30}+w _{\eta 12})^{2}-(w _{\eta 21}+w _{\eta 03})^{2}]$

