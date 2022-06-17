# Morphology features

Let:
$A$ be a set of $Np$ pixels included in the ROI

AREA_PIXELS_COUNT $= ||A||$

AREA_UM2 $= ||A|| s^2$ where $s$ is pixel size in micrometers 

CENTROID_X $= \frac{1}{n} \sum _i ^n  A_{Xi}$

CENTROID_Y $=  \frac{1}{n} \sum _i ^n  A_{Yi}$

WEIGHTED_CENTROID_X $= \frac{1}{n} \sum _i ^n  A_i (A_{Xi}-\text {CENTROID\_X})$

WEIGHTED_CENTROID_Y $= \frac{1}{n} \sum _i ^n  A_i (A_{Yi}-\text {CENTROID\_Y})$


MASS_DISPLACEMENT $= \sqrt {( WEIGHTED\_CENTROID\_X - CENTROID\_X)^2 + ( WEIGHTED\_CENTROID\_Y - CENTROID\_Y)^2}$

COMPACTNESS $= \frac {1}{n} \displaystyle {\sqrt {\operatorname {E} \left[(A-(CENTROID\_X,CENTROID\_Y)) )^{2}\right]}} $

BBOX_YMIN = $= \operatorname {min}A_Y$

BBOX_XMIN $= \operatorname {min}A_X$

BBOX_HEIGHT $= \operatorname {max}A_Y - BBOX\_YMIN$

BBOX_WIDTH $= \operatorname {max}A_X - BBOX\_XMIN$

MAJOR_AXIS_LENGTH $=4 \sqrt {\lambda_1}$ where $\lambda_1$ is the first largest principal component 

MINOR_AXIS_LENGTH $=4 \sqrt {\lambda_2}$ where $\lambda_2$ is the second largest principal component 

PERIMETER $= card(A)$
