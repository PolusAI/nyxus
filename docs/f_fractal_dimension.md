# Fractal dimension features

Suppose $A$ is a shape's area and $P$ is its perimeter, and we are determining $D$.

## FRACT_DIM_BOXCOUNT

The underlying relation is $n = b^{-D}$. The feature is calculated as the slope $-D$ of the plot $\text {log} \: n$ against $\text {log} \: b$.

## FRACT_DIM_PERIMETER

The underlying relation is $n = P^{2/D}$. The feature is calculated as the slope of the plot $\text {log} \: A$ against $\text {log} \: P$.