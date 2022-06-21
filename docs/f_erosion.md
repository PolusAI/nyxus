# Erosion features

These features use the morphological erosion of the binary ROI image with respect to the $3 \times 3$ structuring element ker. The input ROI image is in the form of an image matrix.

## EROSIONS_2_VANISH

This feature calulates the number of erosion operations on a ROI to vanish in its axis aligned bounding box.

## EROSIONS_2_VANISH_COMPLEMENT

Unlike EROSIONS_2_VANISH which is measured with respect to a ROI's axis aligned bounding box, his feature calculates the number of erosion operations on a ROI to vanish in its convex hull.