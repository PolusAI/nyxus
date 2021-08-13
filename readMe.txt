Morphological methods:

+	0-area: total number of the ROI (Region of Interest) pixels
+	1-centroid_y: y coordinate of the ROI's centroid
+	2-centroid_x: x coordinate of the ROI's centroid
+	3-bbox_ymin: y coordinate where the rectangular bounding box encompassing the ROI begins
+	4-bbox_xmin: x coordinate where the rectangular bounding box encompassing the ROI begins
+	5-bbox_height: height of the rectangular bounding box encompassing the ROI
+	6-bbox_width: width of the rectangular bounding box encompassing the ROI
+	7-major_axis_length: length of major axis of the ellipse that has the same normalized second central moments as the region.
+	8-minor_axis_length: length of minor axis of the ellipse that has the same normalized second central moments as the region.
+	9-eccentricity: ratio of focal distance over the major axis length.
+	10-orientation: angle between the x axis and the major axis of the ellipse that has same second moments as the region.
+	11-convex_area: area of the convex hull
+	12-euler_number: Euler characteristic of the ROI
+	13-equivalent_diameter: diameter of a circle with the same area as the ROI
+	14-solidity: ratio of the pixel counts in the ROI to the pixel counts in the convex hull
+	15-perimeter: perimeter length of the ROI
+	16-max_feret: Max Feret diamater
+	17-min_feret: Min Feret diamater
+	18-Neighbors: number of neighbors touching the ROI
+	19-polygonality_score: score ranges from -infinity to 10. Score 10 indicates the object shape is polygon and score -infinity indicates the object shape is not polygon.
+	20-hexagonality_score: score ranges from -infinity to 10. Score 10 indicates the object shape is hexagon and score -infinity indicates the object shape is not hexagon.
+	21-hexagonality_sd: dispersion of hexagonality score relative to its mean.
+	22-circularity: roundness of the ROI which is computed as (4*Area*pi)/(Perimeter2). For a perfect circle, the circularity value is 1.
+	23-extremap1_x: x coordinate of the Extrema point at top-left
+	24-extremap1_y: y coordinate of the Extrema point at top-left
+	25-extremap2_x: x coordinate of the Extrema point at top-right
+	26-extremap2_y: y coordinate of the Extrema point at top-right
+	27-extremap3_x: x coordinate of the Extrema point at right-top
+	28-extremap3_y: y coordinate of the Extrema point at right-top
+	29-extremap4_x: x coordinate of the Extrema point at right-bottom
+	30-extremap4_y: y coordinate of the Extrema point at right-bottom
+	31-extremap5_x: x coordinate of the Extrema point at bottom-right
+	32-extremap5_y: y coordinate of the Extrema point at bottom-right
+	33-extremap6_x: x coordinate of the Extrema point at bottom-left
+	34-extremap6_y: y coordinate of the Extrema point at bottom-left
+	35-extremap7_x: x coordinate of the Extrema point at left-bottom
+	36-extremap7_y: y coordinate of the Extrema point at left-bottom
+	37-extremap8_x: x coordinate of the Extrema point at left-top
+	38-extremap8_y: y coordinate of the Extrema point at left-top
+	39-extent: ratio of the pixel counts in the ROI to the pixel counts in the bounding box encompassing the ROI 
+	40-max_feret_angle: Max Feret angle
+	41-min_feret_angle: Min Feret angle
+	42-convex_hull_perimeter: perimeter of the convex hull
43-filled_area_pixels: area of the ROI after filling the holes
44-max_inclosing_circle_diameter: diameter of the maximum inclosing cricle of the ROI
45-diameter_min_enclosing_circle: diameter of the minimum enclosing circle of the ROI
46-diameter_circumscribing_circle: diameter of the circle circumscribing the ROI
47-diameter_inscribing_circle: diameter of the cricle inscribing the ROI
48-diameter_equal_perimeter: diameter of a circle with the same perimeter as the ROI
49-rotated_bounding_box_height: height of the rotated bounding box encompassing the ROI
50-rotated_bounding_box_width: width of the rotated bounding box encompassing the ROI
51-geodetic_length: length of a rectangle with the same area and perimeter as the ROI
52-thickness: width of a rectangle with the same area and perimeter as the ROI
53-erosion_pixels: number of iterations until ROI is faded out by applying the erosion algorithm
54-erosion_complement_pixels: number of iterations until the complement area between convex hull and the ROI is faded out by applying the erosion algorithm
55-fractal_dimension_box_counting: fractal dimension as estimated by box counting method
56-fractal_dimension_perimeter: fractal dimension as estimated by perimeter method
+	57-feret_diameter_min: min statistics of all the feret diameters
+	58-feret_diameter_max: max statistics of all the feret diameters
+	59-feret_diameter_mean: mean statistics of all the feret diameters
+	60-feret_diameter_median: median statistics of all the feret diameters
+	61-feret_diameter_std: std statistics of all the feret diameters
+	62-feret_diameter_mode: mode statistics of all the feret diameters
+	63-martin_length_min: min statistics of all the martin lengths
+	64-martin_length_max: max statistics of all the martin lengths
+	65-martin_length_mean: mean statistics of all the martin lengths
+	66-martin_length_median: median statistics of all the martin lengths
+	67-martin_length_std: std statistics of all the martin lengths
+	68-martin_length_mode: mode statistics of all the martin lengths
69-nassenstein_diameter_min: min statistics of all the nassenstein diameters
70-nassenstein_diameter_max: max statistics of all the nassenstein diameters
71-nassenstein_diameter_mean: mean statistics of all the nassenstein diameters
72-nassenstein_diameter_median: median statistics of all the nassenstein diameters
73-nassenstein_diameter_std: std statistics of all the nassenstein diameters
74-nassenstein_diameter_mode: mdoe statistics of all the nassenstein diameters
75-maxchords_min: min statistics of all the max Chord lengths
76-maxchords_max: max statistics of all the max Chord lengths
77-maxchords_mean: mean statistics of all the max Chord lengths
78-maxchords_median: median statistics of all the max Chord lengths
79-maxchords_std: std statistics of all the max Chord lengths
80-maxchords_mode: mode statistics of all the max Chord lengths
81-allchords_min: min statistics of all the Chord lengths
82-allchords_max: max statistics of all the Chord lengths
83-allchords_mean: mean statistics of all the Chord lengths
84-allchords_median: median statistics of all the Chord lengths
85-allchords_std: std statistics of all the Chord lengths
86-allchords_mode: mode statistics of all the Chord lengths
87-x_max: overall max chord of the ROI in all possible orientations
88-y_max: longest chord orthogonal to x_max
 
 
And here are the intensity features that are implemented: 
 
0-Mean
1-Median
2-Min
3-Max
4-Range
5-Standard Deviation
6-Skewness
7-Kurtosis
8-Mean Absolute Deviation
9-Energy
10-Root Mean Squared
11-Entropy
12-Mode
13-Uniformity
14-10th Percentile
15-25th Percentile
16-75th Percentile
17-90th Percentile
18-Interquartile Range
19-Robust Mean Absolute Deviation
20-Weighted Centroid in y direction
21-Weighted Centroid in x direction
 


Test command lines:

[Windows] 
C:\WORK\AXLE\data\jayapriya\intensity C:\WORK\AXLE\data\jayapriya\label C:\WORK\AXLE\data\output  

[Unix]
./sensemaker.exe ~/work/data-jayapriya/intensity ~/work/data-jayapriya/label ./output-jayapria/


??? -ffast-math
