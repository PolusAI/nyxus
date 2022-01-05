# Nyxus
<br>
A scalable feature extractor 
<br>
<br>

## Overview
The feature extraction plugin extracts morphology and intensity based features from pairs of intensity/binary mask images and produces a csv file output. The input image should be in tiled [OME TIFF format](https://docs.openmicroscopy.org/ome-model/6.2.0/ome-tiff/specification.html).  The plugin extracts the following features:


__Pixel intensity features:__


------------------
| Nyxus feature code | Description |
|--------|-------|
| INTEGRATED_INTENSITY | Integrated intensity of the region of interest (ROI) |
| MEAN  | Mean intensity value of the ROI | 
| MEDIAN  | The median value of pixels in the ROI | 
| MIN  | Minimum intensity value in the ROI | 
| MAX  | Maximum intensity value in the ROI | 
| RANGE  | Range between the maximmu and minimum | 
| STANDARD_DEVIATION  | Standard deviation | 
| STANDARD_ERROR  | Standard error | 
| SKEWNESS  | Skewness - the 3rd standardized moment | 
| KURTOSIS  | Kurtosis - the 4th standardized moment | 
| HYPERSKEWNESS  | Hyperskewness - the 5th standardized moment | 
| HYPERFLATNESS  | Hyperflatness - the 6th standardized moment  |  
| MEAN_ABSOLUTE_DEVIATION  | Mean absolute devation | ... | 
| ENERGY  | ROI energy | ... | 
| ROOT_MEAN_SQUARED  | Root of mean squared deviation | ... | 
| ENTROPY  | ROI entropy - a measure of the amount of information (that is, randomness) in the ROI | 
| MODE  | The mode value of pixels in the ROI - the value that appears most often in a set of ROI intensity values |  
| UNIFORMITY  | Uniformity - measures how uniform the distribution of ROI intensities is | 
| UNIFORMITY_PIU  | Percent image uniformity, another measure of intensity distribution uniformity | 
| P01, P10, P25, P75, P90, P99  | 1%, 10%, 25%, 75%, 90%, and 99% percentiles of intensity distribution | 
| INTERQUARTILE_RANGE  | Distribution's interquartile range | 
| ROBUST_MEAN_ABSOLUTE_DEVIATION  | Robust mean absolute deviation | 
| MASS_DISPLACEMENT  | ROI mass displacement | 


__Morphology features:__

------------------
| Nyxus feature code | Description |
|--------|-------|
| AREA_PIXELS_COUNT | ROI area in the number of pixels |
| AREA_UM2  | ROI area in metric units |
| CENTROID_X  | X-coordinate of the enter point of the ROI |
| CENTROID_Y  | Y-coordinate of the center point of the ROI |
| COMPACTNESS  | Mean squared distance of the object’s pixels from the centroid divided by the area. Compactness of a filled circle is 1, compactness of irregular objects or objects with holes is greater than 1 |
| BBOX_YMIN | Y-position and size of the smallest axis-aligned box containing the ROI |
| BBOX_XMIN | X-position and size of the smallest axis-aligned box containing the ROI |
| BBOX_HEIGHT | Height of the smallest axis-aligned box containing the ROI |
| BBOX_WIDTH | Width of the smallest axis-aligned box containing the ROI |
| MAJOR_AXIS_LENGTH | Length (in pixels) of the major axis of the ellipse that has the same normalized second central moments as the region |
| MINOR_AXIS_LENGTH | Length (in pixels) of the minor axis of the ellipse that has the same normalized second central moments as the region |
| ECCENTRICITY | Ratio of ROI's inertia ellipse focal distance over the major axis length |
| ORIENTATION | Angle between the 0th axis and the major axis of the ellipse that has same second moments as the region |
| ROUNDNESS | Represents how similar a ROI's inertia ellipse is to circle. Calculated based on the major and minor exis lengths |
| NUM_NEIGHBORS | The number of neighbors bordering the ROI's perimeter. Algorithmically calculating this feature invilves solving the nearest neighbors search problem that in turn involves the proximity measure and the proximity threshold. Particularly, this plugin uses the $L_2$ norm measure over Cartesian space of pixel coordinates and parameter _--pixelDistance_. |
| PERCENT_TOUCHING | Percent of ROI's contour pixels touching neighbor ROIs |
| CLOSEST_NEIGHBOR1_DIST | Distance from ROI's centroid to the nearest neighboring ROI's centroid |
| CLOSEST_NEIGHBOR1_ANG | Angle between ROI's centroid and its nearest neighboring ROI's centroid |
| CLOSEST_NEIGHBOR2_DIST | Distance from ROI's centroid to the second nearest neighboring ROI's centroid |
| CLOSEST_NEIGHBOR2_ANG | Angle between ROI's centroid and its second nearest neighboring ROI's centroid |
| ANG_BW_NEIGHBORS_MEAN | Mean angle between ROI's centroid and centroids of its neighboring ROIs |
| ANG_BW_NEIGHBORS_STDDEV | Standard deviation of angles between ROI's centroid and centroids of its neighboring ROIs |
| ANG_BW_NEIGHBORS_MODE | Mode value of angles between ROI's centroid and centroids of its neighboring ROIs |
| EXTENT | Proportion of the pixels (2D) or voxels (3D) in the bounding box that are also in the region. Computed as the area/volume of the object divided by the area/volume of the bounding box |
| ASPECT_RATIO | The ratio of the major axis to the minor axis of ROI's inertia ellipse |
| CONVEX_HULL_AREA | Area of ROI's convex hull |
| SOLIDITY | Ratio of pixels in the ROI common with its convex hull image |
| PERIMETER | Number of pixels in ROI's contour |
| EQUIVALENT_DIAMETER | Diameter of a circle with the same area as the ROI |
| EDGE_MEAN_INTENSITY | Mean intensity of ROI's contour pixels |
| EDGE_STDDEV_INTENSITY | Standard deviation of ROI's contour pixels |
| EDGE_MAX_INTENSITY | Maximum intensity of ROI's contour pixels |
| EDGE_MIN_INTENSITY | Minimum intensity of ROI's contour pixels |
| CIRCULARITY | Represents how similar a shape is to circle. Clculated based on ROI's area and its convex perimeter |
| EROSIONS_2_VANISH | Number of erosion operations for a ROI to vanish in its axis aligned bounding box |
| EROSIONS_2_VANISH_COMPLEMENT | Number of erosion operations for a ROI to vanish in its convex hull |
| FRACT_DIM_BOXCOUNT | Fractal dimension determined by the box counting method according to ISO 9276-6. If C is a fractal set, with fractal dimension DF < D, then the number N of boxes of size R needed to cover the set scales as R^(-DF). DF is known as the Hausdorf dimension, or Kolmogorov capacity, or Kolmogorov dimension, or simply box-counting dimension |
| FRACT_DIM_PERIMETER | Fractal dimension determined by the perimeter method according to ISO 9276-6. If we approximate ROI's contour with rulers of length lambda, the perimeter based fractal dimension is the slope of the best fit line of log ROI perimeter versus log lambda, subtracted from 1 |
| WEIGHTED_CENTROID_Y  | X-coordinate of centroid | 
| WEIGHTED_CENTROID_X  | Y-coordinate of centroid | 
| MIN_FERET_DIAMETER | Feret diameter (or maximum caliber diameter) is the longest distance between any two ROI points along the same (horizontal) direction. This feature is the minimum Feret diameter for angles ranging 0 to 180 degrees |
| MAX_FERET_DIAMETER | Maximum Feret diameter for angles ranging 0 to 180 degrees |
| MIN_FERET_ANGLE | Angle of the minimum Feret diameter |
| MAX_FERET_ANGLE | Angle of the maximum Feret diameter |
| STAT_FERET_DIAM_MIN | Minimum of Feret diameters of the ROI rotated at angles 0-180 degrees |
| STAT_FERET_DIAM_MAX | Maximum of Feret diameters of the ROI rotated at angles 0-180 degrees |
| STAT_FERET_DIAM_MEAN | Mean Feret diameter of the ROI rotated at angles 0-180 degrees |
| STAT_FERET_DIAM_MEDIAN | Median value of Feret diameters of the ROI rotated at angles 0-180 degrees |
| STAT_FERET_DIAM_STDDEV | Standard deviation of Feret diameter of the ROI rotated at angles 0-180 degrees |
| STAT_FERET_DIAM_MODE | Histogram mode of Feret diameters of the ROI rotated at angles 0-180 degrees |
| STAT_MARTIN_DIAM_MIN | Minimum of Martin diameters of the ROI rotated at angles 0-180 degrees |
| STAT_MARTIN_DIAM_MAX | Maximum of Martin diameters of the ROI rotated at angles 0-180 degrees |
| STAT_MARTIN_DIAM_MEAN | Mean of Martin diameter of the ROI rotated at angles 0-180 degrees |
| STAT_MARTIN_DIAM_MEDIAN | Median value of Martin diameters of the ROI rotated at angles 0-180 degrees |
| STAT_MARTIN_DIAM_STDDEV | Standard deviation of Martin diameter of the ROI rotated at angles 0-180 degrees |
| STAT_MARTIN_DIAM_MODE | Histogram mode of Martin diameters of the ROI rotated at angles 0-180 degrees |
| STAT_NASSENSTEIN_DIAM_MIN | Minimum of Nassenstein diameters of the ROI rotated at angles 0-180 degrees |
| STAT_NASSENSTEIN_DIAM_MAX | Maximum of Nassenstein diameters of the ROI rotated at angles 0-180 degrees |
| STAT_NASSENSTEIN_DIAM_MEAN | Mean of Nassenstein diameter of the ROI rotated at angles 0-180 degrees |
| STAT_NASSENSTEIN_DIAM_MEDIAN | Median value of Nassenstein diameters of the ROI rotated at angles 0-180 degrees |
| STAT_NASSENSTEIN_DIAM_STDDEV | Standard deviation of Nassenstein diameter of the ROI rotated at angles 0-180 degrees |
| STAT_NASSENSTEIN_DIAM_MODE | Histogram mode of Nassenstein diameters of the ROI rotated at angles 0-180 degrees |
| MAXCHORDS_MAX | Maximum of ROI's longest chords built at angles 0-180 degrees |
| MAXCHORDS_MAX_ANG | Angle of the chord referenced in MAXCHORDS_MAX |
| MAXCHORDS_MIN | Minimum of ROI's longest chords built at angles 0-180 degrees |
| MAXCHORDS_MIN_ANG | Angle of the chord referenced in MAXCHORDS_MIN |
| MAXCHORDS_MEDIAN | Median value of ROI's longest chords built at angles 0-180 degrees |
| MAXCHORDS_MEAN | Mean value of ROI's longest chords built at angles 0-180 degrees |
| MAXCHORDS_MODE | Histogram mode of ROI's longest chords built at angles 0-180 degrees |
| MAXCHORDS_STDDEV | Sndard deviation of ROI's longest chords built at angles 0-180 degrees |
| ALLCHORDS_MAX | Maximum of all the ROI's chords built at angles 0-180 degrees |
| ALLCHORDS_MAX_ANG | Angle of the chord referenced in ALLCHORDS_MAX |
| ALLCHORDS_MIN | Minimum of all the ROI's chords built at angles 0-180 degrees |
| ALLCHORDS_MIN_ANG | Angle of the chord referenced in ALLCHORDS_MIN |
| ALLCHORDS_MEDIAN | Median value of all the ROI's chords built at angles 0-180 degrees |
| ALLCHORDS_MEAN | Mean value of all the ROI's chords built at angles 0-180 degrees|
| ALLCHORDS_MODE | Histogram mode of all the ROI's chords built at angles 0-180 degrees |
| ALLCHORDS_STDDEV | Sndard deviation of all the ROI's chords built at angles 0-180 degrees |
| EULER_NUMBER | Euler characteristic of the ROI - the number of objects in the ROI minus the number of holes assuming the 8-neighbor connectivity of ROI's pixels |
| EXTREMA_P1_X | X-ccordinate of ROI's axis aligned bounding box extremum point #1 |
| EXTREMA_P1_Y  | Y-ccordinate of ROI's axis aligned bounding box extremum point #1 |
| EXTREMA_P2_X | X-ccordinate of ROI's axis aligned bounding box extremum point #2|
| EXTREMA_P2_Y | |
| EXTREMA_P3_X | X-ccordinate of ROI's axis aligned bounding box extremum point #3 |
| EXTREMA_P3_Y  | |
| EXTREMA_P4_X | X-ccordinate of ROI's axis aligned bounding box extremum point #4 |
| EXTREMA_P4_Y  | |
| EXTREMA_P5_X | X-ccordinate of ROI's axis aligned bounding box extremum point #5 |
| EXTREMA_P5_Y  | |
| EXTREMA_P6_X | X-ccordinate of ROI's axis aligned bounding box extremum point #6 |
| EXTREMA_P6_Y  | |
| EXTREMA_P7_X | X-ccordinate of ROI's axis aligned bounding box extremum point #7 |
| EXTREMA_P7_Y  | |
| EXTREMA_P8_X | X-ccordinate of ROI's axis aligned bounding box extremum point #8 |
| EXTREMA_P8_Y  | |
| POLYGONALITY_AVE | The score ranges from $-\infty$ to 10. Score 10 indicates the object shape is polygon and score $-\infty$ indicates the ROI shape is not polygon |
| HEXAGONALITY_AVE | The score ranges from $-\infty$ to 10. Score 10 indicates the object shape is hexagon and score $-\infty$ indicates the ROI shape is not hexagon |
| HEXAGONALITY_STDDEV | Standard deviation of hexagonality_score relative to its mean |


The features are calculated using [scikit-image](https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops) library.

## Installation
Nyxus can be installed as a Python package, as a Docker image, or be compiled from source

### Install with pip
You can install Nyxus using [pip package manager](https://pypi.org/project/pip):
```
pip install nyxus 
```
### Install from sources
Another option is cloning the Nyxus repository and installing it manually:

```
git clone https://github.com/friskluft/nyxus.git
cd nyxus
cmake . 
```

### Install from sources and package into a Docker image
The 3rd option alternative to running Nyxus as a Python library or a standalone executable program is making a POLUS plugin of it by packaging it into a Docker image. The latter requires advancing the plugin version number in file VERSION, building the Docker image, uploading it to POLUS repository, and registering the plugin. To build an image, run 

```
./build-docker.sh
```

This feature extractor is designed to be run on POLUS WIPP platform but dry-running it on a local machine before deployment to WIPP after code change is a good idea  

__Example - testing docker image with local data__ 

Assuming the Docker image's hash is \<hash\>, the root of the data directory on the test machine is /images/collections, and intensity and segmentation mask image collections are in subdirectories /images/collections/c1/int and /images/collections/c1/seg respectively, the image can be test-run with command
```
docker run -it --mount type=bind,source=/images/collections,target=/data <hash> --intDir=/data/c1/int --segDir=/data/c1/seg --outDir=/data/output --filePattern=.* --csvfile=separatecsv --features=all
```

Assuming the built image's version as displayed by command 
```
docker images
```
is "labshare/polus-feature-extraction-plugin:1.2.3", the image can be pushed to POLUS organization repository at Docker image cloud with the following 2 commands. The first command 
```
docker tag labshare/polus-feature-extraction-plugin:1.2.3 polusai/polus-feature-extraction-plugin:1.2.3
```
aliases the labshare organization image in a different organization - polusai - permitting image's registering as a POLUS WIPP plugin. The second command 
```
docker push polusai/polus-feature-extraction-plugin:1.2.3
```
uploads the image to the repository of WIPP plugin images. Lastly, to register the plugin in WIPP, edit the text file of plugin's manifest (file __*plugin.json*__) to ensure that the manifest keys __*version*__ and __*containerId*__ refer to the uploaded Docker image version, navigate to WIPP web application's plugins page, and add a new plugin by uploading the updated manifest file.

## Dependencies
Nyxus is tested with Python 3.7+. Building Nyxus from sources requires the following packages:

[NIST Hedgehog](https://github.com/usnistgov/hedgehog) >= 1.0.16 <br>
[NIST Fastloader](https://github.com/usnistgov/FastLoader) >= 2.1.4 <br>
[pybind11](https://github.com/pybind/pybind11) >= 2.8.1 <br>
[libTIFF](http://www.libtiff.org) >= 3.6.1 <br>

## Plugin inputs

__Label image collection:__
The input should be a labeled image in tiled OME TIFF format (.ome.tif). Extracting morphology features, Feret diameter statistics, neighbors, hexagonality and polygonality scores requires the segmentation labels image. If extracting morphological features is not required, the label image collection can be not specified.

__Intensity image collection:__
Extracting intensity-based features requires intensity image in tiled OME TIFF format. This is an optional parameter - the input for this parameter is required only when intensity-based features needs to be extracted.

__File pattern:__
Enter file pattern to match the intensity and labeled/segmented images to extract features (https://pypi.org/project/filepattern/) Filepattern will sort and process files in the labeled and intensity image folders alphabetically if universal selector(.*.ome.tif) is used. If a more specific file pattern is mentioned as input, it will get matches from labeled image folder and intensity image folder based on the pattern implementation.

__Pixel distance:__
Enter value for this parameter if neighbors touching cells needs to be calculated. The default value is 5. This parameter is optional.

__Features:__
Comma separated list of features to be extracted. If all the features are required, then choose option __*all*__.

__Csvfile:__
There are 2 options available under this category. __*Separatecsv*__ - to save all the features extracted for each image in separate csv file. __*Singlecsv*__ - to save all the features extracted from all the images in the same csv file.

__Embedded pixel size:__
This is an optional parameter. Use this parameter only if units are present in the metadata and want to use those embedded units for the features extraction. If this option is selected, value for the length of unit and pixels per unit parameters are not required.

__Length of unit:__
Unit name for conversion. This is also an optional parameter. This parameter will be displayed in plugin's WIPP user interface only when embedded pixel size parameter is not selected (ckrresponding check box checked).

__Pixels per unit:__
If there is a metric mentioned in Length of unit, then Pixels per unit cannot be left blank and hence the scale per unit value must be mentioned in this parameter. This parameter will be displayed in plugin's user interface only when embedded pixel size parameter is not selected.

__Note:__ If Embedded pixel size is not selected and values are entered in Length of unit and Pixels per unit, then the metric unit mentioned in length of unit will be considered.
If Embedded pixel size, Length of unit and Pixels per unit is not selected and the unit and pixels per unit fields are left blank, the unit will be assumed to be pixels.

__Output:__
The output is a csv file containing the value of features required.

For more information on WIPP, visit the [official WIPP page](https://github.com/usnistgov/WIPP/tree/master/user-guide).



## Plugin command line parameters
Running the WIPP feature extraction plugin is controlled via nine named input arguments and one output argument that are passed by WIPP web application to plugin's Docker image - see Table 1.

__Table 1 - Command line parameters__

------
| Parameter | Description | I/O | Type |
|------|-------------|------|----|
--intDir|Intensity image collection|Input|collection|
--pixelDistance|Pixel distance to calculate the neighbors touching cells|Input|integer|
--filePattern|To match intensity and labeled/segmented images|Input|string
--segDir|Labeled image collection|Input|collection
--features|Select intensity and shape features required|Input|array
--csvfile|Save csv file as one csv file for all images or separate csv file for each image|Input|enum
--embeddedpixelsize|Consider the unit embedded in metadata, if present|Input|boolean
--unitLength|Enter the metric for unit conversion|Input|string
--pixelsPerunit|Enter the number of pixels per unit of the metric|Input|number
--outDir|Output collection|Output|csvCollection
---

Input type __*collection*__ is a WIPP's image collection browsable at WIPP web application/Data/Images Collections. Output type __*csvCollection*__ indicates that result of the successfully run plugin will be available to a user as CSV-files. To access the result in WIPP, navigate to WIPP web application/Workflows, choose the task, expand drop-down list 'Ouputs', and navigate to the URL leading to a WIPP data collection. Input type __*enum*__ is a single-selection list of options. Input type __*array*__ is a multi-selection list of options. Input types __*boolean*__ is represented with a check-box in WIPP's user interface. There are 2 parameters referring input type __*string*__ - the file pattern applied to image file names in collections defined by parameters __*intDir*__ and __*segDir*__, and the name of the measurement unit defined by optional parameters __*embeddedpixelsize*__ and __*pixelsPerunit*__. The file pattern parameter is mandatory. Its valid values are regular expression style wildcards to filter file names, for example, .\* to select all the files or .\*c1\\.ome\\.tif to select just files ending in "c1.ome.tif". Input type __*integer*__ is a positive integer value or zero. Input type __*number*__ used in parameter __*pixelsPerunit*__ is a positive real value defining the number of pixels in the measurement unit defined by parameter __*unitLength*__. 

Parameter __*features*__ defines a set of desired features to be calculated. Valid string literal to feature correspondence is listed in the feature table above.

The following command line is an example of running the dockerized feature extractor (image hash 87f3b560bbf2) with only intensity features selected:
```
docker run -it --mount type=bind,source=/images/collections,target=/data 87f3b560bbf2 --intDir=/data/c1/int --segDir=/data/c1/seg --outDir=/data/output --filePattern=.* --csvfile=separatecsv --features=entropy,kurtosis,skewness,max_intensity,mean_intensity,min_intensity,median,mode,standard_deviation
```
or its undockerized equivalent:
```
python main.py --intDir=/images/collections/c1/int --segDir=/images/collections/c1/seg --outDir=/temp_fe/output --filePattern=.* --csvfile=separatecsv --features=entropy,kurtosis,skewness,max_intensity,mean_intensity,min_intensity,median,mode,standard_deviation
```

