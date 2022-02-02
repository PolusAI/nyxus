# Nyxus
<br>
A scalable feature extractor 
<br>
<br>

## Overview
The feature extraction plugin extracts morphology and intensity based features from pairs of intensity/binary mask images and produces a csv file output. The input image should be in tiled [OME TIFF format](https://docs.openmicroscopy.org/ome-model/6.2.0/ome-tiff/specification.html).  The plugin extracts the following features:

Nyxus provides a set of pixel intensity, morphology, texture, intensity distribution features, digital filter based features and image moments

------------------
| Nyxus feature code | Description |
|--------|-------|
| INTEGRATED_INTENSITY | Integrated intensity of the region of interest (ROI) |
| MEAN, MAX, MEDIAN, STANDARD_DEVIATION, MODE | Mean/max/median/stddev/mode intensity value of the ROI | 
| SKEWNESS, KURTOSIS, HYPERSKEWNESS, HYPERFLATNESS  | higher standardized moments | 
| MEAN_ABSOLUTE_DEVIATION  | Mean absolute devation | 
| ENERGY  | ROI energy | 
| ROOT_MEAN_SQUARED  | Root of mean squared deviation | 
| ENTROPY  | ROI entropy - a measure of the amount of information in the ROI | 
| UNIFORMITY  | Uniformity - measures how uniform the distribution of ROI intensities is | 
| UNIFORMITY_PIU  | Percent image uniformity, another measure of intensity distribution uniformity | 
| P01, P10, P25, P75, P90, P99  | 1%, 10%, 25%, 75%, 90%, and 99% percentiles of intensity distribution | 
| INTERQUARTILE_RANGE  | Distribution's interquartile range | 
| ROBUST_MEAN_ABSOLUTE_DEVIATION  | Robust mean absolute deviation | 
| MASS_DISPLACEMENT  | ROI mass displacement | 
| AREA_PIXELS_COUNT | ROI area in the number of pixels |
| COMPACTNESS  | Mean squared distance of the object’s pixels from the centroid divided by the area |
| BBOX_YMIN | Y-position and size of the smallest axis-aligned box containing the ROI |
| BBOX_XMIN | X-position and size of the smallest axis-aligned box containing the ROI |
| BBOX_HEIGHT | Height of the smallest axis-aligned box containing the ROI |
| BBOX_WIDTH | Width of the smallest axis-aligned box containing the ROI |
| MAJOR/MINOR_AXIS_LENGTH, ECCENTRICITY, ORIENTATION, ROUNDNESS | Inertia ellipse features |
| NUM_NEIGHBORS, PERCENT_TOUCHING | The number of neighbors bordering the ROI's perimeter and related neighbor methods |
| EXTENT | Proportion of the pixels in the bounding box that are also in the region |
| CONVEX_HULL_AREA | Area of ROI's convex hull |
| SOLIDITY | Ratio of pixels in the ROI common with its convex hull image |
| PERIMETER | Number of pixels in ROI's contour |
| EQUIVALENT_DIAMETER | Diameter of a circle with the same area as the ROI |
| EDGE_MEAN/MAX/MIN/STDDEV_INTENSITY | Intensity statistics of ROI's contour pixels |
| CIRCULARITY | Represents how similar a shape is to circle. Clculated based on ROI's area and its convex perimeter |
| EROSIONS_2_VANISH | Number of erosion operations for a ROI to vanish in its axis aligned bounding box |
| EROSIONS_2_VANISH_COMPLEMENT | Number of erosion operations for a ROI to vanish in its convex hull |
| FRACT_DIM_BOXCOUNT, FRACT_DIM_PERIMETER | Fractal dimension features |
| GLCM | Gray level co-occurrence Matrix features |
| GLRLM | Gray level run-length matrix based features
| GLSZM | Gray level size zone matrix based features
| GLDM | Gray level dependency matrix based features
| NGTDM | Neighbouring gray tone difference matrix features
| ZERNIKE2D, FRAC_AT_D, RADIAL_CV, MEAN_FRAC | Radial distribution features
| GABOR | A set of Gabor filters of varying frequencies and orientations |

For the complete list of features see [Nyxus provided features](docs/featurelist.md)

## Feature groups

Apart from defining your feature set by explicitly specifying comma-separated feature code, Nyxus lets a user specify popular feature groups. Supported feature groups are:

------------------------------------
| Group code | Belonging features |
|--------------------|-------------|
| \*all_intensity\* | integrated_intensity, mean, median, min, max, range, standard_deviation, standard_error, uniformity, skewness, kurtosis, hyperskewness, hyperflatness, mean_absolute_deviation, energy, root_mean_squared, entropy, mode, uniformity, p01, p10, p25, p75, p90, p99, interquartile_range, robust_mean_absolute_deviation, mass_displacement
| \*all_morphology\* | area_pixels_count, area_um2, centroid_x, centroid_y, weighted_centroid_y, weighted_centroid_x, compactness, bbox_ymin, bbox_xmin, bbox_height, bbox_width, major_axis_length, minor_axis_length, eccentricity, orientation, num_neighbors, extent, aspect_ratio, equivalent_diameter, convex_hull_area, solidity, perimeter, edge_mean_intensity, edge_stddev_intensity, edge_max_intensity, edge_min_intensity, circularity
| \*basic_morphology\* | area_pixels_count, area_um2, centroid_x, centroid_y, bbox_ymin, bbox_xmin, bbox_height, bbox_width
| \*all_glcm\* | glcm_angular2ndmoment, glcm_contrast, glcm_correlation, glcm_variance, glcm_inversedifferencemoment, glcm_sumaverage, glcm_sumvariance, glcm_sumentropy, glcm_entropy, glcm_differencevariance, glcm_differenceentropy, glcm_infomeas1, glcm_infomeas2
| \*all_glrlm\* | glrlm_sre, glrlm_lre, glrlm_gln, glrlm_glnn, glrlm_rln, glrlm_rlnn, glrlm_rp, glrlm_glv, glrlm_rv, glrlm_re, glrlm_lglre, glrlm_hglre, glrlm_srlgle, glrlm_srhgle, glrlm_lrlgle, glrlm_lrhgle
| \*all_glszm\* | glszm_sae, glszm_lae, glszm_gln, glszm_glnn, glszm_szn, glszm_sznn, glszm_zp, glszm_glv, glszm_zv, glszm_ze, glszm_lglze, glszm_hglze, glszm_salgle, glszm_sahgle, glszm_lalgle, glszm_lahgle
| \*all_gldm\* | gldm_sde, gldm_lde, gldm_gln, gldm_dn, gldm_dnn, gldm_glv, gldm_dv, gldm_de, gldm_lgle, gldm_hgle, gldm_sdlgle, gldm_sdhgle, gldm_ldlgle, gldm_ldhgle
| \*all_ngtdm\* | ngtdm_coarseness, ngtdm_contrast, ngtdm_busyness, ngtdm_complexity, ngtdm_strength
| \*all\* | All the features 

### Example: running Nyxus to extract only intensity and basic morphology features
```
./nyxus --features=*all_intensity*,*basic_morphology* --intDir=/home/ec2-user/data-ratbrain/int --segDir=/home/ec2-user/data-ratbrain/seg --outDir=/home/ec2-user/work/OUTPUT-ratbrain --filePattern=.* --csvFile=singlecsv 
```

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
--segDir|Labeled image collection|Input|collection
--intSegMapDir | Data collection of the ad-hoc intensity-to-mask file mapping | Input | Collection
--intSegMapFile | Name of the text file containing an ad-hoc intensity-to-mask file mapping. The files are assumed to reside in corresponding intensity and label collections | Input | string
--features|Select intensity and shape features required|Input|array
--filePattern|To match intensity and labeled/segmented images|Input|string
--csvfile|Save csv file as one csv file for all images or separate csv file for each image|Input|enum
--pixelDistance|Pixel distance to calculate the neighbors touching cells|Input|integer|
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
