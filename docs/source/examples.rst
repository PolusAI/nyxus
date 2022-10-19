
Examples
========

This chapter presents some particular usage cases of Nyxus

1. Requesting specific features
-------------------------------

Suppose we need to extract only Zernike features and first 3 Hu's moments:

.. code-block:: bash

   ./nyxus --features=ZERNIKE2D,HU_M1,HU_M2,HU_M3 --intDir=/home/ec2-user/data-ratbrain/int --segDir=/home/ec2-user/data-ratbrain/seg --outDir=/home/ec2-user/work/OUTPUT-ratbrain --filePattern=.* --csvFile=singlecsv

2. Requesting specific feature groups
-------------------------------------

Suppose we need to extract only intensity features basic morphology features: 

.. code-block:: bash

   ./nyxus --features=*all_intensity*,*basic_morphology* --intDir=/home/ec2-user/data-ratbrain/int --segDir=/home/ec2-user/data-ratbrain/seg --outDir=/home/ec2-user/work/OUTPUT-ratbrain --filePattern=.* --csvFile=singlecsv

3. Mixing specific feature groups and individual features
---------------------------------------------------------

Suppose we need to extract intensity features, basic morphology features, and Zernike features: 

.. code-block:: bash

   ./nyxus --features=*all_intensity*,*basic_morphology*,zernike2d --intDir=/home/ec2-user/data-ratbrain/int --segDir=/home/ec2-user/data-ratbrain/seg --outDir=/home/ec2-user/work/OUTPUT-ratbrain --filePattern=.* --csvFile=singlecsv

4. Specifying a feature list from with a file instead of command line
---------------------------------------------------------------------

Sometimes a list of requested features can be long making Nyxus command line huge. An alternative to dealing with a long command line is specifying all the desired features in a comma, space, and newline delimited text file. Suppose a feature set is in file feature_list.txt:

.. code-block:: bash

   mean,min,kurtosis
   skewness

Then the command line will be:

.. code-block:: bash

   ./nyxus --features=feature_list.txt --intDir=/home/ec2-user/data-ratbrain/int --segDir=/home/ec2-user/data-ratbrain/seg --outDir=/home/ec2-user/work/OUTPUT-ratbrain --filePattern=.* --csvFile=singlecsv

5. Whole-image feature extraction
---------------------------------

The regular operation mode of Nyxus is processing pairs of intensity and mask images treating non-zero pixel values of the mask image as segment label. The other operation mode is the so called "single-ROI mode" - treating the intensity image as segment. To activate it, just reference the intensity image collection as mask in the command line:

.. code-block:: bash

   ./nyxus --features=*basic_morphology* --intDir=/home/ec2-user/data-ratbrain/int --segDir=/home/ec2-user/data-ratbrain/int --outDir=/home/ec2-user/work/OUTPUT-ratbrain --filePattern=.* --csvFile=singlecsv

6. Regular and ad-hoc mapping between intensity and mask image files
--------------------------------------------------------------------

Intensity and mask image collections are specified in the command line (via parameters --intDir and --segDir) and the default mapping between an intensity and mask image, after applying a file name pattern (via parameter --filePattern), is the 1:1 mapping:

.. code-block:: bash

   intensity_image_1       segment_image_1
   intensity_image_2       segment_image_2
   intensity_image_3       segment_image_3
   intensity_image_4       segment_image_4

Here, each intensity and mask image is assumed to reside in the corresponding image collection directory specified with command line arguments --intDir=/home/ec2-user/data-ratbrain/int --segDir=/home/ec2-user/data-ratbrain/seg. More precisely:

.. code-block:: bash

   /home/ec2-user/data-ratbrain/int/image_1.ome.tif    /home/ec2-user/data-ratbrain/seg/image_1.ome.tif
   /home/ec2-user/data-ratbrain/int/image_2.ome.tif    /home/ec2-user/data-ratbrain/seg/image_2.ome.tif
   /home/ec2-user/data-ratbrain/int/image_3.ome.tif    /home/ec2-user/data-ratbrain/seg/image_3.ome.tif
   /home/ec2-user/data-ratbrain/int/image_4.ome.tif    /home/ec2-user/data-ratbrain/seg/image_4.ome.tif

In case the dataset is based on a 1:N mapping, for example

.. code-block:: bash
 
   intensity_image_1       segment_image_A
   intensity_image_2       segment_image_A
   intensity_image_3       segment_image_A
   intensity_image_4       segment_image_B

the user needs to pass such an ad-hoc mapping to Nyxus via referenceing a mapping definition text file in the command line (parameter --intSegMapFile). 

**Note: the order of mapping definition file columns is critical, and the 1-st column is interpreted as the intensity image files column while the 2-nd column is interpreted as the mask image files.** 

Assuming contents of file mapping.txt is

.. code-block:: bash

   image_1.ome.tif       image_A.ome.tif
   image_2.ome.tif       image_A.ome.tif
   image_3.ome.tif       image_A.ome.tif
   image_4.ome.tif       image_B.ome.tif

and the file is passed to Nyxus via parameter --intSegMapFile, the mapping will resolve to mapping

.. code-block:: bash

   /home/ec2-user/data-ratbrain/int/image_1.ome.tif    /home/ec2-user/data-ratbrain/seg/image_A.ome.tif
   /home/ec2-user/data-ratbrain/int/image_2.ome.tif    /home/ec2-user/data-ratbrain/seg/image_A.ome.tif
   /home/ec2-user/data-ratbrain/int/image_3.ome.tif    /home/ec2-user/data-ratbrain/seg/image_A.ome.tif
   /home/ec2-user/data-ratbrain/int/image_4.ome.tif    /home/ec2-user/data-ratbrain/seg/image_B.ome.tif

7. Ad-hoc mapping between intensity and mask image files via Python interface
-----------------------------------------------------------------------------

Alternatively, Nyxus can process explicitly defined pairs of intensity-mask images, for example image "i1" with mask "m1" and image "i2" with mask "m2":

```python 
from nyxus import Nyxus
nyx = Nyxus(["*ALL*"])
features = nyx.featurize(
    [
        "/path/to/images/intensities/i1.ome.tif", 
        "/path/to/images/intensities/i2.ome.tif"
    ], 
    [
        "/path/to/images/labels/m1.ome.tif", 
        "/path/to/images/labels/m2.ome.tif"
    ])
```


