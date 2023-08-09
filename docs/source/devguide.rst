.. role:: raw-html-m2r(raw)
   :format: html

Developer's guide
=================

Adding a new feature
--------------------

Adding a feature is a 9-step procedure.

Step 1 - create a numeric identifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

 Come up with an internal c++ compliant identifier for the feature and its user-facing counterpart. Edit enum AvailableFeatures in file featureset.h keeping constant _COUNT_ , for example :

.. code-block:: c++
   
   enum AvailableFeatures
   {
       ...
       MYFEATURE1,
       MYFEATURE2,
       MYFEATURE3,
       ...
       _COUNT_
   };

Step 2 - create a user facing string identifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

edit the integer to string feature identifier mapping in mapping UserFacingFeatureNames in file featureset.cpp. For example, if we want to give features MYFEATURE1, MYFEATURE2, MYFEATURE3 which so far are just numeric constants user-facing names MY_FEATURE_1, MY_FEATURE_2, and MY_FEATURE_3 that can be used in the command line, we need to edit UserFacingFeatureNames the following way

.. code-block:: c++

   std::map <std::string, AvailableFeatures> UserFacingFeatureNames =
   {
       ...
       {"MY_FEATURE_1", MYFEATURE1},
       {"MY_FEATURE_2", MYFEATURE2},
       {"MY_FEATURE_3", MYFEATURE3 },
       ...
   };

Step 3 - create a feature method class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any Nyxus feature needs to be derived from class FeatureMethod defining a particular calculator of one or multiple features. FeatureMethod is a skeleton of the custom feature calculator responding to image data streamed to it in various ways - pixel by pixel (so called online mode), as a cached pixel cloud in the form of std::vector\ :raw-html-m2r:`<Pixel2>` object for images whose size permits caching a single ROI's data in user computer's RAM (so called trivial ROIs), and as a browser of a mask-intensity image pair for ROIs whose cache exceeds the RAM limit (so called non-trivial or oversized ROIs). All the particular feature calculation logic neds to be placed in your implementation of FeatureMethod's pure virtual methods. The class's header and source files are suggested to be placed in directory "features". For example, if we want to implement a class calculating a 3-segmental ROI intensity statistics (means) weighted by unit perimeter length delivered to user as features MYFEATURE1, MYFEATURE2, and MYFEATURE3: 

.. code-block:: c++

   #include "../feature_method.h"
   class ShamrockFeature: public FeatureMethod
   {
   public:
       ShamrockFeature(); 
       void calculate(LR& r);
       void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
       void osized_calculate(LR& r, ImageLoader& imloader);
       void save_value(std::vector<std::vector<double>>& feature_vals);
       virtual void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);
       static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

       // Constants used in the output
       const static int num_segments = 3;

   private:
       std::vector<double> segment_means;
   };

Step 4 - define feature class' provided features and feature dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Feature methods are run by Nyxus feature manager. The order of their running is determined by their inter-dependencies. Class FeatureMethod's function method provide_features() lets you declare specific features implemented by your class; method add_dependencies() lets you declare features that need to be calculated and saved to ROIs' LR::fvals cache prior to running calculations of your feature method. The feature codes that you use as arguments to add_dependencies() and provide_features() come from file featureset.h . For example

.. code-block:: c++

   ShamrockFeature::ShamrockFeature() : 
       FeatureMethod("ShamrockFeature") 
   {
       // we expose them
       provide_features ({MYFEATURE1, MYFEATURE2, MYFEATURE3}); 

       // we need this feature prior to working on MYFEATURE1, MYFEATURE2, and MYFEATURE3
       add_dependencies ({PERIMETER}); 
   }

Step 5 - plan feature's internal and exposed data; implement saving results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Step 6 - implement feature method's online behavior (for oversized ROIs only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In order to perform some action on the level of individual pixel while scanning a trivial ROI e.g. calculate some statistics using Welford principle, override abstract method

.. code-block:: c++

   void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);


or give it empty body.

Step 7 - implement feature calculation of regular sized ROIs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ROIs are classified to regular ("trivial") or oversized automatically based on their area in pixels. It's developer's responsibility to handle both cases by implementing pure virtual methods of abstract class FeatureMethod, parent of your particular feature method. To implement regular-sized feature calculation, override method

.. code-block:: c++

   void calculate (LR& r);


For example

.. code-block:: c++

   void ShamrockFeature::calculate(LR& r)
   {
      // prepare the results buffer
      segment_means.resize(num_segments);



      // iterate cached ROI pixels
      for (auto& px : raw_pixels)
      {
         // accumulate sums
         ...
      }

      // calculate elements of segment_means
      ...

   }


Step 8 - implement feature calculation of oversized ROIs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An oversized ROI's cached data cannot fit in computer meory so in the oversized ROI scenarios we cannot rely on its pixel cloud or image matrix. Instead, all the calculations should be performed "in place" - using the image browser class ImageLoader (header image_loader.h) similarly to class ImageMatrix (image_matrix.h) and creating out of memory cache using classes OutOfRamPixelCloud, OOR_ReadMatrix, ReadImageMatrix_nontriv, and WriteImageMatrix_nontriv (header image_matrix.nontriv). You are guaranteed to have initialized object LR::osized_pixel_cloud prior to the call of method osized_calculate(). For example:

.. code-block:: c++

   void ShamrockFeature::osized_calculate (LR& r, ImageLoader& imlo)
   {
      // prepare the results buffer
      segment_means.resize(num_segments);



      // iterate ROI pixels directly in the huge source image
      OutOfRamPixelCloud& cloud = r.osized_pixel_cloud;
      for (size_t i = 0; i < cloud.get_size(); i++) // oversized analog for for(auto& px : raw_pixels)
      {
         auto pxA = cloud.get_at(i);
         // accumulate sums
         ...
      }

      // calculate elements of segment_means
      ...

   }


Step 9 - implementing the output of composite features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If your feature method class provides multiple features, like ShamrockFeature calculating intensity statistics in 3 segmental bins in the above example, the output of corresponding values can be managed for the CSV-file and Python bindings in functions

.. code-block:: c++

   save_features_2_csv (std::string intFpath, std::string segFpath, std::string outputDir)


and

.. code-block:: c++

   save_features_2_buffer(std::vectorstd::string& headerBuf, std::vector\ :raw-html-m2r:`<double>`\ & resultBuf, std::vectorstd::string& stringColBuf)


accordingly.

The ROI cache - structure LR
-----------------------------

A mask-intensity image pair is being prescanned and examined before the feature manager runs feature calculation of each feature method. As a result of that examination ROIs are being determined themselves and structure LR (defined in file roi_cache.h) is initialized for each ROI. Some fields are essential to developer's feature calculation in overridable methods of base class FeatureMethod:


.. list-table::
   :header-rows: 1
   
   * - LR field 
     - Description 
   * - int label 
     - ROI's integer ID number 
   * - std::string segFname, intFname 
     - ROI's host mask and intensity image names 
   * - std::vector <Pixel2> raw_pixels 
     - cloud of ROI's cached pixels 
   * - OutOfRamPixelCloud osized_pixel_cloud 
     - cloud of ROI's pixels cached out of memory 
   * - unsigned int aux_area 
     - ROI area in pixels
   * - PixIntens aux_min, aux_max 
     - minimum and maximum pixel intensity within the ROI mask 
   * - AABB aabb 
     - axis aligned bounding box giving ROI's bounding box dimensions and origin position 
   * - std::vector<Pixel2> contour 
     - (trivial ROIs only) pixlels of ROI contour initialized by feature PERIMETER
   * - std::vector<Pixel2> convHull_CH 
     - (trivial ROIs only) pixels of ROI's convex hull initialized as a result of calculating any of features CONVEX_HULL_AREA, SOLIDITY, and CIRCULARITY 
   * - std::vector<std::vector<StatsReal>> fvals 
     - vector of feature value vectors of length AvailableFeatures::\_COUNT\_ (see file featureset.h) 
   * - ImageMatrix aux_image_matrix 
     - (trivial ROIs only) matrix of pixel intensities
   * - std::unordered_set <unsigned int> host_tiles 
     - indices of TIFF tiles hosting the ROI (generally a ROI can span multiple TIFF tiles)  


Adding a feature group
-----------------------
Often multiple features need to be calculated together and the user faces the need to specify a long comma separated list of features. As a result the command line may become cumbersome. For example, calculating some popular morphologic features may involve the following command line

.. code-block:: bash

   nyxus --features=AREA_PIXELS_COUNT,AREA_UM2,CENTROID_X,CENTROID_Y,BBOX_YMIN,BBOX_XMIN,BBOX_HEIGHT,BBOX_WIDTH --intDir=/home/ec2-user/work/datasetXYZ/int --segDir=/home/ec2-user/work/dataXYZ/seg --outDir=/home/ec2-user/work/datasetXYZ --filePattern=.* --outputType=separatecsv


Features can be grouped toegther and gived convenient aliases, for example the above features AREA_PIXELS_COUNT, AREA_UM2, CENTROID_X, CENTROID_Y, BBOX_YMIN, BBOX_XMIN, BBOX_HEIGHT, and BBOX_WIDTH can be refered to as \*BASIC_MORPHOLOGY\* . (Asterisks are a part of the alias and aren't special symbols.) The command line then becomes simpler

.. code-block:: bash

   nyxus --features=\ *BASIC_MORPHOLOGY* AREA_PIXELS_COUNT,AREA_UM2,CENTROID_X,CENTROID_Y,BBOX_YMIN,BBOX_XMIN,BBOX_HEIGHT,BBOX_WIDTH*\ * --intDir=/home/ec2-user/work/datasetXYZ/int --segDir=/home/ec2-user/work/dataXYZ/seg --outDir=/home/ec2-user/work/datasetXYZ --filePattern=.* --outputType=separatecsv

Step 1 - giving an alias to a multiple features 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Given the features that you need to group together are already implemented, to create a feature group define its user-facing identifier in file environment.h, for example create alias MY_FEATURE_GROUP for features MYF1, MYF2, and MYF3

.. code-block:: c++

   define MY_FEATURE_GROUP "MYFEATURES"


Step 2 - reflect the new group in the command line help 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Make sure that the new feature group's alias is visible in the command line help.
Then handle the command line input in file environment.cpp, method Environment::process_feature_list()

.. code-block:: c++

   if (s == MY_FEATURE_GROUP)
   {
      auto F = {MYF1, MYF2, MYF3};
      theFeatureSet.enableFeatures(F);
      continue;
   }


Step 3 - reflect the new group available to plugin users
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In plugin use cases, don't forget to update the plugin manifest with the information about the new feature group! For example, in WIPP:

.. code-block:: c++

   ...

   {
      "description": "MYFEATURES is a group of my few handy features",
      "enum": ["MYFEATURES"]
   },
   ...

