#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <filesystem>

#include "../src/nyx/dirs_and_files.h"
#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"


void test_initialization() {

    int argc = 8;
    char** argv;

    std::filesystem::create_directory("out");

    std::vector<std::string> features = {
        "*ALL*",
        "INTEGRATED_INTENSITY",
        "MEAN", "MAX", "MEDIAN", "STANDARD_DEVIATION", "MODE",
        "SKEWNESS", "KURTOSIS", "HYPERSKEWNESS", "HYPERFLATNESS",
        "MEAN_ABSOLUTE_DEVIATION",
        "ENERGY",
        "ROOT_MEAN_SQUARED",
        "ENTROPY",
        "UNIFORMITY",
        "UNIFORMITY_PIU",
        "P01", "P10", "P25", "P75", "P90", "P99",
        "INTERQUARTILE_RANGE",
        "ROBUST_MEAN_ABSOLUTE_DEVIATION",
        "MASS_DISPLACEMENT",
        "AREA_PIXELS_COUNT",
        "COMPACTNESS",
        "BBOX_YMIN",
        "BBOX_XMIN",
        "BBOX_HEIGHT",
        "BBOX_WIDTH",
        "MAJOR_AXIS_LENGTH", "MINOR_AXIS_LENGTH", "ECCENTRICITY", "ORIENTATION", "ROUNDNESS",
        "NUM_NEIGHBORS", "PERCENT_TOUCHING",
        "EXTENT",
        "CONVEX_HULL_AREA",
        "SOLIDITY",
        "PERIMETER",
        "EQUIVALENT_DIAMETER",
        "EDGE_MEAN_INTENSITY", "EDGE_MAX_INTENSITY", "EDGE_MIN_INTENSITY", "EDGE_STDDEV_INTENSITY",
        "CIRCULARITY",
        "EROSIONS_2_VANISH",
        "EROSIONS_2_VANISH_COMPLEMENT",
        "FRACT_DIM_BOXCOUNT",
        "FRACT_DIM_PERIMETER",
        "GLCM",
        "GLRLM",
        "GLSZM",
        "GLDM",
        "NGTDM",
        "ZERNIKE2D", "FRAC_AT_D", "RADIAL_CV", "MEAN_FRAC",
        "GABOR",
        "*all_intensity*",
        "*all_morphology*",
        "*basic_morphology*",
        "*all_glcm*",
        "*all_glrlm*",
        "*all_glszm*",
        "*all_gldm*",
        "*all_ngtdm*",
        "*all_easy*"
    };

    for(auto& feature: features) {
        
        std::string feature_arg = "--features=" + feature;

        argv = new char*[9];
        argv[0] = const_cast<char*>("./nyxus");
        argv[1] = const_cast<char*>(feature_arg.c_str());
        argv[2] = const_cast<char*>("--intDir=../tests/python/data/dsb2018/train/images");
        argv[3] = const_cast<char*>("--segDir=../tests/python/data/dsb2018/train/masks");
        argv[4] = const_cast<char*>("--outDir=./out");
        argv[5] = const_cast<char*>( "--csvFile=singlecsv");
        argv[6] = const_cast<char*>("--filePattern=.*");
        argv[7] = const_cast<char*>("--loaderThreads=1");
        argv[8] = const_cast<char*>("--verbosity=4");

        std::cerr << "here" << std::endl;

        int parseRes = theEnvironment.parse_cmdline (argc, argv);

        // Have the feature manager prepare the feature toolset reflecting user's selection
        if (!theFeatureMgr.compile())
        {
            ADD_FAILURE_AT(__FILE__, __LINE__);
        }
        theFeatureMgr.apply_user_selection();

        // Scan file names
        std::vector <std::string> intensFiles, labelFiles;
        int errorCode = Nyxus::read_dataset (
            theEnvironment.intensity_dir, 
            theEnvironment.labels_dir, 
            theEnvironment.get_file_pattern(),
            theEnvironment.output_dir, 
            theEnvironment.intSegMapDir, 
            theEnvironment.intSegMapFile, 
            true, 
            intensFiles, labelFiles);

        if (errorCode)
        {
            ADD_FAILURE_AT(__FILE__, __LINE__);
        }

        // One-time initialization
        init_feature_buffers();

        // Process the image sdata
        int min_online_roi_size = 0;
        errorCode = processDataset (
            intensFiles, 
            labelFiles, 
            theEnvironment.n_loader_threads, 
            theEnvironment.n_pixel_scan_threads, 
            theEnvironment.n_reduce_threads,
            min_online_roi_size,
            true, // 'true' to save to csv
            theEnvironment.output_dir);

        // Check the error code 
        if (errorCode) {
            ADD_FAILURE_AT(__FILE__, __LINE__);
        }

        for (auto& lab: uniqueLabels) {
        
            LR& r = roiData[lab];

            const ImageMatrix& im = r.aux_image_matrix;

            ASSERT_GT(im.height, 0);
            ASSERT_GT(im.width, 0);
        }
    }

}
