#include <gtest/gtest.h>


#include "../src/nyx/roi_cache.h"
#include "../src/nyx/parallel.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"

// check values of multiple features
void test_truth(const std::vector<double> values, const std::vector<double>& truth_values, double frac_tolerance) {

    if (values.size() != truth_values.size()) {
        FAIL() << "Number of truth values does not match the number of features calculated" << std::endl;
    }

    for (int i = 0; i < values.size(); ++i) {
        ASSERT_TRUE(agrees_gt(values[i], truth_values[i], frac_tolerance));
    }
}

// check value of single feature
void test_truth(const std::vector<double> values, double truth_value, double frac_tolerance) {
    ASSERT_TRUE(agrees_gt(values[0], truth_value, frac_tolerance));
}

template <class T, class S>
void test_feature(const T& feature_object,
                  const Feature2D& feature_name,
                  const int num_feautures, 
                  const NyxusPixel* intensity_data, 
                  const NyxusPixel* mask_data, 
                  int image_size,
                  const S& truth_value,
                  double frac_tolerance = 1000) {
    
    LR roidata;
    // Calculate features
    T f;

    load_masked_test_roi_data (roidata, intensity_data, mask_data, image_size);
    ASSERT_NO_THROW(f.calculate(roidata));

    // Initialize per-ROI feature value buffer with zeros
    roidata.initialize_fvals();

    // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
    f.save_value(roidata.fvals);

    std::cout << "value: " << roidata.fvals[(int)feature_name][0] << std::endl;

    test_truth(roidata.fvals[(int)feature_name], truth_value, frac_tolerance);
};
