#include <gtest/gtest.h>


#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/pixel.h"
#include "../src/nyx/environment.h"
#include "test_data.h"
#include "test_main_nyxus.h"
namespace Nyxus
{
    template <class T, class S, class F>
    void test_feature(const T& feature_object,
                    const F& feature_name,
                    const int num_feautures, 
                    const NyxusPixel* intensity_data, 
                    const NyxusPixel* mask_data, 
                    int image_size,
                    const S& truth_value,
                    double frac_tolerance = 1000) {
        
        LR roidata;
        // Calculate features
        T f;
        Fsettings s;

        load_masked_test_roi_data (roidata, intensity_data, mask_data, image_size);
        ASSERT_NO_THROW(f.calculate(roidata, s));

        // Initialize per-ROI feature value buffer with zeros
        roidata.initialize_fvals();

        // Retrieve values of the features implemented by class 'PixelIntensityFeatures' into ROI's feature buffer
        f.save_value(roidata.fvals);

        ASSERT_TRUE(agrees_gt(roidata.fvals[(int)feature_name][0], truth_value, frac_tolerance));
    };
};
