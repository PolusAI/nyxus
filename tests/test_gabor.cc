#include <fstream>
#include "../src/nyx/helpers/fsystem.h"
#include "../src/nyx/features/gabor.h"
#include "../src/nyx/features/intensity.h"
#include "test_gabor.h"
#include "test_gabor_truth.h"
#include "test_main_nyxus.h"

using namespace std;
using namespace Nyxus;

void test_gabor(bool gpu)
{
    for(int i = 0; i < dsb_data.size(); ++i) 
    {
        LR roidata;
        roidata.initialize_fvals();

        // Feed data to the ROI
        load_test_roi_data(roidata, i);

        // Calculate features
        GaborFeature f;
        if(gpu) 
        {
            #ifdef USE_GPU
                ASSERT_NO_THROW(f.calculate_gpu_multi_filter(roidata));
            #else
                std::cerr << "GPU build is not enabled. Defaulting to CPU version." << std::endl;
                ASSERT_NO_THROW(f.calculate(roidata));
            #endif
        } 
        else 
        {
            ASSERT_NO_THROW(f.calculate(roidata));
        }

        f.save_value (roidata.fvals);

        ASSERT_TRUE(gabor_truth[i].size() == roidata.fvals[(int)Nyxus::Feature2D::GABOR].size());

        for(int j = 0; j < gabor_truth[i].size(); ++j) 
        {
            ASSERT_TRUE(agrees_gt(gabor_truth[i][j], roidata.fvals[(int)Nyxus::Feature2D::GABOR][j]));
        }
    }
}
