#pragma once

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include "../src/nyx/helpers/fsystem.h"
#include "../src/nyx/dirs_and_files.h"
#include "../src/nyx/cache.h"
#include "../src/nyx/globals.h"

//xxxxxxxxxx using Nyxus::allocateTrivialRoisBuffers;
//xxxxxxxxx using Nyxus::freeTrivialRoisBuffers;

void test_initialization() 
{
    Nyxus::Roidata roiData;
    CpusideCache hostCache;

    std::vector<int> pending {0, 1, 2, 3};

    for(int i = 0 ; i < pending.size(); ++i) 
    {
        LR roi;
        load_test_roi_data (roi, i, false);
        roiData[i] = roi;
    }

    allocateTrivialRoisBuffers (pending, roiData, hostCache);

    for(int i = 0; i < pending.size(); ++i)
    {
        LR& r = roiData[i];
        const ImageMatrix& im = r.aux_image_matrix;
        ASSERT_TRUE(im.height > 0);
        ASSERT_TRUE(im.width > 0);
    }

    freeTrivialRoisBuffers (pending, roiData);

}

