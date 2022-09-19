#pragma once

#include "../src/nyx/roi_cache.h"
#include <src/nyx/parallel.h>
#include "test_dsb2018_data.h"
#include "test_data.h"

namespace Nyxus
{
    extern void init_label_record_2(LR& lr, const std::string& segFile, const std::string& intFile, int x, int y, int label, PixIntens intensity, unsigned int tile_index);
    extern void update_label_record_2(LR& lr, int x, int y, int label, PixIntens intensity, unsigned int tile_index);
    extern void allocateTrivialRoisBuffers(const std::vector<int>& Pending);
    extern void freeTrivialRoisBuffers(const std::vector<int>& Pending);

    /// @brief Tests the agreement with ground truth up to the tolerance specified as a fraction of the ground truth
    static bool agrees_gt(double fval, double ground_truth, double frac_tolerance = 1000.)
    {
        auto diff = fval - ground_truth;
        auto tolerance = ground_truth / frac_tolerance;
        bool good = std::abs(diff) <= std::abs(tolerance);
        return good;
    }

    static void load_test_roi_data(LR& roidata)
    {
        int dummyLabel = 100, dummyTile = 200;

        // -- mocking gatherRoisMetrics():
        for (auto& px : testData)
        {
            // -- mocking feed_pixel_2_metrics ():
            if (roidata.aux_area == 0)
                init_label_record_2(roidata, "theSegFname", "theIntFname", px.x, px.y, dummyLabel, px.intensity, dummyTile);
            else
                update_label_record_2(roidata, px.x, px.y, dummyLabel, px.intensity, dummyTile);
        }

        // -- mocking scanTrivialRois():
        for (auto& px : testData)
            // -- mocking feed_pixel_2_cache ():
            roidata.raw_pixels.push_back(Pixel2(px.x, px.y, px.intensity));
    }

    static void load_test_roi_data(LR& roidata, int data_idx, bool allocate_IM = true)
    {
        int dummyLabel = 100, dummyTile = 200;

        // -- mocking gatherRoisMetrics():
        int i = 0;
        size_t w = dsb_data[data_idx].x;
        size_t h = dsb_data[data_idx].y;
        for (auto& px : dsb_data[data_idx].pixels)
        {
            // -- mocking feed_pixel_2_metrics ():
            if (roidata.aux_area == 0)
                init_label_record_2(roidata, "theSegFname", "theIntFname", i%w, i/w, dummyLabel, px, dummyTile);
            else
                update_label_record_2(roidata, i%w, i/w, dummyLabel, px, dummyTile);

            ++i;
        }

        // -- mocking scanTrivialRois():
        i = 0;
        for (auto& px : dsb_data[data_idx].pixels) {
            // -- mocking feed_pixel_2_cache ():
            roidata.raw_pixels.push_back(Pixel2(i%w, i/w, px));
            ++i;
        }

        if (allocate_IM)
            roidata.aux_image_matrix = ImageMatrix(roidata.raw_pixels);

    }
}