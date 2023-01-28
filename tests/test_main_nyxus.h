#pragma once

#include "../src/nyx/roi_cache.h"
#include "../src/nyx/parallel.h"
#include "test_dsb2018_data.h"
#include "test_data.h"

namespace Nyxus
{
    /// @brief Tests the agreement with ground truth up to the tolerance specified as a fraction of the ground truth
    static bool agrees_gt(double fval, double ground_truth, double frac_tolerance = 1000.)
    {
        auto diff = fval - ground_truth;
        auto tolerance = ground_truth / frac_tolerance;
        bool good = std::abs(diff) <= std::abs(tolerance);
        return good;
    }

    static void load_test_roi_data (LR& roidata, const NyxusPixel* testData, size_t count)
    {
        int dummyLabel = 100, dummyTile = 200;

        // -- mocking gatherRoisMetrics():
        for (auto i=0; i<count; i++)
        {
            const NyxusPixel& px = testData[i];
            // -- mocking feed_pixel_2_metrics ():
            if (roidata.aux_area == 0)
                init_label_record_2(roidata, "theSegFname", "theIntFname", px.x, px.y, dummyLabel, px.intensity, dummyTile);
            else
                update_label_record_2(roidata, px.x, px.y, dummyLabel, px.intensity, dummyTile);
        }

        // -- mocking scanTrivialRois():
        for (auto i = 0; i < count; i++)
        {
            const NyxusPixel& px = testData[i];
            // -- mocking feed_pixel_2_cache ():
            roidata.raw_pixels.push_back(Pixel2(px.x, px.y, px.intensity));
        }
    }

    static void load_masked_test_roi_data (LR& roidata, const NyxusPixel* intensityData, const NyxusPixel* maskData, size_t count)
    {
        int dummyLabel = 100, dummyTile = 200;
        // -- mocking phase 1, gatherRoisMetrics():
        for (auto i = 0; i < count; i++)
        {
            // Check if pixel [i] belongs to the ROI
            const NyxusPixel& maskPixel = maskData[i];
            if (maskPixel.intensity == 0)
                continue;   // Skip this pixel
            // Pixel [i] is within the ROI, feed it to ROI shape and intensity range examiner
            const NyxusPixel& px = intensityData[i];
            // -- mocking feed_pixel_2_metrics ():
            if (roidata.aux_area == 0)
                init_label_record_2(roidata, "theSegFname", "theIntFname", px.x, px.y, dummyLabel, px.intensity, dummyTile);
            else
                update_label_record_2(roidata, px.x, px.y, dummyLabel, px.intensity, dummyTile);
        }
        // -- mocking phase 2, scanTrivialRois():
        for (auto i = 0; i < count; i++)
        {
            // Check if pixel [i] belongs to the ROI
            const NyxusPixel& maskPixel = maskData[i];
            if (maskPixel.intensity == 0)
                continue;   // Skip this pixel
            // Pixel [i] is within the ROI, feed it to ROI pixel accumulator
            const NyxusPixel& px = intensityData[i];
            // -- mocking feed_pixel_2_cache ():
            roidata.raw_pixels.push_back(Pixel2(px.x, px.y, px.intensity));
        }
        // -- allocating the image matrix (roidata.aux_image_matrix)
        //      (Phase 1 creates roidata.aabb giving us ROI's dimensions)
        roidata.aux_image_matrix.allocate(
            roidata.aabb.get_width(),
            roidata.aabb.get_height());
        // -- filling the image matrix
        roidata.aux_image_matrix.calculate_from_pixelcloud (roidata.raw_pixels, roidata.aabb);
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