#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <memory>
#include <unordered_map>
#include <unordered_set> 
#include <algorithm>
#include <iostream>
#include <chrono>
#include <thread>
#include <future>
#include <array>
#include "sensemaker.h"


#ifndef __unix
#define NOMINMAX	// Prevent converting std::min(), max(), ... into macros
#include<windows.h>
#endif


void Contour::calculate(std::vector<Pixel2> rawPixels)
{
    for (auto& pix : rawPixels)
    {
        // check if x-1 exists
        bool found = false;
        for (auto& pix2 : rawPixels)
            if (pix2.y == pix.y && pix2.x == pix.x - 1)
            {
                found = true;
                break;
            }
        if (!found)
        {
            contour_pixels.push_back(pix);	// Register a contour pixel
            continue;	// No need to check other neighboring pixels of this pixel, we've known that it lives on the contour
        }

        // check if x+1 exists
        found = false;
        for (auto& pix2 : rawPixels)
            if (pix2.y == pix.y && pix2.x == pix.x + 1)
            {
                found = true;
                break;
            }
        if (!found)
        {
            contour_pixels.push_back(pix);	// Register a contour pixel
            continue;	// No need to check other neighboring pixels of this pixel, we've known that it lives on the contour
        }

        // check if y-1 exists
        found = false;
        for (auto& pix2 : rawPixels)
            if (pix2.x == pix.x && pix2.y == pix.y - 1)
            {
                found = true;
                break;
            }
        if (!found)
        {
            contour_pixels.push_back(pix);	// Register a contour pixel
            continue;	// No need to check other neighboring pixels of this pixel, we've known that it lives on the contour
        }

        // check if y+1 exists
        found = false;
        for (auto& pix2 : rawPixels)
            if (pix2.x == pix.x && pix2.y == pix.y + 1)
            {
                found = true;
                break;
            }
        if (!found)
        {
            contour_pixels.push_back(pix);	// Register a contour pixel
            continue;	// No need to check other neighboring pixels of this pixel, we've known that it lives on the contour
        }
    }
}

StatsInt Contour::get_roi_perimeter()
{
    return (StatsInt)contour_pixels.size();
}

StatsReal Contour::get_diameter_equal_perimeter()
{
    StatsReal retval = get_roi_perimeter() / M_PI;
    return retval;
}
