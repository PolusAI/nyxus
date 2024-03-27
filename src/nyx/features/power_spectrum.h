#pragma once

#include <vector>
#include "../helpers/helpers.h"
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"
#include "../environment.h"

/// @brief Extract face feature based on gabor filtering
class PowerSpectrumFeature: public FeatureMethod
{
public:

    const constexpr static std::initializer_list<Nyxus::Feature2D> featureset = { Nyxus::Feature2D::FOCUS_SCORE };

    PowerSpectrumFeature();

    static bool required(const FeatureSet& fs);
   
    //=== Trivial ROIs ===
    void calculate(LR& r);

    //=== Non-trivial ROIs ===
    void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}
    void osized_calculate(LR& r, ImageLoader& imloader){};

    // Result saver
    void save_value(std::vector<std::vector<double>>& feature_vals);

    static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

    //-------------- - User interface
    static int ksize;

private:
 
    std::vector<double> fvals;

    //=== Trivial ROIs ===

    std::vector<double> invariant(std::vector<unsigned int> image);
    std::tuple<std::vector<int>, std::vector<double>, std::vector<double>>  rps(std::vector<unsigned int> image, int rows, int cols);
    double power_spectrum_slope(const ImageMatrix& Im);
};

