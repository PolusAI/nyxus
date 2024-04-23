#pragma once

#include <vector>
#include <iostream>
#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <numeric>
#include <unordered_map>
#include "brisque_gaussian.h"
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"
#include "../environment.h"

class BrisqueFeature : public FeatureMethod {

private:
    std::unordered_map<MscnType, std::vector<double>> coefficients_;

    std::vector<double> features_;

    std::vector<double> get_normalized_gaussian_kernel(int kernel_size, float sigma);
    std::vector<double> calculate_features(MscnType type);

public:

    BrisqueFeature();

    void calculate(LR& r);

    static bool required(const FeatureSet& fs);

    void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);
    static void parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

    //=== Non-trivial ROIs ===
    void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}
    void osized_calculate(LR& r, ImageLoader& imloader){};

    // Result saver
    void save_value(std::vector<std::vector<double>>& feature_vals);

    static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

    void brisque(const ImageMatrix& Im);
};
