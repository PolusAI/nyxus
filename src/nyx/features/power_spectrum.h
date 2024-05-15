#pragma once

#include <vector>
#include <complex>
#include "../helpers/helpers.h"
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"
#include "../environment.h"

/// @brief Extract face feature based on gabor filtering
class PowerSpectrumFeature: public FeatureMethod
{
public:

    const constexpr static std::initializer_list<Nyxus::FeatureIMQ> featureset = { Nyxus::FeatureIMQ::FOCUS_SCORE };

    PowerSpectrumFeature();

    static bool required(const FeatureSet& fs);
   
    //=== Trivial ROIs ===
    void calculate(LR& r);

    //=== Non-trivial ROIs ===
    void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}
    void osized_calculate(LR& r, ImageLoader& imloader){};

    static void parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
    void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);

    // Result saver
    void save_value(std::vector<std::vector<double>>& feature_vals);

    static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

    //-------------- - User interface
    static int ksize;

private:
    double slope_;

    //=== Trivial ROIs ===

    static void invariant(const std::vector<unsigned int>& image, std::vector<double>& out);

    static void rps(const std::vector<unsigned int>& image, 
              int rows, int cols, 
              std::vector<double>& mag_sum, 
              std::vector<double>& power_sum);

    static double power_spectrum_slope(const ImageMatrix& Im);

    static unsigned int next_power_of_2(unsigned int x) {
        if (x == 0) return 1;
        --x;
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;
        return ++x;
    }

    static void power_of_2_padding(std::vector<std::complex<double>>& complex_image, int rows, int cols);
};

