#pragma once

#include <vector>
#include "../helpers/helpers.h"
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"
#include "../environment.h"

#define EPSILON 1e-8

/// @brief Extract face feature based on gabor filtering
class SharpnessFeature: public FeatureMethod
{
public:

    const constexpr static std::initializer_list<Nyxus::Feature2D> featureset = { Nyxus::Feature2D::SHARPNESS };

    SharpnessFeature();

    static bool required(const FeatureSet& fs);
   
    //=== Trivial ROIs ===
    void calculate(LR& r);

    static void parallel_process_1_batch(size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
    void parallel_process(std::vector<int>& roi_labels, std::unordered_map <int, LR>& roiData, int n_threads);

    //=== Non-trivial ROIs ===
    void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}
    void osized_calculate(LR& r, ImageLoader& imloader){};

    // Result saver
    void save_value(std::vector<std::vector<double>>& feature_vals);

    static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

    //-------------- - User interface
    static int ksize;

private:

    // Result cache
    double sharpness_;

    std::vector<unsigned int> pad_array(const std::vector<unsigned int>& array, int rows, int cols, int padRows, int padCols);

    std::vector<double> remove_padding(std::vector<double> img, int img_row, int img_col, int row_padding, int col_padding);

    std::vector<double> median_blur(const std::vector<unsigned int>& img, int rows, int cols, int ksize);

    std::vector<double> convolve_1d(const std::vector<double>& img, std::vector<double>& kernel);

    std::vector<double> smooth_image(const std::vector<unsigned int>& image, int rows, int cols, bool transpose=false, double epsilon=1e-8);

    std::tuple<std::vector<double>, std::vector<double>> edges(const std::vector<unsigned int>& image, int rows, int cols, double edge_threshold=0.0001);

    std::vector<double> absolute_difference(const std::vector<double>& mat1, const std::vector<double>& mat2, int numRows, int numCols);

    std::tuple<std::vector<double>, std::vector<double>> dom(std::vector<double>& Im, int rows, int cols);

    std::tuple<std::vector<double>, std::vector<double>> contrast(const std::vector<double>& Im, int rows, int cols);

    double sharpness(const ImageMatrix& Im, int width=2);

};

