#pragma once

#include <vector>
#include "../helpers/helpers.h"
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"
#include "../environment.h"

static constexpr double EPSILON = 1e-8;

/// @brief Extract face feature based on gabor filtering
class SharpnessFeature: public FeatureMethod
{
public:

    const constexpr static std::initializer_list<Nyxus::FeatureIMQ> featureset = { Nyxus::FeatureIMQ::SHARPNESS };

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

    static void pad_array(const std::vector<unsigned int>& array, std::vector<unsigned int>& out, int rows, int cols, int padRows, int padCols);

    static void remove_padding(std::vector<double>& img, int img_row, int img_col, int row_padding, int col_padding);

    static void median_blur(const std::vector<unsigned int>& img, std::vector<double>& blurred_img_out, int rows, int cols, int ksize);

    static std::vector<double> convolve_1d(const std::vector<double>& img, std::vector<double>& kernel);

    static void smooth_image(const std::vector<unsigned int>& image, std::vector<double>& smoothed, std::vector<double>& smoothed_transposed, int rows, int cols, double epsilon=1e-8);

    static void edges(const std::vector<unsigned int>& image, std::vector<double>& edge_x_out, std::vector<double>& edge_y_out, int rows, int cols, double edge_threshold=0.0001);

    static void dom(const std::vector<double>& Im, std::vector<double>& dom_x_out, std::vector<double>& dom_y_out, int rows, int cols);

    static void contrast(const std::vector<double>& Im, std::vector<double>&cx_out, std::vector<double>&cy_out, int rows, int cols);

    static double sharpness(const ImageMatrix& Im, int width=2);

};

