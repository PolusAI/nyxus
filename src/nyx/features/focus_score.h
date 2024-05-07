#pragma once

#include <vector>
#include "../helpers/helpers.h"
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"
#include "../environment.h"

/// @brief Extract face feature based on gabor filtering
class FocusScoreFeature: public FeatureMethod
{
public:

    const constexpr static std::initializer_list<Nyxus::Feature2D> featureset = { Nyxus::Feature2D::FOCUS_SCORE,  Nyxus::Feature2D::LOCAL_FOCUS_SCORE};

    FocusScoreFeature();

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
    double focus_score_;
    double local_focus_score_;

    //=== Trivial ROIs ===

    double get_local_focus_score(const std::vector<PixIntens>& image, int n_image, int m_image, int ksize=1, int scale=2);

    static void laplacian(const std::vector<PixIntens>& image, std::vector<double>& out, int n_image, int m_image, int ksize=1);

    static double mean(const std::vector<double>& image);
    static double variance(const std::vector<double>& image);

};

