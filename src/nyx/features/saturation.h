#pragma once

#include <vector>
#include "../roi_cache.h"
#include "image_matrix.h"
#include "../feature_method.h"
#include "../feature_settings.h"
#include "../environment.h"

class SaturationFeature : public FeatureMethod 
{

public:

    const constexpr static std::initializer_list<Nyxus::FeatureIMQ> featureset = { Nyxus::FeatureIMQ::MIN_SATURATION, Nyxus::FeatureIMQ::MAX_SATURATION};

    SaturationFeature();

    static bool required(const FeatureSet& fs);
   
    //=== Trivial ROIs ===
    void calculate (LR& r, const Fsettings& s);

    static void extract (LR& roi, const Fsettings& s);
    static void parallel_process_1_batch (size_t firstitem, size_t lastitem, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData, const Fsettings & s, const Dataset & ds);

    //=== Non-trivial ROIs ===
    //=== Non-trivial ROIs ===
    void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}
    void osized_calculate (LR& r, const Fsettings& s, ImageLoader& ldr);
    static std::tuple<double, double> get_percent_max_pixels_NT(WriteImageMatrix_nontriv& Im);

    // Result saver
    void save_value(std::vector<std::vector<double>>& feature_vals);

    // User interface
    static int ksize;

private:
 
    double max_saturation_, min_saturation_;

    static std::tuple<double, double> get_percent_max_pixels(const ImageMatrix& Im);

};