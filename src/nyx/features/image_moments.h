#pragma once

#include <unordered_map>
#include "../roi_cache.h"
#include "contour.h"
#include "image_matrix.h"
#include "../feature_method.h"

// Inspired by Yavuz Unver
// 
// Hu Moments and Digit Recognition Algorithm:
// http://www.wseas.us/e-library/conferences/2013/CambridgeUK/AISE/AISE-15.pdf
//

/// @brief Hu invariants, weighted Hu invariants, spatial , central, and normalized central moments.
class ImageMomentsFeature: public FeatureMethod
{
public:
    ImageMomentsFeature(); 

    void calculate(LR& r);
    void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
    void osized_calculate(LR& r, ImageLoader& imloader);
    void save_value(std::vector<std::vector<double>>& feature_vals);
    static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

    // Compatibility with manual reduce
    static bool required(const FeatureSet& fs)
    {
        return fs.anyEnabled({
                SPAT_MOMENT_00,
                SPAT_MOMENT_01,
                SPAT_MOMENT_02,
                SPAT_MOMENT_03,
                SPAT_MOMENT_10,
                SPAT_MOMENT_11,
                SPAT_MOMENT_12,
                SPAT_MOMENT_20,
                SPAT_MOMENT_21,
                SPAT_MOMENT_30,

                CENTRAL_MOMENT_02,
                CENTRAL_MOMENT_03,
                CENTRAL_MOMENT_11,
                CENTRAL_MOMENT_12,
                CENTRAL_MOMENT_20,
                CENTRAL_MOMENT_21,
                CENTRAL_MOMENT_30,

                NORM_CENTRAL_MOMENT_02,
                NORM_CENTRAL_MOMENT_03,
                NORM_CENTRAL_MOMENT_11,
                NORM_CENTRAL_MOMENT_12,
                NORM_CENTRAL_MOMENT_20,
                NORM_CENTRAL_MOMENT_21,
                NORM_CENTRAL_MOMENT_30,

                HU_M1,
                HU_M2,
                HU_M3,
                HU_M4,
                HU_M5,
                HU_M6,
                HU_M7,

                WEIGHTED_HU_M1,
                WEIGHTED_HU_M2,
                WEIGHTED_HU_M3,
                WEIGHTED_HU_M4,
                WEIGHTED_HU_M5,
                WEIGHTED_HU_M6,
                WEIGHTED_HU_M7 });
    }

private:
    double Moment (const pixData& D, int p, int q);
    void calcOrigins (const pixData& D);
    double CentralMom (const pixData& D, int p, int q);
    double NormSpatMom (const pixData& D, int p, int q);
    double NormCentralMom (const pixData& D, int p, int q);

    std::tuple<double, double, double, double, double, double, double> calcHuInvariants_imp (const pixData& D);
    void calcHuInvariants (const pixData& D);
    void calcWeightedHuInvariants (const pixData& D);
    void calcNormCentralMoments(const pixData& D);
    void calcNormSpatialMoments(const pixData& D);
    void calcCentralMoments(const pixData& D);
    void calcWeightedCentralMoments(const pixData& D);
    void calcSpatialMoments(const pixData& D);
    void calcWeightedSpatialMoments(const pixData& D);

    double originOfX = 0, originOfY = 0;
    double m00 = 0, m01 = 0, m02 = 0, m03 = 0, m10 = 0, m11 = 0, m12 = 0, m20 = 0, m21 = 0, m30 = 0;    // spatial moments
    double wm00 = 0, wm01 = 0, wm02 = 0, wm03 = 0, wm10 = 0, wm11 = 0, wm12 = 0, wm20 = 0, wm21 = 0, wm30 = 0;    // weighted spatial moments
    double w00 = 0, w01 = 0, w02 = 0, w03 = 0, w10 = 0, w20 = 0, w30 = 0;   // normalized spatial moments
    double nu02 = 0, nu03 = 0, nu11 = 0, nu12 = 0, nu20 = 0, nu21 = 0, nu30 = 0;    // normalized central moments
    double mu02 = 0, mu03 = 0, mu11 = 0, mu12 = 0, mu20 = 0, mu21 = 0, mu30 = 0;    // central moments
    double wmu02 = 0, wmu03 = 0, wmu11 = 0, wmu12 = 0, wmu20 = 0, wmu21 = 0, wmu30 = 0;    // weighted central moments
    double hm1 = 0, hm2 = 0, hm3 = 0, hm4 = 0, hm5 = 0, hm6 = 0, hm7 = 0;   // Hu invariants
    double whm1 = 0, whm2 = 0, whm3 = 0, whm4 = 0, whm5 = 0, whm6 = 0, whm7 = 0;    // weighted Hu invariants
};

