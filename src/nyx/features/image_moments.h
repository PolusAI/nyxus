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

using pixcloud = std::vector <Pixel2>;          // cloud of pixels
using intcloud = std::vector <RealPixIntens>;   // cloud of pixel intensities
using pixcloud_NT = OutOfRamPixelCloud;

/// @brief Hu invariants, weighted Hu invariants, spatial , central, and normalized central moments.
class ImageMomentsFeature: public FeatureMethod
{
public:
    // Codes of features implemented by this class. Used in feature manager's mechanisms, 
    // in the feature group nickname expansion, and in the feature value output 
    const constexpr static std::initializer_list<Nyxus::Feature2D> featureset =
    {
        // Spatial (raw) moments
        Nyxus::Feature2D::SPAT_MOMENT_00,
        Nyxus::Feature2D::SPAT_MOMENT_01,
        Nyxus::Feature2D::SPAT_MOMENT_02,
        Nyxus::Feature2D::SPAT_MOMENT_03,
        Nyxus::Feature2D::SPAT_MOMENT_10,
        Nyxus::Feature2D::SPAT_MOMENT_11,
        Nyxus::Feature2D::SPAT_MOMENT_12,
        Nyxus::Feature2D::SPAT_MOMENT_13,
        Nyxus::Feature2D::SPAT_MOMENT_20,
        Nyxus::Feature2D::SPAT_MOMENT_21,
        Nyxus::Feature2D::SPAT_MOMENT_22,
        Nyxus::Feature2D::SPAT_MOMENT_23,
        Nyxus::Feature2D::SPAT_MOMENT_30,

        // Weighted spatial moments
        Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_00,
        Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_01,
        Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_02,
        Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_03,
        Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_10,
        Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_11,
        Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_12,
        Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_20,
        Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_21,
        Nyxus::Feature2D::WEIGHTED_SPAT_MOMENT_30,

        // Central moments
        Nyxus::Feature2D::CENTRAL_MOMENT_00,
        Nyxus::Feature2D::CENTRAL_MOMENT_01,
        Nyxus::Feature2D::CENTRAL_MOMENT_02,
        Nyxus::Feature2D::CENTRAL_MOMENT_03,
        Nyxus::Feature2D::CENTRAL_MOMENT_10,
        Nyxus::Feature2D::CENTRAL_MOMENT_11,
        Nyxus::Feature2D::CENTRAL_MOMENT_12,
        Nyxus::Feature2D::CENTRAL_MOMENT_13,
        Nyxus::Feature2D::CENTRAL_MOMENT_20,
        Nyxus::Feature2D::CENTRAL_MOMENT_21,
        Nyxus::Feature2D::CENTRAL_MOMENT_22,
        Nyxus::Feature2D::CENTRAL_MOMENT_23,
        Nyxus::Feature2D::CENTRAL_MOMENT_30,
        Nyxus::Feature2D::CENTRAL_MOMENT_31,
        Nyxus::Feature2D::CENTRAL_MOMENT_32,
        Nyxus::Feature2D::CENTRAL_MOMENT_33,

        // Weighted central moments
        Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_02,
        Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_03,
        Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_11,
        Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_12,
        Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_20,
        Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_21,
        Nyxus::Feature2D::WEIGHTED_CENTRAL_MOMENT_30,
        
        // weighted normalized central moments
        Nyxus::Feature2D::WT_NORM_CTR_MOM_02,
        Nyxus::Feature2D::WT_NORM_CTR_MOM_03,
        Nyxus::Feature2D::WT_NORM_CTR_MOM_11,
        Nyxus::Feature2D::WT_NORM_CTR_MOM_12,
        Nyxus::Feature2D::WT_NORM_CTR_MOM_20,
        Nyxus::Feature2D::WT_NORM_CTR_MOM_21,
        Nyxus::Feature2D::WT_NORM_CTR_MOM_30,

        // Normalized central moments
        Nyxus::Feature2D::NORM_CENTRAL_MOMENT_02,
        Nyxus::Feature2D::NORM_CENTRAL_MOMENT_03,
        Nyxus::Feature2D::NORM_CENTRAL_MOMENT_11,
        Nyxus::Feature2D::NORM_CENTRAL_MOMENT_12,
        Nyxus::Feature2D::NORM_CENTRAL_MOMENT_20,
        Nyxus::Feature2D::NORM_CENTRAL_MOMENT_21,
        Nyxus::Feature2D::NORM_CENTRAL_MOMENT_30,

        // Normalized (standardized) spatial moments
        Nyxus::Feature2D::NORM_SPAT_MOMENT_00,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_01,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_02,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_03,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_10,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_11,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_12,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_13,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_20,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_21,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_22,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_23,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_30,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_31,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_32,
        Nyxus::Feature2D::NORM_SPAT_MOMENT_33,

        // Hu's moments 1-7 
        Nyxus::Feature2D::HU_M1,
        Nyxus::Feature2D::HU_M2,
        Nyxus::Feature2D::HU_M3,
        Nyxus::Feature2D::HU_M4,
        Nyxus::Feature2D::HU_M5,
        Nyxus::Feature2D::HU_M6,
        Nyxus::Feature2D::HU_M7,

        // Weighted Hu's moments 1-7 
        Nyxus::Feature2D::WEIGHTED_HU_M1,
        Nyxus::Feature2D::WEIGHTED_HU_M2,
        Nyxus::Feature2D::WEIGHTED_HU_M3,
        Nyxus::Feature2D::WEIGHTED_HU_M4,
        Nyxus::Feature2D::WEIGHTED_HU_M5,
        Nyxus::Feature2D::WEIGHTED_HU_M6,
        Nyxus::Feature2D::WEIGHTED_HU_M7
    };

    ImageMomentsFeature();

    void calculate(LR& r);
    void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
    void osized_calculate(LR& r, ImageLoader& imloader);
    void save_value(std::vector<std::vector<double>>& feature_vals);
    static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
    static void gpu_process_all_rois (const std::vector<int>& ptrLabels, std::unordered_map <int, LR>& ptrLabelData, size_t batch_offset, size_t batch_len);

    // Compatibility with manual reduce
    static bool required(const FeatureSet& fs)
    {
        return fs.anyEnabled (ImageMomentsFeature::featureset);
    }

private:
    // Trivial ROI
    double moment (const pixcloud& cloud, int p, int q);
    double moment (const pixcloud& cloud, const intcloud& real_intens, int p, int q);
    void calcOrigins (const pixcloud& cloud);
    void calcOrigins (const pixcloud& cloud, const intcloud& real_valued_intensities);
    double centralMom (const pixcloud& c, int p, int q);
    double centralMom(const pixcloud& c, const intcloud& realintens, int p, int q);
    double normRawMom (const pixcloud& cloud, int p, int q);
    double normCentralMom (const pixcloud& c, int p, int q);
    double normCentralMom (const pixcloud& cloud, const intcloud& realintens, int p, int q);
    std::tuple<double, double, double, double, double, double, double> calcHu_imp (double _02, double _03, double _11, double _12, double _20, double _21, double _30);
    void calcRawMoments (const pixcloud& cloud);
    void calcNormRawMoments (const pixcloud& cloud);
    void calcNormCentralMoments (const pixcloud& cloud);
    void calcWeightedRawMoments (const pixcloud& cloud);
    void calcWeightedRawMoments (const pixcloud& cloud, const intcloud& real_valued_intensities);
    void calcCentralMoments (const pixcloud& cloud);
    void calcWeightedCentralMoments (const pixcloud& cloud, const intcloud& real_valued_intensities);
    void calcWeightedNormCentralMoms (const pixcloud& cloud, const intcloud& realintens);
    void calcHuInvariants (const pixcloud& cloud);
    void calcWeightedHuInvariants (const pixcloud& cloud, const intcloud& real_valued_intensities);

    // Non-trivial ROI
    double moment(const pixcloud_NT& cloud, int p, int q);
    void calcOrigins(const pixcloud_NT& cloud);
    double centralMom(const pixcloud_NT& c, int p, int q);
    double normRawMom(const pixcloud_NT& cloud, int p, int q);
    double normCentralMom(const pixcloud_NT& c, int p, int q);
    std::tuple<double, double, double, double, double, double, double> calcHuInvariants_imp(const pixcloud_NT& cloud);
    void calcRawMoments(const pixcloud_NT& cloud);
    void calcNormRawMoments(const pixcloud_NT& cloud);
    void calcNormCentralMoments(const pixcloud_NT& cloud);
    void calcWeightedRawMoments(const pixcloud_NT& cloud);
    void calcCentralMoments(const pixcloud_NT& cloud);
    void calcWeightedCentralMoments(const pixcloud_NT& cloud);
    void calcHuInvariants(const pixcloud_NT& cloud);
    void calcWeightedHuInvariants(const pixcloud_NT& cloud);

#ifdef USE_GPU
    void calculate_via_gpu(LR& r, size_t roi_index);
#endif

    StatsInt baseX = 0, baseY = 0; // cached min X and Y of the ROI. Reason - Pixel2's X and Y are absolute so we need to make them relative. Must be set in calculate() prior to calculating any 2D moment
    double originOfX = 0, originOfY = 0; // centroids
    double m00=0, m01=0, m02=0, m03=0, m10=0, m11=0, m12=0, m13=0, m20=0, m21=0, m22=0, m23=0, m30=0;    // spatial moments
    double wm00 = 0, wm01 = 0, wm02 = 0, wm03 = 0, wm10 = 0, wm11 = 0, wm12 = 0, wm20 = 0, wm21 = 0, wm30 = 0;    // weighted spatial moments
    double w00=0, w01=0, w02=0, w03=0, w10=0, w11=0, w12=0, w13=0, w20=0, w21=0, w22=0, w23=0, w30=0, w31=0, w32=0, w33=0;   // normalized spatial moments
    double nu02 = 0, nu03 = 0, nu11 = 0, nu12 = 0, nu20 = 0, nu21 = 0, nu30 = 0;    // normalized central moments
    double mu00=0, mu01=0, mu02=0, mu03=0, mu10=0, mu11=0, mu12=0, mu13=0, mu20=0, mu21=0, mu22=0, mu23=0, mu30=0, mu31=0, mu32=0, mu33=0;  // central moments
    double wmu02 = 0, wmu03 = 0, wmu11 = 0, wmu12 = 0, wmu20 = 0, wmu21 = 0, wmu30 = 0;    // weighted central moments
    double hm1 = 0, hm2 = 0, hm3 = 0, hm4 = 0, hm5 = 0, hm6 = 0, hm7 = 0;   // Hu invariants
    double wncm02 = 0, wncm03 = 0, wncm11 = 0, wncm12 = 0, wncm20 = 0, wncm21 = 0, wncm30 = 0;  // weighted normalized central moments
    double whm1 = 0, whm2 = 0, whm3 = 0, whm4 = 0, whm5 = 0, whm6 = 0, whm7 = 0;    // weighted Hu invariants

    const double weighting_epsilon = 0.001;
};

namespace NyxusGpu
{
    bool ImageMomentsFeature_calculate(size_t roi_idx);
}

namespace Nyxus
{
    extern size_t largest_roi_imatr_buf_len;    // set in phase 2

    /// @brief Copies integer pixel cloud intensities to real-valued vector
    void copy_pixcloud_intensities (intcloud & dst, const pixcloud & src);

    /// @brief Applies to distance-to-contour weighting to intensities of pixel cloud 
    void apply_dist2contour_weighting(
        // output
        intcloud & weighted_intensities,
        // input
        const pixcloud & cloud,
        const pixcloud & contour,
        const double epsilon);
}

