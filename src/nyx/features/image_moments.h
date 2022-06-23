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
    static void gpu_process_all_rois (const std::vector<int>& ptrLabels, std::unordered_map <int, LR>& ptrLabelData);

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
    // Trivial ROI

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

    #ifdef USE_GPU
        void calculate_via_gpu(LR& r, size_t roi_index);
    #endif

    // Non-trivial (oversized) ROI

    double Moment_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I, int p, int q);
    double Moment_nontriv (WriteImageMatrix_nontriv& I, int p, int q);
    void calcOrigins_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    void calcOrigins_nontriv (WriteImageMatrix_nontriv& I);
    double CentralMom_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I, int p, int q);
    double CentralMom_nontriv (WriteImageMatrix_nontriv& W, int p, int q);
    double NormSpatMom_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I, int p, int q);
    double NormCentralMom_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I, int p, int q);
    double NormCentralMom_nontriv (WriteImageMatrix_nontriv& W, int p, int q);

    std::tuple<double, double, double, double, double, double, double> calcHuInvariants_imp_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    std::tuple<double, double, double, double, double, double, double> calcHuInvariants_imp_nontriv (WriteImageMatrix_nontriv& I);
    void calcHuInvariants_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    void calcWeightedHuInvariants_nontriv (WriteImageMatrix_nontriv& W);
    void calcNormCentralMoments_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    void calcNormSpatialMoments_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    void calcCentralMoments_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    void calcWeightedCentralMoments_nontriv (WriteImageMatrix_nontriv& W);
    void calcSpatialMoments_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    void calcWeightedSpatialMoments_nontriv (WriteImageMatrix_nontriv& W);

    double originOfX = 0, originOfY = 0; // origins
    double m00 = 0, m01 = 0, m02 = 0, m03 = 0, m10 = 0, m11 = 0, m12 = 0, m20 = 0, m21 = 0, m30 = 0;    // spatial moments
    double wm00 = 0, wm01 = 0, wm02 = 0, wm03 = 0, wm10 = 0, wm11 = 0, wm12 = 0, wm20 = 0, wm21 = 0, wm30 = 0;    // weighted spatial moments
    double w00 = 0, w01 = 0, w02 = 0, w03 = 0, w10 = 0, w20 = 0, w30 = 0;   // normalized spatial moments
    double nu02 = 0, nu03 = 0, nu11 = 0, nu12 = 0, nu20 = 0, nu21 = 0, nu30 = 0;    // normalized central moments
    double mu02 = 0, mu03 = 0, mu11 = 0, mu12 = 0, mu20 = 0, mu21 = 0, mu30 = 0;    // central moments
    double wmu02 = 0, wmu03 = 0, wmu11 = 0, wmu12 = 0, wmu20 = 0, wmu21 = 0, wmu30 = 0;    // weighted central moments
    double hm1 = 0, hm2 = 0, hm3 = 0, hm4 = 0, hm5 = 0, hm6 = 0, hm7 = 0;   // Hu invariants
    double whm1 = 0, whm2 = 0, whm3 = 0, whm4 = 0, whm5 = 0, whm6 = 0, whm7 = 0;    // weighted Hu invariants
};

bool ImageMomentsFeature_calculate2 (
    // output:
    double& m00, double& m01, double& m02, double& m03, double& m10, double& m11, double& m12, double& m20, double& m21, double& m30,   // spatial moments
    double& cm02, double& cm03, double& cm11, double& cm12, double& cm20, double& cm21, double& cm30,   // central moments
    double& nu02, double& nu03, double& nu11, double& nu12, double& nu20, double& nu21, double& nu30,    // normalized central moments
    double& w00, double& w01, double& w02, double& w03, double& w10, double& w20, double& w30,   // normalized spatial moments
    double& hm1, double& hm2, double& hm3, double& hm4, double& hm5, double& hm6, double& hm7,  // Hu moments
    double& wm00, double& wm01, double& wm02, double& wm03, double& wm10, double& wm11, double& wm12, double& wm20, double& wm21, double& wm30,   // weighted spatial moments
    double& wmu02, double& wmu03, double& wmu11, double& wmu12, double& wmu20, double& wmu21, double& wmu30,   // weighted central moments
    double& whm1, double& whm2, double& whm3, double& whm4, double& whm5, double& whm6, double& whm7,    // weighted Hum moments
    // input:
    const ImageMatrix& Im,
    size_t roi_idx, 
    StatsInt aabb_min_x, 
    StatsInt aabb_min_y);

bool send_contours_to_gpu (const std::vector<size_t> & hoIndices, const std::vector< StatsInt> & hoContourData);
bool free_contour_data_on_gpu();

