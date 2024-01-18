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
    static void gpu_process_all_rois (const std::vector<int>& ptrLabels, std::unordered_map <int, LR>& ptrLabelData);

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
    std::tuple<double, double, double, double, double, double, double> calcHuInvariants_imp (const pixcloud& cloud);
    std::tuple<double, double, double, double, double, double, double> calcHuInvariants_imp (const pixcloud& cloud, const intcloud& real_valued_intensities);
    void calcRawMoments (const pixcloud& cloud);
    void calcNormRawMoments (const pixcloud& cloud);
    void calcNormCentralMoments (const pixcloud& cloud);
    void calcWeightedRawMoments (const pixcloud& cloud);
    void calcWeightedRawMoments (const pixcloud& cloud, const intcloud& real_valued_intensities);
    void calcCentralMoments (const pixcloud& cloud);
    void calcWeightedCentralMoments (const pixcloud& cloud);
    void calcWeightedCentralMoments (const pixcloud& cloud, const intcloud& real_valued_intensities);
    void calcHuInvariants (const pixcloud& cloud);
    void calcWeightedHuInvariants (const pixcloud& cloud);
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
    double whm1 = 0, whm2 = 0, whm3 = 0, whm4 = 0, whm5 = 0, whm6 = 0, whm7 = 0;    // weighted Hu invariants

    const double weighting_epsilon = 0.001;
};

// References glocal objects 'Nyxus::ImageMatrixBuffer' and 'Nyxus::devImageMatrixBuffer'
bool ImageMomentsFeature_calculate (
    // output:
    double& m00, double& m01, double& m02, double& m03, double& m10, double& m11, double& m12, double& m13, double& m20, double& m21, double& m22, double& m23, double& m30,   // spatial moments
    double& cm00, double& cm01, double& cm02, double& cm03, double& cm10, double& cm11, double& cm12, double& cm13, double& cm20, double& cm21, double& cm22, double& cm23, double& cm30, double& cm31, double& cm32, double& cm33,  // central moments
    double& nu02, double& nu03, double& nu11, double& nu12, double& nu20, double& nu21, double& nu30,    // normalized central moments
    double& w00, double& w01, double& w02, double& w03, double& w10, double& w11, double& w12, double& w13, double& w20, double& w21, double& w22, double& w23, double& w30, double& w31, double& w32, double& w33,   // normalized spatial moments
    double& hm1, double& hm2, double& hm3, double& hm4, double& hm5, double& hm6, double& hm7,  // Hu moments
    double& wm00, double& wm01, double& wm02, double& wm03, double& wm10, double& wm11, double& wm12, double& wm20, double& wm21, double& wm30,   // weighted spatial moments
    double& wmu02, double& wmu03, double& wmu11, double& wmu12, double& wmu20, double& wmu21, double& wmu30,   // weighted central moments
    double& whm1, double& whm2, double& whm3, double& whm4, double& whm5, double& whm6, double& whm7,    // weighted Hum moments
    // input:
    size_t im_buffer_offset,
    size_t roi_idx,
    StatsInt aabb_min_x,
    StatsInt aabb_min_y,
    StatsInt width,
    StatsInt height);

bool allocate_2dmoments_buffers_on_gpu(size_t max_cloudsize);
bool free_2dmoments_buffers_on_gpu ();
bool send_roi_data_2_gpu(Pixel2* data, size_t n);
bool send_contour_data_2_gpu(Pixel2* data, size_t n);
bool free_roi_data_on_gpu();
namespace Nyxus
{
    extern size_t largest_roi_imatr_buf_len;    // set in phase 2

    // Objects implementing GPU-based calculation of geometric moments
    // -- device-side copy of a ROI cloud
    extern Pixel2* devRoiCloudBuffer;
    extern size_t roi_cloud_len;
    extern RealPixIntens* devRealintensBuffer;   // [roi_cloud_len]
    // -- device-side copy of ROI's contour data
    extern Pixel2* devContourCloudBuffer;
    extern size_t contour_cloud_len;
    // -- result of partial geometric moment expression (before sum-reduce)
    extern double* devPrereduce;     // reduction helper [roi_cloud_len]
    // -- reduce helpers
#if 0
    double* devBlockSubsums = nullptr;  // [whole chunks]
    double* hoBlockSubsums = nullptr;   // [whole chunks]
#endif
    extern double* d_out;    // 1 double
    extern void* d_temp_storage;   // allocated [] elements by cub::DeviceReduce::Sum()
    extern size_t temp_storage_bytes;

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
