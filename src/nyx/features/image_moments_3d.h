#pragma once

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
class VolumeMomentsFeature : public FeatureMethod
{
public:
    VolumeMomentsFeature();

    void calculate(LR& r);
    void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity);
    void osized_calculate(LR& r, ImageLoader& imloader) {}
    void save_value(std::vector<std::vector<double>>& feature_vals);
    static void parallel_process_1_batch(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);
    static void gpu_process_all_rois(const std::vector<int>& ptrLabels, std::unordered_map <int, LR>& ptrLabelData);

    // Compatibility with manual reduce
    static bool required(const FeatureSet& fs)
    {
        return fs.anyEnabled ({
            // raw moments
            D3_RAW_MOMENT_000,

            D3_RAW_MOMENT_010,
            D3_RAW_MOMENT_011,
            D3_RAW_MOMENT_012,
            D3_RAW_MOMENT_013,

            D3_RAW_MOMENT_020,
            D3_RAW_MOMENT_021,
            D3_RAW_MOMENT_022,
            D3_RAW_MOMENT_023,

            D3_RAW_MOMENT_030,
            D3_RAW_MOMENT_031,
            D3_RAW_MOMENT_032,
            D3_RAW_MOMENT_033,

            D3_RAW_MOMENT_100,
            D3_RAW_MOMENT_101,
            D3_RAW_MOMENT_102,
            D3_RAW_MOMENT_103,

            D3_RAW_MOMENT_110,
            D3_RAW_MOMENT_111,
            D3_RAW_MOMENT_112,
            D3_RAW_MOMENT_113,

            D3_RAW_MOMENT_120,
            D3_RAW_MOMENT_121,
            D3_RAW_MOMENT_122,
            D3_RAW_MOMENT_123,

            D3_RAW_MOMENT_200,
            D3_RAW_MOMENT_201,
            D3_RAW_MOMENT_202,
            D3_RAW_MOMENT_203,

            D3_RAW_MOMENT_210,
            D3_RAW_MOMENT_211,
            D3_RAW_MOMENT_212,
            D3_RAW_MOMENT_213,

            D3_RAW_MOMENT_300,
            D3_RAW_MOMENT_301,
            D3_RAW_MOMENT_302,
            D3_RAW_MOMENT_303,

            // normalized raw moments
            D3_NORM_RAW_MOMENT_000,
            D3_NORM_RAW_MOMENT_010,
            D3_NORM_RAW_MOMENT_011,
            D3_NORM_RAW_MOMENT_012,
            D3_NORM_RAW_MOMENT_013,
            D3_NORM_RAW_MOMENT_020,
            D3_NORM_RAW_MOMENT_021,
            D3_NORM_RAW_MOMENT_022,
            D3_NORM_RAW_MOMENT_023,
            D3_NORM_RAW_MOMENT_030,
            D3_NORM_RAW_MOMENT_031,
            D3_NORM_RAW_MOMENT_032,
            D3_NORM_RAW_MOMENT_033,
            D3_NORM_RAW_MOMENT_100,
            D3_NORM_RAW_MOMENT_101,
            D3_NORM_RAW_MOMENT_102,
            D3_NORM_RAW_MOMENT_103,
            D3_NORM_RAW_MOMENT_200,
            D3_NORM_RAW_MOMENT_201,
            D3_NORM_RAW_MOMENT_202,
            D3_NORM_RAW_MOMENT_203,
            D3_NORM_RAW_MOMENT_300,
            D3_NORM_RAW_MOMENT_301,
            D3_NORM_RAW_MOMENT_302,
            D3_NORM_RAW_MOMENT_303,

            // central moments
            D3_CENTRAL_MOMENT_020,
            D3_CENTRAL_MOMENT_021,
            D3_CENTRAL_MOMENT_022,
            D3_CENTRAL_MOMENT_023,

            D3_CENTRAL_MOMENT_030,
            D3_CENTRAL_MOMENT_031,
            D3_CENTRAL_MOMENT_032,
            D3_CENTRAL_MOMENT_033,

            D3_CENTRAL_MOMENT_110,
            D3_CENTRAL_MOMENT_111,
            D3_CENTRAL_MOMENT_112,
            D3_CENTRAL_MOMENT_113,

            D3_CENTRAL_MOMENT_120,
            D3_CENTRAL_MOMENT_121,
            D3_CENTRAL_MOMENT_122,
            D3_CENTRAL_MOMENT_123,

            D3_CENTRAL_MOMENT_200,
            D3_CENTRAL_MOMENT_201,
            D3_CENTRAL_MOMENT_202,
            D3_CENTRAL_MOMENT_203,

            D3_CENTRAL_MOMENT_210,
            D3_CENTRAL_MOMENT_211,
            D3_CENTRAL_MOMENT_212,
            D3_CENTRAL_MOMENT_213,

            D3_CENTRAL_MOMENT_300,
            D3_CENTRAL_MOMENT_301,
            D3_CENTRAL_MOMENT_302,
            D3_CENTRAL_MOMENT_303,

            // normalized central moments
            D3_NORM_CENTRAL_MOMENT_020,
            D3_NORM_CENTRAL_MOMENT_021,
            D3_NORM_CENTRAL_MOMENT_022,
            D3_NORM_CENTRAL_MOMENT_023,

            D3_NORM_CENTRAL_MOMENT_030,
            D3_NORM_CENTRAL_MOMENT_031,
            D3_NORM_CENTRAL_MOMENT_032,
            D3_NORM_CENTRAL_MOMENT_033,

            D3_NORM_CENTRAL_MOMENT_110,
            D3_NORM_CENTRAL_MOMENT_111,
            D3_NORM_CENTRAL_MOMENT_112,
            D3_NORM_CENTRAL_MOMENT_113,

            D3_NORM_CENTRAL_MOMENT_120,
            D3_NORM_CENTRAL_MOMENT_121,
            D3_NORM_CENTRAL_MOMENT_122,
            D3_NORM_CENTRAL_MOMENT_123,

            D3_NORM_CENTRAL_MOMENT_200,
            D3_NORM_CENTRAL_MOMENT_201,
            D3_NORM_CENTRAL_MOMENT_202,
            D3_NORM_CENTRAL_MOMENT_203,

            D3_NORM_CENTRAL_MOMENT_210,
            D3_NORM_CENTRAL_MOMENT_211,
            D3_NORM_CENTRAL_MOMENT_212,
            D3_NORM_CENTRAL_MOMENT_213,

            D3_NORM_CENTRAL_MOMENT_300,
            D3_NORM_CENTRAL_MOMENT_301,
            D3_NORM_CENTRAL_MOMENT_302,
            D3_NORM_CENTRAL_MOMENT_303,
            });
    }

private:
    // Trivial ROI

    double moment(const std::vector<Pixel2> & pixelCloud, int p, int q, int z);
    void calcOrigins (const std::vector <Pixel2>& pixelCloud);
    double centralMom (const std::vector<Pixel2>& pixelCloud, int p, int q, int z);
    double normRawMom (const std::vector <Pixel2>& cloud, int p, int q, int z);
    double normCentralMom (const std::vector<Pixel2>& pixelCloud, int p, int q, int z);
    void calcRawMoments (const std::vector<Pixel2>& pixelCloud);
    void calcNormRawMoments(const std::vector<Pixel2>& pixelCloud);
    void calcCentralMoments (const std::vector<Pixel2>& pixelCloud);
    void calcNormCentralMoments (const std::vector<Pixel2>& pixelCloud);

#ifdef USE_GPU
    void calculate_via_gpu(LR& r, size_t roi_index);
    bool ImageMomentsFeature_calculate2(
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

    bool ImageMomentsFeature_calculate3(
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
        size_t im_buffer_offset,
        size_t roi_idx,
        StatsInt aabb_min_x,
        StatsInt aabb_min_y,
        StatsInt width,
        StatsInt height);

    bool send_contours_to_gpu(const std::vector<size_t>& hoIndices, const std::vector<StatsInt>& hoContourData);
    bool free_contour_data_on_gpu();
    bool send_imgmatrices_to_gpu(PixIntens* hoImageMatrixBuffer, size_t buf_len);
    bool free_imgmatrices_on_gpu();
#endif

    // Non-trivial (oversized) ROI
    /*
    double Moment_nontriv(ImageLoader& imlo, ReadImageMatrix_nontriv& I, int p, int q);
    double Moment_nontriv(WriteImageMatrix_nontriv& I, int p, int q);
    void calcOrigins_nontriv(ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    void calcOrigins_nontriv(WriteImageMatrix_nontriv& I);
    double CentralMom_nontriv(ImageLoader& imlo, ReadImageMatrix_nontriv& I, int p, int q);
    double CentralMom_nontriv(WriteImageMatrix_nontriv& W, int p, int q);
    double NormSpatMom_nontriv(ImageLoader& imlo, ReadImageMatrix_nontriv& I, int p, int q);
    double NormCentralMom_nontriv(ImageLoader& imlo, ReadImageMatrix_nontriv& I, int p, int q);
    double NormCentralMom_nontriv(WriteImageMatrix_nontriv& W, int p, int q);

    std::tuple<double, double, double, double, double, double, double> calcHuInvariants_imp_nontriv(ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    std::tuple<double, double, double, double, double, double, double> calcHuInvariants_imp_nontriv(WriteImageMatrix_nontriv& I);
    void calcHuInvariants_nontriv(ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    void calcWeightedHuInvariants_nontriv(WriteImageMatrix_nontriv& W);
    void calcNormCentralMoments_nontriv(ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    void calcNormSpatialMoments_nontriv(ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    void calcCentralMoments_nontriv(ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    void calcWeightedCentralMoments_nontriv(WriteImageMatrix_nontriv& W);
    void calcSpatialMoments_nontriv(ImageLoader& imlo, ReadImageMatrix_nontriv& I);
    void calcWeightedSpatialMoments_nontriv(WriteImageMatrix_nontriv& W);
    */

    double originOfX = 0, originOfY = 0, originOfZ = 0; 
    
    // raw moments
    double m_000, 
        m_010, m_011, m_012, m_013,
        m_020, m_021, m_022, m_023,
        m_030, m_031, m_032, m_033,
        m_100, m_101, m_102, m_103,
        m_110, m_111, m_112, m_113,
        m_120, m_121, m_122, m_123,
        m_200, m_201, m_202, m_203,
        m_210, m_211, m_212, m_213,
        m_300, m_301, m_302, m_303;

    // normalized raw moments
    double w_000,
        w_010, w_011, w_012, w_013,
        w_020, w_021, w_022, w_023,
        w_030, w_031, w_032, w_033,
        w_100, w_101, w_102, w_103,
        w_200, w_201, w_202, w_203,
        w_300, w_301, w_302, w_303;

    // central moments
    double mu_020, mu_021, mu_022, mu_023,
        mu_030, mu_031, mu_032, mu_033,
        mu_110, mu_111, mu_112, mu_113,
        mu_120, mu_121, mu_122, mu_123,
        mu_200, mu_201, mu_202, mu_203,
        mu_210, mu_211, mu_212, mu_213,
        mu_300, mu_301, mu_302, mu_303;

    // normalized central moments
    double nu_020, nu_021, nu_022, nu_023,
        nu_030, nu_031, nu_032, nu_033,
        nu_110, nu_111, nu_112, nu_113,
        nu_120, nu_121, nu_122, nu_123,
        nu_200, nu_201, nu_202, nu_203,
        nu_210, nu_211, nu_212, nu_213,
        nu_300, nu_301, nu_302, nu_303;

    /*
    double wm00 = 0, wm01 = 0, wm02 = 0, wm03 = 0, wm10 = 0, wm11 = 0, wm12 = 0, wm20 = 0, wm21 = 0, wm30 = 0;    // weighted spatial moments
    double w00 = 0, w01 = 0, w02 = 0, w03 = 0, w10 = 0, w20 = 0, w30 = 0;   // normalized spatial moments
    double nu02 = 0, nu03 = 0, nu11 = 0, nu12 = 0, nu20 = 0, nu21 = 0, nu30 = 0;    // normalized central moments
    double mu02 = 0, mu03 = 0, mu11 = 0, mu12 = 0, mu20 = 0, mu21 = 0, mu30 = 0;    // central moments
    double wmu02 = 0, wmu03 = 0, wmu11 = 0, wmu12 = 0, wmu20 = 0, wmu21 = 0, wmu30 = 0;    // weighted central moments
    double hm1 = 0, hm2 = 0, hm3 = 0, hm4 = 0, hm5 = 0, hm6 = 0, hm7 = 0;   // Hu invariants
    double whm1 = 0, whm2 = 0, whm3 = 0, whm4 = 0, whm5 = 0, whm6 = 0, whm7 = 0;    // weighted Hu invariants
    */
};



namespace Nyxus
{
    extern PixIntens* ImageMatrixBuffer;
    extern size_t imageMatrixBufferLen;
}
