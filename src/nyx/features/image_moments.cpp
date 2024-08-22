#include "../environment.h"
#ifdef USE_GPU
    #include "../gpucache.h"
    #include "../gpu/geomoments.cuh"
#endif
#include "image_moments.h"

using namespace Nyxus;

#define MOMENTS_OF_BINARY
#ifdef MOMENTS_OF_BINARY
    #define INTEN(x) 1.0
#else
    #define INTEN(x) x
#endif

ImageMomentsFeature::ImageMomentsFeature() : FeatureMethod("ImageMomentsFeature")
{
    provide_features (ImageMomentsFeature::featureset);
    add_dependencies ({Feature2D::PERIMETER});
}

void ImageMomentsFeature::calculate (LR& r)
{
    // intercept blank ROIs
    if (r.aux_max == r.aux_min)
    {
        // spatial moments
        m00 = m01 = m02 = m03 = m10 = m11 = m12 = m13 = m20 = m21 = m22 = m23 = m30 =   
        // weighted spatial moments
        wm00 = wm01 = wm02 = wm03 = wm10 = wm11 = wm12 = wm20 = wm21 = wm30 =           
        // normalized spatial moments
        w00 = w01 = w02 = w03 = w10 = w11 = w12 = w13 = w20 = w21 = w22 = w23 = w30 = w31 = w32 = w33 =
        // normalized central moments
        nu02 = nu03 = nu11 = nu12 = nu20 = nu21 = nu30 =
        // central moments
        mu00 = mu01 = mu02 = mu03 = mu10 = mu11 = mu12 = mu13 = mu20 = mu21 = mu22 = mu23 = mu30 = mu31 = mu32 = mu33 =
        // weighted central moments
        wmu02 = wmu03 = wmu11 = wmu12 = wmu20 = wmu21 = wmu30 =
        // Hu invariants
        hm1 = hm2 = hm3 = hm4 = hm5 = hm6 = hm7 =
        // weighted Hu invariants
        whm1 = whm2 = whm3 = whm4 = whm5 = whm6 = whm7 = theEnvironment.nan_substitute;

        return;
    }

    // Cache ROI frame of reference
    baseX = r.aabb.get_xmin();
    baseY = r.aabb.get_ymin();

    // Calculate non-weighted moments
    auto& c = r.raw_pixels;
    calcOrigins (c);
    calcRawMoments (c);
    calcCentralMoments (c);
    calcNormRawMoments (c);
    calcNormCentralMoments (c);
    calcHuInvariants (c);

    // Prepare weighted pixel cloud
    std::vector<RealPixIntens> w;
    Nyxus::copy_pixcloud_intensities (w, c);
    Nyxus::apply_dist2contour_weighting (w, c, r.contour, weighting_epsilon);

    // Calculate weighted moments
    calcOrigins (c, w);
    calcWeightedRawMoments (c, w);
    calcWeightedCentralMoments (c, w);
    calcWeightedNormCentralMoms (c, w);
    calcWeightedHuInvariants (c, w);
}

#ifdef USE_GPU
void ImageMomentsFeature::calculate_via_gpu (LR& r, size_t roi_idx)
{
    bool ok = NyxusGpu::ImageMomentsFeature_calculate (roi_idx);
    if (!ok)
        std::cerr << "Geometric moments: error calculating features on GPU\n";

}
#endif

void ImageMomentsFeature::osized_add_online_pixel (size_t x, size_t y, uint32_t intensity) {} // Not supporting online for image moments

void ImageMomentsFeature::save_value(std::vector<std::vector<double>>& fvals)
{
    fvals[(int)Feature2D::SPAT_MOMENT_00][0] = m00;
    fvals[(int)Feature2D::SPAT_MOMENT_01][0] = m01;
    fvals[(int)Feature2D::SPAT_MOMENT_02][0] = m02;
    fvals[(int)Feature2D::SPAT_MOMENT_03][0] = m03;
    fvals[(int)Feature2D::SPAT_MOMENT_10][0] = m10;
    fvals[(int)Feature2D::SPAT_MOMENT_11][0] = m11;
    fvals[(int)Feature2D::SPAT_MOMENT_12][0] = m12;
    fvals[(int)Feature2D::SPAT_MOMENT_13][0] = m13;
    fvals[(int)Feature2D::SPAT_MOMENT_20][0] = m20;
    fvals[(int)Feature2D::SPAT_MOMENT_21][0] = m21;
    fvals[(int)Feature2D::SPAT_MOMENT_22][0] = m22;
    fvals[(int)Feature2D::SPAT_MOMENT_23][0] = m23;
    fvals[(int)Feature2D::SPAT_MOMENT_30][0] = m30;

    fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_00][0] = wm00;
    fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_01][0] = wm01;
    fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_02][0] = wm02;
    fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_03][0] = wm03;
    fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_10][0] = wm10;
    fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_11][0] = wm11;
    fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_12][0] = wm12;
    fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_20][0] = wm20;
    fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_21][0] = wm21;
    fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_30][0] = wm30;

    fvals[(int)Feature2D::CENTRAL_MOMENT_00][0] = mu00;
    fvals[(int)Feature2D::CENTRAL_MOMENT_01][0] = mu01;
    fvals[(int)Feature2D::CENTRAL_MOMENT_02][0] = mu02;
    fvals[(int)Feature2D::CENTRAL_MOMENT_03][0] = mu03;
    fvals[(int)Feature2D::CENTRAL_MOMENT_10][0] = mu10;
    fvals[(int)Feature2D::CENTRAL_MOMENT_11][0] = mu11;
    fvals[(int)Feature2D::CENTRAL_MOMENT_12][0] = mu12;
    fvals[(int)Feature2D::CENTRAL_MOMENT_13][0] = mu13;
    fvals[(int)Feature2D::CENTRAL_MOMENT_20][0] = mu20;
    fvals[(int)Feature2D::CENTRAL_MOMENT_21][0] = mu21;
    fvals[(int)Feature2D::CENTRAL_MOMENT_22][0] = mu22;
    fvals[(int)Feature2D::CENTRAL_MOMENT_23][0] = mu23;
    fvals[(int)Feature2D::CENTRAL_MOMENT_30][0] = mu30;
    fvals[(int)Feature2D::CENTRAL_MOMENT_31][0] = mu31;
    fvals[(int)Feature2D::CENTRAL_MOMENT_32][0] = mu32;
    fvals[(int)Feature2D::CENTRAL_MOMENT_33][0] = mu33;

    fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_02][0] = wmu02;
    fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_03][0] = wmu03;
    fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_11][0] = wmu11;
    fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_12][0] = wmu12;
    fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_20][0] = wmu20;
    fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_21][0] = wmu21;
    fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_30][0] = wmu30;

    fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_02][0] = nu02;
    fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_03][0] = nu03;
    fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_11][0] = nu11;
    fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_12][0] = nu12;
    fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_20][0] = nu20;
    fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_21][0] = nu21;
    fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_30][0] = nu30;

    fvals[(int)Feature2D::NORM_SPAT_MOMENT_00][0] = w00;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_01][0] = w01;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_02][0] = w02;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_03][0] = w03;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_10][0] = w10;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_11][0] = w11;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_12][0] = w12;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_13][0] = w13;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_20][0] = w20;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_21][0] = w21;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_22][0] = w22;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_23][0] = w23;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_30][0] = w30;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_31][0] = w31;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_32][0] = w32;
    fvals[(int)Feature2D::NORM_SPAT_MOMENT_33][0] = w33;

    fvals[(int)Feature2D::HU_M1][0] = hm1;
    fvals[(int)Feature2D::HU_M2][0] = hm2;
    fvals[(int)Feature2D::HU_M3][0] = hm3;
    fvals[(int)Feature2D::HU_M4][0] = hm4;
    fvals[(int)Feature2D::HU_M5][0] = hm5;
    fvals[(int)Feature2D::HU_M6][0] = hm6;
    fvals[(int)Feature2D::HU_M7][0] = hm7;

    fvals[(int)Feature2D::WT_NORM_CTR_MOM_02][0] = wncm02;
    fvals[(int)Feature2D::WT_NORM_CTR_MOM_03][0] = wncm03;
    fvals[(int)Feature2D::WT_NORM_CTR_MOM_11][0] = wncm11;
    fvals[(int)Feature2D::WT_NORM_CTR_MOM_12][0] = wncm12;
    fvals[(int)Feature2D::WT_NORM_CTR_MOM_20][0] = wncm20;
    fvals[(int)Feature2D::WT_NORM_CTR_MOM_21][0] = wncm21;
    fvals[(int)Feature2D::WT_NORM_CTR_MOM_30][0] = wncm30;

    fvals[(int)Feature2D::WEIGHTED_HU_M1][0] = whm1;
    fvals[(int)Feature2D::WEIGHTED_HU_M2][0] = whm2;
    fvals[(int)Feature2D::WEIGHTED_HU_M3][0] = whm3;
    fvals[(int)Feature2D::WEIGHTED_HU_M4][0] = whm4;
    fvals[(int)Feature2D::WEIGHTED_HU_M5][0] = whm5;
    fvals[(int)Feature2D::WEIGHTED_HU_M6][0] = whm6;
    fvals[(int)Feature2D::WEIGHTED_HU_M7][0] = whm7;
}

/// @brief Calculates a spatial 2D moment of order q,p of ROI pixel cloud
double ImageMomentsFeature::moment (const pixcloud& cloud, int p, int q)
{
    double q_ = q, p_ = p, sum = 0;
    for (auto& pxl : cloud)
        sum += INTEN(double(pxl.inten)) * pow(double(pxl.x-baseX), p_) * pow(double(pxl.y-baseY), q_);
    return sum;
}

/// @brief Calculates a spatial 2D moment of order q,p of ROI pixel cloud 'c' using real-valued intensities 'real_intens'
double ImageMomentsFeature::moment (const pixcloud & c, const intcloud & real_intens, int p, int q)
{
    double q_ = q, p_ = p, sum = 0;
    size_t n = c.size();
    for (size_t i=0; i<n; i++)
    {
        const Pixel2& pxl = c[i];
        sum += real_intens[i] * pow(double(pxl.x - baseX), p_) * pow(double(pxl.y - baseY), q_);
    }
    return sum;
}

void ImageMomentsFeature::calcOrigins (const pixcloud & cloud)
{
    double m00 = moment (cloud, 0, 0), 
        m10 = moment (cloud, 1, 0),
        m01 = moment (cloud, 0, 1);
    originOfX =  m10 / m00;
    originOfY =  m01 / m00;
}

void ImageMomentsFeature::calcOrigins (const pixcloud & cloud, const intcloud & real_valued_intensities)
{
    double m00 = moment (cloud, real_valued_intensities, 0, 0);
    originOfX = moment (cloud, real_valued_intensities, 1, 0) / m00;
    originOfY = moment (cloud, real_valued_intensities, 0, 1) / m00;
}

/// @brief Calculates the central 2D moment of order q,p of ROI pixel cloud
double ImageMomentsFeature::centralMom (const pixcloud & cloud, int p, int q)
{
    double sum = 0;
    for (auto& pxl : cloud)
        sum += INTEN(double(pxl.inten)) * pow(double(pxl.x-baseX) - originOfX, p) * pow(double(pxl.y-baseY) - originOfY, q);
    return sum;
}

/// @brief Calculates the central 2D moment of order q,p of ROI pixel cloud using real-valued intensities
double ImageMomentsFeature::centralMom (const pixcloud & cloud, const intcloud & realintens, int p, int q)
{
    double sum = 0;
    size_t n = cloud.size();
    for (size_t i = 0; i < n; i++)
    {
        auto& pxl = cloud[i];
        sum += realintens[i] * pow(double(pxl.x - baseX) - originOfX, p) * pow(double(pxl.y - baseY) - originOfY, q);
    }
    return sum;
}

/// @brief Calculates the normalized spatial 2D moment of order q,p of ROI pixel cloud
double ImageMomentsFeature::normRawMom (const pixcloud & cloud, int p, int q)
{
    double k = ((double(p) + double(q)) / 2.0) + 1.0;
    double retval = moment(cloud, p, q) / pow(moment(cloud, 0, 0), k);
    return retval;
}

/// @brief Calculates the normalized central 2D moment of order q,p of ROI pixel cloud
double ImageMomentsFeature::normCentralMom (const pixcloud & cloud, int p, int q)
{
    double temp = ((double(p) + double(q)) / 2.0) + 1.0;
    double retval = centralMom(cloud, p, q) / pow(moment (cloud, 0, 0), temp);
    return retval;
}

/// @brief Calculates the normalized central 2D moment of order q,p of ROI pixel cloud using real-valued intensities 'realintens'
double ImageMomentsFeature::normCentralMom (const pixcloud & cloud, const intcloud & realintens, int p, int q)
{
    double temp = ((double(p) + double(q)) / 2.0) + 1.0;
    double retval = centralMom(cloud, realintens, p, q) / pow(moment(cloud, realintens, 0, 0), temp);
    return retval;
}

// Hu-1962 invariants
// _02, _03, _11, _12, _20, _21, _30 are normed central moments
std::tuple<double, double, double, double, double, double, double> ImageMomentsFeature::calcHu_imp (double _02, double _03, double _11, double _12, double _20, double _21, double _30)
{
    double h1 = _20 + _02;
    double h2 = pow((_20 - _02), 2) + 4 * (pow(_11, 2));
    double h3 = pow((_30 - 3 * _12), 2) +
        pow((3 * _21 - _03), 2);
    double h4 = pow((_30 + _12), 2) +
        pow((_21 + _03), 2);
    double h5 = (_30 - 3 * _12) *
        (_30 + _12) *
        (pow(_30 + _12, 2) - 3 * pow(_21 + _03, 2)) +
        (3 * _21 - _03) * (_21 + _03) *
        (pow(3 * (_30 + _12), 2) - pow(_21 + _03, 2));
    double h6 = (_20 - _02) * (pow(_30 + _12, 2) -
        pow(_21 + _03, 2)) + (4 * _11 * (_30 + _12) *
            _21 + _03);
    double h7 = (3 * _21 - _03) * (_30 + _12) * (pow(_30 + _12, 2) -
        3 * pow(_21 + _03, 2)) - (_30 - 3 * _12) * (_21 + _03) *
        (3 * pow(_30 + _12, 2) - pow(_21 + _03, 2));

    return { h1, h2, h3, h4, h5,h6, h7 };
}

// Prerequisite: precalculated normed central moments 'nu02 ... nu30'
void ImageMomentsFeature::calcHuInvariants (const pixcloud & cloud)
{
    std::tie(hm1, hm2, hm3, hm4, hm5, hm6, hm7) = calcHu_imp (nu02, nu03, nu11, nu12, nu20, nu21, nu30);
}

// Prerequisite: precalculated weighted normed central moments 'wncm02 ... wncm30'
void ImageMomentsFeature::calcWeightedHuInvariants (const pixcloud & cloud, const intcloud & realintens)
{
    std::tie(whm1, whm2, whm3, whm4, whm5, whm6, whm7) = calcHu_imp (wncm02, wncm03, wncm11, wncm12, wncm20, wncm21, wncm30);
}

void ImageMomentsFeature::calcRawMoments (const pixcloud & cloud)
{
    m00 = moment (cloud, 0, 0);
    m01 = moment (cloud, 0, 1);
    m02 = moment (cloud, 0, 2);
    m03 = moment (cloud, 0, 3);
    m10 = moment (cloud, 1, 0);
    m11 = moment (cloud, 1, 1);
    m12 = moment (cloud, 1, 2);
    m13 = moment (cloud, 1, 3);
    m20 = moment (cloud, 2, 0);
    m21 = moment (cloud, 2, 1);
    m22 = moment (cloud, 2, 2);
    m23 = moment (cloud, 2, 3);
    m30 = moment (cloud, 3, 0);
}

void ImageMomentsFeature::calcWeightedRawMoments (const pixcloud & cloud, const intcloud & real_intens)
{
    wm00 = moment (cloud, real_intens, 0, 0);
    wm01 = moment (cloud, real_intens, 0, 1);
    wm02 = moment (cloud, real_intens, 0, 2);
    wm03 = moment (cloud, real_intens, 0, 3);
    wm10 = moment (cloud, real_intens, 1, 0);
    wm11 = moment (cloud, real_intens, 1, 1);
    wm12 = moment (cloud, real_intens, 1, 2);
    wm20 = moment (cloud, real_intens, 2, 0);
    wm21 = moment (cloud, real_intens, 2, 1);
    wm30 = moment (cloud, real_intens, 3, 0);
}

void ImageMomentsFeature::calcCentralMoments (const pixcloud & cloud)
{
    mu00 = centralMom (cloud, 0, 0);
    mu01 = centralMom (cloud, 0, 1);
    mu02 = centralMom (cloud, 0, 2);
    mu03 = centralMom (cloud, 0, 3);

    mu10 = centralMom (cloud, 1, 0);
    mu11 = centralMom (cloud, 1, 1);
    mu12 = centralMom (cloud, 1, 2);
    mu13 = centralMom (cloud, 1, 3);

    mu20 = centralMom (cloud, 2, 0);
    mu21 = centralMom (cloud, 2, 1);
    mu22 = centralMom (cloud, 2, 2);
    mu23 = centralMom (cloud, 2, 3);

    mu30 = centralMom (cloud, 3, 0);
    mu31 = centralMom (cloud, 3, 1);
    mu32 = centralMom (cloud, 3, 2);
    mu33 = centralMom (cloud, 3, 3);
}

void ImageMomentsFeature::calcWeightedCentralMoments (const pixcloud& cloud, const intcloud& realintens)
{
    wmu02 = centralMom (cloud, realintens, 0, 2);
    wmu03 = centralMom (cloud, realintens, 0, 3);
    wmu11 = centralMom (cloud, realintens, 1, 1);
    wmu12 = centralMom (cloud, realintens, 1, 2);
    wmu20 = centralMom (cloud, realintens, 2, 0);
    wmu21 = centralMom (cloud, realintens, 2, 1);
    wmu30 = centralMom (cloud, realintens, 3, 0);
}

void ImageMomentsFeature::calcNormCentralMoments (const pixcloud & cloud)
{
    nu02 = normCentralMom (cloud, 0, 2);
    nu03 = normCentralMom (cloud, 0, 3);
    nu11 = normCentralMom (cloud, 1, 1);
    nu12 = normCentralMom (cloud, 1, 2);
    nu20 = normCentralMom (cloud, 2, 0);
    nu21 = normCentralMom (cloud, 2, 1);
    nu30 = normCentralMom (cloud, 3, 0);
}

void ImageMomentsFeature::calcWeightedNormCentralMoms (const pixcloud & cloud, const intcloud& realintens)
{
    wncm20 = normCentralMom (cloud, realintens, 2, 0);
    wncm02 = normCentralMom (cloud, realintens, 0, 2);
    wncm11 = normCentralMom (cloud, realintens, 1, 1);
    wncm30 = normCentralMom (cloud, realintens, 3, 0);
    wncm12 = normCentralMom (cloud, realintens, 1, 2);
    wncm21 = normCentralMom (cloud, realintens, 2, 1);
    wncm03 = normCentralMom (cloud, realintens, 0, 3);
}

void ImageMomentsFeature::calcNormRawMoments (const pixcloud & cloud)
{
    w00 = normRawMom (cloud, 0, 0);
    w01 = normRawMom (cloud, 0, 1);
    w02 = normRawMom (cloud, 0, 2);
    w03 = normRawMom (cloud, 0, 3);

    w10 = normRawMom (cloud, 1, 0);
    w11 = normRawMom (cloud, 1, 1);
    w12 = normRawMom (cloud, 1, 2);
    w13 = normRawMom (cloud, 1, 3);

    w20 = normRawMom (cloud, 2, 0);
    w21 = normRawMom (cloud, 2, 1);
    w22 = normRawMom (cloud, 2, 2);
    w23 = normRawMom (cloud, 2, 3);

    w30 = normRawMom (cloud, 3, 0);
    w31 = normRawMom (cloud, 3, 1);
    w32 = normRawMom (cloud, 3, 2);
    w33 = normRawMom (cloud, 3, 3);
}

/// @brief Calculates the features for a subset of ROIs in a thread-safe way with other ROI subsets
/// @param start Start index of the ROI label vector
/// @param end End index of the ROI label vector
/// @param ptrLabels ROI label vector
/// @param ptrLabelData ROI data
void ImageMomentsFeature::parallel_process_1_batch (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        if (r.has_bad_data())
            continue;

        ImageMomentsFeature imf;
        imf.calculate(r);
        imf.save_value(r.fvals);
    }
}

#ifdef USE_GPU

void save_values_from_gpu_buffer(
    std::unordered_map <int, LR>& roidata,
    const std::vector<int>& roilabels,
    const GpuCache<gpureal>& intermediate_already_hostside,
    size_t batch_offset,
    size_t batch_len)
{
    for (size_t i = 0; i < batch_len; i++)
    {
        size_t roiidx = batch_offset + i;
        auto lbl = roilabels[roiidx];
        LR& roi = roidata[lbl];
        auto& fvals = roi.fvals;

        size_t offs = i * GpusideState::__COUNT__;
        const gpureal* ptrBuf = &intermediate_already_hostside.hobuffer[offs];

        fvals[(int)Feature2D::SPAT_MOMENT_00][0] = ptrBuf[GpusideState::RM00];
        fvals[(int)Feature2D::SPAT_MOMENT_01][0] = ptrBuf[GpusideState::RM01];
        fvals[(int)Feature2D::SPAT_MOMENT_02][0] = ptrBuf[GpusideState::RM02];
        fvals[(int)Feature2D::SPAT_MOMENT_03][0] = ptrBuf[GpusideState::RM03];
        fvals[(int)Feature2D::SPAT_MOMENT_10][0] = ptrBuf[GpusideState::RM10];
        fvals[(int)Feature2D::SPAT_MOMENT_11][0] = ptrBuf[GpusideState::RM11];
        fvals[(int)Feature2D::SPAT_MOMENT_12][0] = ptrBuf[GpusideState::RM12];
        fvals[(int)Feature2D::SPAT_MOMENT_13][0] = ptrBuf[GpusideState::RM13];
        fvals[(int)Feature2D::SPAT_MOMENT_20][0] = ptrBuf[GpusideState::RM20];
        fvals[(int)Feature2D::SPAT_MOMENT_21][0] = ptrBuf[GpusideState::RM21];
        fvals[(int)Feature2D::SPAT_MOMENT_22][0] = ptrBuf[GpusideState::RM22];
        fvals[(int)Feature2D::SPAT_MOMENT_23][0] = ptrBuf[GpusideState::RM23];
        fvals[(int)Feature2D::SPAT_MOMENT_30][0] = ptrBuf[GpusideState::RM30];

        fvals[(int)Feature2D::CENTRAL_MOMENT_00][0] = ptrBuf[GpusideState::CM00];
        fvals[(int)Feature2D::CENTRAL_MOMENT_01][0] = ptrBuf[GpusideState::CM01];
        fvals[(int)Feature2D::CENTRAL_MOMENT_02][0] = ptrBuf[GpusideState::CM02];
        fvals[(int)Feature2D::CENTRAL_MOMENT_03][0] = ptrBuf[GpusideState::CM03];
        fvals[(int)Feature2D::CENTRAL_MOMENT_10][0] = ptrBuf[GpusideState::CM10];
        fvals[(int)Feature2D::CENTRAL_MOMENT_11][0] = ptrBuf[GpusideState::CM11];
        fvals[(int)Feature2D::CENTRAL_MOMENT_12][0] = ptrBuf[GpusideState::CM12];
        fvals[(int)Feature2D::CENTRAL_MOMENT_13][0] = ptrBuf[GpusideState::CM13];
        fvals[(int)Feature2D::CENTRAL_MOMENT_20][0] = ptrBuf[GpusideState::CM20];
        fvals[(int)Feature2D::CENTRAL_MOMENT_21][0] = ptrBuf[GpusideState::CM21];
        fvals[(int)Feature2D::CENTRAL_MOMENT_22][0] = ptrBuf[GpusideState::CM22];
        fvals[(int)Feature2D::CENTRAL_MOMENT_23][0] = ptrBuf[GpusideState::CM23];
        fvals[(int)Feature2D::CENTRAL_MOMENT_30][0] = ptrBuf[GpusideState::CM30];
        fvals[(int)Feature2D::CENTRAL_MOMENT_31][0] = ptrBuf[GpusideState::CM31];
        fvals[(int)Feature2D::CENTRAL_MOMENT_32][0] = ptrBuf[GpusideState::CM32];
        fvals[(int)Feature2D::CENTRAL_MOMENT_33][0] = ptrBuf[GpusideState::CM33];

        fvals[(int)Feature2D::NORM_SPAT_MOMENT_00][0] = ptrBuf[GpusideState::W00];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_01][0] = ptrBuf[GpusideState::W01];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_02][0] = ptrBuf[GpusideState::W02];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_03][0] = ptrBuf[GpusideState::W03];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_10][0] = ptrBuf[GpusideState::W10];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_11][0] = ptrBuf[GpusideState::W11];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_12][0] = ptrBuf[GpusideState::W12];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_13][0] = ptrBuf[GpusideState::W13];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_20][0] = ptrBuf[GpusideState::W20];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_21][0] = ptrBuf[GpusideState::W21];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_22][0] = ptrBuf[GpusideState::W22];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_23][0] = ptrBuf[GpusideState::W23];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_30][0] = ptrBuf[GpusideState::W30];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_31][0] = ptrBuf[GpusideState::W31];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_32][0] = ptrBuf[GpusideState::W32];
        fvals[(int)Feature2D::NORM_SPAT_MOMENT_33][0] = ptrBuf[GpusideState::W33];

        fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_02][0] = ptrBuf[GpusideState::NU02];
        fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_03][0] = ptrBuf[GpusideState::NU03];
        fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_11][0] = ptrBuf[GpusideState::NU11];
        fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_12][0] = ptrBuf[GpusideState::NU12];
        fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_20][0] = ptrBuf[GpusideState::NU20];
        fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_21][0] = ptrBuf[GpusideState::NU21];
        fvals[(int)Feature2D::NORM_CENTRAL_MOMENT_30][0] = ptrBuf[GpusideState::NU30];

        fvals[(int)Feature2D::HU_M1][0] = ptrBuf[GpusideState::H1];
        fvals[(int)Feature2D::HU_M2][0] = ptrBuf[GpusideState::H2];
        fvals[(int)Feature2D::HU_M3][0] = ptrBuf[GpusideState::H3];
        fvals[(int)Feature2D::HU_M4][0] = ptrBuf[GpusideState::H4];
        fvals[(int)Feature2D::HU_M5][0] = ptrBuf[GpusideState::H5];
        fvals[(int)Feature2D::HU_M6][0] = ptrBuf[GpusideState::H6];
        fvals[(int)Feature2D::HU_M7][0] = ptrBuf[GpusideState::H7];

        fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_00][0] = ptrBuf[GpusideState::WRM00];
        fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_01][0] = ptrBuf[GpusideState::WRM01];
        fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_02][0] = ptrBuf[GpusideState::WRM02];
        fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_03][0] = ptrBuf[GpusideState::WRM03];
        fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_10][0] = ptrBuf[GpusideState::WRM10];
        fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_11][0] = ptrBuf[GpusideState::WRM11];
        fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_12][0] = ptrBuf[GpusideState::WRM12];
        fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_20][0] = ptrBuf[GpusideState::WRM20];
        fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_21][0] = ptrBuf[GpusideState::WRM21];
        fvals[(int)Feature2D::WEIGHTED_SPAT_MOMENT_30][0] = ptrBuf[GpusideState::WRM30];

        fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_02][0] = ptrBuf[GpusideState::WCM02];
        fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_03][0] = ptrBuf[GpusideState::WCM03];
        fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_11][0] = ptrBuf[GpusideState::WCM11];
        fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_12][0] = ptrBuf[GpusideState::WCM12];
        fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_20][0] = ptrBuf[GpusideState::WCM20];
        fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_21][0] = ptrBuf[GpusideState::WCM21];
        fvals[(int)Feature2D::WEIGHTED_CENTRAL_MOMENT_30][0] = ptrBuf[GpusideState::WCM30];

        fvals[(int)Feature2D::WT_NORM_CTR_MOM_02][0] = ptrBuf[GpusideState::WNU02];
        fvals[(int)Feature2D::WT_NORM_CTR_MOM_03][0] = ptrBuf[GpusideState::WNU03];
        fvals[(int)Feature2D::WT_NORM_CTR_MOM_11][0] = ptrBuf[GpusideState::WNU11];
        fvals[(int)Feature2D::WT_NORM_CTR_MOM_12][0] = ptrBuf[GpusideState::WNU12];
        fvals[(int)Feature2D::WT_NORM_CTR_MOM_20][0] = ptrBuf[GpusideState::WNU20];
        fvals[(int)Feature2D::WT_NORM_CTR_MOM_21][0] = ptrBuf[GpusideState::WNU21];
        fvals[(int)Feature2D::WT_NORM_CTR_MOM_30][0] = ptrBuf[GpusideState::WNU30];

        fvals[(int)Feature2D::WEIGHTED_HU_M1][0] = ptrBuf[GpusideState::WH1];
        fvals[(int)Feature2D::WEIGHTED_HU_M2][0] = ptrBuf[GpusideState::WH2];
        fvals[(int)Feature2D::WEIGHTED_HU_M3][0] = ptrBuf[GpusideState::WH3];
        fvals[(int)Feature2D::WEIGHTED_HU_M4][0] = ptrBuf[GpusideState::WH4];
        fvals[(int)Feature2D::WEIGHTED_HU_M5][0] = ptrBuf[GpusideState::WH5];
        fvals[(int)Feature2D::WEIGHTED_HU_M6][0] = ptrBuf[GpusideState::WH6];
        fvals[(int)Feature2D::WEIGHTED_HU_M7][0] = ptrBuf[GpusideState::WH7];
    }
}

void ImageMomentsFeature::gpu_process_all_rois (
    const std::vector<int> & Labels, 
    std::unordered_map <int, LR>& RoiData,
    size_t batch_offset, 
    size_t batch_len)
{
    for (auto i=0; i < batch_len; i++)
    {
        size_t far_i = i + batch_offset;
        auto lab = Labels[far_i];
        LR& r = RoiData[lab];
        
        // Calculate features        
        ImageMomentsFeature imf;
        imf.calculate_via_gpu (r, i);
        //---delayed until we process all the ROIs on GPU-side--->  imf.save_value (r.fvals);

        // Pull the result from GPU cache and save it
        if (!NyxusGpu::gpu_featurestatebuf.download())
        { 
            std::cerr << "error in " << __FILE__ << ":" << __LINE__ << "\n";
            return;
        }
        
        save_values_from_gpu_buffer (RoiData, Labels, NyxusGpu::gpu_featurestatebuf, batch_offset, batch_len);
    }
}
#endif // USE_GPU

namespace Nyxus
{
    void copy_pixcloud_intensities (intcloud & dst, const pixcloud & src)
    {
        dst.reserve (src.size());
        for (auto pxl : src)
            dst.push_back (RealPixIntens(pxl.inten));
    }

    /// @brief Applies to distance-to-contour weighting to intensities of pixel cloud. Saves the result in 'realintens' 
    void apply_dist2contour_weighting(
        // input & output
        intcloud & realintens,
        // input
        const pixcloud & cloud,
        const pixcloud & contour,
        const double epsilon)
    {
        size_t n = cloud.size();
        for (size_t i = 0; i < n; i++)
        {
            auto& p = cloud[i];

            // pixel distance
            double mind2 = p.min_sqdist (contour);
            double dist = std::sqrt(mind2);

            // weighted intensity
            double I = 1.0; // shape moments => constant intensity within the ROI
            realintens[i] = I / (dist + epsilon);
        }
    }
}
