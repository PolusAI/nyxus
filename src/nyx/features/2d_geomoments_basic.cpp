#include "../environment.h"
#ifdef USE_GPU
    #include "../cache.h"
    #include "../gpu/geomoments.cuh"
#endif
#include "2d_geomoments.h"

using namespace Nyxus;

#ifdef USE_GPU

    void save_values_from_gpu_buffer(
        std::unordered_map <int, LR>& roidata,
        const std::vector<int>& roilabels,
        const GpuCache<gpureal>& intermediate_already_hostside,
        size_t batch_offset,
        size_t batch_len);

#endif

namespace Nyxus
{
    void copy_pixcloud_intensities(reintenvec& dst, const pixcloud& src)
    {
        dst.reserve(src.size());
        for (auto pxl : src)
            dst.push_back(RealPixIntens(pxl.inten));
    }

}

/// @brief Applies to distance-to-contour weighting to intensities of pixel cloud. Saves the result in 'realintens' 
void BasicGeomoms2D::apply_dist2contour_weighting(
    // input & output
    reintenvec& realintens,
    // input
    const pixcloud& cloud,
    const pixcloud& contour,
    const double epsilon)
{
    size_t n = cloud.size();
    for (size_t i = 0; i < n; i++)
    {
        auto& p = cloud[i];

        // pixel distance
        double mind2 = p.min_sqdist(contour);
        double dist = std::sqrt(mind2);

        // weighted intensity
        double I = INTEN(double(p.inten));
        realintens[i] = I * std::log(dist + epsilon);
    }
}

/// @brief Applies to wholeslide distance-to-contour weighting to intensities of pixel cloud. Saves the result in 'realintens' 
/// Elements of 'contour' are expected to be four connected vertices of a whole slide
/// 
void BasicGeomoms2D::apply_dist2contour_weighting_wholeslide(
    // input & output
    reintenvec& realintens,
    // input
    const pixcloud& cloud,
    const pixcloud& contour,
    const double epsilon)
{
    size_t n = cloud.size();
    for (size_t i = 0; i < n; i++)
    {
        auto& p = cloud[i];

        auto& c0 = contour[0],
            & c1 = contour[1],
            & c2 = contour[2],
            & c3 = contour[3];

        // skip contour pixels
        if (p.aligned(c0) || p.aligned(c1) || p.aligned(c2) || p.aligned(c3))
        {
            realintens[i] = epsilon;
            continue;
        }

        // min distance
        double d1 = p.dist_to_segment(c0, c1),
            d2 = p.dist_to_segment(c1, c2),
            d3 = p.dist_to_segment(c2, c3),
            d4 = p.dist_to_segment(c3, c0);

        double dist = std::min(std::min(d1, d2), std::min(d3, d4));

        // weighted intensity
        double I = INTEN(double(p.inten));
        realintens[i] = I * std::log(dist + epsilon); 
    }
}

void BasicGeomoms2D::calculate(LR& r, const Fsettings& s, intenfunction ifun)
{
    INTEN = ifun;
  
    // Cache ROI frame of reference
    baseX = r.aabb.get_xmin();
    baseY = r.aabb.get_ymin();

    // -- non-weighted moments
    auto& c = r.raw_pixels;
    calcOrigins(c);
    calcRawMoments(c);
    calcCentralMoments(c);
    calcNormRawMoments(c);
    calcNormCentralMoments(c);
    calcHuInvariants(c);

    // -- prepare weighted pixel cloud
    std::vector<RealPixIntens> w;
    Nyxus::copy_pixcloud_intensities(w, c);
    if (STNGS_SINGLEROI(s))   // former theEnvironment.singleROI
        apply_dist2contour_weighting_wholeslide (w, c, r.contour, weighting_epsilon);
    else
        apply_dist2contour_weighting (w, c, r.contour, weighting_epsilon);

    // -- weighted moments
    calcOrigins(c, w);
    calcWeightedRawMoments(c, w);
    calcWeightedCentralMoments(c, w);
    calcWeightedNormCentralMoms(c, w);
    calcWeightedHuInvariants(c, w);

}

double BasicGeomoms2D::moment(const pixcloud& cloud, int p, int q)
{
    double q_ = q, p_ = p, sum = 0;
    for (auto& pxl : cloud)
        sum += INTEN(double(pxl.inten)) * pow(double(pxl.x - baseX), p_) * pow(double(pxl.y - baseY), q_);
    return sum;
}

double BasicGeomoms2D::moment(const pixcloud& c, const reintenvec& real_intens, int p, int q)
{
    double q_ = q, p_ = p, sum = 0.0;
    size_t n = c.size();
    for (size_t i = 0; i < n; i++)
    {
        const Pixel2& pxl = c[i];
        double I = real_intens[i];
        sum += I * pow(double(pxl.x - baseX), p_) * pow(double(pxl.y - baseY), q_);
    }
    return sum;
}

void BasicGeomoms2D::calcOrigins(const pixcloud& cloud)
{
    double m00 = moment(cloud, 0, 0),
        m10 = moment(cloud, 1, 0),
        m01 = moment(cloud, 0, 1);
    originOfX = m10 / m00;
    originOfY = m01 / m00;
}

void BasicGeomoms2D::calcOrigins(const pixcloud& cloud, const reintenvec& real_valued_intensities)
{
    double m00 = moment(cloud, real_valued_intensities, 0, 0);
    originOfX = moment(cloud, real_valued_intensities, 1, 0) / m00;
    originOfY = moment(cloud, real_valued_intensities, 0, 1) / m00;
}

inline double int_pow (double a, int b)
{
    double retval = 1.0;
    for (int i = 0; i < b; i++)
        retval *= a;
    return retval;
}

/// @brief Calculates the central 2D moment of order q,p of ROI pixel cloud
double BasicGeomoms2D::centralMom(const pixcloud& cloud, int p, int q)
{
    double sum = 0;
    for (auto& pxl : cloud)
    {
        double I = INTEN(double(pxl.inten));
        sum += I * int_pow(double(pxl.x - baseX) - (int)originOfX, p) * int_pow(double(pxl.y - baseY) - (int)originOfY, q);
    }
    return sum;
}

/// @brief Calculates the central 2D moment of order q,p of ROI pixel cloud using real-valued intensities
double BasicGeomoms2D::centralMom(const pixcloud& cloud, const reintenvec& realintens, int p, int q)
{
    double sum = 0;
    size_t n = cloud.size();
    for (size_t i = 0; i < n; i++)
    {
        auto& pxl = cloud[i];
        sum += realintens[i] * int_pow(double(pxl.x - baseX) - (int)originOfX, p) * int_pow(double(pxl.y - baseY) - (int)originOfY, q);
    }
    return sum;
}

/// @brief Calculates the normalized spatial 2D moment of order q,p of ROI pixel cloud
double BasicGeomoms2D::normRawMom(const pixcloud& cloud, int p, int q)
{
    double k = ((double(p) + double(q)) / 2.0) + 1.0;
    double retval = moment(cloud, p, q) / pow(moment(cloud, 0, 0), k);
    return retval;
}

/// @brief Calculates the normalized central 2D moment of order q,p of ROI pixel cloud
double BasicGeomoms2D::normCentralMom(const pixcloud& cloud, int p, int q)
{
    double temp = ((double(p) + double(q)) / 2.0) + 1.0;
    double retval = centralMom(cloud, p, q) / pow(moment(cloud, 0, 0), temp);
    return retval;
}

/// @brief Calculates the normalized central 2D moment of order q,p of ROI pixel cloud using real-valued intensities 'realintens'
double BasicGeomoms2D::normCentralMom(const pixcloud& cloud, const reintenvec& realintens, int p, int q)
{
    double temp = ((double(p) + double(q)) / 2.0) + 1.0;
    double retval = centralMom(cloud, realintens, p, q) / pow(moment(cloud, realintens, 0, 0), temp);
    return retval;
}

// Hu-1962 invariants
// _02, _03, _11, _12, _20, _21, _30 are normed central moments
std::tuple<double, double, double, double, double, double, double> BasicGeomoms2D::calcHu_imp(double _02, double _03, double _11, double _12, double _20, double _21, double _30)
{
    double h1 = _20 + _02;
    double h2 = int_pow((_20 - _02), 2) + 4 * (int_pow(_11, 2));
    double h3 = int_pow((_30 - 3 * _12), 2) +
        int_pow((3 * _21 - _03), 2);
    double h4 = int_pow((_30 + _12), 2) +
        int_pow((_21 + _03), 2);
    double h5 = (_30 - 3 * _12) *
        (_30 + _12) *
        (int_pow(_30 + _12, 2) - 3 * int_pow(_21 + _03, 2)) +
        (3 * _21 - _03) * (_21 + _03) *
        (int_pow(3 * (_30 + _12), 2) - int_pow(_21 + _03, 2));
    double h6 = (_20 - _02) * (int_pow(_30 + _12, 2) -
        int_pow(_21 + _03, 2)) + (4 * _11 * (_30 + _12) *
            _21 + _03);
    double h7 = (3 * _21 - _03) * (_30 + _12) * (int_pow(_30 + _12, 2) -
        3 * int_pow(_21 + _03, 2)) - (_30 - 3 * _12) * (_21 + _03) *
        (3 * int_pow(_30 + _12, 2) - int_pow(_21 + _03, 2));

    return { h1, h2, h3, h4, h5,h6, h7 };
}

// Prerequisite: precalculated normed central moments 'nu02 ... nu30'
void BasicGeomoms2D::calcHuInvariants(const pixcloud& cloud)
{
    std::tie(hm1, hm2, hm3, hm4, hm5, hm6, hm7) = calcHu_imp(nu02, nu03, nu11, nu12, nu20, nu21, nu30);
}

// Prerequisite: precalculated weighted normed central moments 'wncm02 ... wncm30'
void BasicGeomoms2D::calcWeightedHuInvariants(const pixcloud& cloud, const reintenvec& realintens)
{
    std::tie(whm1, whm2, whm3, whm4, whm5, whm6, whm7) = calcHu_imp(wncm02, wncm03, wncm11, wncm12, wncm20, wncm21, wncm30);
}

void BasicGeomoms2D::calcRawMoments(const pixcloud& cloud)
{
    m00 = moment(cloud, 0, 0);
    m01 = moment(cloud, 0, 1);
    m02 = moment(cloud, 0, 2);
    m03 = moment(cloud, 0, 3);
    m10 = moment(cloud, 1, 0);
    m11 = moment(cloud, 1, 1);
    m12 = moment(cloud, 1, 2);
    m13 = moment(cloud, 1, 3);
    m20 = moment(cloud, 2, 0);
    m21 = moment(cloud, 2, 1);
    m22 = moment(cloud, 2, 2);
    m23 = moment(cloud, 2, 3);
    m30 = moment(cloud, 3, 0);
}

void BasicGeomoms2D::calcWeightedRawMoments(const pixcloud& cloud, const reintenvec& real_intens)
{
    wm00 = moment(cloud, real_intens, 0, 0);
    wm01 = moment(cloud, real_intens, 0, 1);
    wm02 = moment(cloud, real_intens, 0, 2);
    wm03 = moment(cloud, real_intens, 0, 3);
    wm10 = moment(cloud, real_intens, 1, 0);
    wm11 = moment(cloud, real_intens, 1, 1);
    wm12 = moment(cloud, real_intens, 1, 2);
    wm20 = moment(cloud, real_intens, 2, 0);
    wm21 = moment(cloud, real_intens, 2, 1);
    wm30 = moment(cloud, real_intens, 3, 0);
}

void BasicGeomoms2D::calcCentralMoments(const pixcloud& cloud)
{
    mu00 = centralMom(cloud, 0, 0);
    mu01 = centralMom(cloud, 0, 1);
    mu02 = centralMom(cloud, 0, 2);
    mu03 = centralMom(cloud, 0, 3);

    mu10 = centralMom(cloud, 1, 0);
    mu11 = centralMom(cloud, 1, 1);
    mu12 = centralMom(cloud, 1, 2);
    mu13 = centralMom(cloud, 1, 3);

    mu20 = centralMom(cloud, 2, 0);
    mu21 = centralMom(cloud, 2, 1);
    mu22 = centralMom(cloud, 2, 2);
    mu23 = centralMom(cloud, 2, 3);

    mu30 = centralMom(cloud, 3, 0);
    mu31 = centralMom(cloud, 3, 1);
    mu32 = centralMom(cloud, 3, 2);
    mu33 = centralMom(cloud, 3, 3);
}

void BasicGeomoms2D::calcWeightedCentralMoments(const pixcloud& cloud, const reintenvec& realintens)
{
    wmu02 = centralMom(cloud, realintens, 0, 2);
    wmu03 = centralMom(cloud, realintens, 0, 3);
    wmu11 = centralMom(cloud, realintens, 1, 1);
    wmu12 = centralMom(cloud, realintens, 1, 2);
    wmu20 = centralMom(cloud, realintens, 2, 0);
    wmu21 = centralMom(cloud, realintens, 2, 1);
    wmu30 = centralMom(cloud, realintens, 3, 0);
}

void BasicGeomoms2D::calcNormCentralMoments(const pixcloud& cloud)
{
    nu02 = normCentralMom(cloud, 0, 2);
    nu03 = normCentralMom(cloud, 0, 3);
    nu11 = normCentralMom(cloud, 1, 1);
    nu12 = normCentralMom(cloud, 1, 2);
    nu20 = normCentralMom(cloud, 2, 0);
    nu21 = normCentralMom(cloud, 2, 1);
    nu30 = normCentralMom(cloud, 3, 0);
}

void BasicGeomoms2D::calcWeightedNormCentralMoms(const pixcloud& cloud, const reintenvec& realintens)
{
    wncm20 = normCentralMom(cloud, realintens, 2, 0);
    wncm02 = normCentralMom(cloud, realintens, 0, 2);
    wncm11 = normCentralMom(cloud, realintens, 1, 1);
    wncm30 = normCentralMom(cloud, realintens, 3, 0);
    wncm12 = normCentralMom(cloud, realintens, 1, 2);
    wncm21 = normCentralMom(cloud, realintens, 2, 1);
    wncm03 = normCentralMom(cloud, realintens, 0, 3);
}

void BasicGeomoms2D::calcNormRawMoments(const pixcloud& cloud)
{
    w00 = normRawMom(cloud, 0, 0);
    w01 = normRawMom(cloud, 0, 1);
    w02 = normRawMom(cloud, 0, 2);
    w03 = normRawMom(cloud, 0, 3);

    w10 = normRawMom(cloud, 1, 0);
    w11 = normRawMom(cloud, 1, 1);
    w12 = normRawMom(cloud, 1, 2);
    w13 = normRawMom(cloud, 1, 3);

    w20 = normRawMom(cloud, 2, 0);
    w21 = normRawMom(cloud, 2, 1);
    w22 = normRawMom(cloud, 2, 2);
    w23 = normRawMom(cloud, 2, 3);

    w30 = normRawMom(cloud, 3, 0);
    w31 = normRawMom(cloud, 3, 1);
    w32 = normRawMom(cloud, 3, 2);
    w33 = normRawMom(cloud, 3, 3);
}

