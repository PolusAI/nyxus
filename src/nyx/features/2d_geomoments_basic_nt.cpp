#include "2d_geomoments.h"

void BasicGeomoms2D::osized_calculate(LR& r, ImageLoader&)
{
    // Cache ROI frame of reference
    baseX = r.aabb.get_xmin();
    baseY = r.aabb.get_ymin();

    // Calculate non-weighted moments
    auto& c = r.raw_pixels_NT;
    calcOrigins(c);
    calcRawMoments(c);
    calcCentralMoments(c);
    calcNormRawMoments(c);
    calcNormCentralMoments(c);
    calcHuInvariants(c);

    // Prepare weighted pixel cloud
    pixcloud_NT w;
    w.init(r.label, "BasicGeomoms2D-osized_calculate-w");

    // Implement apply_dist2contour_weighting (w, r.contour, weighting_epsilon) :
    for (auto p : c)
    {
        // pixel distance
        auto mind2 = p.min_sqdist(r.contour);
        double dist = std::sqrt(mind2);

        // adjusted intensity
        PixIntens wi = PixIntens((double)p.inten / (dist + weighting_epsilon));

        // save
        Pixel2 p_weighted = p;
        p_weighted.inten = wi;
        w.add_pixel(p_weighted);
    }

    // Calculate weighted moments
    calcOrigins(w);
    calcWeightedRawMoments(w);
    calcWeightedCentralMoments(w);
    calcWeightedHuInvariants(w);
}

/// @brief Calculates the spatial 2D moment of order q,p f ROI pixel cloud
double BasicGeomoms2D::moment(const pixcloud_NT& cloud, int p, int q)
{
    double q_ = q, p_ = p, sum = 0;
    for (auto pxl : cloud)
        sum += double(pxl.inten) * pow(double(pxl.x - baseX), p_) * pow(double(pxl.y - baseY), q_);
    return sum;
}

void BasicGeomoms2D::calcOrigins(const pixcloud_NT& cloud)
{
    double m00 = moment(cloud, 0, 0);
    originOfX = moment(cloud, 1, 0) / m00;
    originOfY = moment(cloud, 0, 1) / m00;
}

/// @brief Calculates the central 2D moment of order q,p of ROI pixel cloud
double BasicGeomoms2D::centralMom(const pixcloud_NT& cloud, int p, int q)
{
    double sum = 0;
    for (auto pxl : cloud)
        sum += double(pxl.inten) * pow(double(pxl.x - baseX) - originOfX, p) * pow(double(pxl.y - baseY) - originOfY, q);
    return sum;
}

/// @brief Calculates the normalized spatial 2D moment of order q,p of ROI pixel cloud
double BasicGeomoms2D::normRawMom(const pixcloud_NT& cloud, int p, int q)
{
    double stddev = centralMom(cloud, 2, 2);
    int w = std::max(q, p);
    double normCoef = pow(stddev, (double)w);
    double cmPQ = centralMom(cloud, p, q);
    double retval = cmPQ / normCoef;
    return retval;
}

/// @brief Calculates the normalized central 2D moment of order q,p of ROI pixel cloud
double BasicGeomoms2D::normCentralMom(const pixcloud_NT& cloud, int p, int q)
{
    double temp = ((double(p) + double(q)) / 2.0) + 1.0;
    double retval = centralMom(cloud, p, q) / pow(moment(cloud, 0, 0), temp);
    return retval;
}

std::tuple<double, double, double, double, double, double, double> BasicGeomoms2D::calcHuInvariants_imp(const pixcloud_NT& cloud)
{
    // calculate the 7 Hu-1962 invariants

    auto _20 = normCentralMom(cloud, 2, 0),
        _02 = normCentralMom(cloud, 0, 2),
        _11 = normCentralMom(cloud, 1, 1),
        _30 = normCentralMom(cloud, 3, 0),
        _12 = normCentralMom(cloud, 1, 2),
        _21 = normCentralMom(cloud, 2, 1),
        _03 = normCentralMom(cloud, 0, 3);

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

    return { h1, h2, h3, h4, h5, h6, h7 };
}

void BasicGeomoms2D::calcHuInvariants(const pixcloud_NT& cloud)
{
    std::tie(hm1, hm2, hm3, hm4, hm5, hm6, hm7) = calcHuInvariants_imp(cloud);
}

void BasicGeomoms2D::calcWeightedHuInvariants(const pixcloud_NT& cloud)
{
    std::tie(whm1, whm2, whm3, whm4, whm5, whm6, whm7) = calcHuInvariants_imp(cloud);
}

void BasicGeomoms2D::calcRawMoments(const pixcloud_NT& cloud)
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

/// @brief 
/// @param cloud Cloud of weighted ROI pixels
void BasicGeomoms2D::calcWeightedRawMoments(const pixcloud_NT& cloud)
{
    wm00 = moment(cloud, 0, 0);
    wm01 = moment(cloud, 0, 1);
    wm02 = moment(cloud, 0, 2);
    wm03 = moment(cloud, 0, 3);
    wm10 = moment(cloud, 1, 0);
    wm11 = moment(cloud, 1, 1);
    wm12 = moment(cloud, 1, 2);
    wm20 = moment(cloud, 2, 0);
    wm21 = moment(cloud, 2, 1);
    wm30 = moment(cloud, 3, 0);
}

void BasicGeomoms2D::calcCentralMoments(const pixcloud_NT& cloud)
{
    mu02 = centralMom(cloud, 0, 2);
    mu03 = centralMom(cloud, 0, 3);
    mu11 = centralMom(cloud, 1, 1);
    mu12 = centralMom(cloud, 1, 2);
    mu20 = centralMom(cloud, 2, 0);
    mu21 = centralMom(cloud, 2, 1);
    mu30 = centralMom(cloud, 3, 0);
}

void BasicGeomoms2D::calcWeightedCentralMoments(const pixcloud_NT& cloud)
{
    wmu02 = centralMom(cloud, 0, 2);
    wmu03 = centralMom(cloud, 0, 3);
    wmu11 = centralMom(cloud, 1, 1);
    wmu12 = centralMom(cloud, 1, 2);
    wmu20 = centralMom(cloud, 2, 0);
    wmu21 = centralMom(cloud, 2, 1);
    wmu30 = centralMom(cloud, 3, 0);
}

void BasicGeomoms2D::calcNormCentralMoments(const pixcloud_NT& cloud)
{
    nu02 = normCentralMom(cloud, 0, 2);
    nu03 = normCentralMom(cloud, 0, 3);
    nu11 = normCentralMom(cloud, 1, 1);
    nu12 = normCentralMom(cloud, 1, 2);
    nu20 = normCentralMom(cloud, 2, 0);
    nu21 = normCentralMom(cloud, 2, 1);
    nu30 = normCentralMom(cloud, 3, 0);
}

void BasicGeomoms2D::calcNormRawMoments(const pixcloud_NT& cloud)
{
    w00 = normRawMom(cloud, 0, 0);
    w01 = normRawMom(cloud, 0, 1);
    w02 = normRawMom(cloud, 0, 2);
    w03 = normRawMom(cloud, 0, 3);
    w10 = normRawMom(cloud, 1, 0);
    w20 = normRawMom(cloud, 2, 0);
    w30 = normRawMom(cloud, 3, 0);
}

