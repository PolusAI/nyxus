#include "image_moments.h"

ImageMoments_features::ImageMoments_features (int minI, int maxI, const ImageMatrix& im, const ImageMatrix& weighted_im)
{
    const pixData& I = im.ReadablePixels();
    calcOrigins(I);
    calcSpatialMoments(I);
    calcCentralMoments(I);
    calcNormCentralMoments(I);
    calcNormSpatialMoments(I);
    calcHuInvariants(I);

    const pixData& W = weighted_im.ReadablePixels();
    calcOrigins (W);
    calcWeightedSpatialMoments (W);
    calcWeightedCentralMoments(W);
    calcWeightedHuInvariants(W);
}

double ImageMoments_features::Moment (const pixData& D, int p, int q)
{
    // calc (p+q)th moment of object
    double sum = 0;
    for (int x = 0; x < D.width(); x++)
    {
        for (int y = 0; y < D.height(); y++)
        {
            sum += D(y,x) * pow(x, p) * pow(y, q);
        }
    }
    return sum;
}

void ImageMoments_features::calcOrigins(const pixData& D)
{
    // calc orgins
    originOfX = Moment (D, 1, 0) / Moment (D, 0, 0);
    originOfY = Moment (D, 0, 1) / Moment (D, 0, 0);
}

double ImageMoments_features::CentralMom(const pixData& D, int p, int q)
{
    // calculate central moment
    double sum = 0;
    for (int x = 0; x < D.width(); x++)
    {
        for (int y = 0; y < D.height(); y++)
        {
            sum += D(y,x) * pow((double(x) - originOfX), p) * pow((double(y) - originOfY), q);
        }
    }
    return sum;
}

// https://handwiki.org/wiki/Standardized_moment
double ImageMoments_features::NormSpatMom(const pixData& D, int p, int q)
{
    double stddev = CentralMom(D, 2, 2);
    int w = std::max(q, p);
    double normCoef = pow(stddev, w);
    double retval = CentralMom(D, p, q) / normCoef;
    return retval;
}

double ImageMoments_features::NormCentralMom(const pixData& D, int p, int q)
{
    double temp = ((double(p) + double(q)) / 2.0) + 1.0;
    double retval = CentralMom(D, p, q) / pow(Moment(D, 0, 0), temp);
    return retval;
}

std::tuple<double, double, double, double, double, double, double> ImageMoments_features::calcHuInvariants_imp (const pixData& D)
{
    // calculate 7 invariant moments
    double h1 = NormCentralMom(D, 2, 0) + NormCentralMom(D, 0, 2);
    double h2 = pow((NormCentralMom(D, 2, 0) - NormCentralMom(D, 0, 2)), 2) + 4 * (pow(NormCentralMom(D, 1, 1), 2));
    double h3 = pow((NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)), 2) +
        pow((3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)), 2);
    double h4 = pow((NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)), 2) +
        pow((NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)), 2);
    double h5 = (NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)) *
        (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) *
        (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) - 3 * pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) +
        (3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)) * (NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)) *
        (pow(3 * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)), 2) - pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2));
    double h6 = (NormCentralMom(D, 2, 0) - NormCentralMom(D, 0, 2)) * (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) -
        pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) + (4 * NormCentralMom(D, 1, 1) * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) *
            NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3));
    double h7 = (3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)) * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) * (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) -
        3 * pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) - (NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)) * (NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)) *
        (3 * pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) - pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2));
    return {h1, h2, h3, h4, h5, h6, h7};
}

void ImageMoments_features::calcHuInvariants (const pixData& D)
{
    std::tie(hm1, hm2, hm3, hm4, hm5, hm6, hm7) = calcHuInvariants_imp(D);
}

void ImageMoments_features::calcWeightedHuInvariants (const pixData& D)
{
    std::tie(whm1, whm2, whm3, whm4, whm5, whm6, whm7) = calcHuInvariants_imp(D);
}

void ImageMoments_features::calcSpatialMoments (const pixData& D)
{
    m00 = Moment (D, 0, 0);
    m01 = Moment (D, 0, 1);
    m02 = Moment (D, 0, 2);
    m03 = Moment (D, 0, 3);
    m10 = Moment (D, 1, 0);
    m11 = Moment (D, 1, 1);
    m12 = Moment (D, 1, 2);
    m20 = Moment (D, 2, 0);
    m21 = Moment (D, 2, 1);
    m30 = Moment (D, 3, 0);
}

void ImageMoments_features::calcWeightedSpatialMoments (const pixData& D)
{
    wm00 = Moment (D, 0, 0);
    wm01 = Moment (D, 0, 1);
    wm02 = Moment (D, 0, 2);
    wm03 = Moment (D, 0, 3);
    wm10 = Moment (D, 1, 0);
    wm11 = Moment (D, 1, 1);
    wm12 = Moment (D, 1, 2);
    wm20 = Moment (D, 2, 0);
    wm21 = Moment (D, 2, 1);
    wm30 = Moment (D, 3, 0);
}

void ImageMoments_features::calcCentralMoments(const pixData& D)
{
    mu02 = CentralMom(D, 0, 2);
    mu03 = CentralMom(D, 0, 3);
    mu11 = CentralMom(D, 1, 1);
    mu12 = CentralMom(D, 1, 2);
    mu20 = CentralMom(D, 2, 0);
    mu21 = CentralMom(D, 2, 1);
    mu30 = CentralMom(D, 3, 0);
}

void ImageMoments_features::calcWeightedCentralMoments(const pixData& D)
{
    wmu02 = CentralMom(D, 0, 2);
    wmu03 = CentralMom(D, 0, 3);
    wmu11 = CentralMom(D, 1, 1);
    wmu12 = CentralMom(D, 1, 2);
    wmu20 = CentralMom(D, 2, 0);
    wmu21 = CentralMom(D, 2, 1);
    wmu30 = CentralMom(D, 3, 0);
}

void ImageMoments_features::calcNormCentralMoments (const pixData& D)
{
    nu02 = NormCentralMom(D, 0, 2);
    nu03 = NormCentralMom(D, 0, 3);
    nu11 = NormCentralMom(D, 1, 1);
    nu12 = NormCentralMom(D, 1, 2);
    nu20 = NormCentralMom(D, 2, 0);
    nu21 = NormCentralMom(D, 2, 1);
    nu30 = NormCentralMom(D, 3, 0);
}

void ImageMoments_features::calcNormSpatialMoments (const pixData& D)
{
    w00 = NormSpatMom (D, 0, 0);
    w01 = NormSpatMom (D, 0, 1);
    w02 = NormSpatMom (D, 0, 2);
    w03 = NormSpatMom (D, 0, 3);
    w10 = NormSpatMom (D, 1, 0);
    w20 = NormSpatMom (D, 2, 0);
    w30 = NormSpatMom (D, 3, 0);
}

std::tuple<double, double, double, double, double, double, double, double, double, double> ImageMoments_features::getSpatialMoments()
{
    return { m00, m01, m02, m03, m10, m11, m12, m20, m21, m30 };
}

std::tuple<double, double, double, double, double, double, double, double, double, double> ImageMoments_features::getWeightedSpatialMoments()
{
    return { wm00, wm01, wm02, wm03, wm10, wm11, wm12, wm20, wm21, wm30 };
}

std::tuple<double, double, double, double, double, double, double> ImageMoments_features::getNormSpatialMoments()
{
    return { w00, w01, w02, w03, w10, w20, w30 };
}

std::tuple<double, double, double, double, double, double, double> ImageMoments_features::getCentralMoments()
{
    return { mu02, mu03, mu11, mu12, mu20, mu21, mu30 };
}

std::tuple<double, double, double, double, double, double, double> ImageMoments_features::getWeightedCentralMoments()
{
    return { wmu02, wmu03, wmu11, wmu12, wmu20, wmu21, wmu30 };
}

std::tuple<double, double, double, double, double, double, double> ImageMoments_features::getNormCentralMoments()
{
    return { nu02, nu03, nu11, nu12, nu20, nu21, nu30 };
}

std::tuple<double, double, double, double, double, double, double> ImageMoments_features::getHuMoments()
{
    return { hm1, hm2, hm3, hm4, hm5, hm6, hm7 };
}

std::tuple<double, double, double, double, double, double, double> ImageMoments_features::getWeightedHuMoments()
{
    return { whm1, whm2, whm3, whm4, whm5, whm6, whm7 };
}

void ImageMoments_features::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        if (r.has_bad_data())
            continue;


        // Prepare the contour if necessary
        if (r.contour.contour_pixels.size() == 0)
            r.contour.calculate(r.aux_image_matrix);

        ImageMatrix weighted_im(r.raw_pixels, r.aabb);
        weighted_im.apply_distance_to_contour_weights(r.raw_pixels, r.contour.contour_pixels);
        ImageMoments_features immo ((int)r.fvals[MIN][0], (int)r.fvals[MAX][0], r.aux_image_matrix, weighted_im);

        double m1, m2, m3, m4, m5, m6, m7, m8, m9, m10;
        std::tie(m1, m2, m3, m4, m5, m6, m7, m8, m9, m10) = immo.getSpatialMoments();
        r.fvals[SPAT_MOMENT_00][0] = m1;
        r.fvals[SPAT_MOMENT_01][0] = m2;
        r.fvals[SPAT_MOMENT_02][0] = m3;
        r.fvals[SPAT_MOMENT_03][0] = m4;
        r.fvals[SPAT_MOMENT_10][0] = m5;
        r.fvals[SPAT_MOMENT_11][0] = m6;
        r.fvals[SPAT_MOMENT_12][0] = m7;
        r.fvals[SPAT_MOMENT_20][0] = m8;
        r.fvals[SPAT_MOMENT_21][0] = m9;
        r.fvals[SPAT_MOMENT_30][0] = m10;

        std::tie(m1, m2, m3, m4, m5, m6, m7, m8, m9, m10) = immo.getWeightedSpatialMoments();
        r.fvals[WEIGHTED_SPAT_MOMENT_00][0] = m1;
        r.fvals[WEIGHTED_SPAT_MOMENT_01][0] = m2;
        r.fvals[WEIGHTED_SPAT_MOMENT_02][0] = m3;
        r.fvals[WEIGHTED_SPAT_MOMENT_03][0] = m4;
        r.fvals[WEIGHTED_SPAT_MOMENT_10][0] = m5;
        r.fvals[WEIGHTED_SPAT_MOMENT_11][0] = m6;
        r.fvals[WEIGHTED_SPAT_MOMENT_12][0] = m7;
        r.fvals[WEIGHTED_SPAT_MOMENT_20][0] = m8;
        r.fvals[WEIGHTED_SPAT_MOMENT_21][0] = m9;
        r.fvals[WEIGHTED_SPAT_MOMENT_30][0] = m10;

        std::tie(m1, m2, m3, m4, m5, m6, m7) = immo.getCentralMoments();
        r.fvals[CENTRAL_MOMENT_02][0] = m1;
        r.fvals[CENTRAL_MOMENT_03][0] = m2;
        r.fvals[CENTRAL_MOMENT_11][0] = m3;
        r.fvals[CENTRAL_MOMENT_12][0] = m4;
        r.fvals[CENTRAL_MOMENT_20][0] = m5;
        r.fvals[CENTRAL_MOMENT_21][0] = m6;
        r.fvals[CENTRAL_MOMENT_30][0] = m7;

        std::tie(m1, m2, m3, m4, m5, m6, m7) = immo.getWeightedCentralMoments();
        r.fvals[WEIGHTED_CENTRAL_MOMENT_02][0] = m1;
        r.fvals[WEIGHTED_CENTRAL_MOMENT_03][0] = m2;
        r.fvals[WEIGHTED_CENTRAL_MOMENT_11][0] = m3;
        r.fvals[WEIGHTED_CENTRAL_MOMENT_12][0] = m4;
        r.fvals[WEIGHTED_CENTRAL_MOMENT_20][0] = m5;
        r.fvals[WEIGHTED_CENTRAL_MOMENT_21][0] = m6;
        r.fvals[WEIGHTED_CENTRAL_MOMENT_30][0] = m7;

        std::tie(m1, m2, m3, m4, m5, m6, m7) = immo.getNormCentralMoments();
        r.fvals[NORM_CENTRAL_MOMENT_02][0] = m1;
        r.fvals[NORM_CENTRAL_MOMENT_03][0] = m2;
        r.fvals[NORM_CENTRAL_MOMENT_11][0] = m3;
        r.fvals[NORM_CENTRAL_MOMENT_12][0] = m4;
        r.fvals[NORM_CENTRAL_MOMENT_20][0] = m5;
        r.fvals[NORM_CENTRAL_MOMENT_21][0] = m6;
        r.fvals[NORM_CENTRAL_MOMENT_30][0] = m7;

        std::tie(m1, m2, m3, m4, m5, m6, m7) = immo.getNormSpatialMoments();
        r.fvals[NORM_SPAT_MOMENT_00][0] = m1;
        r.fvals[NORM_SPAT_MOMENT_01][0] = m2;
        r.fvals[NORM_SPAT_MOMENT_02][0] = m3;
        r.fvals[NORM_SPAT_MOMENT_03][0] = m4;
        r.fvals[NORM_SPAT_MOMENT_10][0] = m5;
        r.fvals[NORM_SPAT_MOMENT_20][0] = m6;
        r.fvals[NORM_SPAT_MOMENT_30][0] = m7;

        std::tie(m1, m2, m3, m4, m5, m6, m7) = immo.getHuMoments();
        r.fvals[HU_M1][0] = m1;
        r.fvals[HU_M2][0] = m2;
        r.fvals[HU_M3][0] = m3;
        r.fvals[HU_M4][0] = m4;
        r.fvals[HU_M5][0] = m5;
        r.fvals[HU_M6][0] = m6;
        r.fvals[HU_M7][0] = m7;

        std::tie(m1, m2, m3, m4, m5, m6, m7) = immo.getWeightedHuMoments();
        r.fvals[WEIGHTED_HU_M1][0] = m1;
        r.fvals[WEIGHTED_HU_M2][0] = m2;
        r.fvals[WEIGHTED_HU_M3][0] = m3;
        r.fvals[WEIGHTED_HU_M4][0] = m4;
        r.fvals[WEIGHTED_HU_M5][0] = m5;
        r.fvals[WEIGHTED_HU_M6][0] = m6;
        r.fvals[WEIGHTED_HU_M7][0] = m7;
    }
}

