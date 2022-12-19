#include "image_moments.h"

void ImageMomentsFeature::osized_calculate(LR& r, ImageLoader& imlo)
{
    const ImageMatrix& im = r.aux_image_matrix;

    ReadImageMatrix_nontriv I(r.aabb); 
    calcOrigins_nontriv (imlo, I);
    calcSpatialMoments_nontriv (imlo, I);
    calcCentralMoments_nontriv (imlo, I);
    calcNormCentralMoments_nontriv (imlo, I);
    calcNormSpatialMoments_nontriv (imlo, I);
    calcHuInvariants_nontriv (imlo, I);

    WriteImageMatrix_nontriv W ("ImageMomentsFeature_osized_calculate_W", r.label); 
    W.init_with_cloud_distance_to_contour_weights (r.raw_pixels_NT, r.aabb, r.contour);
    calcOrigins_nontriv (W);
    calcWeightedSpatialMoments_nontriv (W);
    calcWeightedCentralMoments_nontriv (W);
    calcWeightedHuInvariants_nontriv (W);
}

double ImageMomentsFeature::Moment_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I, int p, int q)
{
    // calc (p+q)th moment of object
    double sum = 0;
    for (size_t x = 0; x < I.get_width(); x++)
    {
        for (size_t y = 0; y < I.get_height(); y++)
        {
            sum += I.get_at(imlo, y, x) * pow(x, p) * pow(y, q);
        }
    }
    return sum;
}

double ImageMomentsFeature::Moment_nontriv (WriteImageMatrix_nontriv & I, int p, int q)
{
    // calc (p+q)th moment of object
    double sum = 0;
    for (size_t x = 0; x < I.get_width(); x++)
    {
        for (size_t y = 0; y < I.get_height(); y++)
        {
            sum += I.get_at(y, x) * pow(x, p) * pow(y, q);
        }
    }
    return sum;
}

void ImageMomentsFeature::calcOrigins_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I)
{
    // calc orgins
    double m00 = Moment_nontriv (imlo, I, 0, 0);
    originOfX = Moment_nontriv (imlo, I, 1, 0) / m00;
    originOfY = Moment_nontriv (imlo, I, 0, 1) / m00;
}

void ImageMomentsFeature::calcOrigins_nontriv (WriteImageMatrix_nontriv & I)
{
    // calc orgins
    double m00 = Moment_nontriv (I, 0, 0);
    originOfX = Moment_nontriv (I, 1, 0) / m00;
    originOfY = Moment_nontriv (I, 0, 1) / m00;
}

double ImageMomentsFeature::CentralMom_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I, int p, int q)
{
    // calculate central moment
    double sum = 0;
    for (int x = 0; x < I.get_width(); x++)
    {
        for (int y = 0; y < I.get_height(); y++)
        {
            sum += I.get_at(imlo, y, x) * pow((double(x) - originOfX), p) * pow((double(y) - originOfY), q);
        }
    }
    return sum;
}

double ImageMomentsFeature::CentralMom_nontriv (WriteImageMatrix_nontriv& W, int p, int q)
{
    // calculate central moment
    double sum = 0;
    for (int x = 0; x < W.get_width(); x++)
    {
        for (int y = 0; y < W.get_height(); y++)
        {
            sum += W.get_at (y, x) * pow((double(x) - originOfX), p) * pow((double(y) - originOfY), q);
        }
    }
    return sum;
}

// https://handwiki.org/wiki/Standardized_moment
double ImageMomentsFeature::NormSpatMom_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I, int p, int q)
{
    double stddev = CentralMom_nontriv (imlo, I, 2, 2);
    int w = std::max(q, p);
    double normCoef = pow(stddev, w);
    double retval = CentralMom_nontriv(imlo, I, p, q) / normCoef;
    return retval;
}

double ImageMomentsFeature::NormCentralMom_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I, int p, int q)
{
    double temp = ((double(p) + double(q)) / 2.0) + 1.0;
    double retval = CentralMom_nontriv (imlo, I, p, q) / pow(Moment_nontriv (imlo, I, 0, 0), temp);
    return retval;
}

double ImageMomentsFeature::NormCentralMom_nontriv (WriteImageMatrix_nontriv& W, int p, int q)
{
    double temp = ((double(p) + double(q)) / 2.0) + 1.0;
    double retval = CentralMom_nontriv (W, p, q) / pow(Moment_nontriv (W, 0, 0), temp);
    return retval;
}

void ImageMomentsFeature::calcSpatialMoments_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I)
{
    m00 = Moment_nontriv (imlo, I, 0, 0);
    m01 = Moment_nontriv (imlo, I, 0, 1);
    m02 = Moment_nontriv (imlo, I, 0, 2);
    m03 = Moment_nontriv (imlo, I, 0, 3);
    m10 = Moment_nontriv (imlo, I, 1, 0);
    m11 = Moment_nontriv (imlo, I, 1, 1);
    m12 = Moment_nontriv (imlo, I, 1, 2);
    m20 = Moment_nontriv (imlo, I, 2, 0);
    m21 = Moment_nontriv (imlo, I, 2, 1);
    m30 = Moment_nontriv (imlo, I, 3, 0);
}

void ImageMomentsFeature::calcWeightedSpatialMoments_nontriv (WriteImageMatrix_nontriv& W)
{
    wm00 = Moment_nontriv (W, 0, 0);
    wm01 = Moment_nontriv (W, 0, 1);
    wm02 = Moment_nontriv (W, 0, 2);
    wm03 = Moment_nontriv (W, 0, 3);
    wm10 = Moment_nontriv (W, 1, 0);
    wm11 = Moment_nontriv (W, 1, 1);
    wm12 = Moment_nontriv (W, 1, 2);
    wm20 = Moment_nontriv (W, 2, 0);
    wm21 = Moment_nontriv (W, 2, 1);
    wm30 = Moment_nontriv (W, 3, 0);
}

void ImageMomentsFeature::calcCentralMoments_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I)
{
    mu02 = CentralMom_nontriv (imlo, I, 0, 2);
    mu03 = CentralMom_nontriv (imlo, I, 0, 3);
    mu11 = CentralMom_nontriv (imlo, I, 1, 1);
    mu12 = CentralMom_nontriv (imlo, I, 1, 2);
    mu20 = CentralMom_nontriv (imlo, I, 2, 0);
    mu21 = CentralMom_nontriv (imlo, I, 2, 1);
    mu30 = CentralMom_nontriv (imlo, I, 3, 0);
}

void ImageMomentsFeature::calcWeightedCentralMoments_nontriv (WriteImageMatrix_nontriv& W)
{
    wmu02 = CentralMom_nontriv (W, 0, 2);
    wmu03 = CentralMom_nontriv (W, 0, 3);
    wmu11 = CentralMom_nontriv (W, 1, 1);
    wmu12 = CentralMom_nontriv (W, 1, 2);
    wmu20 = CentralMom_nontriv (W, 2, 0);
    wmu21 = CentralMom_nontriv (W, 2, 1);
    wmu30 = CentralMom_nontriv (W, 3, 0);
}

void ImageMomentsFeature::calcNormCentralMoments_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I)
{
    nu02 = NormCentralMom_nontriv (imlo, I, 0, 2);
    nu03 = NormCentralMom_nontriv (imlo, I, 0, 3);
    nu11 = NormCentralMom_nontriv (imlo, I, 1, 1);
    nu12 = NormCentralMom_nontriv (imlo, I, 1, 2);
    nu20 = NormCentralMom_nontriv (imlo, I, 2, 0);
    nu21 = NormCentralMom_nontriv (imlo, I, 2, 1);
    nu30 = NormCentralMom_nontriv (imlo, I, 3, 0);
}

void ImageMomentsFeature::calcNormSpatialMoments_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I)
{
    w00 = NormSpatMom_nontriv (imlo, I, 0, 0);
    w01 = NormSpatMom_nontriv (imlo, I, 0, 1);
    w02 = NormSpatMom_nontriv (imlo, I, 0, 2);
    w03 = NormSpatMom_nontriv (imlo, I, 0, 3);
    w10 = NormSpatMom_nontriv (imlo, I, 1, 0);
    w20 = NormSpatMom_nontriv (imlo, I, 2, 0);
    w30 = NormSpatMom_nontriv (imlo, I, 3, 0);
}

std::tuple<double, double, double, double, double, double, double> ImageMomentsFeature::calcHuInvariants_imp_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I)
{
    // calculate 7 invariant moments
    double h1 = NormCentralMom_nontriv (imlo, I, 2, 0) + NormCentralMom_nontriv (imlo, I, 0, 2);
    double h2 = pow((NormCentralMom_nontriv (imlo, I, 2, 0) - NormCentralMom_nontriv (imlo, I, 0, 2)), 2) + 4 * (pow(NormCentralMom_nontriv (imlo, I, 1, 1), 2));
    double h3 = pow((NormCentralMom_nontriv (imlo, I, 3, 0) - 3 * NormCentralMom_nontriv (imlo, I, 1, 2)), 2) +
        pow((3 * NormCentralMom_nontriv (imlo, I, 2, 1) - NormCentralMom_nontriv (imlo, I, 0, 3)), 2);
    double h4 = pow((NormCentralMom_nontriv (imlo, I, 3, 0) + NormCentralMom_nontriv (imlo, I, 1, 2)), 2) +
        pow((NormCentralMom_nontriv (imlo, I, 2, 1) + NormCentralMom_nontriv (imlo, I, 0, 3)), 2);
    double h5 = (NormCentralMom_nontriv (imlo, I, 3, 0) - 3 * NormCentralMom_nontriv (imlo, I, 1, 2)) *
        (NormCentralMom_nontriv (imlo, I, 3, 0) + NormCentralMom_nontriv (imlo, I, 1, 2)) *
        (pow(NormCentralMom_nontriv (imlo, I, 3, 0) + NormCentralMom_nontriv (imlo, I, 1, 2), 2) - 3 * pow(NormCentralMom_nontriv (imlo, I, 2, 1) + NormCentralMom_nontriv (imlo, I, 0, 3), 2)) +
        (3 * NormCentralMom_nontriv (imlo, I, 2, 1) - NormCentralMom_nontriv (imlo, I, 0, 3)) * (NormCentralMom_nontriv (imlo, I, 2, 1) + NormCentralMom_nontriv (imlo, I, 0, 3)) *
        (pow(3 * (NormCentralMom_nontriv (imlo, I, 3, 0) + NormCentralMom_nontriv (imlo, I, 1, 2)), 2) - pow(NormCentralMom_nontriv (imlo, I, 2, 1) + NormCentralMom_nontriv (imlo, I, 0, 3), 2));
    double h6 = (NormCentralMom_nontriv (imlo, I, 2, 0) - NormCentralMom_nontriv (imlo, I, 0, 2)) * (pow(NormCentralMom_nontriv (imlo, I, 3, 0) + NormCentralMom_nontriv (imlo, I, 1, 2), 2) -
        pow(NormCentralMom_nontriv (imlo, I, 2, 1) + NormCentralMom_nontriv (imlo, I, 0, 3), 2)) + (4 * NormCentralMom_nontriv (imlo, I, 1, 1) * (NormCentralMom_nontriv (imlo, I, 3, 0) + NormCentralMom_nontriv (imlo, I, 1, 2)) *
            NormCentralMom_nontriv (imlo, I, 2, 1) + NormCentralMom_nontriv (imlo, I, 0, 3));
    double h7 = (3 * NormCentralMom_nontriv (imlo, I, 2, 1) - NormCentralMom_nontriv (imlo, I, 0, 3)) * (NormCentralMom_nontriv (imlo, I, 3, 0) + NormCentralMom_nontriv (imlo, I, 1, 2)) * (pow(NormCentralMom_nontriv (imlo, I, 3, 0) + NormCentralMom_nontriv (imlo, I, 1, 2), 2) -
        3 * pow(NormCentralMom_nontriv (imlo, I, 2, 1) + NormCentralMom_nontriv (imlo, I, 0, 3), 2)) - (NormCentralMom_nontriv (imlo, I, 3, 0) - 3 * NormCentralMom_nontriv (imlo, I, 1, 2)) * (NormCentralMom_nontriv (imlo, I, 2, 1) + NormCentralMom_nontriv (imlo, I, 0, 3)) *
        (3 * pow(NormCentralMom_nontriv (imlo, I, 3, 0) + NormCentralMom_nontriv (imlo, I, 1, 2), 2) - pow(NormCentralMom_nontriv (imlo, I, 2, 1) + NormCentralMom_nontriv (imlo, I, 0, 3), 2));
    return { h1, h2, h3, h4, h5, h6, h7 };
}

std::tuple<double, double, double, double, double, double, double> ImageMomentsFeature::calcHuInvariants_imp_nontriv (WriteImageMatrix_nontriv& W)
{
    // calculate 7 invariant moments
    double h1 = NormCentralMom_nontriv (W, 2, 0) + NormCentralMom_nontriv (W, 0, 2);
    double h2 = pow((NormCentralMom_nontriv (W, 2, 0) - NormCentralMom_nontriv (W, 0, 2)), 2) + 4 * (pow(NormCentralMom_nontriv (W, 1, 1), 2));
    double h3 = pow((NormCentralMom_nontriv (W, 3, 0) - 3 * NormCentralMom_nontriv (W, 1, 2)), 2) +
        pow((3 * NormCentralMom_nontriv (W, 2, 1) - NormCentralMom_nontriv (W, 0, 3)), 2);
    double h4 = pow((NormCentralMom_nontriv (W, 3, 0) + NormCentralMom_nontriv (W, 1, 2)), 2) +
        pow((NormCentralMom_nontriv (W, 2, 1) + NormCentralMom_nontriv (W, 0, 3)), 2);
    double h5 = (NormCentralMom_nontriv (W, 3, 0) - 3 * NormCentralMom_nontriv (W, 1, 2)) *
        (NormCentralMom_nontriv (W, 3, 0) + NormCentralMom_nontriv (W, 1, 2)) *
        (pow(NormCentralMom_nontriv (W, 3, 0) + NormCentralMom_nontriv (W, 1, 2), 2) - 3 * pow(NormCentralMom_nontriv (W, 2, 1) + NormCentralMom_nontriv (W, 0, 3), 2)) +
        (3 * NormCentralMom_nontriv (W, 2, 1) - NormCentralMom_nontriv (W, 0, 3)) * (NormCentralMom_nontriv (W, 2, 1) + NormCentralMom_nontriv (W, 0, 3)) *
        (pow(3 * (NormCentralMom_nontriv (W, 3, 0) + NormCentralMom_nontriv (W, 1, 2)), 2) - pow(NormCentralMom_nontriv (W, 2, 1) + NormCentralMom_nontriv (W, 0, 3), 2));
    double h6 = (NormCentralMom_nontriv (W, 2, 0) - NormCentralMom_nontriv (W, 0, 2)) * (pow(NormCentralMom_nontriv (W, 3, 0) + NormCentralMom_nontriv (W, 1, 2), 2) -
        pow(NormCentralMom_nontriv (W, 2, 1) + NormCentralMom_nontriv (W, 0, 3), 2)) + (4 * NormCentralMom_nontriv (W, 1, 1) * (NormCentralMom_nontriv (W, 3, 0) + NormCentralMom_nontriv (W, 1, 2)) *
            NormCentralMom_nontriv (W, 2, 1) + NormCentralMom_nontriv (W, 0, 3));
    double h7 = (3 * NormCentralMom_nontriv (W, 2, 1) - NormCentralMom_nontriv (W, 0, 3)) * (NormCentralMom_nontriv (W, 3, 0) + NormCentralMom_nontriv (W, 1, 2)) * (pow(NormCentralMom_nontriv (W, 3, 0) + NormCentralMom_nontriv (W, 1, 2), 2) -
        3 * pow(NormCentralMom_nontriv (W, 2, 1) + NormCentralMom_nontriv (W, 0, 3), 2)) - (NormCentralMom_nontriv (W, 3, 0) - 3 * NormCentralMom_nontriv (W, 1, 2)) * (NormCentralMom_nontriv (W, 2, 1) + NormCentralMom_nontriv (W, 0, 3)) *
        (3 * pow(NormCentralMom_nontriv (W, 3, 0) + NormCentralMom_nontriv (W, 1, 2), 2) - pow(NormCentralMom_nontriv (W, 2, 1) + NormCentralMom_nontriv (W, 0, 3), 2));
    return { h1, h2, h3, h4, h5, h6, h7 };
}

void ImageMomentsFeature::calcHuInvariants_nontriv (ImageLoader& imlo, ReadImageMatrix_nontriv& I)
{
    std::tie(hm1, hm2, hm3, hm4, hm5, hm6, hm7) = calcHuInvariants_imp_nontriv (imlo, I);
}

void ImageMomentsFeature::calcWeightedHuInvariants_nontriv (WriteImageMatrix_nontriv& W)
{
    std::tie(whm1, whm2, whm3, whm4, whm5, whm6, whm7) = calcHuInvariants_imp_nontriv (W);
}


