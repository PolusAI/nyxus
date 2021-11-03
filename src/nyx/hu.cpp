#include "hu.h"

void HuMoments::initialize (int minI, int maxI, const ImageMatrix& im, const ImageMatrix& weighted_im)
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

double HuMoments::Moment (const pixData& D, int p, int q)
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

void HuMoments::calcOrigins(const pixData& D)
{
    // calc orgins
    originOfX = Moment (D, 1, 0) / Moment (D, 0, 0);
    originOfY = Moment (D, 0, 1) / Moment (D, 0, 0);
}

double HuMoments::CentralMom(const pixData& D, int p, int q)
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
double HuMoments::NormSpatMom(const pixData& D, int p, int q)
{
    double stddev = CentralMom(D, 2, 2);
    int w = std::max(q, p);
    double normCoef = pow(stddev, w);
    double retval = CentralMom(D, p, q) / normCoef;
    return retval;
}

double HuMoments::NormCentralMom(const pixData& D, int p, int q)
{
    double temp = ((double(p) + double(q)) / 2.0) + 1.0;
    double retval = CentralMom(D, p, q) / pow(Moment(D, 0, 0), temp);
    return retval;
}

std::tuple<double, double, double, double, double, double, double> HuMoments::calcHuInvariants_imp (const pixData& D)
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

void HuMoments::calcHuInvariants (const pixData& D)
{
    std::tie(hm1, hm2, hm3, hm4, hm5, hm6, hm7) = calcHuInvariants_imp(D);
}

void HuMoments::calcWeightedHuInvariants (const pixData& D)
{
    std::tie(whm1, whm2, whm3, whm4, whm5, whm6, whm7) = calcHuInvariants_imp(D);
}

void HuMoments::calcSpatialMoments (const pixData& D)
{
    m00 = Moment (D, 0, 0);
    m01 = Moment (D, 0, 1);
    m02 = Moment (D, 0, 2);
    m03 = Moment (D, 0, 3);
    m10 = Moment (D, 1, 0);
    m11 = Moment (D, 1, 1);
    m12 = Moment (D, 1, 2);
    m20 = Moment(D, 2, 0);
    m21 = Moment(D, 2, 1);
    m30 = Moment(D, 3, 0);
}

void HuMoments::calcWeightedSpatialMoments (const pixData& D)
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

void HuMoments::calcCentralMoments(const pixData& D)
{
    mu02 = CentralMom(D, 0, 2);
    mu03 = CentralMom(D, 0, 3);
    mu11 = CentralMom(D, 1, 1);
    mu12 = CentralMom(D, 1, 2);
    mu20 = CentralMom(D, 2, 0);
    mu21 = CentralMom(D, 2, 1);
    mu30 = CentralMom(D, 3, 0);
}

void HuMoments::calcWeightedCentralMoments(const pixData& D)
{
    wmu02 = CentralMom(D, 0, 2);
    wmu03 = CentralMom(D, 0, 3);
    wmu11 = CentralMom(D, 1, 1);
    wmu12 = CentralMom(D, 1, 2);
    wmu20 = CentralMom(D, 2, 0);
    wmu21 = CentralMom(D, 2, 1);
    wmu30 = CentralMom(D, 3, 0);
}

void HuMoments::calcNormCentralMoments (const pixData& D)
{
    nu02 = NormCentralMom(D, 0, 2);
    nu03 = NormCentralMom(D, 0, 3);
    nu11 = NormCentralMom(D, 1, 1);
    nu12 = NormCentralMom(D, 1, 2);
    nu20 = NormCentralMom(D, 2, 0);
    nu21 = NormCentralMom(D, 2, 1);
    nu30 = NormCentralMom(D, 3, 0);
}

void HuMoments::calcNormSpatialMoments (const pixData& D)
{
    w00 = NormSpatMom (D, 0, 0);
    w01 = NormSpatMom (D, 0, 1);
    w02 = NormSpatMom (D, 0, 2);
    w03 = NormSpatMom (D, 0, 3);
    w10 = NormSpatMom (D, 1, 0);
    w20 = NormSpatMom (D, 2, 0);
    w30 = NormSpatMom (D, 3, 0);
}

std::tuple<double, double, double, double, double, double, double, double, double, double> HuMoments::getSpatialMoments()
{
    return { m00, m01, m02, m03, m10, m11, m12, m20, m21, m30 };
}

std::tuple<double, double, double, double, double, double, double, double, double, double> HuMoments::getWeightedSpatialMoments()
{
    return { wm00, wm01, wm02, wm03, wm10, wm11, wm12, wm20, wm21, wm30 };
}

std::tuple<double, double, double, double, double, double, double> HuMoments::getNormSpatialMoments()
{
    return { w00, w01, w02, w03, w10, w20, w30 };
}

std::tuple<double, double, double, double, double, double, double> HuMoments::getCentralMoments()
{
    return { mu02, mu03, mu11, mu12, mu20, mu21, mu30 };
}

std::tuple<double, double, double, double, double, double, double> HuMoments::getWeightedCentralMoments()
{
    return { wmu02, wmu03, wmu11, wmu12, wmu20, wmu21, wmu30 };
}

std::tuple<double, double, double, double, double, double, double> HuMoments::getNormCentralMoments()
{
    return { nu02, nu03, nu11, nu12, nu20, nu21, nu30 };
}

std::tuple<double, double, double, double, double, double, double> HuMoments::getHuMoments()
{
    return { hm1, hm2, hm3, hm4, hm5, hm6, hm7 };
}

std::tuple<double, double, double, double, double, double, double> HuMoments::getWeightedHuMoments()
{
    return { whm1, whm2, whm3, whm4, whm5, whm6, whm7 };
}
