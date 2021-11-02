#include "hu.h"

void HuMoments::initialize (int minI, int maxI, const ImageMatrix& im)
{
    const pixData& D = im.ReadablePixels();
    calcOrigins(D);
    calcSpatialMoments(D);
    calcCentralMoments(D);
    calcNormCentralMoments(D);
    calcNormSpatialMoments(D);
    calcHuInvariantMoments(D);
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

void HuMoments::calcHuInvariantMoments(const pixData& D)
{
    // calculate 7 invariant moments
    hm1 = NormCentralMom(D, 2, 0) + NormCentralMom(D, 0, 2);
    hm2 = pow((NormCentralMom(D, 2, 0) - NormCentralMom(D, 0, 2)), 2) + 4 * (pow(NormCentralMom(D, 1, 1), 2));
    hm3 = pow((NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)), 2) +
        pow((3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)), 2);
    hm4 = pow((NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)), 2) +
        pow((NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)), 2);
    hm5 = (NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)) *
        (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) *
        (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) - 3 * pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) +
        (3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)) * (NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)) *
        (pow(3 * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)), 2) - pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2));
    hm6 = (NormCentralMom(D, 2, 0) - NormCentralMom(D, 0, 2)) * (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) -
        pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) + (4 * NormCentralMom(D, 1, 1) * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) *
            NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3));
    hm7 = (3 * NormCentralMom(D, 2, 1) - NormCentralMom(D, 0, 3)) * (NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2)) * (pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) -
        3 * pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2)) - (NormCentralMom(D, 3, 0) - 3 * NormCentralMom(D, 1, 2)) * (NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3)) *
        (3 * pow(NormCentralMom(D, 3, 0) + NormCentralMom(D, 1, 2), 2) - pow(NormCentralMom(D, 2, 1) + NormCentralMom(D, 0, 3), 2));

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

std::tuple<double, double, double, double, double, double, double> HuMoments::getNormSpatialMoments()
{
    return { w00, w01, w02, w03, w10, w20, w30 };
}

std::tuple<double, double, double, double, double, double, double> HuMoments::getCentralMoments()
{
    return { mu02, mu03, mu11, mu12, mu20, mu21, mu30 };
}

std::tuple<double, double, double, double, double, double, double> HuMoments::getNormCentralMoments()
{
    return { nu02, nu03, nu11, nu12, nu20, nu21, nu30 };
}

std::tuple<double, double, double, double, double, double, double> HuMoments::getHuMoments()
{
    return { hm1, hm2, hm3, hm4, hm5, hm6, hm7 };
}
