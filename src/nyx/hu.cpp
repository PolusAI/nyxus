#include "hu.h"

void HuMoments::initialize (int minI, int maxI, const ImageMatrix& im)
{
    const pixData& D = im.ReadablePixels();
    calcOrgins(D);
    calcInvariantMoments(D);
}

double HuMoments::calcMoment(const pixData& D, int p, int q)
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

void HuMoments::calcOrgins(const pixData& D)
{
    // calc orgins
    orginOfX = calcMoment(D, 1, 0) / calcMoment(D, 0, 0);
    orginOfY = calcMoment(D, 0, 1) / calcMoment(D, 0, 0);
}

double HuMoments::calcCentralMoment(const pixData& D, int p, int q)
{
    // calculate central moment
    double sum = 0;
    for (int x = 0; x < D.width(); x++)
    {
        for (int y = 0; y < D.height(); y++)
        {
            sum += D(y,x) * pow((double(x) - orginOfX), p) * pow((double(y) - orginOfY), q);
        }
    }
    return sum;
}

double HuMoments::calcNormalizedMoment(const pixData& D, int p, int q)
{
    // calculate normalized moments
    double temp = ((double(p) + double(q)) / 2.0) + 1.0;
    return calcCentralMoment(D, p, q) / pow(calcMoment(D, 0, 0), temp);
}

void HuMoments::calcInvariantMoments(const pixData& D)
{
    // calculate 7 invariant moments
    m1 = calcNormalizedMoment(D, 2, 0) + calcNormalizedMoment(D, 0, 2);
    m2 = pow((calcNormalizedMoment(D, 2, 0) - calcNormalizedMoment(D, 0, 2)), 2) + 4 * (pow(calcNormalizedMoment(D, 1, 1), 2));
    m3 = pow((calcNormalizedMoment(D, 3, 0) - 3 * calcNormalizedMoment(D, 1, 2)), 2) +
        pow((3 * calcNormalizedMoment(D, 2, 1) - calcNormalizedMoment(D, 0, 3)), 2);
    m4 = pow((calcNormalizedMoment(D, 3, 0) + calcNormalizedMoment(D, 1, 2)), 2) +
        pow((calcNormalizedMoment(D, 2, 1) + calcNormalizedMoment(D, 0, 3)), 2);
    m5 = (calcNormalizedMoment(D, 3, 0) - 3 * calcNormalizedMoment(D, 1, 2)) *
        (calcNormalizedMoment(D, 3, 0) + calcNormalizedMoment(D, 1, 2)) *
        (pow(calcNormalizedMoment(D, 3, 0) + calcNormalizedMoment(D, 1, 2), 2) - 3 * pow(calcNormalizedMoment(D, 2, 1) + calcNormalizedMoment(D, 0, 3), 2)) +
        (3 * calcNormalizedMoment(D, 2, 1) - calcNormalizedMoment(D, 0, 3)) * (calcNormalizedMoment(D, 2, 1) + calcNormalizedMoment(D, 0, 3)) *
        (pow(3 * (calcNormalizedMoment(D, 3, 0) + calcNormalizedMoment(D, 1, 2)), 2) - pow(calcNormalizedMoment(D, 2, 1) + calcNormalizedMoment(D, 0, 3), 2));
    m6 = (calcNormalizedMoment(D, 2, 0) - calcNormalizedMoment(D, 0, 2)) * (pow(calcNormalizedMoment(D, 3, 0) + calcNormalizedMoment(D, 1, 2), 2) -
        pow(calcNormalizedMoment(D, 2, 1) + calcNormalizedMoment(D, 0, 3), 2)) + (4 * calcNormalizedMoment(D, 1, 1) * (calcNormalizedMoment(D, 3, 0) + calcNormalizedMoment(D, 1, 2)) *
            calcNormalizedMoment(D, 2, 1) + calcNormalizedMoment(D, 0, 3));
    m7 = (3 * calcNormalizedMoment(D, 2, 1) - calcNormalizedMoment(D, 0, 3)) * (calcNormalizedMoment(D, 3, 0) + calcNormalizedMoment(D, 1, 2)) * (pow(calcNormalizedMoment(D, 3, 0) + calcNormalizedMoment(D, 1, 2), 2) -
        3 * pow(calcNormalizedMoment(D, 2, 1) + calcNormalizedMoment(D, 0, 3), 2)) - (calcNormalizedMoment(D, 3, 0) - 3 * calcNormalizedMoment(D, 1, 2)) * (calcNormalizedMoment(D, 2, 1) + calcNormalizedMoment(D, 0, 3)) *
        (3 * pow(calcNormalizedMoment(D, 3, 0) + calcNormalizedMoment(D, 1, 2), 2) - pow(calcNormalizedMoment(D, 2, 1) + calcNormalizedMoment(D, 0, 3), 2));

}

std::tuple<double, double, double, double, double, double, double> HuMoments::getMoments()
{
    return { m1, m2, m3, m4, m5, m6, m7 };
}
