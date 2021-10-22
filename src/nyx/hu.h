#pragma once

#include "image_matrix.h"

// Inspired by Yavuz Unver
// 
// Hu Moments and Digit Recognition Algorithm:
// http://www.wseas.us/e-library/conferences/2013/CambridgeUK/AISE/AISE-15.pdf
//

#ifndef HUMOMENTS_H
#define HUMOMENTS_H

#include <math.h>

class HuMoments
{
public:

    HuMoments() {}
    void initialize (int minI, int maxI, const ImageMatrix& im);
    std::tuple<double, double, double, double, double, double, double> getMoments();

protected:

    double calcMoment (const pixData& D, int p, int q);
    void calcOrgins(const pixData& D);
    double calcCentralMoment(const pixData& D, int p, int q);
    double calcNormalizedMoment(const pixData& D, int p, int q);
    void calcInvariantMoments(const pixData& D);

    double orginOfX = 0.0, orginOfY = 0.0, m1 = 0.0, m2 = 0.0, m3 = 0.0, m4 = 0.0, m5 = 0.0, m6 = 0.0, m7 = 0.0;
};

#endif // HUMOMENTS_H