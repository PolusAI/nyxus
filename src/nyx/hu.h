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
    std::tuple<double, double, double, double, double, double, double, double, double, double> getSpatialMoments();
    std::tuple<double, double, double, double, double, double, double> getNormSpatialMoments();
    std::tuple<double, double, double, double, double, double, double> getCentralMoments();
    std::tuple<double, double, double, double, double, double, double> getNormCentralMoments();
    std::tuple<double, double, double, double, double, double, double> getHuMoments();

protected:

    double Moment (const pixData& D, int p, int q);
    void calcOrigins (const pixData& D);
    double CentralMom (const pixData& D, int p, int q);
    double NormSpatMom (const pixData& D, int p, int q);
    double NormCentralMom (const pixData& D, int p, int q);

    void calcHuInvariantMoments(const pixData& D);
    void calcNormCentralMoments(const pixData& D);
    void calcNormSpatialMoments(const pixData& D);
    void calcCentralMoments(const pixData& D);
    void calcSpatialMoments(const pixData& D);

    double originOfX = 0, originOfY = 0;
    double m00 = 0, m01 = 0, m02 = 0, m03 = 0, m10 = 0, m11 = 0, m12 = 0, m20 = 0, m21 = 0, m30 = 0;    
    double w00 = 0, w01 = 0, w02 = 0, w03 = 0, w10 = 0, w20 = 0, w30 = 0;   // normalized spatial moments
    double nu02 = 0, nu03 = 0, nu11 = 0, nu12 = 0, nu20 = 0, nu21 = 0, nu30 = 0;
    double mu02 = 0, mu03 = 0, mu11 = 0, mu12 = 0, mu20 = 0, mu21 = 0, mu30 = 0;    
    double hm1 = 0, hm2 = 0, hm3 = 0, hm4 = 0, hm5 = 0, hm6 = 0, hm7 = 0;
};

#endif // HUMOMENTS_H