#pragma once

#include "contour.h"
#include "image_matrix.h"

/*
class Dist2edgeMatrix
{
public:

    Dist2edgeMatrix (const std::vector <Pixel2>& labels_raw_pixels, const Contour& cont, AABB& aabb) :
        original_aabb(aabb)
    {
        _pix_plane.allocate(aabb.get_width(), aabb.get_height(), -1);

        // Dimensions
        auto width = aabb.get_width();
        auto height = aabb.get_height();
        auto n = height * width;

        // Zero the matrix
        _pix_plane.reserve(n);
        for (auto i = 0; i < n; i++)
            _pix_plane.push_back(0);

        // Read pixels
        for (auto& pxl : labels_raw_pixels)
        {
            double dist = cont.get_dist_to_contour(pxl.x, pxl.y);
            auto c = pxl.x - aabb.get_xmin(),
                r = pxl.y - aabb.get_ymin();
            _pix_plane[r * width + c] = dist;
        }
    }

    double operator() (int y, int x) const
    {
        double val = this->_pix_plane(y, x);
        return val;
    }

    bool safe(int y, int x) const
    {
        return this->_pix_plane.safe(y, x);
    }

protected:
    SimpleMatrix<double> _pix_plane;
    AABB original_aabb;
};
*/

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
    void initialize (int minI, int maxI, const ImageMatrix& im, const ImageMatrix& weighted_im);
    std::tuple<double, double, double, double, double, double, double, double, double, double> getSpatialMoments();
    std::tuple<double, double, double, double, double, double, double, double, double, double> getWeightedSpatialMoments();
    std::tuple<double, double, double, double, double, double, double> getNormSpatialMoments();
    std::tuple<double, double, double, double, double, double, double> getCentralMoments();
    std::tuple<double, double, double, double, double, double, double> getWeightedCentralMoments();
    std::tuple<double, double, double, double, double, double, double> getNormCentralMoments();
    std::tuple<double, double, double, double, double, double, double> getHuMoments();
    std::tuple<double, double, double, double, double, double, double> getWeightedHuMoments();

protected:

    double Moment (const pixData& D, int p, int q);
    void calcOrigins (const pixData& D);
    double CentralMom (const pixData& D, int p, int q);
    double NormSpatMom (const pixData& D, int p, int q);
    double NormCentralMom (const pixData& D, int p, int q);

    std::tuple<double, double, double, double, double, double, double> calcHuInvariants_imp (const pixData& D);
    void calcHuInvariants (const pixData& D);
    void calcWeightedHuInvariants (const pixData& D);
    void calcNormCentralMoments(const pixData& D);
    void calcNormSpatialMoments(const pixData& D);
    void calcCentralMoments(const pixData& D);
    void calcWeightedCentralMoments(const pixData& D);
    void calcSpatialMoments(const pixData& D);
    void calcWeightedSpatialMoments(const pixData& D);

    double originOfX = 0, originOfY = 0;
    double m00 = 0, m01 = 0, m02 = 0, m03 = 0, m10 = 0, m11 = 0, m12 = 0, m20 = 0, m21 = 0, m30 = 0;    // spatial moments
    double wm00 = 0, wm01 = 0, wm02 = 0, wm03 = 0, wm10 = 0, wm11 = 0, wm12 = 0, wm20 = 0, wm21 = 0, wm30 = 0;    // weighted spatial moments
    double w00 = 0, w01 = 0, w02 = 0, w03 = 0, w10 = 0, w20 = 0, w30 = 0;   // normalized spatial moments
    double nu02 = 0, nu03 = 0, nu11 = 0, nu12 = 0, nu20 = 0, nu21 = 0, nu30 = 0;
    double mu02 = 0, mu03 = 0, mu11 = 0, mu12 = 0, mu20 = 0, mu21 = 0, mu30 = 0;    // central moments
    double wmu02 = 0, wmu03 = 0, wmu11 = 0, wmu12 = 0, wmu20 = 0, wmu21 = 0, wmu30 = 0;    // weighted central moments
    double hm1 = 0, hm2 = 0, hm3 = 0, hm4 = 0, hm5 = 0, hm6 = 0, hm7 = 0;   // Hu invariants
    double whm1 = 0, whm2 = 0, whm3 = 0, whm4 = 0, whm5 = 0, whm6 = 0, whm7 = 0;    // weighted Hu invariants


};

#endif // HUMOMENTS_H