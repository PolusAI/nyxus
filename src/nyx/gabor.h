#pragma once

#include "image_matrix.h"

#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include "gabor.h"
#include <omp.h>

class GaborFeatures
{
public:

    GaborFeatures() {}

    static int num_features;

    void calc_GaborTextureFilters2D(const ImageMatrix& Im0, std::vector<double>& ratios);

protected:

    void conv(double* c, double* a, double* b, int na, int ma, int nb, int mb);

    /*
    Creates a non-normalized Gabor filter
    */

    //function Gex = Gabor(f0,sig2lam,gamma,theta,fi,n),
    double* Gabor(double f0, double sig2lam, double gamma, double theta, double fi, int n);

    /* Computes Gabor energy */
    //Function [e2] = GaborEnergy(Im,f0,sig2lam,gamma,theta,n),
    PixIntens* /*double**/ GaborEnergy(const ImageMatrix& Im, PixIntens* /* double* */ out, double f0, double sig2lam, double gamma, double theta, int n);

};

