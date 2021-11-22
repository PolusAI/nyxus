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

    void conv_ddd (double* c, double* a, double* b, int na, int ma, int nb, int mb);
    void conv_dud (double* c, const unsigned int* a, double* b, int na, int ma, int nb, int mb);
    void conv_parallel (double* c, double* a, double* b, int na, int ma, int nb, int mb);

    // Creates a non-normalized Gabor filter
    void Gabor (
        double* Gex,    // buffer of size n*n*2
        double f0, 
        double sig2lam, 
        double gamma, 
        double theta, 
        double fi, 
        int n);

    // Computes Gabor energy 
    void GaborEnergy (
        const ImageMatrix& Im, 
        PixIntens* /* double* */ out, 
        double* auxC, 
        double* Gex, 
        double f0, 
        double sig2lam, 
        double gamma, 
        double theta, 
        int n);
};

