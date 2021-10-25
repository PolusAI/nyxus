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

    //  conv2comp - conv2 when the smaller matrix is of complex numbers

    //    DOUBLE *c;	/* Result matrix (ma+mb-1)-by-(na+nb-1) */
    //    DOUBLE *a;	/* Larger matrix */
    //    DOUBLE *b;	/* Smaller matrix */
    //    INT ma;		/* Row size of a */
    //    INT na;		/* Column size of a */
    //    INT mb;		/* Row size of b */
    //    INT nb;		/* Column size of b */
    //    INT plusminus;	/* add or subtract from result */
    //    int *flopcnt;	/* flop count */

    void conv2comp(double* c, double* a, double* b, int na, int ma, int nb, int mb);


    //  conv2 - the conv2 matlab function

    //    DOUBLE *c;	/* Result matrix (ma+mb-1)-by-(na+nb-1) */
    //    DOUBLE *a;	/* Larger matrix */
    //    DOUBLE *b;	/* Smaller matrix */
    //    INT ma;		/* Row size of a */
    //    INT na;		/* Column size of a */
    //    INT mb;		/* Row size of b */
    //    INT nb;		/* Column size of b */
    //    INT plusminus;	/* add or subtract from result */
    //    int *flopcnt;	/* flop count */

    void conv2(double* c, double* a, double* b, int ma, int na, int mb, int nb, int plusminus);

    /*
    Creates a non-normalized Gabor filter
    */

    //function Gex = Gabor(f0,sig2lam,gamma,theta,fi,n),
    double* Gabor(double f0, double sig2lam, double gamma, double theta, double fi, int n);

    /* Computes Gabor energy */
    //Function [e2] = GaborEnergy(Im,f0,sig2lam,gamma,theta,n),
    PixIntens* /*double**/ GaborEnergy(const ImageMatrix& Im, PixIntens* /* double* */ out, double f0, double sig2lam, double gamma, double theta, int n);

};

