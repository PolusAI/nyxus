#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <omp.h>
#include "gabor.h"


int GaborFeatures::num_features = 7;

void GaborFeatures::calc_GaborTextureFilters2D (const ImageMatrix& Im0, std::vector<double> & ratios)
{
    double GRAYthr;
    /* parameters set up in complience with the paper */
    double gamma = 0.5, sig2lam = 0.56;
    int n = 38;
    double f0[7] = { 1,2,3,4,5,6,7 };       // frequencies for several HP Gabor filters
    double f0LP = 0.1;     // frequencies for one LP Gabor filter
    double theta = 3.14159265 / 2;
    unsigned int ii;
    unsigned long originalScore = 0;

    readOnlyPixels in_plane = Im0.ReadablePixels();
    ImageMatrix Im;

    if (false /*Im0.BoundingBoxFlag == true*/) {
        //MM: Create another Image Matrix with a padding of size n/2 around it. n is assumed to be an even number.
        Im.allocate(Im0.width + n, Im0.height + n);
        writeablePixels new_plane = Im.WriteablePixels();
        //MM: First copy the original pixel values
        for (auto y = 0; y < Im0.height; ++y) {
            for (auto x = 0; x < Im0.width; ++x) {
                new_plane(y + (n / 2), x + (n / 2)) = in_plane(y, x);
            }
        }
        //MM: Assign NaN to the padding area around the image
        for (auto x = 0; x < n / 2; ++x) {
            for (auto y = 0; y < Im.height; ++y) {
                new_plane(y, x) = std::numeric_limits<double>::quiet_NaN();
            }
        }
        for (auto x = Im.width - (n / 2); x < Im.width; ++x) {
            for (auto y = 0; y < Im.height; ++y) {
                new_plane(y, x) = std::numeric_limits<double>::quiet_NaN();
            }
        }
        for (auto y = 0; y < n / 2; ++y) {
            for (auto x = n / 2; x < Im.width - n / 2; ++x) {
                new_plane(y, x) = std::numeric_limits<double>::quiet_NaN();
            }
        }
        for (auto y = Im.height - (n / 2); y < Im.height; ++y) {
            for (auto x = n / 2; x < Im.width - n / 2; ++x) {
                new_plane(y, x) = std::numeric_limits<double>::quiet_NaN();
            }
        }
    }
    else {
        Im.allocate(Im0.width, Im0.height);
        writeablePixels new_plane = Im.WriteablePixels();

        for (auto y = 0; y < Im0.height; ++y) 
        {
            for (auto x = 0; x < Im0.width; ++x) 
            {
                auto a = in_plane (y, x);
                new_plane(y, x) = a;
            }
        }
    }

    ImageMatrix e2img;
    e2img.allocate(Im.width, Im.height);

    // compute the original score before Gabor
    GaborEnergy(Im, e2img.writable_data_ptr(), f0LP, sig2lam, gamma, theta, n);
    readOnlyPixels pix_plane = e2img.ReadablePixels();
    // N.B.: for the base of the ratios, the threshold is 0.4 of max energy,
    // while the comparison thresholds are Otsu.

    //MM originalScore = (pix_plane.array() > pix_plane.maxCoeff() * 0.4).count();
    //MM:
    Moments2 local_stats;
    e2img.GetStats(local_stats);
    double max_val = local_stats.max__();

    //
    //originalScore = (pix_plane.array() > max_val * 0.4).count();
    //
    int cnt = 0;
    double cmp_a = max_val * 0.4;
    for (auto a : pix_plane)
        if (double(a) > cmp_a)
            cnt++;
    originalScore = cnt;

    ratios.resize(8, 0.0);
    for (ii = 0; ii < GaborFeatures::num_features; ii++) {
        unsigned long afterGaborScore = 0;
        GaborEnergy(Im, e2img.writable_data_ptr(), f0[ii], sig2lam, gamma, theta, n);
        writeablePixels e2_pix_plane = e2img.WriteablePixels();

        //MM e2_pix_plane.array() = (e2_pix_plane.array() / e2_pix_plane.maxCoeff()).unaryExpr (Moments2func(e2img.stats));
        //MM:
        Moments2 local_stats2;
        e2img.GetStats(local_stats2);
        double max_val2 = local_stats2.max__();

        //
        //e2_pix_plane.array() = (e2_pix_plane.array() / max_val2).unaryExpr(Moments2func(e2img.stats));
        //
        for (auto& a : e2_pix_plane)
            a = double(a) / max_val2;

        GRAYthr = e2img.Otsu();

        //
        //afterGaborScore = (e2_pix_plane.array() > GRAYthr).count();
        //
        afterGaborScore = 0;
        for (auto a : e2_pix_plane)
            if (double(a) > GRAYthr)
                afterGaborScore++;

        ratios[ii] = (double)afterGaborScore / (double)originalScore;
    }
}

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

void GaborFeatures::conv2comp(double* c, double* a, double* b, int na, int ma, int nb, int mb) 
{
    //	double *p,*q;	/* Pointer to elements in 'a' and 'c' matrices */
    //	double wr,wi;     	/* Imaginary and real weights from matrix b    */
    int mc, nc;
    //	int k,l,i,j;
    //	double *r;				/* Pointer to elements in 'b' matrix */

    mc = ma + mb - 1;
    nc = (na + nb - 1) * 2;

    /* initalize the output matrix */
    //MM   for (int j = 0; j < mc; ++j)     /* For each element in b */
    //       for (int i = 0; i < nc; ++i)
    //           c[j*nc+i] = 0;

    /* Perform convolution */
    //	r = b;
    //	for (j = 0; j < mb; ++j) {    /* For each element in b */
    //		for (i = 0; i < nb; ++i) {
    //			wr = *(r++);			/* Get weight from b matrix */
    //			wi = *(r++);
    //			p = c + j*nc + i*2;                 /* Start at first row of a in c. */
    //			q = a;
    //			for (l = 0; l < ma; l++) {               /* For each row of a ... */
    //				for (k = 0; k < na; k++) {
    //					*(p++) += *(q) * wr;	        /* multiply by the real weight and add.      */
    //					*(p++) += *(q++) * wi;       /* multiply by the imaginary weight and add. */
    //				}
    //				p += (nb-1)*2;	                /* Jump to next row position of a in c */
    //		*flopcnt += 2*ma*na;
    //			}
    //		}
    //	}

#pragma omp parallel
    {
        double* cThread = new double[mc * nc];

        for (int aa = 0; aa < mc * nc; aa++) cThread[aa] = std::numeric_limits<double>::quiet_NaN();

#pragma omp for schedule(dynamic)
        for (int j = 0; j < mb; ++j) {    /* For each element in b */
            for (int i = 0; i < nb; ++i) {
                double* r = b + (j * nb + i) * 2;
                double wr = *(r++);			/* Get weight from b matrix */
                double wi = *(r);
                double* p = cThread + j * nc + i * 2;                 /* Start at first row of a in c. */
                double* q = a;
                for (int l = 0; l < ma; l++) {               /* For each row of a ... */
                    for (int k = 0; k < na; k++) {

                        //MM *(p++) += *(q) * wr;	        /* multiply by the real weight and add.      */
                        //MM *(p++) += *(q++) * wi;       /* multiply by the imaginary weight and add. */
                        //MM:
                        if (!std::isnan(*q)) {
                            if (std::isnan(*p))
                            {
                                *(p++) = *(q)*wr;	        /* multiply by the real weight and add.      */
                                *(p++) = *(q++) * wi;       /* multiply by the imaginary weight and add. */
                            }
                            else {
                                *(p++) += *(q)*wr;	        /* multiply by the real weight and add.      */
                                *(p++) += *(q++) * wi;       /* multiply by the imaginary weight and add. */
                            }
                        }
                        else { q++; p = p + 2; }


                    }
                    p += (nb - 1) * 2;	                /* Jump to next row position of a in c */
                    //		*flopcnt += 2*ma*na;
                }
            }
        }
#pragma omp critical
        {
            for (int j = 0; j < mc; ++j) {    /* For each element in b */
                for (int i = 0; i < nc; ++i) {
                    c[j * nc + i] += *(cThread++);
                }
            }
        }
    }
}


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

void GaborFeatures::conv2(double* c, double* a, double* b, int ma, int na, int mb, int nb, int plusminus)
{
    double* p, * q;	/* Pointer to elements in 'a' and 'c' matrices */
    double w;		/* Weight (element of 'b' matrix) */
    int mc, nc;
    int k, l, i, j;
    double* r;				/* Pointer to elements in 'b' matrix */

    mc = ma + mb - 1;
    nc = na + nb - 1;

    /* Perform convolution */

    r = b;
    for (j = 0; j < nb; ++j) {			/* For each non-zero element in b */
        for (i = 0; i < mb; ++i) {
            w = *(r++);				/* Get weight from b matrix */
            if (w != 0.0) {
                p = c + i + j * mc;	/* Start at first column of a in c. */
                for (l = 0, q = a; l < na; l++) {		/* For each column of a ... */
                    for (k = 0; k < ma; k++) {
                        *(p++) += *(q++) * w * plusminus;	/* multiply by weight and add. */
                    }
                    p += mb - 1;	/* Jump to next column position of a in c */
                }
                //		*flopcnt += 2*ma*na;
            } /* end if */
        }
    }
}


/*
Creates a non-normalized Gabor filter
*/

//function Gex = Gabor(f0,sig2lam,gamma,theta,fi,n),
double* GaborFeatures::Gabor(double f0, double sig2lam, double gamma, double theta, double fi, int n) 
{
    double* tx, * ty;
    double lambda = 2 * M_PI / f0;
    double cos_theta = cos(theta), sin_theta = sin(theta);
    double sig = sig2lam * lambda;
    double sum;
    double* Gex;
    int x, y;
    int nx = n;
    int ny = n;
    tx = new double[nx + 1];
    ty = new double[ny + 1];

    if (nx % 2 > 0) {
        tx[0] = -((nx - 1) / 2);
        for (x = 1; x < nx; x++)
            tx[x] = tx[x - 1] + 1;
    }
    else {
        tx[0] = -(nx / 2);
        for (x = 1; x <= nx; x++)
            tx[x] = tx[x - 1] + 1;
    }

    if (ny % 2 > 0) {
        ty[0] = -((ny - 1) / 2);
        for (y = 1; y < ny; y++)
            ty[y] = ty[y - 1] + 1;
    }
    else {
        ty[0] = -(ny / 2);
        for (y = 1; y <= ny; y++)
            ty[y] = ty[y - 1] + 1;
    }

    Gex = new double[n * n * 2];

    sum = 0;
    for (y = 0; y < n; y++) {
        for (x = 0; x < n; x++) {
            double argm, xte, yte, rte, ge;
            xte = tx[x] * cos_theta + ty[y] * sin_theta;
            yte = ty[y] * cos_theta - tx[x] * sin_theta;
            rte = xte * xte + gamma * gamma * yte * yte;
            ge = exp(-1 * rte / (2 * sig * sig));
            argm = xte * f0 + fi;
            Gex[y * n * 2 + x * 2] = ge * cos(argm);             // ge .* exp(j.*argm);
            Gex[y * n * 2 + x * 2 + 1] = ge * sin(argm);
            sum += sqrt(pow(Gex[y * n * 2 + x * 2], 2) + pow(Gex[y * n * 2 + x * 2 + 1], 2));
        }
    }

    /* normalize */
    for (y = 0; y < n; y++)
        for (x = 0; x < n * 2; x += 1)
            Gex[y * n * 2 + x] = Gex[y * n * 2 + x] / sum;

    delete[] tx;
    delete[] ty;

    return(Gex);
}

/* Computes Gabor energy */
//Function [e2] = GaborEnergy(Im,f0,sig2lam,gamma,theta,n),
PixIntens* /*double**/ GaborFeatures::GaborEnergy(const ImageMatrix& Im, PixIntens* /* double* */ out, double f0, double sig2lam, double gamma, double theta, int n) 
{
    double* Gexp, * image, * c;
    double fi = 0;
    Gexp = Gabor(f0, sig2lam, gamma, theta, fi, n);
    readOnlyPixels pix_plane = Im.ReadablePixels();

    c = new double[(Im.width + n - 1) * (Im.height + n - 1) * 2];

    for (int i = 0; i < (Im.width + n - 1) * (Im.height + n - 1) * 2; i++) { c[i] = 0; } //MM

    image = new double[Im.width * Im.height];
    for (auto y = 0; y < Im.height; y++)
        for (auto x = 0; x < Im.width; x++)
            image[y * Im.width + x] = pix_plane(y, x);

    conv2comp(c, image, Gexp, Im.width, Im.height, n, n);

    decltype(Im.height) b = 0;
    for (auto y = (int)ceil((double)n / 2); b < Im.height; y++) 
    {
        decltype(Im.width) a = 0;
        for (auto x = (int)ceil((double)n / 2); a < Im.width; x++) {
            //MM:
            if (std::isnan(c[y * 2 * (Im.width + n - 1) + x * 2]) || std::isnan(c[y * 2 * (Im.width + n - 1) + x * 2 + 1])) {
                out[b * Im.width + a] = std::numeric_limits<double>::quiet_NaN();
                a++;
                continue;
            }

            out[b * Im.width + a] = sqrt(pow(c[y * 2 * (Im.width + n - 1) + x * 2], 2) + pow(c[y * 2 * (Im.width + n - 1) + x * 2 + 1], 2));
            a++;
        }
        b++;
    }

    delete[] image;
    delete[] Gexp;
    delete[] c;
    return(out);
}
