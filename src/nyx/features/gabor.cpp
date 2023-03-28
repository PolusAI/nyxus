#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <omp.h>
#include "gabor.h"

using namespace std;

void GaborFeature::calculate (LR& r)
{
    // Skip calculation in case of noninformative data
    if (r.aux_max == r.aux_min)
    {
        fvals.resize (GaborFeature::num_features, 0);   // 'fvals' will then be picked up by save_values()
        return;
    }

    const ImageMatrix& Im0 = r.aux_image_matrix;
    int ii;
    unsigned long originalScore = 0;
    double f0[7] = { 1, 2, 3, 4, 5, 6, 7 };       // frequencies for several HP Gabor filters

    // Temp buffers

    // --1
    ImageMatrix e2img;
    e2img.allocate (Im0.width, Im0.height);

    // --2
    std::vector<double> auxC ((Im0.width + n - 1) * (Im0.height + n - 1) * 2);

    // --3
    std::vector<double> auxG (n * n * 2);

    // --4
    tx.resize (n + 1);
    ty.resize (n + 1);

    // compute the original score before Gabor
    GaborEnergy (Im0, e2img.writable_data_ptr(), auxC.data(), auxG.data(), f0LP, sig2lam, gamma, theta, n);
    readOnlyPixels pix_plane = e2img.ReadablePixels();
    // N.B.: for the base of the ratios, the threshold is 0.4 of max energy,
    // while the comparison thresholds are Otsu.

    Moments2 local_stats;
    e2img.GetStats(local_stats);
    double max_val = local_stats.max__();

    int cnt = 0;
    double cmp_a = max_val * 0.4;
    for (auto a : pix_plane)
        if (double(a) > cmp_a)
            cnt++;
    originalScore = cnt;

    if (fvals.size() != GaborFeature::num_features)
        fvals.resize (GaborFeature::num_features, 0.0);

    for (ii = 0; ii < GaborFeature::num_features; ii++)
    {
        unsigned long afterGaborScore = 0;
        writeablePixels e2_pix_plane = e2img.WriteablePixels();
        GaborEnergy (Im0, e2_pix_plane.data(), auxC.data(), auxG.data(), f0[ii], sig2lam, gamma, theta, n);

        afterGaborScore = 0;
        for (auto a : e2_pix_plane)
            if (double(a)/max_val > GRAYthr)
                afterGaborScore++;

        fvals[ii] = (double)afterGaborScore / (double)originalScore;
    }
}

#ifdef USE_GPU
void GaborFeature::calculate_gpu (LR& r)
{
    // Skip calculation in case of noninformative data
    if (r.aux_max == r.aux_min)
    {
        fvals.resize (GaborFeature::num_features, 0);   // 'fvals' will then be picked up by save_values()
        return;
    }

    const ImageMatrix& Im0 = r.aux_image_matrix;

    double GRAYthr;
    /* parameters set up in complience with the paper */
    double gamma = 0.5, sig2lam = 0.56;
    int n = 38;
    double f0[7] = { 1, 2, 3, 4, 5, 6, 7 };       // frequencies for several HP Gabor filters
    double f0LP = 0.1;     // frequencies for one LP Gabor filter
    double theta = 3.14159265 / 2;
    int ii;
    unsigned long originalScore = 0;

    readOnlyPixels im0_plane = Im0.ReadablePixels();

    if (fvals.size() != GaborFeature::num_features)
        fvals.resize (GaborFeature::num_features, 0.0);
    
    // Temp buffers

    // --1
    ImageMatrix e2img;
    e2img.allocate (Im0.width, Im0.height);


    // --2
    std::vector<double> auxC ((Im0.width + n - 1) * (Im0.height + n - 1) * 2);

    // --3
    std::vector<double> auxG (n * n * 2);

    // compute the original score before Gabor
    GaborEnergyGPU (Im0, e2img.writable_data_ptr(), auxC.data(), auxG.data(), f0LP, sig2lam, gamma, theta, n);
    readOnlyPixels pix_plane = e2img.ReadablePixels();
    // N.B.: for the base of the ratios, the threshold is 0.4 of max energy,
    // while the comparison thresholds are Otsu.

    Moments2 local_stats;
    e2img.GetStats(local_stats);
    double max_val = local_stats.max__();

    int cnt = 0;
    double cmp_a = max_val * 0.4;
    for (auto a : pix_plane)
        if (double(a) > cmp_a)
            cnt++;
    originalScore = cnt;

    for (ii = 0; ii < GaborFeature::num_features; ii++)
    {
        unsigned long afterGaborScore = 0;
        writeablePixels e2_pix_plane = e2img.WriteablePixels();
        GaborEnergyGPU (Im0, e2_pix_plane.data(), auxC.data(), auxG.data(), f0[ii], sig2lam, gamma, theta, n);

        GRAYthr = 0.25; // --Using simplified thresholding-- GRAYthr = e2img.Otsu();

        afterGaborScore = 0;
        for (auto a : e2_pix_plane)
            if (double(a)/max_val > GRAYthr)
                afterGaborScore++;

        fvals[ii] = (double)afterGaborScore / (double)originalScore;
    }
}

void GaborFeature::calculate_gpu_multi_filter (LR& r)
{

    if (r.aux_max == r.aux_min)
    {
        fvals.resize (GaborFeature::num_features, 0);   // 'fvals' will then be picked up by save_values()
        return;
    }

    const ImageMatrix& Im0 = r.aux_image_matrix;

    double GRAYthr;
    
    double gamma = 0.5, sig2lam = 0.56;
    int n = 38;
    double f0[8] = { 0.1, 1., 2., 3., 4., 5., 6., 7. };       // frequencies for several HP Gabor filters
    double f0LP = 0.1;     // frequencies for one LP Gabor filter
    double theta = 3.14159265 / 2;
    int ii;
    unsigned long originalScore = 0;
    double max_val;
    
    readOnlyPixels im0_plane = Im0.ReadablePixels();

    long int cufft_size = 2 * ((Im0.width + n - 1) * (Im0.height + n - 1));

    int step_size = ceil(cufft_size / CUFFT_MAX_SIZE);
    if(step_size == 0) step_size = 1;
    int num_filters = ceil(8/step_size);

    ImageMatrix e2img;
    e2img.allocate (Im0.width, Im0.height);

    // --2

    std::vector<double> auxC (num_filters * (Im0.width + n - 1) * (Im0.height + n - 1) * 2);

    // --3
    std::vector<double> auxG (n * n * 2);

    for(int i = 0; i < 8; i += num_filters){

        // compute the original score before Gabor
        vector<vector<PixIntens>> e2_pix_plane_vec(num_filters, vector<PixIntens>(Im0.width * Im0.height));

        std::vector<double> f(num_filters);

        for(int j = i; j < i + num_filters; j++) {
            if(j >= 8) break;
            
            f[j-i] = f0[j]; 
        }

        GaborEnergyGPUMultiFilter (Im0, e2_pix_plane_vec, auxC.data(), auxG.data(), f, sig2lam, gamma, theta, n, num_filters);

        if (i == 0) {
            vector<PixIntens>& pix_plane = e2_pix_plane_vec[0];

            // N.B.: for the base of the ratios, the threshold is 0.4 of max energy,
            // while the comparison thresholds are Otsu.

            PixIntens* e2img_ptr = e2img.writable_data_ptr();

            for(int k = 0; k < Im0.height * Im0.width; ++k){
                e2img_ptr[k] = pix_plane[k];
            } 
            
            Moments2 local_stats;
            //e2img._pix_plane = pix_plane;
            e2img.GetStats(local_stats);
            max_val = local_stats.max__();

            int cnt = 0;
            double cmp_a = max_val * 0.4;
            for (auto a : pix_plane)
                if (double(a) > cmp_a)
                    cnt++;
            originalScore = cnt;
        }
        
        if (fvals.size() != GaborFeature::num_features)
            fvals.resize (GaborFeature::num_features, 0.0);

        for (ii = 0; ii < num_filters-1; ii++)
        {
            vector<PixIntens>& e2_pix_plane_temp = e2_pix_plane_vec[ii+1];

            unsigned long afterGaborScore = 0;


            GRAYthr = 0.25; // --Using simplified thresholding-- GRAYthr = e2img.Otsu();

            afterGaborScore = 0;
            for (auto a : e2_pix_plane_temp)
                if (double(a)/max_val > GRAYthr)
                    afterGaborScore++;

            fvals[ii] = (double)afterGaborScore / (double)originalScore;
        }
    }
}
#endif

void GaborFeature::save_value(std::vector<std::vector<double>>& feature_vals)
{
    feature_vals[GABOR].resize(fvals.size());
    for (int i = 0; i < fvals.size(); i++)
        feature_vals[GABOR][i] = fvals[i];
}

//  conv
//
//    double *c;	Result matrix (ma+mb-1)-by-(na+nb-1)
//    double *a;	Larger matrix 
//    double *b;	Smaller matrix 
//    int ma;		Row size of a 
//    int na;		Column size of a 
//    int mb;		Row size of b 
//    int nb;		Column size of b 
void GaborFeature::conv_ddd (
    double* c, 
    double* a, 
    double* b, 
    int na, int ma, int nb, int mb) 
{
    	double *p,*q;	/* Pointer to elements in 'a' and 'c' matrices */
    	double wr,wi;     	/* Imaginary and real weights from matrix b    */
    int mc, nc;
    	int k,l,i,j;
    	double *r;				/* Pointer to elements in 'b' matrix */

    mc = ma + mb - 1;
    nc = (na + nb - 1) * 2;

    /* initalize the output matrix */
     for (int j = 0; j < mc; ++j)     /* For each element in b */
           for (int i = 0; i < nc; ++i)
               c[j*nc+i] = 0;

    /* Perform convolution */
    	r = b;
    	for (j = 0; j < mb; ++j) {    /* For each element in b */
    		for (i = 0; i < nb; ++i) {
    			wr = *(r++);			/* Get weight from b matrix */
    			wi = *(r++);
    			p = c + j*nc + i*2;                 /* Start at first row of a in c. */
    			q = a;
    			for (l = 0; l < ma; l++) {               /* For each row of a ... */
    				for (k = 0; k < na; k++) {
    					*(p++) += *(q) * wr;	        /* multiply by the real weight and add.      */
    					*(p++) += *(q++) * wi;       /* multiply by the imaginary weight and add. */
    				}
    				p += (nb-1)*2;	                /* Jump to next row position of a in c */
    		//*flopcnt += 2*ma*na;
    			}
    		}
    	}
}

void GaborFeature::conv_dud(
    double* C,  // must be zeroed before call
    const unsigned int* A,
    double* B,
    int na, int ma, int nb, int mb)
{
    int ip, iq;     // Pointer to elements in 'A' and 'C' matrices

    double wr, wi;  // Imaginary and real weights from matrix B
    int mc, nc;

    int ir = 0;         // Pointer to elements in 'b' matrix

    mc = ma + mb - 1;
    nc = (na + nb - 1) * 2;

    // initalize the output matrix 
    int mcnc = mc * nc;
    for (int i = 0; i < mcnc; i++)
        C[i] = 0.0;

    for (int j = 0; j < mb; ++j) 
    {    
        // For each element in b 
        for (int i = 0; i < nb; ++i)
        {
            // Get weight from B matrix
            wr = B[ir];
            wi = B[ir+1];
            ir += 2;

            // Start at first row of A in C 
            ip = j * nc + i * 2;
            iq = 0;

            for (int l = 0; l < ma; l++) 
            {               
                // For each row of A ... 
                for (int k = 0; k < na; k++) 
                {
                    // cache A[iq]
                    double a = A[iq];
                    iq++;

                    // multiply by the real weight and add
                    C[ip] += a * wr;

                    // multiply by the imaginary weight and add
                    C[ip+1] += a * wi;
                    ip += 2;
                }

                // Jump to next row position of A in C
                ip += (nb - 1) * 2;
            }
        }
    }
}

void GaborFeature::conv_parallel (
    double* c,
    double* a,
    double* b,
    int na, int ma, int nb, int mb)
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
        //---double* cThread = new double[mc * nc];
        std::vector<double> cThread(mc * nc);

        for (int aa = 0; aa < mc * nc; aa++)
            cThread[aa] = std::numeric_limits<double>::quiet_NaN();

        #pragma omp for schedule(dynamic)
        for (int j = 0; j < mb; ++j)
        {    /* For each element in b */
            for (int i = 0; i < nb; ++i)
            {
                double* r = b + (j * nb + i) * 2;
                double wr = *(r++);			/* Get weight from b matrix */
                double wi = *(r);

                //--- double* p = cThread + j * nc + i * 2;                 /* Start at first row of a in c. */
                int p = j * nc + i * 2;

                double* q = a;
                for (int l = 0; l < ma; l++)
                {
                    /* For each row of a ... */
                    for (int k = 0; k < na; k++)
                    {

                        //MM *(p++) += *(q) * wr;	        /* multiply by the real weight and add.      */
                        //MM *(p++) += *(q++) * wi;       /* multiply by the imaginary weight and add. */
                        //MM:
                        if (!std::isnan(*q))
                        {
                            if (std::isnan(cThread[p]))//--- if (std::isnan(*p))
                            {
                                cThread[p++] = *(q)*wr; //--- *(p++) = *(q)*wr;	        /* multiply by the real weight and add.      */
                                cThread[p++] = *(q++) * wi; //--- *(p++) = *(q++) * wi;       /* multiply by the imaginary weight and add. */
                            }
                            else
                            {
                                cThread[p++] += *(q)*wr; //--- *(p++) += *(q)*wr;	        /* multiply by the real weight and add.      */
                                cThread[p++] += *(q++) * wi; //--- *(p++) += *(q++) * wi;       /* multiply by the imaginary weight and add. */
                            }
                        }
                        else
                        {
                            q++;
                            p = p + 2;
                        }


                    }
                    p += (nb - 1) * 2;	                /* Jump to next row position of a in c */
                    //		*flopcnt += 2*ma*na;
                }
            }
        }
        #pragma omp critical
        {
            int p = 0; // index in cThread
            for (int j = 0; j < mc; ++j) {    /* For each element in b */
                for (int i = 0; i < nc; ++i)
                {
                    c[j * nc + i] += cThread[p++]; //--- c[j * nc + i] += *(cThread++);
                }
            }
        }
    }
}

void GaborFeature::conv_parallel_dud(
    double* c,
    const unsigned int* a,
    double* b,
    int na, int ma, int nb, int mb)
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
        //---double* cThread = new double[mc * nc];
        std::vector<double> cThread(mc * nc);

        for (int aa = 0; aa < mc * nc; aa++)
            cThread[aa] = std::numeric_limits<double>::quiet_NaN();

        #pragma omp for schedule(dynamic)
        for (int j = 0; j < mb; ++j)
        {    /* For each element in b */
            for (int i = 0; i < nb; ++i)
            {
                double* r = b + (j * nb + i) * 2;
                double wr = *(r++);			/* Get weight from b matrix */
                double wi = *(r);

                //--- double* p = cThread + j * nc + i * 2;                 /* Start at first row of a in c. */
                int p = j * nc + i * 2;

                const unsigned int* q = a;
                for (int l = 0; l < ma; l++)
                {
                    /* For each row of a ... */
                    for (int k = 0; k < na; k++)
                    {

                        //MM *(p++) += *(q) * wr;	        /* multiply by the real weight and add.      */
                        //MM *(p++) += *(q++) * wi;       /* multiply by the imaginary weight and add. */
                        //MM:
                        if (!std::isnan((double)*q))
                        {
                            if (std::isnan(cThread[p]))//--- if (std::isnan(*p))
                            {
                                cThread[p++] = *(q)*wr; //--- *(p++) = *(q)*wr;	        /* multiply by the real weight and add.      */
                                cThread[p++] = *(q++) * wi; //--- *(p++) = *(q++) * wi;       /* multiply by the imaginary weight and add. */
                            }
                            else
                            {
                                cThread[p++] += *(q)*wr; //--- *(p++) += *(q)*wr;	        /* multiply by the real weight and add.      */
                                cThread[p++] += *(q++) * wi; //--- *(p++) += *(q++) * wi;       /* multiply by the imaginary weight and add. */
                            }
                        }
                        else
                        {
                            q++;
                            p = p + 2;
                        }


                    }
                    p += (nb - 1) * 2;	                /* Jump to next row position of a in c */
                    //		*flopcnt += 2*ma*na;
                }
            }
        }
        #pragma omp critical
        {
            int p = 0; // index in cThread
            for (int j = 0; j < mc; ++j) {    /* For each element in b */
                for (int i = 0; i < nc; ++i)
                {
                    c[j * nc + i] += cThread[p++]; //--- c[j * nc + i] += *(cThread++);
                }
            }
        }
    }
}

// Creates a normalized Gabor filter
void GaborFeature::Gabor (double* Gex, double f0, double sig2lam, double gamma, double theta, double fi, int n)
{
    double lambda = 2 * M_PI / f0;
    double cos_theta = cos(theta), sin_theta = sin(theta);
    double sig = sig2lam * lambda;
    double sum;
    int x, y;
    int nx = n;
    int ny = n;

    if (nx % 2 > 0) {
        tx[0] = -((nx - 1) / 2);
        for (x = 1; x < nx; x++)
            tx[x] = tx[x - 1] + 1;
    }
    else {
        tx[0] = -(nx / 2);
        for (x = 1; x < nx; x++)
            tx[x] = tx[x - 1] + 1;
    }

    if (ny % 2 > 0) {
        ty[0] = -((ny - 1) / 2);
        for (y = 1; y < ny; y++)
            ty[y] = ty[y - 1] + 1;
    }
    else {
        ty[0] = -(ny / 2);
        for (y = 1; y < ny; y++)
            ty[y] = ty[y - 1] + 1;
    }

    sum = 0;
    for (y = 0; y < n; y++) 
    {
        for (x = 0; x < n; x++) 
{
            double argm, xte, yte, rte, ge;
            xte = tx[x] * cos_theta + ty[y] * sin_theta;
            yte = ty[y] * cos_theta - tx[x] * sin_theta;
            rte = xte * xte + gamma * gamma * yte * yte;
            ge = exp(-1 * rte / (2 * sig * sig));
            argm = xte * f0 + fi;
            int idx = y * n * 2 + x * 2;
            Gex[idx] = ge * cos(argm);             // ge .* exp(j.*argm);
            Gex[idx + 1] = ge * sin(argm);
            sum += sqrt(pow(Gex[idx], 2) + pow(Gex[idx + 1], 2));
        }
    }

    // normalize
    for (y = 0; y < n; y++)
        for (x = 0; x < n * 2; x++)
            Gex[y * n * 2 + x] /= sum;

}

// Computes Gabor energy
void GaborFeature::GaborEnergy (
    const ImageMatrix& Im, 
    PixIntens* out, 
    double* auxC, 
    double* Gexp,
    double f0, 
    double sig2lam, 
    double gamma, 
    double theta, 
    int n) 
{
    int n_gab = n;

    double fi = 0;
    Gabor (Gexp, f0, sig2lam, gamma, theta, fi, n_gab);

    readOnlyPixels pix_plane = Im.ReadablePixels();

    #if 0
    //=== Version 1 (using the cached image)
    for (int i = 0; i < (Im.width + n - 1) * (Im.height + n - 1) * 2; i++) 
    { 
        c[i] = 0; 
    } //MM    
    
    double *image = new double[Im.width * Im.height];
    //std::vector<double> image(Im.width * Im.height);

    for (auto y = 0; y < Im.height; y++)
        for (auto x = 0; x < Im.width; x++)
            image[y * Im.width + x] = pix_plane(y, x);

    conv (c, image, Gexp, Im.width, Im.height, n, n);

    delete[] image;
    #endif

    //=== Version 2
    conv_dud (auxC, pix_plane.data(), Gexp, Im.width, Im.height, n_gab, n_gab);

    decltype(Im.height) b = 0;
    for (auto y = (int)ceil((double)n / 2); b < Im.height; y++) 
    {
        decltype(Im.width) a = 0;
        for (auto x = (int)ceil((double)n / 2); a < Im.width; x++) 
        {
            if (std::isnan(auxC[y * 2 * (Im.width + n - 1) + x * 2]) || std::isnan(auxC[y * 2 * (Im.width + n - 1) + x * 2 + 1])) 
            {
                out[b * Im.width + a] = (PixIntens) std::numeric_limits<double>::quiet_NaN();
                a++;
                continue;
            }

            out[b * Im.width + a] = (PixIntens) sqrt(pow(auxC[y * 2 * (Im.width + n - 1) + x * 2], 2) + pow(auxC[y * 2 * (Im.width + n - 1) + x * 2 + 1], 2));
            a++;
        }
        b++;
    }
}

#ifdef USE_GPU
void GaborFeature::GaborEnergyGPU (
    const ImageMatrix& Im, 
    PixIntens* /* double* */ out, 
    double* auxC, 
    double* Gexp,
    double f0, 
    double sig2lam, 
    double gamma, 
    double theta, 
    int n) 
{
    int n_gab = n;

    double fi = 0;
    Gabor (Gexp, f0, sig2lam, gamma, theta, fi, n_gab);

    readOnlyPixels pix_plane = Im.ReadablePixels();

    //=== Version 2
    bool success = CuGabor::conv_dud_gpu_fft (auxC, pix_plane.data(), Gexp, Im.width, Im.height, n_gab, n_gab);
    if(!success) {
        std::cerr << "Unable to calculate Gabor features on GPU." << endl;
    }

    decltype(Im.height) b = 0;
    for (auto y = (int)ceil((double)n / 2); b < Im.height; y++) 
    {
        decltype(Im.width) a = 0;
        for (auto x = (int)ceil((double)n / 2); a < Im.width; x++) 
        {
            if (std::isnan(auxC[y * 2 * (Im.width + n - 1) + x * 2]) || std::isnan(auxC[y * 2 * (Im.width + n - 1) + x * 2 + 1])) 
            {
                out[b * Im.width + a] = (PixIntens) std::numeric_limits<double>::quiet_NaN();
                a++;
                continue;
            }

            out[b * Im.width + a] = (PixIntens) sqrt(pow(auxC[y * 2 * (Im.width + n - 1) + x * 2], 2) + pow(auxC[y * 2 * (Im.width + n - 1) + x * 2 + 1], 2));
            a++;
        }
        b++;
    }
}

void GaborFeature::GaborEnergyGPUMultiFilter (
    const ImageMatrix& Im, 
    vector<vector<PixIntens>>& /* double* */ out, 
    double* auxC, 
    double* Gexp,
    std::vector<double>& f, 
    double sig2lam, 
    double gamma, 
    double theta, 
    int n,
    int num_filters) 
{
    int n_gab = n;

    vector<double> g_filters;
    g_filters.resize(2 * n * n * num_filters);

    double fi = 0;

    int idx = 0;
    double f0;
    for(int i = 0; i < num_filters; ++i) 
    {   
        f0 = f[i];
        Gabor (Gexp, f0, sig2lam, gamma, theta, fi, n_gab);
        for(int i = 0; i < 2*n*n; ++i) {
            g_filters[idx+i] = Gexp[i];
        }

        idx += 2*n*n;
    }

    readOnlyPixels pix_plane = Im.ReadablePixels();
    
    bool success = CuGabor::conv_dud_gpu_fft_multi_filter (auxC, pix_plane.data(), g_filters.data(), Im.width, Im.height, n_gab, n_gab, num_filters);
    if(!success) {
        std::cerr << "Unable to calculate Gabor features on GPU." << endl;
    }

    for (int i = 0; i < num_filters; ++i){
        idx = 2 * i * (Im.width + n - 1) * (Im.height + n - 1);
        decltype(Im.height) b = 0;
        
        for (auto y = (int)ceil((double)n / 2); b < Im.height; y++) 
        {
            decltype(Im.width) a = 0;

            for (auto x = (int)ceil((double)n / 2); a < Im.width; x++) 
            {
                if (std::isnan(auxC[idx + (y * 2 * (Im.width + n - 1) + x * 2)]) || std::isnan(auxC[idx + (y * 2 * (Im.width + n - 1) + x * 2 + 1)])) 
                {
                    out[i][(b * Im.width + a)] = (PixIntens) std::numeric_limits<double>::quiet_NaN();
                    a++;
                    continue;
                }
                out[i][(b * Im.width + a)] = (PixIntens) sqrt(pow(auxC[idx + (y * 2 * (Im.width + n - 1) + x * 2)], 2) + pow(auxC[idx + (y * 2 * (Im.width + n - 1) + x * 2 + 1)], 2));
                a++;
            }
            b++;
        }
        
    }

}
#endif

void GaborFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];

        GaborFeature gf;

        gf.calculate (r);

        gf.save_value (r.fvals);
    }
}

#ifdef USE_GPU
void GaborFeature::gpu_process_all_rois( std::vector<int>& ptrLabels, std::unordered_map <int, LR>& ptrLabelData) 
{
    for (auto& lab: ptrLabels) {
        LR& r = ptrLabelData[lab];

        GaborFeature gf;

        gf.calculate_gpu_multi_filter(r);

        gf.save_value (r.fvals);
    }
}
#endif


