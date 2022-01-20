#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include "gabor.h"
#include "image_matrix_nontriv.h"


GaborFeature::GaborFeature() {}

void GaborFeature::osized_calculate (LR& r, ImageLoader& imloader)
{
    double GRAYthr;

    // Parameters set up in complience with the paper 
    double gamma = 0.5, sig2lam = 0.56;
    int n = 38;
    double f0[7] = { 1,2,3,4,5,6,7 };       // frequencies for several HP Gabor filters
    double f0LP = 0.1;     // frequencies for one LP Gabor filter
    double theta = 3.14159265 / 2;
    int ii;
    unsigned long originalScore = 0;

    //---readOnlyPixels im0_plane = Im0.ReadablePixels();
    ReadImageMatrix_nontriv Im0 (r.aabb);

    // --1
    WriteImageMatrix_nontriv e2img ("e2img", r.label);
    auto roiWidth = r.aabb.get_width(),
        roiHeight = r.aabb.get_height();
    e2img.allocate (roiWidth, roiHeight);

    // --2
    //---std::vector<double> auxC((Im0.width + n - 1) * (Im0.height + n - 1) * 2);
    WriteImageMatrix_nontriv auxC ("auxC", r.label);
    auxC.allocate((roiWidth + n - 1) * (roiHeight + n - 1) * 2);

    // --3
    std::vector<double> auxG (n*n*2);

    // compute the original score before Gabor
    osized_GaborEnergy (imloader, Im0, e2img, auxC, auxG.data(), f0LP, sig2lam, gamma, theta, n);

    double max_val = e2img.get_max();

    int cnt = 0;
    double cmp_a = max_val * 0.4;
    for (size_t i = 0; i < e2img.size(); i++) //--- for (auto a : pix_plane)
    {
        double a = e2img.get_at(i);
        if (a > cmp_a)
            cnt++;
    }
    originalScore = cnt;

    if (fvals.size() != GaborFeature::num_features)
        fvals.resize(GaborFeature::num_features, 0.0);

    for (ii = 0; ii < GaborFeature::num_features; ii++)
    {
        unsigned long afterGaborScore = 0;
        osized_GaborEnergy (imloader, Im0, e2img, auxC, auxG.data(), f0[ii], sig2lam, gamma, theta, n);

        GRAYthr = 0.25; // --Using simplified thresholding-- GRAYthr = e2img.Otsu();

        afterGaborScore = 0;
        for (size_t i = 0; i < e2img.size(); i++) 
        {
            double a = e2img.get_at(i);
            if (a / max_val > GRAYthr)
                afterGaborScore++;
        }

        fvals[ii] = (double)afterGaborScore / (double)originalScore;
    }
}

void GaborFeature::osized_conv_dud (
    ImageLoader& imloader, 
    WriteImageMatrix_nontriv& C, //--- double* C,
    ReadImageMatrix_nontriv& A, //--- const unsigned int* A,
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
        C.set_at(i, 0.0); //--- C[i] = 0.0;

    for (int j = 0; j < mb; ++j)
    {
        // For each element in b 
        for (int i = 0; i < nb; ++i)
        {
            // Get weight from B matrix
            wr = B[ir];
            wi = B[ir + 1];
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
                    double a = A.get_at(imloader, iq); //--- A[iq];
                    iq++;

                    // multiply by the real weight and add
                    //--- C[ip] += a * wr;
                    double tmp = C.get_at(ip) +  a * wr;
                    C.set_at(ip, tmp);

                    // multiply by the imaginary weight and add
                    //--- C[ip + 1] += a * wi;
                    tmp = C.get_at(ip+1) + a * wi;
                    C.set_at(ip + 1, tmp);
                    ip += 2;
                }

                // Jump to next row position of A in C
                ip += (nb - 1) * 2;
            }
        }
    }
}

void GaborFeature::osized_GaborEnergy(
    ImageLoader& imloader, 
    ReadImageMatrix_nontriv & Im, //--- const ImageMatrix& Im,
    WriteImageMatrix_nontriv& out, //--- PixIntens* /* double* */ out,
    WriteImageMatrix_nontriv& auxC, //--- double* auxC,
    double* Gexp,
    double f0,
    double sig2lam,
    double gamma,
    double theta,
    int n)
{
    int n_gab = n;

    double fi = 0;
    osized_Gabor (Gexp, f0, sig2lam, gamma, theta, fi, n_gab);

    //--- readOnlyPixels pix_plane = Im.ReadablePixels();

    auto w = Im.get_width(), 
        h = Im.get_height();

    osized_conv_dud (imloader, auxC, Im, Gexp, w, h, n_gab, n_gab);

    decltype(h) b = 0;
    for (auto y = (int)ceil((double)n / 2); b < h; y++)
    {
        decltype(w) a = 0;
        for (auto x = (int)ceil((double)n / 2); a < w; x++)
        {
            if (std::isnan(auxC.get_at(y*2*(w+n-1)+x*2)) || std::isnan(auxC.get_at(y*2*(w+n-1)+x*2+1)))
            {
                out.set_at (b*w+a, (PixIntens)std::numeric_limits<double>::quiet_NaN());
                a++;
                continue;
            }

            out.set_at (b*w+a, (PixIntens)sqrt(pow(auxC.get_at(y*2*(w+n-1)+x*2), 2) + pow(auxC.get_at(y*2*(w+n-1)+x*2+1), 2)));
            a++;
        }
        b++;
    }
}

void GaborFeature::osized_Gabor (double* Gex, double f0, double sig2lam, double gamma, double theta, double fi, int n)
{
    //double* tx, * ty;
    double lambda = 2 * M_PI / f0;
    double cos_theta = cos(theta), sin_theta = sin(theta);
    double sig = sig2lam * lambda;
    double sum;
    //A double* Gex;
    int x, y;
    int nx = n;
    int ny = n;
    //A tx = new double[nx + 1];
    std::vector<double> tx(nx + 1);
    //A ty = new double[ny + 1];
    std::vector<double> ty(nx + 1);

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

    //A Gex = new double[n * n * 2];

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

    // normalize
    for (y = 0; y < n; y++)
        for (x = 0; x < n * 2; x += 1)
            Gex[y * n * 2 + x] = Gex[y * n * 2 + x] / sum;
}


