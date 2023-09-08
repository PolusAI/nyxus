#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include "gabor.h"
#include "image_matrix_nontriv.h"


GaborFeature::GaborFeature() : FeatureMethod("GaborFeature")
{
    provide_features ({ GABOR });
}

void GaborFeature::osized_calculate (LR& r, ImageLoader&)
{
    int nF = (int)GaborFeature::f0_theta_pairs.size();

    if (fvals.size() != nF)
        fvals.resize(nF);
    std::fill (fvals.begin(), fvals.end(), 0.0);

    // Skip calculation in case of noninformative data
    if (r.aux_max == r.aux_min)
        return;

    // Prepare the image matrix
    WriteImageMatrix_nontriv Im0 ("GaborFeature-osized_calculate-Im0", r.label);
    Im0.allocate_from_cloud (r.raw_pixels_NT, r.aabb, false);

    // Buffer for Gabor filters
    std::vector<double> auxG (n * n * 2);

    // Prepare temp objects
    tx.resize(n + 1);
    ty.resize(n + 1);

    // Compute the original score before Gabor
    double max0 = 0.0;
    size_t cnt0 = 0;
    GaborEnergy_NT2 (Im0, auxG.data(), f0LP, sig2lam, gamma, M_PI_2, n, true/*request max*/, 0/*threshold*/, max0/*out*/, cnt0/*out*/); // compromise pi/2 theta

    double cmp_a = max0 * GRAYthr;

    size_t originalScore = 0;
    for (int i=0; i < nF; i++)
    {
        VERBOSLVL3 (std::cout << "\tcalculating Gabor frequency " << i+1 << "/" << nF << "\n");

        // -- unpack a frequency-angle pair
        const auto& ft = f0_theta_pairs[i];
        auto f0 = ft.first;
        auto theta = ft.second;

        size_t afterGaborScore = 0;
        GaborEnergy_NT2 (Im0, auxG.data(), f0, sig2lam, gamma, theta, n, false/*request count*/, cmp_a/*threshold*/, max0/*out*/, afterGaborScore/*out*/);
        fvals[i] = (double)afterGaborScore / (double)originalScore;

        VERBOSLVL3 (std::cout << "\t\tfeature [" << i << "] = " << fvals[i] << "\n");
    }
}

void GaborFeature::GetStats_NT (WriteImageMatrix_nontriv & I, Moments2& moments2)
{
    // Feed all the image pixels into the Moments2 object:
    moments2.reset();
    for (size_t i = 0; i < I.size(); i++)
    {
        auto inten = I.get_at(i);
        moments2.add (inten);
    }
}

// Computes Gabor energy
void GaborFeature::GaborEnergy_NT2 (
    WriteImageMatrix_nontriv& Im,
    double* Gexp,
    double f0,
    double sig2lam,
    double gamma,
    double theta,
    int n,
    bool max_or_threshold,  // 'true' to return max, 'false' to return count
    double threshold,
    double & max_val/*out*/,
    size_t & over_threshold_count/*out*/)
{
    // Reset the output variables
    over_threshold_count = 0;
    max_val = -1;

    // Prepare a filter
    double fi = 0;
    Gabor (Gexp, f0, sig2lam, gamma, theta, fi, n);

    // Helpful temps
    auto width = Im.get_width(),
        height = Im.get_height();
    auto xy0 = (int)ceil(double(n) / 2.);

    // Window (N-fold kernel size)
    int winX = n * 10,
        winY = n * 10;
    std::vector<unsigned int> W (winY * winX);

    // Convolution result buffer
    std::vector<double> auxC ((winY + n - 1) * (winX + n - 1) * 2);

    // Iterate the image window by window
    int n_winHor = width / winX, //ceil (float(width) / float(win)),
        n_winVert = height / winY; //ceil (float(height) / float(win));

    // ROI smaller than kernel?
    if (n_winVert == 0 || n_winHor == 0)
    {
        // Fill the window with data
        for (int row=0; row < height; row++)
            for (int col = 0; col < width; col++)
            {
                size_t idx = row * width + col;
                W[idx] = Im.get_at(idx);
            }

        // Convolve
        conv_dud (auxC.data(), W.data(), Gexp, winY, winX, n, n);

        // Collect the result
        for (auto y = xy0, b = 0; b < winY; y++, b++)
            for (auto x = xy0, a = 0; a < winX; x++, a++)
            {
                auto inten = (PixIntens)sqrt(pow(auxC[y * 2 * (winX + n - 1) + x * 2], 2) + pow(auxC[y * 2 * (winX + n - 1) + x * 2 + 1], 2));
                // Calculate requested summary
                if (max_or_threshold)
                    max_val = std::max(max_val, double(inten));
                else
                    if (double(inten) > threshold)
                        over_threshold_count++;
            }

        return;
    }

    // ROI larger than kernel
    for (int winVert = 0; winVert < n_winVert; winVert++)
    {
        for (int winHor = 0; winHor < n_winHor; winHor++)
        {
            // Fill the window with data
            for (int row=0; row<winY; row++)
                for (int col = 0; col < winX; col++)
                {
                    size_t imIdx = winVert * n_winHor * winY * winX // skip whole windows above
                        + row * n_winHor * winX  // skip 'row' image-wide horizontal lines
                        + winHor * n_winHor * winX   // skip winHor-wide line
                        + col;
                    W[row * winX + col] = Im.get_at(imIdx);
                }

            // Convolve
            conv_dud (auxC.data(), W.data(), Gexp, winY, winX, n, n);

            // Collect the result
            for (auto y = xy0, b = 0; b < height; y++, b++)
                for (auto x = xy0, a = 0; a < width; x++, a++)
                {
                    auto inten = (PixIntens)sqrt(pow(auxC[y * 2 * (winX + n - 1) + x * 2], 2) + pow(auxC[y * 2 * (winX + n - 1) + x * 2 + 1], 2));
                    // Calculate requested summary
                    if (max_or_threshold)
                        max_val = std::max(max_val, double(inten));
                    else
                        if (double(inten) > threshold)
                            over_threshold_count++;
                }
        }
    }
}

void GaborFeature::conv_dud_NT (
    double* C,
    WriteImageMatrix_nontriv& A,
    double* B,
    int na, int ma, int nb, int mb)
{
    int ip, iq;     // Pointer to elements in 'A' and 'C' matrices

    double wr, wi;  // Imaginary and real weights from matrix B
    int mc, nc;

    int ir = 0;         // Pointer to elements in 'b' matrix

    mc = ma + mb - 1;
    nc = (na + nb - 1) * 2;

    // initialize the output matrix
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
                    double a = A[iq];
                    iq++;

                    // multiply by the real weight and add
                    C[ip] += a * wr;

                    // multiply by the imaginary weight and add
                    C[ip + 1] += a * wi;
                    ip += 2;
                }

                // Jump to next row position of A in C
                ip += (nb - 1) * 2;
            }
        }
    }
}
