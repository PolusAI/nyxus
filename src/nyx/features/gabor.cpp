#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include <omp.h>
#include "gabor.h"

using namespace std;

// Static members changeable by user via class 'Environment'
double GaborFeature::gamma = 0.1;           
double GaborFeature::sig2lam = 0.8;         
int GaborFeature::n = 16;                    
double GaborFeature::f0LP = 0.1;            
double GaborFeature::GRAYthr = 0.025;      
std::vector<std::pair<double, double>> GaborFeature::f0_theta_pairs
{
    {0,         4.0}, 
    {M_PI_4,    16.0}, 
    {M_PI_2,    32.0},  
    {M_PI_4*3.0, 64.0} 
};

void GaborFeature::calculate (LR& r)
{
    // Number of frequencies (feature values) calculated
    int nFreqs = (int) GaborFeature::f0_theta_pairs.size();

    // Prepare the feature value buffer
    if (fvals.size() != nFreqs)
        fvals.resize (nFreqs);

    // Skip calculation in case of noninformative data and return all zeros as feature values
    if (r.aux_max == r.aux_min)
    {
        std::fill (fvals.begin(), fvals.end(), 0.0);   // 'fvals' will then be picked up by save_values()
        return;
    }

    // Get ahold of the ROI image matrix
    const ImageMatrix& Im0 = r.aux_image_matrix;

    // Temp buffers

    // --1
    ImageMatrix e2img;
    e2img.allocate (Im0.width, Im0.height);
    readOnlyPixels pix_plane = e2img.ReadablePixels();

    // --2
    std::vector<double> auxC ((Im0.width + n - 1) * (Im0.height + n - 1) * 2);

    // --3
    std::vector<double> auxG (n * n * 2);

    // --4
    tx.resize (n + 1);
    ty.resize (n + 1);

    // Compute the baseline score before applying high-pass Gabor filters
    GaborEnergy (Im0, e2img.writable_data_ptr(), auxC.data(), auxG.data(), f0LP, sig2lam, gamma, M_PI_2, n);    // compromise pi/2 theta

    // Values that we need for scoring filter responses
    Moments2 local_stats;
    e2img.GetStats(local_stats);
    double maxval = local_stats.max__(), 
        cmpval = local_stats.min__();

    // Score the baseline signal    
    unsigned long baselineScore = 0;
    for (auto a : pix_plane)
        if (double(a) > cmpval)
            baselineScore++;

    // Iterate frequencies and score corresponding filter response over the baseline
    for (int i=0; i < nFreqs; i++)
    {
        // filter response for i-th frequency
        writeablePixels e2_pix_plane = e2img.WriteablePixels();
        
        // -- unpack a frequency-angle pair
        const auto& ft = f0_theta_pairs[i];
        auto f0 = ft.first;
        auto theta = ft.second;

        GaborEnergy (Im0, e2_pix_plane.data(), auxC.data(), auxG.data(), f0, sig2lam, gamma, theta, n);

        // score it
        unsigned long afterGaborScore = 0;
        for (auto a : e2_pix_plane)
            if (double(a)/maxval > GRAYthr)
                afterGaborScore++;

        // save the score as feature value
        fvals[i] = (double)afterGaborScore / (double)baselineScore;
    }
}

#ifdef USE_GPU

void GaborFeature::calculate_gpu (LR& r)
{
    // Number of frequencies (feature values) calculated
    int nFreqs = (int)GaborFeature::f0_theta_pairs.size();

    // Prepare the feature value buffer
    if (fvals.size() != nFreqs)
        fvals.resize(nFreqs);

    // Skip calculation in case of noninformative data and return all zeros as feature values
    if (r.aux_max == r.aux_min)
    {
        std::fill(fvals.begin(), fvals.end(), 0.0);   // 'fvals' will then be picked up by save_values()
        return;
    }

    // Get ahold of the ROI image matrix
    const ImageMatrix& Im0 = r.aux_image_matrix;
    readOnlyPixels im0_plane = Im0.ReadablePixels();
    
    // Temp buffers

    // --1
    ImageMatrix e2img;
    e2img.allocate (Im0.width, Im0.height);
    readOnlyPixels pix_plane = e2img.ReadablePixels();

    // --2
    std::vector<double> auxC ((Im0.width + n - 1) * (Im0.height + n - 1) * 2);

    // --3
    std::vector<double> auxG (n * n * 2);

    // --4
    tx.resize (n + 1);
    ty.resize (n + 1);

    // Compute the baseline score before applying high-pass Gabor filters
    GaborEnergyGPU (Im0, e2img.writable_data_ptr(), auxC.data(), auxG.data(), f0LP, sig2lam, gamma, M_PI_2, n); // compromise pi/2 theta

    // Values that we need for scoring filter responses
    Moments2 local_stats;
    e2img.GetStats(local_stats);
    double maxval = local_stats.max__(),
        cmpval = local_stats.min__();

    // Score the baseline signal    
    unsigned long baselineScore = 0;
    for (auto a : pix_plane)
        if (double(a) > cmpval)
            baselineScore++;

    // Iterate frequencies and score corresponding filter response over the baseline
    for (int i=0; i < nFreqs; i++)
    {
        // filter response for i-th frequency
        writeablePixels e2_pix_plane = e2img.WriteablePixels();

        // -- unpack a frequency-angle pair
        const auto& ft = f0_theta_pairs[i];
        auto f0 = ft.first;
        auto theta = ft.second;

        GaborEnergyGPU (Im0, e2_pix_plane.data(), auxC.data(), auxG.data(), f0, sig2lam, gamma, theta, n);

        // score it
        unsigned long afterGaborScore = 0;
        for (auto a : e2_pix_plane)
            if (double(a)/maxval > GRAYthr)
                afterGaborScore++;

        // save the score as feature value
        fvals[i] = (double)afterGaborScore / (double)baselineScore;
    }
}

void GaborFeature::calculate_gpu_multi_filter (LR& r)
{
    // Number of frequencies (feature values) calculated
    int nFreqs = (int)GaborFeature::f0_theta_pairs.size();

    // Prepare the feature value buffer
    if (fvals.size() != nFreqs)
        fvals.resize(nFreqs);

    // Skip calculation in case of noninformative data and return all zeros as feature values
    if (r.aux_max == r.aux_min)
    {
        std::fill(fvals.begin(), fvals.end(), 0.0);   // 'fvals' will then be picked up by save_values()
        return;
    }

    // Get ahold of the ROI image matrix
    const ImageMatrix& Im0 = r.aux_image_matrix;
    readOnlyPixels im0_plane = Im0.ReadablePixels();

    // cuFFT dimensions
    long int cufft_size = 2 * ((Im0.width + n - 1) * (Im0.height + n - 1));
    int step_size = ceil(cufft_size / CUFFT_MAX_SIZE);
    if(step_size == 0) 
        step_size = 1;
    int num_filters = ceil(8/step_size);

    // Temp buffers
    // 
    // --1
    ImageMatrix e2img;
    e2img.allocate (Im0.width, Im0.height);

    // --2
    std::vector<double> auxC (num_filters * (Im0.width + n - 1) * (Im0.height + n - 1) * 2);

    // --3
    std::vector<double> auxG (n * n * 2);

    // --4
    tx.resize (n + 1);
    ty.resize (n + 1);

    // Variables that will be initialized in the 1-st iteration of the following loop
    unsigned long baselineScore = 0;    // abundance of baseline signal pixels over its average
    double maxval = 0.0;                // max value of the baseline signal

    // Common frequency vector: merge the low-pass (baseline related) and high-pass frequencies
    std::vector<double> freqs = { f0LP }, 
        thetas = { M_PI_2 };    // the lowpass filter goes at compromise pi/2 theta
    for (auto& ft : f0_theta_pairs)
    {
        freqs.push_back(ft.first);
        thetas.push_back(ft.second);
    }

    for(int i = 0; i < 8; i += num_filters)
    {
        // Compute the baseline score before applying high-pass Gabor filters
        vector<vector<PixIntens>> e2_pix_plane_vec (num_filters, vector<PixIntens>(Im0.width * Im0.height));

        std::vector<double> f(num_filters);

        for(int j = i; j < i + num_filters; j++) 
        {
            if(j >= 8) 
                break;
            f[j-i] = freqs[j]; 
        }

        // Calculate low-passed baseline and high-passed filter responses
        GaborEnergyGPUMultiFilter (Im0, e2_pix_plane_vec, auxC.data(), auxG.data(), f, sig2lam, gamma, thetas, n, num_filters);

        // Examine the baseline signal
        if (i == 0) 
        {
            vector<PixIntens>& pix_plane = e2_pix_plane_vec[0];

            PixIntens* e2img_ptr = e2img.writable_data_ptr();

            for(int k = 0; k < Im0.height * Im0.width; ++k)
            {
                e2img_ptr[k] = pix_plane[k];
            } 
            
            // Values that we need for scoring filter responses
            Moments2 local_stats;
            e2img.GetStats(local_stats);
            maxval = local_stats.max__();
            double cmpval = local_stats.min__();

            // Score the baseline signal    
            for (auto a : pix_plane)
                if (double(a) > cmpval)
                    baselineScore++;
        }
        
        // Iterate high-pass filter response signals and score them 
        for (int i=0; i < nFreqs; i++)
        {
            vector<PixIntens>& e2_pix_plane_temp = e2_pix_plane_vec[i+1];

            // score it
            unsigned long afterGaborScore = 0;
            for (auto a : e2_pix_plane_temp)
                if (double(a) / maxval > GRAYthr)
                    afterGaborScore++;

            // save the score as feature value
            fvals[i] = (double)afterGaborScore / (double)baselineScore;
        }
    }
}
#endif

void GaborFeature::save_value(std::vector<std::vector<double>>& feature_vals)
{
    int nFreqs = (int) GaborFeature::f0_theta_pairs.size();

    if (feature_vals[GABOR].size() != nFreqs)
        feature_vals[GABOR].resize(fvals.size());
    for (int i=0; i < nFreqs; i++)
        feature_vals[GABOR][i] = fvals[i];
}

// conv_dud -- producing a double-valued image by convolving an unsigned int valued image with double-valued kernel
//  ('conv_dud' is double <- uint * double)
//
//    C	-- Result matrix (ma+mb-1)-by-(na+nb-1). Must be zero-filled before call!
//    A -- Larger matrix 
//    B -- Smaller matrix of convolution kernel
//    ma -- Row size of A 
//    na -- Column size of A
//    mb -- Row size of B
//    nb -- Column size of B

void GaborFeature::conv_dud(
    double* C,
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

// Creates a normalized Gabor filter
void GaborFeature::Gabor (double* Gex, double f0, double sig2lam, double gamma, double theta, double fi, int n)
{
    double lambda = 2 * M_PI / f0,
        cos_theta = cos(theta), 
        sin_theta = sin(theta),
        sig = sig2lam * lambda;
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
    std::vector<std::vector<PixIntens>>& out, 
    double* auxC, 
    double* Gexp,
    const std::vector<double>& f0s,     // f0-s matching 'thetas'
    double sig2lam, 
    double gamma, 
    const std::vector<double>& thetas, // thetas matching frequencies in 'f0s'
    int n,
    int num_filters) 
{
    int n_gab = n;

    vector<double> g_filters;
    g_filters.resize(2 * n * n * num_filters);

    double fi = 0;

    int idx = 0;
    for(int i = 0; i < num_filters; ++i) 
    {   
        Gabor (Gexp, f0s[i], sig2lam, gamma, thetas[i], fi, n_gab);
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


double GaborFeature::get_theta_in_degrees (int i) 
{
    // theta needs to be unpacked from a pair
    const auto& ft = GaborFeature::f0_theta_pairs[i];
    auto theta = ft.second;
    return theta / M_PI * 180;
}

