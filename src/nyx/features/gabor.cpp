#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include "gabor.h"
#ifdef USE_GPU
    #include "../gpucache.h"
#endif

using namespace Nyxus;

//
// Static members changeable by user via class 'Environment'
//

double GaborFeature::gamma = 0.1;           
double GaborFeature::sig2lam = 0.8;         
int GaborFeature::n = 16;   // convolution kernel size
double GaborFeature::f0LP = 0.1;            
double GaborFeature::GRAYthr = 0.025;      
std::vector<std::pair<double, double>> GaborFeature::f0_theta_pairs
{
    {0,         4.0}, 
    {M_PI_4,    16.0}, 
    {M_PI_2,    32.0},  
    {M_PI_4*3.0, 64.0} 
};

#ifdef USE_GPU
    int GaborFeature::n_bank_filters = -1;
    std::vector<std::vector<double>> GaborFeature::filterbank;
    std::vector<double> GaborFeature::ho_filterbank;
    double* GaborFeature::dev_filterbank = nullptr;
#endif

//
//
//

bool GaborFeature::required(const FeatureSet& fs) 
{ 
    return fs.isEnabled (Feature2D::GABOR); 
}

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

    // --4  Memory for Gabor harmonics amplitudes
    tx.resize (n + 1);
    ty.resize (n + 1);

    // Compute the baseline score before applying high-pass Gabor filters
    GaborEnergy (Im0, e2img.writable_data_ptr(), auxC.data(), auxG.data(), f0LP, sig2lam, gamma, M_PI_2, n);    // compromise pi/2 theta

    // Values that we need for scoring filter responses
    Moments2 local_stats;
    e2img.GetStats(local_stats);
    double maxval = local_stats.max__(), 
        cmpval = local_stats.min__();

    // intercept blank baseline filter response
    if (maxval == cmpval)
    {
        for (int i = 0; i < nFreqs; i++)
            fvals[i] = theEnvironment.resultOptions.noval();

        return;
    }

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

namespace CuGabor
{
    bool drvImatFromCloud(size_t ri, size_t w, size_t h);
}

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

    // --4  Memory for Gabor harmonics amplitudes
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

void GaborFeature::calculate_gpu_multi_filter (LR& r, size_t roiidx)
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

    // create a GPU-side image matrix
    if (!CuGabor::drvImatFromCloud (roiidx, r.aabb.get_width(), r.aabb.get_height()))
    {
        std::cerr << "ERROR: image matrix from pixel cloud failed \n";
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
    int num_filters = ceil(nFreqs/step_size) + 1;

    // Temp buffers
    // 

    // ROI image padded w.r.t. convolution kernel in complex layout
    std::vector<double> auxC (num_filters * (Im0.width + n - 1) * (Im0.height + n - 1) * 2);

    // convolution kernel in complex layout
    std::vector<double> auxG (n * n * 2);

    // Memory for Gabor harmonics amplitudes
    tx.resize (n + 1);
    ty.resize (n + 1);

    // Common frequency vector: merge the low-pass (baseline related) and high-pass frequencies
    std::vector<double> freqs = { f0LP }, 
        thetas = { M_PI_2 };    // the lowpass filter goes at compromise pi/2 theta
    for (auto& ft : f0_theta_pairs)
    {
        freqs.push_back(ft.first);
        thetas.push_back(ft.second);
    }

    // Compute the baseline score before applying high-pass Gabor filters
    std::vector<std::vector<PixIntens>> responses (num_filters, std::vector<PixIntens>(Im0.width * Im0.height));

    // Calculate low-passed baseline and high-passed filter responses
    //      'responses' is montage of Gabor energy of filter responses
    GaborEnergyGPUMultiFilter (Im0, responses, auxC.data(), auxG.data(), freqs, sig2lam, gamma, thetas, n, freqs.size());

    // Examine the baseline signal

    // we need to get these 3 values from response[0], the baseline signal
    PixIntens maxval = 0,
        cmpval = UINT16_MAX; // min
    size_t baselineScore = 0;

    size_t wh = Im0.width * Im0.height;
    for (size_t i = 0; i < wh; i++)
    {
        auto a = responses[0][i];
        maxval = std::max(a, maxval);
        cmpval = std::min(a, cmpval);
    }

    for (size_t i = 0; i < wh; i++)
    {
        auto a = responses[0][i];
        if (a > cmpval)
            baselineScore++;
    }

    // Iterate high-pass filter response signals and score them 
    for (auto k = 1; k < freqs.size(); k++)
    {
        size_t offs = k * wh;
        size_t afterGaborScore = 0;
        // score this filter response
        for (size_t i = 0; i < wh; i++)
        {
            double a = responses[k][i]; 
            if (a / maxval > GRAYthr)
                afterGaborScore++;
        }
        // save the score as feature value
        fvals[k - 1] = (double)afterGaborScore / (double)baselineScore;
    }

}
#endif

void GaborFeature::save_value(std::vector<std::vector<double>>& feature_vals)
{
    int nFreqs = (int) GaborFeature::f0_theta_pairs.size();

    if (feature_vals[(int)Feature2D::GABOR].size() != nFreqs)
        feature_vals[(int)Feature2D::GABOR].resize(fvals.size());
    for (int i=0; i < nFreqs; i++)
        feature_vals[(int)Feature2D::GABOR][i] = fvals[i];
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
void GaborFeature::Gabor (double* Gex, double f0, double sig2lam, double gamma, double theta, double fi, int n, std::vector<double> & tx, std::vector<double> & ty)
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
    Gabor (Gexp, f0, sig2lam, gamma, theta, fi, n_gab, tx, ty);

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
    Gabor (Gexp, f0, sig2lam, gamma, theta, fi, n_gab, tx, ty);

    readOnlyPixels pix_plane = Im.ReadablePixels();

    //=== Version 2
    bool success = CuGabor::conv_dud_gpu_fft (auxC, pix_plane.data(), Gexp, Im.width, Im.height, n_gab, n_gab);
    if(!success) {
        std::cerr << "Unable to calculate Gabor features on GPU \n";
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
    std::vector<std::vector<PixIntens>>& out,   // energy image 
    double* auxC,   // batch of filter responses in complex layout 
    double* Gexp,
    const std::vector<double>& f0s,     // f0-s matching 'thetas'
    double sig2lam, 
    double gamma, 
    const std::vector<double>& thetas, // thetas matching frequencies in 'f0s'
    int kerside,
    int num_filters) 
{
    readOnlyPixels pix_plane = Im.ReadablePixels();

    // result in NyxusGpu::gabor_energy_image.hobuffer 
    // of length [ num_filters * Im.width * Im.height ]
    bool success = CuGabor::conv_dud_gpu_fft_multi_filter (
        auxC, 
        pix_plane.data(), 
        GaborFeature::ho_filterbank.data(), 
        Im.width, 
        Im.height, 
        kerside,
        kerside,
        num_filters,
        GaborFeature::dev_filterbank);
    if (!success)
    {
        std::cerr << "\n\n\nUnable to calculate Gabor features on GPU \n\n\n";
        return;
    }

    // GPU: 'NyxusGpu::gabor_energy_image.hobuffer' already has the above out[i][.] result

    // Convert it to a 'paged' layout
    size_t wh = Im.width * Im.height;
    for (int f = 0; f < num_filters; f++)
    {
        size_t skip = f * wh;
        for (size_t i = 0; i < wh; i++)
            out[f][i] = NyxusGpu::gabor_energy_image.hobuffer [skip + i];
    }

}
#endif

void GaborFeature::extract (LR& r)
{
    GaborFeature gf;
    gf.calculate (r);
    gf.save_value (r.fvals);
}

void GaborFeature::reduce (size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData)
{
    for (auto i = start; i < end; i++)
    {
        int lab = (*ptrLabels)[i];
        LR& r = (*ptrLabelData)[lab];
        extract (r);
    }
}

#ifdef USE_GPU
void GaborFeature::gpu_process_all_rois(
    std::vector<int>& L, 
    std::unordered_map <int, LR>& RoiData, 
    size_t batch_offset,
    size_t batch_len)
{
    for (size_t i = 0; i < batch_len; i++)
    {
        size_t far_i = i + batch_offset;
        auto lab = L [far_i];
        LR& r = RoiData [lab];

        // Calculate features        
        GaborFeature f;
        f.calculate_gpu_multi_filter (r, i);
        f.save_value (r.fvals);
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

#ifdef USE_GPU

void GaborFeature::create_filterbank()
{
    // frequencies and angles
    std::vector<double> freqs = { f0LP },
        thetas = { M_PI_2 };    // the lowpass filter goes at compromise pi/2 theta
    for (auto& ft : GaborFeature::f0_theta_pairs)
    {
        freqs.push_back(ft.first);
        thetas.push_back(ft.second);
    }

    // allocate the conv kernels buffer
    GaborFeature::n_bank_filters = (int)freqs.size();
    GaborFeature::filterbank.resize(GaborFeature::n_bank_filters);
    for (auto& f : GaborFeature::filterbank)
        f.resize (2 * GaborFeature::n * GaborFeature::n * GaborFeature::n_bank_filters);

    ho_filterbank.resize(2 * GaborFeature::n * GaborFeature::n * GaborFeature::n_bank_filters);

    // temps
    std::vector<double> temp_tx, temp_ty;
    temp_tx.resize (GaborFeature::n + 1);
    temp_ty.resize (GaborFeature::n + 1);
    double fi = 0;  // offset

    // filter bank as a single chunk of memory
    int idx = 0;
    for (int i = 0; i < GaborFeature::n_bank_filters; i++)
    {
        auto filterData = GaborFeature::filterbank[i];
        Gabor (filterData.data(), freqs[i], GaborFeature::sig2lam, GaborFeature::gamma, thetas[i], fi, GaborFeature::n, temp_tx, temp_ty);

        for (int i = 0; i < 2 * GaborFeature::n * GaborFeature::n; ++i)
            GaborFeature::ho_filterbank [idx + i] = filterData[i];

        idx += 2 * GaborFeature::n * GaborFeature::n;
    }
}

bool GaborFeature::send_filterbank_2_gpuside()
{
    return CuGabor::send_filterbank_2_gpuside (&GaborFeature::dev_filterbank, GaborFeature::ho_filterbank.data(), GaborFeature::ho_filterbank.size());
}

#endif

bool GaborFeature::init_class()
{
    #ifdef USE_GPU
    GaborFeature::create_filterbank();
    if (! GaborFeature::send_filterbank_2_gpuside())
        return false;
    #endif

    return true;
}