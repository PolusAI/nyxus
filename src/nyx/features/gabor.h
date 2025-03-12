#pragma once

#include <vector>
#include <unordered_map>
#include "../roi_cache.h"
#include "image_matrix.h"

#define _USE_MATH_DEFINES	// For M_PI, etc.
#include <cmath>
#include "../feature_method.h"
#include "../environment.h"
#ifdef USE_GPU
    #include "../gpu/gabor.cuh"
#endif

/// @brief Extract face feature based on gabor filtering
class GaborFeature: public FeatureMethod
{
public:

    const constexpr static std::initializer_list<Nyxus::Feature2D> featureset = { Nyxus::Feature2D::GABOR };

    GaborFeature();

    static bool required(const FeatureSet& fs);
   
    //=== Trivial ROIs ===
    void calculate(LR& r);

    // Trivial ROI on GPU
    #ifdef USE_GPU
        void calculate_gpu(LR& r);
        void calculate_gpu_multi_filter (LR& r, size_t roiidx);
        static void gpu_process_all_rois (std::vector<int>& ptrLabels, std::unordered_map <int, LR>& ptrLabelData, size_t batch_offset, size_t batch_len);
    #endif

    //=== Non-trivial ROIs ===
    void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}
    void osized_calculate(LR& r, ImageLoader& imloader);

    // Result saver
    void save_value(std::vector<std::vector<double>>& feature_vals);

    static void extract (LR& roi);
    static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

    //-------------- - User interface

    // Aspect ratio of the Gaussian
    static double gamma;            
    // spatial frequency bandwidth (sigma to lambda)
    static double sig2lam;          
    // Size of the filter kernel
    static int n;                    
    // Frequency of the low-pass Gabor filter used to produce the reference baseline image
    static double f0LP;             
    // Threshold of the filtered at frequency f0[i] to baseline image ratio
    static double GRAYthr;  

    // Pairs of orientation angles of the Gaussian (in radians) and frequency of corresponding highpass filters
    static std::vector<std::pair<double, double>> f0_theta_pairs;

    // Returns the angle of i-th pair of 'f0_theta_pairs'
    static double get_theta_in_degrees (int i);

    static bool init_class();

private:

    // Result cache
    std::vector<double> fvals;

    //=== Trivial ROIs ===

    // Convolves an uint-valued image with double-valued kernel
    void conv_dud (double* c, const unsigned int* a, double* b, int na, int ma, int nb, int mb);

    // Creates a non-normalized Gabor filter
    static void Gabor (
        double* Gex,    // buffer of size n*n*2
        double f0, 
        double sig2lam, 
        double gamma, 
        double theta, 
        double fi, 
        int n, 
        std::vector<double>& tx, 
        std::vector<double>& ty);

    // Buffers for Gabor amplitudes. Used by method Gabor()
    std::vector<double> tx, ty;

    // Computes Gabor energy 
    void GaborEnergy (
        const ImageMatrix& Im, 
        PixIntens* out, 
        double* auxC, 
        double* Gex, 
        double f0, 
        double sig2lam, 
        double gamma, 
        double theta, 
        int n);

    #ifdef USE_GPU
    void GaborEnergyGPU (
        const ImageMatrix& Im, 
        PixIntens* /* double* */ out, 
        double* auxC, 
        double* Gex, 
        double f0, 
        double sig2lam, 
        double gamma, 
        double theta, 
        int n);

    void GaborEnergyGPUMultiFilter(
        const ImageMatrix& Im,
        std::vector<std::vector<PixIntens>>& out,
        double* auxC,
        double* Gexp,
        const std::vector<double>& f0,     // frequencies matching 'thetas'
        double sig2lam,
        double gamma,
        const std::vector<double>& thetas, // thetas matching frequencies in 'f'
        int n,
        int num_filters);
    #endif

    //=== Nontrivial ROIs ===

    void GaborEnergy_NT2 (
        WriteImageMatrix_nontriv& Im,
        double* Gexp,
        double f0,
        double sig2lam,
        double gamma,
        double theta,
        int n, 
        bool max_or_threshold,
        double threshold, 
        double & max_val, 
        size_t & cnt);

    void conv_dud_NT (
        double* C,
        WriteImageMatrix_nontriv& A,
        double* B,
        int na, int ma, int nb, int mb);

    void GetStats_NT (WriteImageMatrix_nontriv& I, Moments2& moments2);

    // members referenced from init_class()
    static void create_filterbank();
    static int n_bank_filters;
    static std::vector<std::vector<double>> filterbank;
    #ifdef USE_GPU
        static bool send_filterbank_2_gpuside();
        static std::vector<double> ho_filterbank;
        static double* dev_filterbank;
    #endif
};

