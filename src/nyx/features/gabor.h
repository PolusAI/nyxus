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
    static bool required(const FeatureSet& fs) { return fs.isEnabled(GABOR); }

    static constexpr int num_features = 7;

    GaborFeature();
    
    // Trivial ROI
    void calculate(LR& r);

    // Trivial ROI on GPU
    #ifdef USE_GPU
        void calculate_gpu(LR& r);
        void calculate_gpu_multi_filter (LR& r);
        static void gpu_process_all_rois( std::vector<int>& ptrLabels, std::unordered_map <int, LR>& ptrLabelData);
    #endif

    // Non-trivial
    void osized_add_online_pixel(size_t x, size_t y, uint32_t intensity) {}
    void osized_calculate(LR& r, ImageLoader& imloader);

    // Result saver
    void save_value(std::vector<std::vector<double>>& feature_vals);

    static void reduce(size_t start, size_t end, std::vector<int>* ptrLabels, std::unordered_map <int, LR>* ptrLabelData);

private:
    // Trivial ROIs
    void conv_ddd (double* c, double* a, double* b, int na, int ma, int nb, int mb);
    void conv_dud (double* c, const unsigned int* a, double* b, int na, int ma, int nb, int mb);
    void conv_parallel (double* c, double* a, double* b, int na, int ma, int nb, int mb);
    void conv_parallel_dud (double* c, const unsigned int* a, double* b, int na, int ma, int nb, int mb);

    // Creates a non-normalized Gabor filter
    void Gabor (
        double* Gex,    // buffer of size n*n*2
        double f0, 
        double sig2lam, 
        double gamma, 
        double theta, 
        double fi, 
        int n);
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

    void GaborEnergyGPUMultiFilter (
        const ImageMatrix& Im, 
        std::vector<std::vector<PixIntens>>& /* double* */ out, 
        double* auxC, 
        double* Gexp,
        std::vector<double>& f, 
        double sig2lam, 
        double gamma, 
        double theta, 
        int n,
        int num_filters);
    #endif

    // Nontrivial ROIs

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

    // Result cache
    std::vector<double> fvals;

    // Parameters
    static constexpr double gamma = 0.5; 
    static constexpr double sig2lam = 0.8;
    static constexpr int n = 20;                    // Gabor filter size
    static constexpr double f0LP = 0.1;             // frequency of the baseline LP Gabor filter
    static constexpr double theta = 3.14159265/3;   // 60 deg
    static constexpr double GRAYthr = 0.25/10;      // simplified thresholding as GRAYthr=e2img.Otsu()
};

