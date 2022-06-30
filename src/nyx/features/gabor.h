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

const int MAX_SIZE = pow(2, 27);

/// @brief Extract face feature based on gabor filtering
class GaborFeature: public FeatureMethod
{
public:
    static bool required(const FeatureSet& fs) { return fs.isEnabled(GABOR); }

    static const int num_features = 7;

    GaborFeature();
    
    // Trivial ROI
    void calculate(LR& r);

    // Trivial ROI on GPU
    void calculate_gpu(LR& r);

    void calculate_gpu_multi_filter (LR& r);

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
        double f[8], 
        double sig2lam, 
        double gamma, 
        double theta, 
        int n,
        int num_filters);

    // Nontrivial ROIs
    void osized_GaborEnergy(
        ImageLoader& imloader,
        ReadImageMatrix_nontriv& Im,
        WriteImageMatrix_nontriv& out,
        WriteImageMatrix_nontriv& auxC,
        double* Gexp,
        double f0,
        double sig2lam,
        double gamma,
        double theta,
        int n);
    void osized_Gabor(double* Gex, double f0, double sig2lam, double gamma, double theta, double fi, int n);
    void osized_conv_dud(
        ImageLoader& imloader,
        WriteImageMatrix_nontriv& C,
        ReadImageMatrix_nontriv& A,
        double* B,
        int na, int ma, int nb, int mb);

    // Result cache
    std::vector<double> fvals;
};

