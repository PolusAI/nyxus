#pragma once

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <unordered_map>
#include <functional>

#include "../helpers/convolution2d.h"

enum class MscnType {
    mscn = 1,
    horizontal = 2,
    vertical = 3,
    main_diagonal = 4,
    secondary_diagonal = 5
};


const static std::vector<MscnType> mscn_types {
    MscnType::mscn, 
    MscnType::horizontal, 
    MscnType::vertical, 
    MscnType::main_diagonal, 
    MscnType::secondary_diagonal
};


class Mscn {

private:

    std::vector<double> mscn_;

    std::vector<unsigned int> image_;
    int image_rows_;
    int image_cols_;

    std::vector<double> kernel_;
    int kernel_rows_;
    int kernel_cols_;

    std::unordered_map<MscnType, std::vector<double>> coefficients_;

    std::vector<double> local_mean_;
    std::vector<double> local_deviation_;


    std::vector<double> get_local_mean() {

        if (local_mean_.size() == 0) {
            local_mean_ = convolution2d(image_, kernel_, image_rows_, image_cols_, kernel_rows_, kernel_cols_);
        }

        return local_mean_;

    }

    std::vector<double> get_local_deviation() {

        if (local_mean_.size() == 0) {
            auto temp = get_local_mean();
        }

        if (local_deviation_.size() == 0) {

            auto image_squared = image_;

            for (auto& pix: image_squared) {
                pix *= pix; // square each pixel
            }

            auto sigma = convolution2d(image_squared, kernel_, image_rows_, image_cols_, kernel_rows_, kernel_cols_);
            
            local_deviation_.resize(image_rows_*image_cols_);

            for (int i = 0; i < image_rows_*image_cols_; ++i) {
                local_deviation_[i] = std::sqrt(std::abs( (local_mean_[i] * local_mean_[i]) - sigma[i]));
            }

        }

        return local_deviation_;

    }

    void calculate_mscn() {
        double c = 1./255.;

        get_local_mean();
        get_local_deviation();
        
        mscn_.resize(image_rows_*image_cols_);

        for (int i = 0; i < image_rows_*image_cols_; ++i) {
            mscn_[i] = (image_[i] - local_mean_[i]) / (local_deviation_[i] + c);
        }
    }


    std::vector<double> get_mscn() {
        return mscn_;
    }

    std::vector<double> mscn_horizontal() {
        std::vector<double> result(image_rows_ * (image_cols_ - 1));
        for (int i = 0; i < image_rows_; ++i) {
            for (int j = 0; j < image_cols_ - 1; ++j) {
                result[i * (image_rows_ - 1) + j] = mscn_[i * image_cols_ + j] * mscn_[i * image_cols_ + j + 1];
            }
        }
        return result;
    }

    std::vector<double> get_mscn_horizontal() {
        std::vector<double> result(image_rows_ * (image_cols_-1));

        for (int i = 0; i < image_rows_; ++i) {
            for (int j = 0; j < image_cols_-1; ++j) {
                result[i * (image_cols_-1) + j] = mscn_[i * image_cols_ + j] * mscn_[i * image_cols_ + j+1];
            }
        }

        return result;
    }

    std::vector<double> get_mscn_vertical() {

        std::vector<double> result((image_rows_ - 1) * image_cols_);

        for (int i = 0; i < image_rows_ - 1; ++i) {
            for (int j = 0; j < image_cols_; ++j) {
                result[i * image_cols_ + j] = mscn_[i * image_cols_ + j] * mscn_[(i + 1) * image_cols_ + j];
            }
        }

        return result;
    }

    std::vector<double> get_mscn_diagonal() {

        std::vector<double> result((image_rows_ - 1) * (image_cols_ - 1));

        for (int i = 0; i < image_rows_ - 1; ++i) {
            for (int j = 0; j < image_cols_ - 1; ++j) {
                result[i * (image_cols_ - 1) + j] = mscn_[i * image_cols_ + j] * mscn_[(i + 1) * image_cols_ + j + 1];
            }
        }

        return result;
    }

    std::vector<double> get_mscn_secondary_diagonal() {

        std::vector<double> result((image_rows_ - 1) * (image_cols_ - 1));

        for (int i = 0; i < image_rows_ - 1; ++i) {
            for (int j = 0; j < image_cols_ - 1; ++j) {
                result[i * (image_cols_ - 1) + j] = mscn_[(i + 1) * image_cols_ + j] * mscn_[i * image_cols_ + j + 1];
            }
        }

        return result;
    }

public:

    Mscn(const std::vector<unsigned int>& img, int rows, int cols, 
         const std::vector<double>& kernel, int k_rows, int k_cols) {

        image_ = img;   
        image_rows_ = rows;
        image_cols_ = cols;
        
        kernel_ = kernel;
        kernel_rows_ = k_rows;
        kernel_cols_ = k_cols;

        calculate_mscn(); // initialize mscn

    }

    std::unordered_map<MscnType, std::vector<double>> get_coefficients() {

        if (coefficients_.size() == 0) {
            coefficients_[MscnType::mscn] = get_mscn(); 
            coefficients_[MscnType::horizontal] = get_mscn_horizontal();
            coefficients_[MscnType::vertical] = get_mscn_vertical();
            coefficients_[MscnType::main_diagonal] = get_mscn_diagonal();
            coefficients_[MscnType::secondary_diagonal] = get_mscn_secondary_diagonal();
        }

        return coefficients_;
         
    }

};

enum DistributionSide { left, right };

class AsymmetricGeneralizedGaussian {
private:
    std::vector<double> x_;
    double alpha_;

    std::vector<double> calculate_x(DistributionSide side) {

        std::vector<double> result;

        if (side == DistributionSide::left) {

            for (double val : x_) {
                if (val < 0) {
                    result.push_back(val);
                }
            }

        } else if (side == DistributionSide::right) {

            for (double val : x_) {
                if (val >= 0) {
                    result.push_back(val);
                }
            }

        } else {
            throw std::invalid_argument("Side was not recognized");
        }

        return result;
    }

    double sigma(DistributionSide side) {

        std::vector<double> _x_vec;

        if (side == DistributionSide::right) {

            _x_vec = calculate_x(DistributionSide::right);

        } else if (side == DistributionSide::left) {

            _x_vec = calculate_x(DistributionSide::left);

        } else {
            throw std::invalid_argument("Side was not recognized");
        }

        double sum = 0.0;
        for (double val : _x_vec) {
            sum += val * val;
        }

        return std::sqrt(sum / _x_vec.size());
    }

    double mean_squares(std::vector<double> x_vec) {

        double sum = 0.0;
        for (double val : x_vec) {
            sum += val * val;
        }

        return sum / x_vec.size();
    }

    double phi(double alpha) {
        return std::pow(std::tgamma(2 / alpha), 2) / (std::tgamma(1 / alpha) * std::tgamma(3 / alpha));
    }

   double find_root(std::function<double(double)> f,
                 double min, double max, double epsilon)
    {
        auto f_min = f(min);
        while (min + epsilon < max) {
            auto const mid = 0.5 * min + 0.5 * max;
            auto const f_mid = f(mid);

            if ((f_min < 0) == (f_mid < 0)) {
                min = mid;
                f_min = f_mid;
            } else {
                max = mid;
            }
        }

        return min;
    }

public:
    AsymmetricGeneralizedGaussian(const std::vector<double>& x_vec) {
        x_ = x_vec;
        alpha_ = -1; // Set an invalid alpha initially
    }

    std::vector<double> x_left() {
        return calculate_x(DistributionSide::left);
    }

    std::vector<double> x_right() {
        return calculate_x(DistributionSide::right);
    }

    double sigma_left() {
        return sigma(DistributionSide::left);
    }

    double sigma_right() {
        return sigma(DistributionSide::right);
    }

    double gamma() {
        return sigma_left() / sigma_right();
    }

    double r_hat() {

        double sum = 0.0;
        for (double val : x_) {
            sum += std::abs(val);
        }

        return std::pow(sum / x_.size(), 2) / mean_squares(x_);
    }

    double R_hat() {
        double gamma_val = gamma();
        return r_hat() * (std::pow(gamma_val, 3) + 1) * (gamma_val + 1) / std::pow(gamma_val * gamma_val + 1, 2);
    }

    double constant() {
        return std::sqrt(std::tgamma(1 / alpha()) / std::tgamma(3 / alpha()));
    }

    double mean() {
        return (sigma_right() - sigma_left()) * constant() * (std::tgamma(2 / alpha()) / std::tgamma(1 / alpha()));
    }

    double alpha() {

        if (alpha_ < 0) {
            throw std::runtime_error("The distribution has no alpha estimated. Run method fit() to calculate.");
        }

        return alpha_;
    }

    double estimate_alpha(double x0 = 0.2) {

        auto func = [&](double alpha) {
            return phi(alpha) - R_hat();
        };

        double solution = find_root(func, -1., 1., x0);

        return solution;
    }

    AsymmetricGeneralizedGaussian fit(double x0 = 0.2) {

        alpha_ = estimate_alpha(x0);

        return *this;
    }
};
