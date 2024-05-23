#pragma once

#include <iostream>
#include <vector>
#include <cmath>


std::vector<double> lstsq(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
    int m = A.size(); // Number of equations
    int n = A[0].size(); // Number of unknowns

    // Initialize the matrix A transpose and the vector b transpose
    std::vector<std::vector<double>> At(n, std::vector<double>(m));
    std::vector<double> bt(b.size());

    // Compute A transpose and b transpose
    for (int i = 0; i < m; ++i) {

        for (int j = 0; j < n; ++j) {
            At[j][i] = A[i][j];
        }

        bt[i] = b[i];
    }

    // Compute A transpose * A and A transpose * b
    std::vector<std::vector<double>> AtA(n, std::vector<double>(n));
    std::vector<double> Atb(n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {

            double sum = 0.0;
            for (int k = 0; k < m; ++k) {
                sum += At[i][k] * A[k][j];
            }

            AtA[i][j] = sum;
        }

        double sum = 0.0;
        for (int k = 0; k < m; ++k) {
            sum += At[i][k] * bt[k];
        }
        
        Atb[i] = sum;
    }

    // Solve the system AtA * x = Atb using Gaussian Elimination
    for (int i = 0; i < n; ++i) {
        
        int max_index = i;
        double max_value = AtA[i][i];

        for (int j = i + 1; j < n; ++j) {

            if (abs(AtA[j][i]) > abs(max_value)) {
                max_value = AtA[j][i];
                max_index = j;
            }
        }

        // Swap rows
        if (max_index != i) {
            std::swap(AtA[i], AtA[max_index]);
            std::swap(Atb[i], Atb[max_index]);
        }

        // Elimination
        for (int j = i + 1; j < n; ++j) {

            double ratio = AtA[j][i] / AtA[i][i];
            
            for (int k = i; k < n; ++k) {
                AtA[j][k] -= ratio * AtA[i][k];
            }

            Atb[j] -= ratio * Atb[i];
        }
    }

    // Back substitution
    std::vector<double> x(n);

    for (int i = n - 1; i >= 0; --i) {

        double sum = 0.0;
        for (int j = i + 1; j < n; ++j) {
            sum += AtA[i][j] * x[j];
        }

        x[i] = (Atb[i] - sum) / AtA[i][i];
    }

    return x;
}