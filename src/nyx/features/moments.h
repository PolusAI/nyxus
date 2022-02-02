#pragma once

#include <stddef.h>
#include <cfloat> // DBL_MAX
#include <cmath>
#define MIN_VAL -FLT_MAX
#define MAX_VAL FLT_MAX

/// @brief Helper class for 2nd order moments calculation 
class Moments2 
{
public:
    Moments2() { reset(); }
    void reset() { _mean = M2 = 0.0; _min = DBL_MAX; _max = -DBL_MAX; _n = 0; }
    inline double add(const double x) {
        size_t n1;
        double delta, delta_n, term1;
        if (std::isnan(x) || x > MAX_VAL || x < MIN_VAL)
            return (x);

        n1 = _n;
        _n = _n + 1;
        delta = x - _mean;
        delta_n = delta / _n;
        term1 = delta * delta_n * n1;
        _mean = _mean + delta_n;
        M2 += term1;

        if (x > _max) _max = x;
        if (x < _min) _min = x;
        return (x);
    }

    size_t n()    const { return _n; }
    double min__()  const { return _min; }
    double max__()  const { return _max; }
    double mean() const { return _mean; }
    double std() const { return (_n > 2 ? sqrt(M2 / (_n - 1)) : 0.0); }
    double var() const { return (_n > 2 ? (M2 / (_n - 1)) : 0.0); }
    void momentVector(double* z) const { z[0] = mean(); z[1] = std(); }

private:
    double _min, _max, _mean, M2;
    size_t _n;
};

/// @brief Helper class for 4-th order moments calculation 
class Moments4 {

public:
    Moments4() { reset(); }
    void reset() { _mean = M2 = M3 = M4 = 0.0; _min = MAX_VAL; _max = MIN_VAL; _n = 0; }
    inline double add(const double x) {
        size_t n1;
        double delta, delta_n, delta_n2, term1;
        if (std::isnan(x) || x > MAX_VAL || x < MIN_VAL) return (x);

        n1 = _n;
        _n = _n + 1;
        delta = x - _mean;
        delta_n = delta / _n;
        delta_n2 = delta_n * delta_n;
        term1 = delta * delta_n * n1;
        _mean = _mean + delta_n;
        M4 = M4 + term1 * delta_n2 * (_n * _n - 3 * _n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3;
        M3 = M3 + term1 * delta_n * (_n - 2) - 3 * delta_n * M2;
        M2 = M2 + term1;
        M5 = M4 * delta;
        M6 = M4 * delta;

        if (x > _max) _max = x;
        if (x < _min) _min = x;
        return (x);
    }

    size_t n()         const { return _n; }
    double min__()       const { return _min; }
    double max__()       const { return _max; }
    double mean()      const { return _mean; }
    double std()      const { return (_n > 2 ? sqrt(M2 / (_n - 1)) : 0.0); }
    double var()      const { return (_n > 2 ? M2 / (_n - 1) : 0.0); }
    double skewness() const { return (_n > 3 ? (sqrt(_n) * M3) / pow(M2, 1.5) : 0.0); }
    double kurtosis() const { return (_n > 4 ? (_n * M4) / (M2 * M2) : 0.0); } // matlab does not subtract 3
    void momentVector(double* z) { z[0] = mean(); z[1] = std(); z[2] = skewness(); z[3] = kurtosis(); }
    double hyperskewness() const { return (_n > 5 ? (_n * M4) / (pow(M2, 5./2.)) : 0.0); }
    double hyperflatness() const { return (_n > 6 ? (_n * M5) / (M2*M2*M2) : 0.0); }

private:
    double _min, _max, _mean, M2, M3, M4, M5, M6;
    size_t _n;
};



