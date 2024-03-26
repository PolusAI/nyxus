#include <iostream>
#include <complex>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <vector>

#define _USE_MATH_DEFINES
#include <math.h>

using cd = std::complex<double>;
const double PI = acos(-1);

void _fft_helper(std::vector<cd> & a, bool invert);

template <class T>
void fft(std::vector<T> & a) {

    if (std::is_same<T, cd>::value) _fft_helper(a, false);

    std::vector<cd> vec;

    for (const auto& num: a) {

        vec.push_back(std::complex(num, 0));
    }

    _fft_helper(vec, false);
};

void inverse_fft(std::vector<cd> & a) {
    _fft_helper(a, true);
};

void _fft_helper(std::vector<cd> & a, bool invert) {
    int n = a.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            std::swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i+j], v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (cd & x : a)
            x /= n;
    }
}