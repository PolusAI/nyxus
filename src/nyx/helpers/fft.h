/* dj_fft.h - public domain FFT library
by Jonathan Dupuy

   INTERFACING

   define DJ_ASSERT(x) to avoid using assert.h.

   QUICK NOTES

*/

#ifndef DJ_INCLUDE_FFT_H
#define DJ_INCLUDE_FFT_H

#include <complex> // std::complex
#include <vector>  // std::vector

namespace dj {

// FFT argument: std::vector<std::complex>
template <typename T> using fft_arg = std::vector<std::complex<T>>;

// FFT direction specifier
enum class fft_dir {DIR_FWD = +1, DIR_BWD = -1};

// FFT routines
template<typename T> fft_arg<T> fft2d(const fft_arg<T> &xi, const fft_dir &dir);

} // namespace dj

//
//
//// end header file ///////////////////////////////////////////////////////////


#include <cmath>
#include <cstdint>
#include <cstring> // std::memcpy

#ifndef DJ_ASSERT
#   include <cassert>
#   define DJ_ASSERT(x) assert(x)
#endif

namespace dj {

constexpr auto Pi = 3.141592653589793238462643383279502884;

/*
 * Returns offset to most significant bit
 * NOTE: only works for positive power of 2s
 * examples:
 * 1b      -> 0d
 * 100b    -> 2d
 * 100000b -> 5d
 */
inline int findMSB(int x)
{
    DJ_ASSERT(x > 0 && "invalid input");
    int p = 0;

    while (x > 1) {
        x>>= 1;
        ++p;
    }

    return p;
}


/*
 *  Bit reverse an integer given a word of nb bits
 *  NOTE: Only works for 32-bit words max
 *  examples:
 *  10b      -> 01b
 *  101b     -> 101b
 *  1011b    -> 1101b
 *  0111001b -> 1001110b
 */
inline int bitr(uint32_t x, int nb)
{
    DJ_ASSERT(nb > 0 && 32 > nb && "invalid bit count");
    x = ( x               << 16) | ( x               >> 16);
    x = ((x & 0x00FF00FF) <<  8) | ((x & 0xFF00FF00) >>  8);
    x = ((x & 0x0F0F0F0F) <<  4) | ((x & 0xF0F0F0F0) >>  4);
    x = ((x & 0x33333333) <<  2) | ((x & 0xCCCCCCCC) >>  2);
    x = ((x & 0x55555555) <<  1) | ((x & 0xAAAAAAAA) >>  1);

    return ((x >> (32 - nb)) & (0xFFFFFFFF >> (32 - nb)));
}

/*
 * Computes a 2D Fourier transform
 * with O(N^2 log N) complexity using the butterfly technique
 *
 * NOTE: the input must be a square matrix whose size is a power-of-two
 */
template <typename T> 
void r2r_fft2d_inplace(std::vector<T> &xi, const fft_dir &dir)
{   
    DJ_ASSERT((xi.size() & (xi.size() - 1)) == 0 && "invalid input size");
    int cnt2 = (int)xi.size();   // NxN
    int msb = findMSB(cnt2) / 2; // lg2(N) = lg2(sqrt(NxN))
    int cnt = 1 << msb;          // N = 2^lg2(N)
    T nrm = T(1) / T(cnt);
    fft_arg<T> xo(cnt2);

    // pre-process the input data
    for (int j2 = 0; j2 < cnt; ++j2)
    for (int j1 = 0; j1 < cnt; ++j1) {
        int k2 = bitr(j2, msb);
        int k1 = bitr(j1, msb);

        xo[j1 + cnt * j2] = nrm * std::complex<T>(xi[k1 + cnt * k2], 0);
    }

    // fft passes
    for (int i = 0; i < msb; ++i) {
        int bm = 1 << i; // butterfly mask
        int bw = 2 << i; // butterfly width
        float ang = T(dir) * Pi / T(bm); // precomputation

        // fft butterflies
        for (int j2 = 0; j2 < (cnt/2); ++j2)
        for (int j1 = 0; j1 < (cnt/2); ++j1) {
            int i11 = ((j1 >> i) << (i + 1)) + j1 % bm; // xmin wing
            int i21 = ((j2 >> i) << (i + 1)) + j2 % bm; // ymin wing
            int i12 = i11 ^ bm;                         // xmax wing
            int i22 = i21 ^ bm;                         // ymax wing
            int k11 = i11 + cnt * i21; // array offset
            int k12 = i12 + cnt * i21; // array offset
            int k21 = i11 + cnt * i22; // array offset
            int k22 = i12 + cnt * i22; // array offset

            // FFT-X
            std::complex<T> z11 = std::polar(T(1), ang * T(i11 ^ bw)); // left rotation
            std::complex<T> z12 = std::polar(T(1), ang * T(i12 ^ bw)); // right rotation
            std::complex<T> tmp1 = xo[k11];
            std::complex<T> tmp2 = xo[k21];

            xo[k11]+= z11 * xo[k12];
            xo[k12] = tmp1 + z12 * xo[k12];
            xo[k21]+= z11 * xo[k22];
            xo[k22] = tmp2 + z12 * xo[k22];

            // FFT-Y
            std::complex<T> z21 = std::polar(T(1), ang * T(i21 ^ bw)); // top rotation
            std::complex<T> z22 = std::polar(T(1), ang * T(i22 ^ bw)); // bottom rotation
            std::complex<T> tmp3 = xo[k11];
            std::complex<T> tmp4 = xo[k12];

            xo[k11]+= z21 * xo[k21];
            xo[k21] = tmp3 + z22 * xo[k21];
            xo[k12]+= z21 * xo[k22];
            xo[k22] = tmp4 + z22 * xo[k22];
        }
    }

    for (int i = 0; i < xo.size(); ++i) {
        xi[i] = std::abs(xo[i]);
    }
}


} // namespace dj

//
//
//// end inline file ///////////////////////////////////////////////////////////
#endif // DJ_INCLUDE_FFT_H

/*
------------------------------------------------------------------------------
This software is available under 2 licenses -- choose whichever you prefer.
------------------------------------------------------------------------------
ALTERNATIVE A - MIT License
Copyright (c) 2019 Jonathan Dupuy
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
ALTERNATIVE B - Public Domain (www.unlicense.org)
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or distribute this
software, either in source code form or as a compiled binary, for any purpose,
commercial or non-commercial, and by any means.
In jurisdictions that recognize copyright laws, the author or authors of this
software dedicate any and all copyright interest in the software to the public
domain. We make this dedication for the benefit of the public at large and to
the detriment of our heirs and successors. We intend this dedication to be an
overt act of relinquishment in perpetuity of all present and future rights to
this software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
------------------------------------------------------------------------------
*/