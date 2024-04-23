#pragma once

#include <vector>

inline std::vector<double> convolution2d(std::vector<unsigned int>& image,
                                std::vector<double>& kernel, 
                                int n_image, int m_image, 
                                int n_kernel, int m_kernel){

    std::vector<double> out(m_image * n_image, 0);


    int xKSize = n_kernel; // number of columns
    int yKSize = m_kernel; // number of rows

    int kernelCenterX = xKSize / 2.;
    int kernelCenterY = yKSize / 2.;

    int ikFlip, jkFlip;
    int ii, jj;

    for(int i = 0; i < m_image; ++i){
        for(int j = 0; j < n_image; ++j){
            for(int ik = 0; ik < yKSize; ++ik){
                ikFlip = yKSize - 1 - ik;
                for(int jk = 0; jk < xKSize; ++jk){
                    jkFlip = xKSize - 1 - jk;

                    ii = i + (kernelCenterY - ikFlip);
                    jj = j + (kernelCenterX - jkFlip);

                    if(ii >= 0 && ii < m_image && jj >= 0 && jj < n_image &&
                       ikFlip >= 0 && jkFlip >=0 && ikFlip < m_kernel && jkFlip < n_kernel){
                        out[i* n_image + j] += ((int)image[ii * n_image + jj]) * kernel[ikFlip * n_kernel + jkFlip];
                    }
                }
            }
        }
    }
    
    return out;
}