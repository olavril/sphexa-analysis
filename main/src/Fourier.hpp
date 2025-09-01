#pragma once 

#include <complex>
#include <vector>
#include <cassert>









// compute the k frequencies belonging to the computed power-spectra (1D)
std::vector<double> computeK(const int NumGridPoints)
{   
    // resulting k-vector
    std::vector<double> k(NumGridPoints,0.0);

    // assigning the k-values
    if (NumGridPoints % 2 == 0){
        for (size_t iGrid = 0; iGrid < NumGridPoints/2; iGrid++){
            k[iGrid] = static_cast<double>(iGrid);
        }
        for (size_t iGrid = NumGridPoints/2; iGrid < NumGridPoints; iGrid++){
            k[iGrid] = static_cast<double>(iGrid) - static_cast<double>(NumGridPoints);
        }
    } else {
        for (size_t iGrid = 0; iGrid < (NumGridPoints-1)/2; iGrid++){
            k[iGrid] = static_cast<double>(iGrid);
        }
        for (size_t iGrid = (NumGridPoints-1)/2; iGrid < NumGridPoints; iGrid++){
            k[iGrid] = static_cast<double>(iGrid) - static_cast<double>(NumGridPoints) + 1;
        }
    }
    return k;
}

