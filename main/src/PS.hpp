#pragma once

#include <complex>
#include <cmath>
#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <cassert>
#include <omp.h>

#include "heffte.h"

#include "paths.h"
#include "grid.hpp"



// function to compute the 3D-power spectrum for a given (grid)-data field
std::vector<std::complex<double>> computePS(const std::vector<double>& data, const int numGridPoints, const int Rank, const int numRanks)
{   
    // box of the full domain
    heffte::box3d<> AllIndizes({0,0,0},{numGridPoints-1,numGridPoints-1,numGridPoints-1});

    // find a processor grid that minimizes the total surface area of all boxes
    std::array<int,3> ProcessorGrid = heffte::proc_setup_min_surface(AllIndizes, numRanks);
    // print information on the grid

    // get all boxes by splitting the Indizes onto the Processor Grid
    // local Box stored at entry Rank
    std::vector<heffte::box3d<>> AllBoxes = heffte::split_world(AllIndizes,ProcessorGrid);
    heffte::box3d<> inbox   = AllBoxes[Rank];
    heffte::box3d<> outbox  = AllBoxes[Rank];

    // create fft plan
    heffte::plan_options options    = heffte::default_options<heffte::backend::fftw>();
    // use pencils format for intermediate computation steps
    // this works better for a large number of MPI ranks
    options.use_pencils             = true;
    // create fftw fft routine
    heffte::fft3d<heffte::backend::fftw> fft(inbox, outbox, MPI_COMM_WORLD, options);

    // create result container
    std::vector<std::complex<double>> output(fft.size_outbox());
    assert(("ERROR: wrong input sizes for fft", fft.size_outbox() == data.size()));

    // compute the fft
    fft.forward(data.data(), output.data());

    return output;
}

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

// select the k-bin for a given grid point
// used for binning according to Rubens version
int select_bin(const double i){
    const int min = std::floor(i);
    const int max = min + 1;

    const double d1 = i - static_cast<double>(min);
    const double d2 = static_cast<double>(max) - i;

    int result = -1;
    if (d1 < d2){
        result = min;
    } else {
        result = max;
    }
    return result;
}

// funtion for sorting the PS into radial bins
// input:       k1D             : 1D k-values
//              PS1Dcomp        : complex Fourier-values
//              numGridPoints   : number of grid points per dimension
//              numBins         : number of PS bins
//              rank, numRanks  : local rank ID & total number of ranks
// output:      k               : k-values for the 1D power spectrum
//              PS              : 1D power spectrum  values that belong to k
//              NumPoints       : number of grid points that contributed to each bin
std::tuple<std::vector<double>,std::vector<double>,std::vector<double>> PSbinning(
                    const std::vector<double> k1D, const std::vector<std::complex<double>> PS1Dcomp, const int numGridPoints,
                    const int numBins, const int Rank, const int numRanks)
{   
    // container for the absolute values of the power spectrum
    std::vector<double> PS1D(PS1Dcomp.size(), 0.0);
    // normalize the fft results by the number of grid points & compute the absolute value of the complex result
    const double totalNumGridPoints = std::pow(static_cast<double>(numGridPoints), 3);
    #pragma omp parallel for
    for(size_t iGrid = 0; iGrid < PS1Dcomp.size(); iGrid++){
        PS1D[iGrid] = std::abs(PS1Dcomp[iGrid]) / totalNumGridPoints;
    }

    // maximum k-value that should be binned
    // Niquist frequency
    const double kMax = numGridPoints/2;
    // binned k output
    std::vector<double> k(numBins);
    // binned PS output
    std::vector<double> PS(numBins);
    // number of grid-points per bin
    std::vector<double> NumPoints(numBins);
    // width of each bin
    const double BinWidth = kMax / static_cast<double>(numBins);

    // box of the full domain
    heffte::box3d<> AllIndizes({0,0,0},{numGridPoints-1,numGridPoints-1,numGridPoints-1});
    // find a processor grid that minimizes the total surface area of all boxes
    std::array<int,3> ProcessorGrid = heffte::proc_setup_min_surface(AllIndizes, numRanks);
    // get all boxes by splitting the Indizes onto the Processor Grid
    // local Box stored at entry Rank
    std::vector<heffte::box3d<>> AllBoxes = heffte::split_world(AllIndizes,ProcessorGrid);

    // loop over the local grid points
    for (size_t ix = 0; ix < AllBoxes[Rank].size[0]; ix++){
        for (size_t iy = 0; iy < AllBoxes[Rank].size[1]; iy++){
            for (size_t iz = 0; iz < AllBoxes[Rank].size[2]; iz++){
                uint64_t GridIndex = getGridIndex(ix+AllBoxes[Rank].low[0],iy+AllBoxes[Rank].low[1],iz+AllBoxes[Rank].low[2],
                                        AllBoxes[Rank].low[0],AllBoxes[Rank].high[0],
                                        AllBoxes[Rank].low[1],AllBoxes[Rank].high[1],
                                        AllBoxes[Rank].low[2],AllBoxes[Rank].high[2]);
                double kvalue = std::sqrt( std::pow(k1D[ix+AllBoxes[Rank].low[0]],2)+ 
                                           std::pow(k1D[iy+AllBoxes[Rank].low[1]],2)+ 
                                           std::pow(k1D[iz+AllBoxes[Rank].low[2]],2));
                // if (kvalue < kMax){
                //     int iBin = std::floor(kvalue / BinWidth);
                //     k[iBin]         += kvalue;
                //     PS[iBin]        += PS1D[GridIndex];
                //     NumPoints[iBin] += 1.0;

                // matching Rubéns binning
                int iBin = select_bin(kvalue);
                if (iBin < k.size()){
                    k[iBin]         += static_cast<double>(iBin);
                    PS[iBin]        += PS1D[GridIndex];
                    NumPoints[iBin] += 1.0;
                }
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // collapse the local results
    collapse(k,         'd');
    collapse(PS,        'd');
    collapse(NumPoints, 'd');
    MPI_Barrier(MPI_COMM_WORLD);

    // normalize the k and PS vectors
    if ( Rank == 0){
        #pragma omp parallel for
        for (size_t iBin = 0; iBin < numBins; iBin++){
            if (NumPoints[iBin] > 0){
                k[iBin]  =                              k[iBin] / NumPoints[iBin];
                PS[iBin] = 4*M_PI*std::pow(k[iBin],2) * PS[iBin] / NumPoints[iBin] / std::pow(numGridPoints,3);
            } else {
                k[iBin]  = (static_cast<double>(iBin) + 0.5) * BinWidth;
                PS[iBin] = 0.0;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    return std::make_tuple(k, PS, NumPoints);
}

void save_PS(const std::vector<double> k, const std::vector<double> PS, const std::vector<double> NumPoints,
            const std::string SimFile, const double SimEkin, const double GridEkin, 
            const double time, const std::string key, const int stepNo, const bool NearestNeighbor, 
            filePathsCls filePaths, const int numGridPoints)
{
    // Output filename 
    std::string filename = "PS-" + key + "-" + std::to_string(stepNo) + ".txt";
    // assemble the output path
    std::string path = filePaths.getPSPath(filename);

    // open the file
    std::ofstream outFile(path);
    if (outFile.is_open())
    {   
        // header
        outFile << "# datakey           : " << key << std::endl;
        outFile << "# simfile           : " << SimFile << std::endl;
        outFile << "# stepNo            : " << stepNo << std::endl;
        outFile << "# time              : " << time << std::endl;
        outFile << "# nearestNeighbor   : " << NearestNeighbor << std::endl;
        outFile << "# numGridPoints     : " << numGridPoints << std::endl;
        outFile << "# SimEkin           : " << SimEkin << std::endl;
        outFile << "# GridEkin          : " << GridEkin << std::endl;
        outFile << "### k   PS(" << key << ")   numPoint" << std::endl;
    
        // loop over the bins 
        for (size_t iBin = 0; iBin < k.size(); iBin++){
            //mean data in each bin
            outFile << k[iBin] << " ";
            // number of grid Points n each bin
            outFile << PS[iBin] << " ";
            // total number of ceonnections per bin
            outFile << NumPoints[iBin] << std::endl;
        }
        // close the file
        outFile.close();
    }
    return;
}


// this function takes the 3D-grid in Fourier space and cuts out a sphere
// input:   data of the 3D-grid in Fourier space
//          1D k-vector to get the k-values
//          radius of the sphere in k-space
//          do you want to get the space inside (true) or outside (false) of the sphere
//          number of gridpoints per dimension
//          number of the local rank & total number of ranks
//
// output:  modified data in Fourier space only containing the selected volume
std::vector<std::complex<double>> cutFourierSphere(const std::vector<std::complex<double>> data,  const std::vector<double> k1D, const double kRadius, const bool UseInner,
                                    const int numGridPoints,
                                    const int Rank, const int numRanks)
{   
    // result vector for the cut version of the grid in Fourier space
    std::vector<std::complex<double>> result = data;

    // get the domain-decomposition
    heffte::box3d<> AllIndizes({0,0,0},{numGridPoints-1,numGridPoints-1,numGridPoints-1});
    std::array<int,3> ProcessorGrid = heffte::proc_setup_min_surface(AllIndizes, numRanks);
    std::vector<heffte::box3d<>> AllBoxes = heffte::split_world(AllIndizes,ProcessorGrid);

    // complex zero value to remove unwanted data
    std::complex<double> zero(0.0,0.0);
    // loop over the local grid points
    for (size_t ix = 0; ix < AllBoxes[Rank].size[0]; ix++){
        for (size_t iy = 0; iy < AllBoxes[Rank].size[1]; iy++){
            for (size_t iz = 0; iz < AllBoxes[Rank].size[2]; iz++){
                double kvalue = std::sqrt( std::pow(k1D[ix+AllBoxes[Rank].low[0]],2)+ 
                                           std::pow(k1D[iy+AllBoxes[Rank].low[1]],2)+ 
                                           std::pow(k1D[iz+AllBoxes[Rank].low[2]],2));
                bool Remove = (kvalue <= kRadius);
                if (UseInner){Remove = not Remove;}
                // if the point was selected to be removed, set it to zero
                if (Remove){
                    uint64_t GridIndex = getGridIndex(ix+AllBoxes[Rank].low[0],iy+AllBoxes[Rank].low[1],iz+AllBoxes[Rank].low[2],
                                                    AllBoxes[Rank].low[0],AllBoxes[Rank].high[0],
                                                    AllBoxes[Rank].low[1],AllBoxes[Rank].high[1],
                                                    AllBoxes[Rank].low[2],AllBoxes[Rank].high[2]);
                    result[GridIndex] = zero;
                }
            }
        }
    }

    return result;
}


std::vector<double> inverseTransform(std::vector<std::complex<double>> data, const int numGridPoints,
                                    const int Rank, const int numRanks)
{
    // get the domain-decomosition
    heffte::box3d<> AllIndizes({0,0,0},{numGridPoints-1,numGridPoints-1,numGridPoints-1});
    std::array<int,3> ProcessorGrid = heffte::proc_setup_min_surface(AllIndizes, numRanks);
    std::vector<heffte::box3d<>> AllBoxes = heffte::split_world(AllIndizes,ProcessorGrid);
    heffte::box3d<> inbox   = AllBoxes[Rank];
    heffte::box3d<> outbox  = AllBoxes[Rank];

    // create fft plan
    heffte::plan_options options    = heffte::default_options<heffte::backend::fftw>();
    options.use_pencils             = true;
    // create fftw fft routine
    heffte::fft3d<heffte::backend::fftw> fft(inbox, outbox, MPI_COMM_WORLD, options);

    // create result container
    std::vector<std::complex<double>> output(fft.size_outbox());
    std::vector<double> ABSoutput(data.size(), 0.0);
    assert(("ERROR: wrong input sizes for fft", fft.size_outbox() == data.size()));

    // compute the fft
    fft.backward(data.data(), output.data());

    // // normalize the fft results by the number of grid points
    const double totalNumGridPoints = std::pow(static_cast<double>(numGridPoints), 3);
    #pragma omp parallel for
    for(size_t iGrid = 0; iGrid < output.size(); iGrid++){
        ABSoutput[iGrid] = std::abs(output[iGrid]) / totalNumGridPoints;
    }

    return ABSoutput;
}