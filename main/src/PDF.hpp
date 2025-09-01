#pragma once

#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <tuple>
#include <stdint.h>
#include <cmath>

#include <optional>
#include "statistics.hpp"

#include "paths.hpp"
#include "MPI_comms.hpp"


// function to save the results of a PDF to a txt file (ONLY execute at rank 0)
// inputs:  filename        : name of the file where the PDF should be saved (NO path, ONLY filename) (default: "" -> automatic filename is created using quantity)
//          quantity        : quantity for which this PDF was computed
//          x,y,N           : PDF data (output of the PDF() function below)
//          filePaths       : object that contains the paths to where the result files should be saved
//          massWeighted    : bool, is ths a mass-weighted PDF? (true: mass-weighted (default); false: volume-weighted)
//          simFile         : name of the simulation file that was analysed
//          stepNo          : step number in the simulation file
//          time            : physical time of the main timestep
//          numPoints       : total number of SPH-particles (mass-weighted) / total number of grid points (volume-weighted)
//          nearestNeighbor : bool, was nearestNeighbor mapping used for the grid? (true: nearest neighbor (default); false: SPH)
//          ftleFile        : filename of the second FTLE step (default: "")
//          ftleStepNo      : step number in the ftleFile for the second FTLE step (default: 0)
//          ftleTime        : physical time of the ftle timestep (default: 0.0)
// output:  NONE, the data is written to a .txt file
void savePDF(const std::string quantity, const std::vector<double>& x, const std::vector<double>& y, const std::vector<uint64_t>& N,
            filePathsCls filePaths, const bool massWeighted, const std::string simFile, const int stepNo, const double time, 
            const uint64_t numPoints,
            const std::optional<statisticsCls>& stats = std::nullopt,
            const bool nearestNeighbor = true, 
            const std::string ftleFile = "", const int ftleStepNo = 0, const double ftleTime = 0.0, 
            std::string filename="")
{
    // check inputs
    assert(("to save the PDF, x must have more than size 0", x.size() > 0));
    assert(("to save the PDF, x and y must have the same size", x.size() == y.size()));
    assert(("to save the PDF, x and N must have the same size", x.size() == N.size()));
    assert(("to save the PDF, stepNo must be non-negative", stepNo >= 0));
    assert(("to save the PDF, ftleStepNo must be non-negative", ftleStepNo >= 0));
    assert(("to save the PDF, the quantity input must not be an empty string", quantity.length() > 0));
    assert(("to save the PDF, the simFile input must not be an empty string", simFile.length() > 0));
    // get the filename
    if (filename.length() == 0){
        if (massWeighted){
            filename = "mass-PDF-" + quantity + "-" + std::to_string(stepNo) + ".txt";
        } else {
            filename = "volume-PDF-" + quantity + "-" + std::to_string(stepNo) + ".txt";
        }
    }
    // get the path to the file
    std::string path = filePaths.getPDFPath(massWeighted,filename);
    
    // write the file
    std::ofstream outFile(path);
    if (outFile.is_open())
    {   
        // header
        outFile << "# variable  : " << quantity << std::endl;
        outFile << "# simfile   : " << simFile << std::endl;
        outFile << "# stepNo    : " << stepNo << std::endl;
        outFile << "# time      : " << time << std::endl;
        outFile << "# numPoints : " << numPoints << std::endl;
        outFile << "# mass-weighted : " << massWeighted << std::endl;
        if (not massWeighted){outFile << "# nearest-neighbor : " << nearestNeighbor << std::endl;}
        if (ftleFile.length() > 0){
            outFile << "# ftleFile   : " << ftleFile << std::endl;
            outFile << "# ftleStepNo : " << ftleStepNo << std::endl;
            outFile << "# ftleTime   : " << ftleTime << std::endl;
        }
        if (stats){
            outFile << "# rms  : " << stats->getRMS()  << std::endl;
            outFile << "# mean : " << stats->getMEAN() << std::endl;
            outFile << "# std  : " << stats->getSTD()  << std::endl;
        }
        outFile << "### x     y     #N" << std::endl;
     
        // loop over the bins 
        for (size_t iBin = 0; iBin < x.size(); iBin++){
            //mean data in each bin
            outFile << x[iBin] << " ";
            // number of grid Points n each bin
            outFile << y[iBin] << " ";
            // total number of ceonnections per bin
            outFile << N[iBin] << std::endl;
        }
        // close the file
        outFile.close();
    }
    return;
}

// function to compute the PDF of a given data vector
// inputs:  data            : data vector for which the PDF will be computed
//          min             : minimum data value (lower bound of the PDF). if logScale=true the bound is 10^min
//          max             : maximum data value (upper bound of the PDF). if logScale=true the bound is 10^max
//          numBins         : number of Bins in the histogram
//          numParticles    : total number of SPH-particles/Grid-points in the simulation
//          rank            : local rank on which this process lives
//          logScale        : do you want a linear or logarithmic equal distribution of the bins (false: linear(default) ; true: logarithmic)
//          numProcessors   : number of Processors for parallel computing (default = 1)
// output:  data values associated with the bins
//          PDF values associated with the bins
//          number of contributing particles to each bin
template<typename T>
std::tuple<std::vector<double>,std::vector<double>,std::vector<uint64_t>> PDF(const std::vector<T>& data, 
                                                        const double min, const double max, const int numBins, const uint64_t numParticles, const int rank, 
                                                        const bool logScale = false, const int numProcessors = 1)
{   
    // check inputs
    assert(("max must be larger than min in PDF()", max > min));
    assert(("numBins must be larger than zero in PDF()", numBins > 0));
    if (logScale){
        assert(("to use logscale (linearScale=false) in PDF(), min must be larger than zero", min > 0));
        for (size_t i = 0; i < data.size(); i++){
            assert(("to use logscale (linearScale=false) in PDF(), all data values must be larger than zero", data[i] > 0));
        }
    }
    // result container x-values of the plot
    std::vector<double> x;
    // result container y-values of the plot
    std::vector<double> y;
    // result container number of particles per Bin
    std::vector<uint64_t> N(numBins, 0);
    // lower bound of the PDF
    double lowerBound;
    // upper bound of the PDF
    double upperBound;
    if (logScale){
        lowerBound = std::pow(10,min);
        upperBound = std::pow(10,max);
    } else {
        lowerBound = min;
        upperBound = max;
    }
    // size of the bins
    const double binSize = (static_cast<double>(max) - static_cast<double>(min)) / static_cast<double>(numBins);
    // number of local particles per processor
    uint64_t numPartsPerProc = static_cast<uint64_t>(std::ceil(static_cast<double>(data.size()) / static_cast<double>(numProcessors)));
    // distributed result container
    std::vector<std::vector<uint64_t>> distN(numProcessors);
    for (size_t i = 0; i < numProcessors; i++){
        distN[i].resize(numBins);
        for (size_t j = 0; j < numBins; j++){
            distN[i][j] = 0;
        }
    }

    // compute the local PDF
    if (logScale){
        #pragma omp parallel for
        for (size_t iProc = 0; iProc < numProcessors; iProc++){
            for (size_t iPart = iProc * numPartsPerProc; (iPart < (iProc+1)*numPartsPerProc) && (iPart < data.size()); iPart++){
                if ((static_cast<double>(data[iPart]) >= lowerBound) && (static_cast<double>(data[iPart]) < upperBound)){
                    int iBin = static_cast<int>(std::floor((std::log10(static_cast<double>(data[iPart])) - min) / binSize));
                    distN[iProc][iBin]++;
                }
            }
        }
    } else {
        #pragma omp parallel for
        for (size_t iProc = 0; iProc < numProcessors; iProc++){
            for (size_t iPart = iProc * numPartsPerProc; (iPart < (iProc+1)*numPartsPerProc) && (iPart < data.size()); iPart++){
                if ((static_cast<double>(data[iPart]) >= lowerBound) && (static_cast<double>(data[iPart]) < upperBound)){
                    int iBin = static_cast<int>(std::floor((static_cast<double>(data[iPart]) - lowerBound) / binSize));
                    distN[iProc][iBin]++;
                }
            }
        }
    }
    // combine distributed results
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t iBin = 0; iBin < numBins; iBin++){
            N[iBin] += distN[iProc][iBin];
        }
    }
    // memory cleanup
    distN.clear();
    distN.shrink_to_fit();


    // collapse results to rank 0
    collapse(N, 'u');

    // finish the results
    if (rank == 0){
        // create x values
        x.resize(numBins);
        if (logScale){
            for (size_t iBin = 0; iBin < numBins; iBin++){x[iBin] = std::pow(min + (static_cast<double>(iBin) + 0.5)*binSize ,2);}
        } else {
            for (size_t iBin = 0; iBin < numBins; iBin++){x[iBin] = min + (static_cast<double>(iBin) + 0.5)*binSize;}
        }
        // create PDF y values
        y.resize(numBins);
        if (logScale){
            for (size_t iBin = 0; iBin < numBins; iBin++){
                double binWidth = std::pow(min+iBin*binSize,2) - std::pow(min+(iBin+1*binSize),2);
                y[iBin] = static_cast<double>(N[iBin]) / (static_cast<double>(numParticles) * binWidth);
            }
        } else {
            for (size_t iBin = 0; iBin < numBins; iBin++){
                y[iBin] = static_cast<double>(N[iBin]) / (static_cast<double>(numParticles) * binSize);
            }
        }
    }


    return std::make_tuple(x,y,N);
}