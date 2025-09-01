#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

#include "grid.hpp"
#include "MPI_comms.hpp"
#include "paths.hpp"
#include "heffte.h"


// function to output a slice of the grid as .txt file for plotting
template<typename T>
void plotBoxSlice(const std::vector<T> data, const char orthogonal, const int layer, const std::string key,
                const int numGridPoints, const double time, const int rank, const int numRanks, 
                const std::string simFile, const int stepNo, filePathsCls filePaths, const bool nearestNeighbor,
                const std::string ftleFile = "", const int ftleStepNo = 0,
                std::string filename = "")
{   
    assert(("in plotBoxSlice, orthogonal input must be 'x', 'y' or 'z'", (((orthogonal == 'x') || (orthogonal == 'y')) || (orthogonal == 'z'))));
    assert(("in plotBoxSlice, time must be larger than 0", time > 0));
    assert(("in plotBoxSlice, rank must be smaller than numRanks", rank < numRanks));
    assert(("in plotBoxSlice, layer must be smaller than numGridPoints", layer < numGridPoints));
    // size of the box
    const double boxSize = 1.0;
    // step size of the grid
    const double gridStep = boxSize / static_cast<double>(numGridPoints);

    // creating result buffer
    std::vector<double> result(std::pow(numGridPoints,2), 0.0);

    // get the local grid section
    int dim;
    if (orthogonal == 'x'){dim = 0;}
    else if (orthogonal == 'y'){dim = 1;}
    else if (orthogonal == 'z'){dim = 2;}
    heffte::box3d<> allIndizes({0,0,0},{numGridPoints-1,numGridPoints-1,numGridPoints-1});
    std::array<int,3> processorGrid = heffte::proc_setup_min_surface(allIndizes, numRanks);
    std::vector<heffte::box3d<>> allBoxes = heffte::split_world(allIndizes,processorGrid);
    const int imin     = static_cast<unsigned>(allBoxes[rank].low[dim]);
    const int imax     = static_cast<unsigned>(allBoxes[rank].high[dim]);
    // does this rank contain relevant data?
    const bool contributing = (imin <= layer) and (layer <= imax);
    // finding relevant datapoints on the local data
    if (contributing){
        // find local boundaries along the plane
        std::vector<double> min(2, 0.0);
        std::vector<double> max(2, 0.0);
        int temp = 0;
        for (size_t i = 0; i < 3; i++){
            if (static_cast<int>(i) != dim){
                min[temp] = allBoxes[rank].low[i];
                max[temp] = allBoxes[rank].high[i];
                temp++;
            }
        }
        

        // loop over local points
        // pragma omp parallel for
        for (size_t i1 = min[0]; i1 <= max[0]; i1++){
            for (size_t i2 = min[1]; i2 <= max[1]; i2++){
                // get gridIndex for accessing the grid
                uint64_t gridIndex;
                switch (dim){
                    case 0:
                        gridIndex = getGridIndex(layer,i1,i2, 
                                                allBoxes[rank].low[0],allBoxes[rank].high[0],
                                                allBoxes[rank].low[1],allBoxes[rank].high[1],
                                                allBoxes[rank].low[2],allBoxes[rank].high[2]);
                        break;
                    case 1:
                        gridIndex = getGridIndex(i1,layer,i2, 
                                                allBoxes[rank].low[0],allBoxes[rank].high[0],
                                                allBoxes[rank].low[1],allBoxes[rank].high[1],
                                                allBoxes[rank].low[2],allBoxes[rank].high[2]);
                        break;
                    case 2:
                        gridIndex = getGridIndex(i1,i2,layer, 
                                                allBoxes[rank].low[0],allBoxes[rank].high[0],
                                                allBoxes[rank].low[1],allBoxes[rank].high[1],
                                                allBoxes[rank].low[2],allBoxes[rank].high[2]);
                        break;
                }
                // get resIndex for accessing the result
                uint64_t resIndex = i1*numGridPoints + i2;
                // assign the data
                if (resIndex >= result.size()){std::cout <<rank<<"res ERROR" << std::endl;}
                if (gridIndex >=  data.size()){std::cout <<rank<< ": data ERROR "<<i1<<" "<<i2<<" "<<layer<<" "<<allBoxes[rank].low[0]<<"-"<<allBoxes[rank].high[0]<<" "<<allBoxes[rank].low[1]<<"-"<<allBoxes[rank].high[1]<<" "<<allBoxes[rank].low[2]<<"-"<<allBoxes[rank].high[2]<<" | " <<gridIndex<< "/" << data.size() << std::endl;}
                result[resIndex] = data[gridIndex];
            }
        }
    }


    // collapsing results to rank 0
    collapse(result, 'd');

    // printing the results to file
    if (rank == 0){
        // creating the filename if none was provided
        if (filename.length() == 0){filename = "box-slice-" + key + "-" + std::to_string(stepNo) + "-" + orthogonal + "-" + std::to_string(layer) + ".txt";}
        // adding the correct path
        filename = filePaths.getSlicesPath(filename);

        // save to file
        std::ofstream outFile(filename);
        if (outFile.is_open()){
            // header
            outFile << "# datakey    : " << key << std::endl;
            outFile << "# simFile    : " << simFile << std::endl;
            outFile << "# stepNo     : " << stepNo << std::endl;
            outFile << "# time       : " << time << std::endl;
            outFile << "# orthogonal : " << orthogonal << std::endl;
            outFile << "# layer      : " << layer << std::endl;
            outFile << "# height     : " << (layer+0.5)*gridStep - boxSize/2 << std::endl;
            outFile << "# numGridPoints   : " << numGridPoints   << std::endl;
            outFile << "# nearestNeighbor : " << nearestNeighbor << std::endl;
            if (ftleFile.length() > 0){
                outFile << "# ftleFile:  " << ftleFile << std::endl;
                outFile << "# ftleStepNo:" << ftleStepNo << std::endl;
            }
            // data
            for (size_t i = 0; i < std::pow(numGridPoints,2); i++){outFile << result[i] << std::endl;}
        }   
    }

    return;
}