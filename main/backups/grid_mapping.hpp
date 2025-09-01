#pragma once

#include <iostream>
#include <cmath>
#include <vector>
#include <stdint.h>
#include <chrono>
#include <cassert>

#include "periodic_conditions.hpp"
#include "MPI_comms.hpp"

#include "heffte.h"

// find out if a particles interpolation sphere (given by its smoothing length) overlaps with
// the grid region of a given rank
// input:       minPart, minimum index of the particles interpolation sphere
//              maxPart, maximum index of the particles interpolation sphere
//              minRank, minimum index of the particles interpolation sphere
//              maxRank, maximum index of the particles interpolation sphere
//              numGridPoints, number of grid points per dimension
// output:      bool (true: does overlap; false: no overlap)
bool doesOverlap(const int minPart, const int maxPart, const int minRank, const int maxRank, const int numGridPoints)
{   
    // does the particle sphere overlap with the rank volume
    bool overlap = false;
    // determine, if there is an overlap
    if ((minRank <= minPart) && (minPart <= maxRank)){overlap = true;} 
    else if ((minRank <= maxPart) && (maxPart <= maxRank)){overlap = true;} 
    else if ((minPart <= minRank) && (maxRank <= maxPart)){overlap = true;} 
    else {
        const int CorrMinPart = correct_periodic_cond(minPart, numGridPoints);
        const int CorrMaxPart = correct_periodic_cond(maxPart, numGridPoints);
        if (CorrMinPart > CorrMaxPart){
            if (minRank <= CorrMaxPart){overlap = true;} 
            else if (maxRank >= CorrMinPart){overlap = true;}
        }
    }
    return overlap;
}

// find out if a particle is contributing to the grid section on a given rank (true: is contributing; false: not contributing)
// input:       x, y, z position of the particle
//              h, smoothing length of the particle
//              halfBox, half the size of the box (usually 0.5)
//              gridStep, stepsize of the grid
//              ixmin, ixmax, ... grid index ranges of the rank that is checked
//              numGridPoints, number of grid points per dimension
// output:      bool (true: is contributing; false: not contributing)
bool isContributing(const double x, const double y, const double z, const double h, 
                    const double halfBox, const double gridStep,
                    const int ixRankMin, const int ixRankMax, const int iyRankMin, const int iyRankMax, const int izRankMin, const int izRankMax,
                    const int numGridPoints)
{   
    // term to shift the positions
    const double CorrectionTerm = halfBox - 0.5*gridStep; 
    // borders of the particle contribution sphere
    int ixPartMin, ixPartMax, iyPartMin, iyPartMax, izPartMin, izPartMax;
    // compute the borders of the particle contribution sphere
    ixPartMin = std::ceil(( x - 2*h + CorrectionTerm ) / gridStep);
    ixPartMax = std::floor((x + 2*h + CorrectionTerm ) / gridStep);
    iyPartMin = std::ceil(( y - 2*h + CorrectionTerm ) / gridStep);
    iyPartMax = std::floor((y + 2*h + CorrectionTerm ) / gridStep);
    izPartMin = std::ceil(( z - 2*h + CorrectionTerm ) / gridStep);
    izPartMax = std::floor((z + 2*h + CorrectionTerm ) / gridStep);
    // find out if the particle contribution sphere overlaps with the box
    bool xOverlap, yOverlap, zOverlap;
    // check overlaps
    xOverlap = doesOverlap(ixPartMin,ixPartMax, ixRankMin,ixRankMax, numGridPoints);
    yOverlap = doesOverlap(iyPartMin,iyPartMax, iyRankMin,iyRankMax, numGridPoints);
    zOverlap = doesOverlap(izPartMin,izPartMax, izRankMin,izRankMax, numGridPoints);
    return ((xOverlap && yOverlap) && zOverlap);
}


// function to create sendIndexLists
// input:       xSim, ySim, zSim    : particle positions of the local SPH-particles
//              hSim                : smoothing lengths of the local SPH-particles
//              numSend             : vector containing the number of particles that this rank sends to each other rank
//              sendPlan            : send Plan for communication
//              allBoxes            : box information for each rank
//              halfBox             : half the side length of the simulation box
//              gridStep            : grid Step size
//              numGridPoints       : number of grid points per dimension
//              numRanks            : number of ranks for this process
// output:      sendIndexLists      : a 2D vector containing indizes for creating send-buffers (access via sendIndexLists[iCommStep][iParticle])
std::vector<std::vector<uint64_t>> createSendIndexLists(const std::vector<double> xSim, const std::vector<double> ySim, const std::vector<double> zSim,
            const std::vector<double> hSim, const std::vector<uint64_t> numSend,
            const std::vector<int> sendPlan, std::vector<heffte::box3d<>> allBoxes,
            const double halfBox, const double gridStep, const int numGridPoints, const int numRanks)
{   
    assert(("in createSendIndexLists xSim & ySim must habve the same size", xSim.size() == ySim.size()));
    assert(("in createSendIndexLists xSim & zSim must habve the same size", xSim.size() == zSim.size()));
    assert(("in createSendIndexLists xSim & hSim must habve the same size", xSim.size() == hSim.size()));
    // result container
    std::vector<std::vector<uint64_t>> sendIndexLists(sendPlan.size());
    for (size_t iStep = 0; iStep < sendPlan.size(); iStep++){
        int sendRank = sendPlan[iStep];
        uint64_t numSendParticles = 0;
        if (sendRank >= 0){numSendParticles = numSend[sendRank];}
        sendIndexLists[iStep].resize(numSendParticles);
        if (numSendParticles > 0){for (size_t i = 0; i < numSendParticles; i++){sendIndexLists[iStep][i] = 0;}}
    }

    // map, to match a rank to its communication step according to the sendPlan (rankStep[rank] = commStep)
    std::vector<int> rankStep(sendPlan.size(), -1);
    for (size_t iStep = 0; iStep < sendPlan.size(); iStep++){rankStep[sendPlan[iStep]] = iStep;}

    // next open position in each sendList
    std::vector<uint64_t> nextOpenPos(sendPlan.size(), 0);
    // creating the index Lists
    for (size_t iPart = 0; iPart < xSim.size(); iPart++){
        //#pragma omp parallel for
        for (size_t iRank = 0; iRank < numRanks; iRank++){
            bool contributing  = isContributing(xSim[iPart],ySim[iPart],zSim[iPart],hSim[iPart],halfBox,gridStep,
                                                allBoxes[iRank].low[0],allBoxes[iRank].high[0],
                                                allBoxes[iRank].low[1],allBoxes[iRank].high[1],
                                                allBoxes[iRank].low[2],allBoxes[iRank].high[2], numGridPoints);
            if (contributing){
                int iStep = rankStep[iRank];
                sendIndexLists[iStep][nextOpenPos[iStep]] = static_cast<uint64_t>(iPart);
                nextOpenPos[iStep]++;
            }
        }
    }
    // return the index lists
    return sendIndexLists;
}


// function to communicate particle data before interpolating the grid
// input:       data            : sph-particle data vector that should be communicated (will be manipulated in-place)
//              sendIndexLists  : SendIndexLists 2D vector for creating the send Buffers
//              numRecv         : number of particles that will be recieved from each other rank
//              send/recvPlan   : communication plans
//              numLocalParts   : number of particles that should be on this rank after communicaation
//              rank, numRanks  : rank on which this process lives & total number of ranks
//              commMode        : char, identifying the type of the data ('d': double (default); 'u': uint64_t; 'i': int)
// output:      None, the data vector will be manipulated
template<typename T>
void dataComm(std::vector<T>& data, const std::vector<std::vector<uint64_t>> sendIndexLists, std::vector<uint64_t> numRecv,
              const std::vector<int> sendPlan, const std::vector<int> recvPlan, const uint64_t numLocalParts, const int rank, const int numRanks,
              char commMode = 'd')
{   
    assert(("in dataComm commMode must be 'd', 'u' or 'i'", (((commMode == 'd') || (commMode == 'u')) || (commMode == 'i'))));
    assert(("in dataComm sendPlan & recvPlan must have the same size", sendPlan.size() == recvPlan.size()));
    assert(("in dataComm sendPlan & recvPlan must have the same size", sendPlan.size() == recvPlan.size()));

    std::vector<T> sendBuffer;
    std::vector<T> recvBuffer;
    int numSending;
    int numRecieving;
    // number of particles that have been recieved & sucessfully sorted in
    uint64_t numFinished = 0;
    // result buffer
    std::vector<T> resBuffer(numLocalParts, 0.0);
    for (size_t iStep = 0; iStep < sendPlan.size(); iStep++){
        // rank to which this rank sends its data
        int sendRank = sendPlan[iStep];
        // rank from which this rank recieves data
        int recvRank = recvPlan[iStep];
        // is this rank sending / recieving during the current comm step?
        bool isSending   = sendRank >= 0;
        bool isRecieving = recvRank >= 0;
        // is the rank communicating with itself?
        bool selfComm = (sendRank == rank) && isSending;
        
        // communicate the data
        if (selfComm){
            numRecieving = numRecv[recvRank];
            recvBuffer.resize(numRecieving);
            #pragma omp parallel for
            for (size_t i = 0; i < numRecieving; i++){
                recvBuffer[i] = data[sendIndexLists[iStep][i]];
            }
        } else {
            // resize the buffers
            if (isSending){
                numSending = sendIndexLists[iStep].size();
                sendBuffer.resize(numSending);
            }
            if (isRecieving){
                numRecieving = numRecv[recvRank];
                recvBuffer.resize(numRecieving);
            }
            // fill the sendBuffer
            if (isSending){
                #pragma omp parallel for
                for (size_t i = 0; i < numSending; i++){
                    sendBuffer[i] = data[sendIndexLists[iStep][i]];
                }
            }

            // communicate
            if (isSending && isRecieving){
                if (commMode == 'd'){
                    MPI_Sendrecv((void *)&sendBuffer[0], numSending,  MPI_DOUBLE, sendRank, 1, 
                                (void *)&recvBuffer[0], numRecieving, MPI_DOUBLE, recvRank, 1,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (commMode == 'u'){
                    MPI_Sendrecv((void *)&sendBuffer[0], numSending,  MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                                (void *)&recvBuffer[0], numRecieving, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (commMode == 'i'){
                    MPI_Sendrecv((void *)&sendBuffer[0], numSending,  MPI_INT, sendRank, 1, 
                                (void *)&recvBuffer[0], numRecieving, MPI_INT, recvRank, 1,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } 
            } else if (isSending){
                if (commMode == 'd'){
                    MPI_Send((void *)&sendBuffer[0], numSending, MPI_DOUBLE, sendRank, 1, MPI_COMM_WORLD);
                } else if (commMode == 'u'){
                    MPI_Send((void *)&sendBuffer[0], numSending, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
                } else if (commMode == 'i'){
                    MPI_Send((void *)&sendBuffer[0], numSending, MPI_INT, sendRank, 1, MPI_COMM_WORLD);
                } 
            } else if (isRecieving){
                if (commMode == 'd'){
                    MPI_Recv((void *)&recvBuffer[0], numRecieving, MPI_DOUBLE, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                } else if (commMode == 'u'){
                    MPI_Recv((void *)&recvBuffer[0], numRecieving, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                } else if (commMode == 'i'){
                    MPI_Recv((void *)&recvBuffer[0], numRecieving, MPI_INT, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                } 
            }
        }
        // sort in the recieved data
        #pragma omp parallel for
        for (size_t iRecv = 0; iRecv < recvBuffer.size(); iRecv++){
            resBuffer[numFinished] = recvBuffer[iRecv];
            numFinished++;
        }
    }

    // memory cleanup
    sendBuffer.clear();
    recvBuffer.clear();
    sendBuffer.shrink_to_fit();
    recvBuffer.shrink_to_fit();
    // put the result where it belongs
    data.resize(resBuffer.size());
    #pragma omp parallel for
    for (size_t iRes = 0; iRes < resBuffer.size(); iRes++){data[iRes] = resBuffer[iRes];}
    // memory cleanup
    resBuffer.clear();
    resBuffer.shrink_to_fit();
    return;
}


// function to find a local gridindex from global ix, iy and iz values
uint64_t getGridIndex(const int ix,const int iy,const int iz, 
                    const int ixmin,const int ixmax, 
                    const int iymin,const int iymax, 
                    const int izmin,const int izmax)
{   
    assert(("ERROR when determining local gridindex. ix < ixmin", ix >= ixmin));
    assert(("ERROR when determining local gridindex. ix > ixmax", ix <= ixmax));
    assert(("ERROR when determining local gridindex. iy < iymin", iy >= iymin));
    assert(("ERROR when determining local gridindex. iy > iymax", iy <= iymax));
    assert(("ERROR when determining local gridindex. iz < izmin", iz >= izmin));
    assert(("ERROR when determining local gridindex. iz > izmax", iz <= izmax));

    const uint64_t localix = static_cast<uint64_t>(ix - ixmin);
    const uint64_t localiy = static_cast<uint64_t>(iy - iymin);
    const uint64_t localiz = static_cast<uint64_t>(iz - izmin);

    const uint64_t xSize = static_cast<uint64_t>(ixmax - ixmin + 1);
    const uint64_t ySize = static_cast<uint64_t>(iymax - iymin + 1);
    //const uint64_t zSize = static_cast<uint64_t>(izmax - izmin + 1);

    // uint64_t GridIndex = ((localix * ySize + localiy) * zSize + localiz);
    uint64_t GridIndex = localix + localiy*xSize + localiz*xSize*ySize;
    return GridIndex;
}


// function to interpolate a grid based on the sph-particles
// input:       xSim, ySim, zSim    : positions of the SPH-particles
//              vxSim, vySim, vzSim : velocities of the SPH-particles
//              hSim                : smoothing lengths of the SPH-particles
//              rhoSim              : densities of the SPH-particles
//              ftleSim             : FTLE values associated with the SPH-particles (pass empty vector if no ftle is used)
//              numGridPoints       : number of grid points per dimension
//              numParticles        : total number of SPH-particles
//              kernelChoice        : which kernel should be used for the interpolation (only SPH) (1: new kernel; 0: old kernel)
//              nearestNeighbor     : use nearest neighbor mapping or SPH-interpolation (true: NN; false: SPH)
//              rank, numRanks      : local rank & total number of ranks
// output:      vxGrid, vyGrid, vzGrid, rhoGrid, ftleGrid, ncGrid
std::tuple<std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>> mapToGrid(
            std::vector<double>&  xSim, std::vector<double>&  ySim, std::vector<double>&  zSim,
            std::vector<double>& vxSim, std::vector<double>& vySim, std::vector<double>& vzSim,
            std::vector<double>&  hSim, std::vector<double>& rhoSim,std::vector<double>& ftleSim, 
            const int numGridPoints, const uint64_t numParticles, 
            const int kernelChoice, const bool nearestNeighbor, const int rank, const int numRanks)
{
    // runtimes
    auto RuntimeStart = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> CommRuntime;
    std::chrono::duration<double> GeometryRuntime;
    std::chrono::duration<double> InterpolatingRuntime;
    std::chrono::duration<double> TotalRuntime;
    // decide if FTLE is used
    const bool useFTLE = ftleSim.size() > 0;
    // side length of the box
    const double boxSize = 1.0;
    // side length of half the box
    const double halfBox = boxSize/2;
    // step size of the grid in 1 dimension
    const double gridStep = boxSize / static_cast<double>(numGridPoints);

    // ------------------------------------------------------------------------------------------------------------------------
    // identify the local index box -------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------

    // box of the full domain
    heffte::box3d<> allIndizes({0,0,0},{numGridPoints-1,numGridPoints-1,numGridPoints-1});
    // find a processor grid that minimizes the total surface area of all boxes
    std::array<int,3> processorGrid = heffte::proc_setup_min_surface(allIndizes, numRanks);
    // get all boxes by splitting the Indizes onto the Processor Grid
    // local Box stored at entry Rank
    std::vector<heffte::box3d<>> allBoxes = heffte::split_world(allIndizes,processorGrid);
    // size of the local Gird-Box
    uint64_t localBoxSize =   static_cast<uint64_t>(allBoxes[rank].size[0]) 
                            * static_cast<uint64_t>(allBoxes[rank].size[1]) 
                            * static_cast<uint64_t>(allBoxes[rank].size[2]);
    
    if (rank == 0){
        for (size_t iRank = 0; iRank < numRanks; iRank++){
            std::cout << iRank << ": " << allBoxes[iRank].low[0] << "-" << allBoxes[iRank].high[0] << " " << allBoxes[iRank].low[1] << "-" << allBoxes[iRank].high[1] << " " << allBoxes[iRank].low[2] << "-" << allBoxes[iRank].high[2] << std::endl;
        }
    }

    // if more than one rank is used, data needs to be communicated
    if (numRanks > 1){
        MPI_Barrier(MPI_COMM_WORLD);
        // time at the start of the particle communication
        auto CommRuntimeStart = std::chrono::high_resolution_clock::now();
        if (rank == 0){std::cout << "\tcommunicating particles; ";}

        // ------------------------------------------------------------------------------------------------------------------------
        // count how many of the local SPH-particles contribute to which box ------------------------------------------------------
        // ------------------------------------------------------------------------------------------------------------------------

        // number of local particles that contribute to each rank
        std::vector<uint64_t> numContributingParticles(numRanks, 0);
        // loop over the local particles
        for (size_t iParticle = 0; iParticle < xSim.size(); iParticle++){
            // loop over the ranks to check whether this particle contributes to each rank or not
            //#pragma omp parallel for
            for (size_t iRank = 0; iRank < numRanks; iRank++){
                bool contributing  = isContributing(xSim[iParticle],ySim[iParticle],zSim[iParticle],hSim[iParticle],halfBox,gridStep,
                                                    allBoxes[iRank].low[0],allBoxes[iRank].high[0],
                                                    allBoxes[iRank].low[1],allBoxes[iRank].high[1],
                                                    allBoxes[iRank].low[2],allBoxes[iRank].high[2], numGridPoints);
                if (contributing){numContributingParticles[iRank]++;}
            }
        }
        // ------------------------------------------------------------------------------------------------------------------------
        // communicate how many contributing particles there are for each rank ----------------------------------------------------
        // ------------------------------------------------------------------------------------------------------------------------

        // total number of particles contributing to THIS rank
        uint64_t localNumContrParticles = numContributingParticles[rank];
        // number of particles that contribute to THIS rank, but come from a different rank
        std::vector<uint64_t> numRecv(numRanks);
        numRecv[rank] = numContributingParticles[rank];
        // send Buffer
        uint64_t SendBufferUINT;
        // recieve Buffer
        uint64_t RecvBufferUINT;
        // communicate
        for (size_t iRank = 1; iRank < numRanks; iRank++){
            // rank to which this rank sends data
            int sendRank = correct_periodic_cond(rank + iRank, numRanks);
            // rank fromo which this rank recieves data
            int recvRank = correct_periodic_cond(rank - iRank, numRanks);

            // get data for sending
            SendBufferUINT = numContributingParticles[sendRank];
            // send / recieve communication
            MPI_Sendrecv((void *)&SendBufferUINT, 1, MPI_UNSIGNED_LONG, sendRank, 1, 
                        (void *)&RecvBufferUINT, 1, MPI_UNSIGNED_LONG, recvRank, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // sort in the recieved data
            localNumContrParticles += RecvBufferUINT;
            numRecv[recvRank] = RecvBufferUINT;
        }
        // for (size_t iRank = 0; iRank < numRanks; iRank++){
        //     if (rank == iRank){
        //         std::cout << rank << ": number of particles that contribute to this rank: " << localNumContrParticles << std::endl;
        //         for (size_t iRank = 0; iRank < numRanks; iRank++){
        //             std::cout << " " << numRecv[iRank];
        //         }
        //         std::cout << std::endl;
        //     }
        //     MPI_Barrier(MPI_COMM_WORLD);
        // }

        // ------------------------------------------------------------------------------------------------------------------------
        // communicate such that each rank has only the particles that contribute to it -------------------------------------------
        // ------------------------------------------------------------------------------------------------------------------------

        // creating the communication plan
        auto [sendPlan,recvPlan] = communicationPlan(numContributingParticles, rank,numRanks);

        // creating sendIndexLists
        std::vector<std::vector<uint64_t>> sendIndexLists = createSendIndexLists(xSim,ySim,zSim,hSim,numContributingParticles,sendPlan,allBoxes,
                                                                                 halfBox,gridStep,numGridPoints,numRanks);

        // communication
        dataComm(xSim,  sendIndexLists,numRecv, sendPlan,recvPlan, localNumContrParticles, rank,numRanks);
        dataComm(ySim,  sendIndexLists,numRecv, sendPlan,recvPlan, localNumContrParticles, rank,numRanks);
        dataComm(zSim,  sendIndexLists,numRecv, sendPlan,recvPlan, localNumContrParticles, rank,numRanks);
        dataComm(hSim,  sendIndexLists,numRecv, sendPlan,recvPlan, localNumContrParticles, rank,numRanks);
        dataComm(vxSim, sendIndexLists,numRecv, sendPlan,recvPlan, localNumContrParticles, rank,numRanks);
        dataComm(vySim, sendIndexLists,numRecv, sendPlan,recvPlan, localNumContrParticles, rank,numRanks);
        dataComm(vzSim, sendIndexLists,numRecv, sendPlan,recvPlan, localNumContrParticles, rank,numRanks);
        dataComm(rhoSim,sendIndexLists,numRecv, sendPlan,recvPlan, localNumContrParticles, rank,numRanks);
        if (useFTLE){dataComm(ftleSim, sendIndexLists,numRecv, sendPlan,recvPlan, localNumContrParticles, rank,numRanks);}

        MPI_Barrier(MPI_COMM_WORLD);
        auto CommRuntimeStop = std::chrono::high_resolution_clock::now();
        CommRuntime = CommRuntimeStop - CommRuntimeStart;
        if (rank == 0){std::cout << "\truntime: " << CommRuntime.count() << " sec" << std::endl;}
    } else {
        std::cout << "only one rank in play -> no rearrangement necessary" << std::endl;
    }

    // ------------------------------------------------------------------------------------------------------------------------
    // create fft grid geometry -----------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    // time at the start of creating the grid geometry
    auto GeomRuntimeStart = std::chrono::high_resolution_clock::now();
    if (rank == 0){std::cout << "\tcreating grid geometry; ";}
    // define transformation geometry
    heffte::fft3d<heffte::backend::fftw> fft(allBoxes[rank],allBoxes[rank], MPI_COMM_WORLD);
    // resize the input boxes
    std::vector<double> ftleGrid;
    std::vector<double> vxGrid(fft.size_inbox(),0.0);
    std::vector<double> vyGrid(fft.size_inbox(),0.0);
    std::vector<double> vzGrid(fft.size_inbox(),0.0);
    std::vector<double> rhoGrid(fft.size_inbox(),0.0);
    // grid data for diagnostics (distance to the neighbor for nearest neighbor, neighborcount for SPH)
    std::vector<double> ncGrid(fft.size_inbox(),0.0);
    if (useFTLE){
        ftleGrid.resize(fft.size_inbox());
        #pragma omp parallel for
        for (size_t i = 0; i < ftleGrid.size(); i++){ftleGrid[i] = 0.0;}
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto GeomRuntimeStop = std::chrono::high_resolution_clock::now();
    GeometryRuntime = GeomRuntimeStop - GeomRuntimeStart;
    if (rank == 0){std::cout << "\truntime: " << GeometryRuntime.count() << " sec" << std::endl;}


    // ------------------------------------------------------------------------------------------------------------------------
    // interpolate the simulation data onto the grid --------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------

    // true if the new kernel should be used
    const bool newKernel = kernelChoice == 1;
    // if nearest neighbor is selected set a unreasonably high value as distance to the nearest neighbor
    double initialNcValue = 1000;
    #pragma omp parallel for 
    for (size_t i = 0; i < ncGrid.size(); i++){ncGrid[i] = initialNcValue;}

    //std::cout << rank << ": " << allBoxes[rank].low[0] << " " << allBoxes[rank].high[0] << " " << allBoxes[rank].low[1] << " " << allBoxes[rank].high[1] << " " << allBoxes[rank].low[2] << " " << allBoxes[rank].high[2] << std::endl;
    
    MPI_Barrier(MPI_COMM_WORLD);
    // time at the start of grid interpolation
    auto InterpolRuntimeStart = std::chrono::high_resolution_clock::now();
    if (rank == 0){std::cout << "\tinterpolating the grid using kernel " << kernelChoice << "; ";}


    uint64_t calls = 0;
    int ixmax;
    int ixmin;
    for (size_t iParticle = 0; iParticle < xSim.size(); iParticle++){
        double h2 = 2*hSim[iParticle];
        // find grid points close to the SPH-particle
        ixmin = std::ceil( (xSim[iParticle]-h2+0.5*(1-gridStep)) / gridStep);
        ixmax = std::floor((xSim[iParticle]+h2+0.5*(1-gridStep)) / gridStep);
        //if(rank == 0){std::cout << xSim[iParticle] << "\t" << ySim[iParticle] << "\t" << zSim[iParticle] << "\t" << hSim[iParticle] << std::endl;}

        // loop over these grid point to compute contributions from this SPH-particle
        //#pragma omp parallel for
        for (int ix = ixmin; ix <= ixmax; ix++){
            int ixCorr = correct_periodic_cond(ix,numGridPoints);
            if ((ixCorr < allBoxes[rank].low[0]) || (ixCorr > allBoxes[rank].high[0])){continue;}
            // compute the distance in x-direction
            double dx = correct_periodic_cond(xSim[iParticle] - (gridStep*((double)(correct_periodic_cond(ix,numGridPoints))+0.5) - halfBox));
            double reducedSearchRange = std::sqrt(std::pow(h2,2) - std::pow(dx,2));
            // minimum y-index on the grid to which the current SPH-particle can contribute
            int iymin = std::ceil( (ySim[iParticle]-h2+0.5*(1-gridStep)) / gridStep);
            // maximum y-index on the grid to which the current SPH-particle can contribute
            int iymax = std::floor((ySim[iParticle]+h2+0.5*(1-gridStep)) / gridStep);
            // loop over the y-indizes in that range
            for (int iy = iymin; iy <= iymax; iy++){
                int iyCorr = correct_periodic_cond(iy,numGridPoints);
                if ((iyCorr < allBoxes[rank].low[1]) || (iyCorr > allBoxes[rank].high[1])){continue;}
                // compute the distance in the y-direction
                double dy = correct_periodic_cond(ySim[iParticle] - (gridStep*((double)(correct_periodic_cond(iy,numGridPoints))+0.5) - halfBox));
                reducedSearchRange = std::sqrt(std::pow(reducedSearchRange,2) - std::pow(dy,2));
                // minimum z-index on the grid to which the current SPH-particle can contribute
                int izmin = std::ceil( (zSim[iParticle]-h2+0.5*(1-gridStep)) / gridStep);
                // maximum z-index on the grid to which the current SPH-particle can contribute
                int izmax = std::floor((zSim[iParticle]+h2+0.5*(1-gridStep)) / gridStep);
                // loop over the z-indizes in that range
                for (int iz = izmin; iz <= izmax; iz++){
                    int izCorr = correct_periodic_cond(iz,numGridPoints);
                    if ((izCorr < allBoxes[rank].low[2]) || (izCorr > allBoxes[rank].high[2])){continue;}
                    // compute the distance in the z-direction
                    double dz = correct_periodic_cond(zSim[iParticle] - (gridStep*((double)(correct_periodic_cond(iz,numGridPoints))+0.5) - halfBox));
                    // compute the total distance between the SPH-particle and the grid-point
                    double dist = std::sqrt(std::pow(dx,2) + std::pow(dy,2) + std::pow(dz,2));
                    
                    if (nearestNeighbor){
                        // identify position of the current grid point in the result vectors
                        uint64_t GridIndex = getGridIndex(ixCorr,iyCorr,izCorr, 
                                                        allBoxes[rank].low[0],allBoxes[rank].high[0],
                                                        allBoxes[rank].low[1],allBoxes[rank].high[1],
                                                        allBoxes[rank].low[2],allBoxes[rank].high[2]);

                        // check if this SPH-particle is closer that the current nearest neighbor
                        if (dist < ncGrid[GridIndex]){
                            calls++;
                            vxGrid[GridIndex]  = vxSim[iParticle];
                            vyGrid[GridIndex]  = vySim[iParticle];
                            vzGrid[GridIndex]  = vzSim[iParticle];
                            rhoGrid[GridIndex] = rhoSim[iParticle];
                            ncGrid[GridIndex]  = dist;
                            if (useFTLE){ftleGrid[GridIndex] = ftleSim[iParticle];}
                        }
                    } else {                                                 
                        // compute the kernel for this distance
                        double kern;
                        if (newKernel){
                            kern = kernel2(dist, hSim[iParticle]);
                        } else {
                            kern = kernel1(dist, hSim[iParticle]);
                        }

                        // check if the kernel is larger than zero
                        // if not, the distance is too great and the SPH-particle does not contribute
                        if (kern > 0){
                            // identify position of the current grid point in the result vectors
                            uint64_t GridIndex = getGridIndex(ixCorr,iyCorr,izCorr, 
                                                            allBoxes[rank].low[0],allBoxes[rank].high[0],
                                                            allBoxes[rank].low[1],allBoxes[rank].high[1],
                                                            allBoxes[rank].low[2],allBoxes[rank].high[2]);
                            // interpolate data to this Grid Index
                            vxGrid[GridIndex]  += vxSim[iParticle]*kern/rhoSim[iParticle];
                            vyGrid[GridIndex]  += vySim[iParticle]*kern/rhoSim[iParticle];
                            vzGrid[GridIndex]  += vzSim[iParticle]*kern/rhoSim[iParticle];
                            rhoGrid[GridIndex] += kern;
                            if (useFTLE){ftleGrid[GridIndex]  += ftleSim[iParticle]*kern/rhoSim[iParticle];}
                            ncGrid[GridIndex]  += 1.0;
                        }
                    } // end of SPH-interpolation routine
                }   
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto InterpolRuntimeStop = std::chrono::high_resolution_clock::now();
    InterpolatingRuntime = InterpolRuntimeStop - InterpolRuntimeStart;
    if (rank == 0){std::cout << "\t" << calls << "\truntime: " << InterpolatingRuntime.count() << " sec\n" << std::endl;}


    if (not nearestNeighbor){
        // mass of a single SPH particle
        const double m = 1/static_cast<double>(numParticles);
        if (rank == 0){ std::cout << "\tSPH-particle mass:" << m << std::endl;}
        // muliply all data with the mass (last part of the interpolation)
        #pragma omp parallel for
        for (size_t iGrid = 0; iGrid < vxGrid.size(); iGrid++){
            vxGrid[iGrid]  = m*vxGrid[iGrid];
            vyGrid[iGrid]  = m*vyGrid[iGrid];
            vzGrid[iGrid]  = m*vzGrid[iGrid];
            rhoGrid[iGrid] = m*rhoGrid[iGrid];
            if (useFTLE){ftleGrid[iGrid] = m*ftleGrid[iGrid];}
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } else {
        // if nearest-neighbor was used, check if any gridpoints remained unassigned
        uint64_t numUnassigned = 0;
        for (size_t iGrid = 0; iGrid < ncGrid.size(); iGrid++){
            if (ncGrid[iGrid] == initialNcValue){numUnassigned++;}
        }
        if (numUnassigned > 0){ 
            std::cout << rank << ": " << numUnassigned << " grid points remain unassigned because no nearest neighbor could be identified" << std::endl;
        }
    }



    MPI_Barrier(MPI_COMM_WORLD);
    auto RuntimeStop = std::chrono::high_resolution_clock::now();
    TotalRuntime = RuntimeStop - RuntimeStart;
    if (rank == 0){std::cout << "\truntime: " << TotalRuntime.count() << " sec\n" << std::endl;}
    // return the grid
    return std::make_tuple(vxGrid, vyGrid, vzGrid, rhoGrid, ftleGrid, ncGrid);
}
