#pragma once

#include <iostream>
#include <tuple>
#include <vector>
#include <stdint.h>
#include <random>
#include <cassert>
#include <limits>
#include <mpi.h>

#include "paths.hpp"
#include "statistics.hpp"
#include "MPI_comms.hpp"
#include "periodic_conditions.hpp"
#include "heffte.h"


// function to compute the mean position in x, y and z of the data on a rank
std::tuple<double,double,double> computeMeanPos(const int rank, const int numRanks,
        const std::vector<double> x = {}, const std::vector<double> y = {}, const std::vector<double> z = {},
        const int ixmin = -1, const int ixmax = -1, const int iymin = -1, const int iymax = -1, const int izmin = -1, const int izmax = -1, const int numGridPoints = 0)
{
    // should the SPH-version be used?
    const bool SPH  = ((x.size() > 0) && (x.size() == y.size())) && (x.size() == z.size());
    // should the grid version be used?
    const bool grid = ((ixmin >= 0) && (ixmax >= ixmin)) && ((iymin >= 0) && (iymax >= iymin)) && ((izmin >= 0) && (izmax >= izmin)) && (numGridPoints > 0);
    // check inputs
    assert(("in computeMeanPos, you have to either provide x,y,z values for SPH particles or min/max/numGridPoints values for a grid", (grid || SPH)));
    assert(("in computeMeanPos, you can either provide x,y,z values for SPH particles or min/max values for a grid. not both.", not (grid && SPH)));
    // result containers
    double xMean, yMean, zMean = 0.0;
    // grid version
    if (grid){
        const double BoxSize = 1.0;
        const double gridStep = BoxSize / static_cast<double>(numGridPoints);
        xMean = (static_cast<double>(ixmax) + static_cast<double>(ixmin))/2 * gridStep - BoxSize/2;
        yMean = (static_cast<double>(iymax) + static_cast<double>(iymin))/2 * gridStep - BoxSize/2;
        zMean = (static_cast<double>(izmax) + static_cast<double>(izmin))/2 * gridStep - BoxSize/2;
    // SPH version
    } else if (SPH){
        xMean = std::accumulate(x.begin(),x.end(),0.0) / x.size();
        yMean = std::accumulate(y.begin(),y.end(),0.0) / y.size();
        xMean = std::accumulate(z.begin(),z.end(),0.0) / z.size();
    }
    // return results
    return std::tie(xMean,yMean,zMean);
}

// function to distribute the total number of connections onto the ranks
std::vector<uint64_t> distributeConnections(const uint64_t limConnections, const int rank, const int numRanks,
        const std::vector<double> x = {}, const std::vector<double> y = {}, const std::vector<double> z = {},
        const int ixmin = -1, const int ixmax = -1, const int iymin = -1, const int iymax = -1, const int izmin = -1, const int izmax = -1, const int numGridPoints = 0)
{
    // number of connections that need to be evaluated by this individual rank
    uint64_t numConnections = std::ceil(static_cast<double>(limConnections) / static_cast<double>(numRanks));
    if (rank == numRanks-1){numConnections = limConnections - (numRanks-1)*numConnections;}
    std::cout << rank << ": numConnections = " << numConnections << "/" << limConnections << std::endl;

    // compute the center of this ranks local data
    auto [xMean,yMean,zMean] = computeMeanPos(rank,numRanks,x,y,z,ixmin,ixmax,iymin,iymax,izmin,izmax,numGridPoints);
    std::cout << rank << ": mean position is [" << xMean << "," << yMean << "," << zMean << "]" << std::endl;
    // distances to the center of each other rank
    std::vector<double> dist(numRanks, 0.0);
    // compute the distances
    for (size_t iRank = 0; iRank < numRanks; iRank++){
        double xBuffer, yBuffer, zBuffer;
        if (rank == iRank){xBuffer=xMean; yBuffer=yMean; zBuffer=zMean;}
        MPI_Bcast((void *)&xBuffer, 1,MPI_DOUBLE,iRank,MPI_COMM_WORLD);
        MPI_Bcast((void *)&yBuffer, 1,MPI_DOUBLE,iRank,MPI_COMM_WORLD);
        MPI_Bcast((void *)&zBuffer, 1,MPI_DOUBLE,iRank,MPI_COMM_WORLD);
        if (rank == iRank){
            dist[iRank] = 0.0;
        } else {
            double dx = correct_periodic_cond(xMean - xBuffer);
            double dy = correct_periodic_cond(yMean - yBuffer);
            double dz = correct_periodic_cond(zMean - zBuffer);
            dist[iRank] = std::sqrt(std::pow(dx,2) + std::pow(dy,2) + std::pow(dz,2));
        }
    }

    // list of how many connections this rank should compute to every other rank
    std::vector<uint64_t> result(numRanks, 0);
    // fraction of the connections that should go to each rank
    std::vector<double> frac(numRanks, 0.0);
    // square the distances for accurate scaling with sphere surfaces
    double max = 0;
    for (size_t i = 0; i < dist.size(); i++){
        if (i != rank){frac[i] = 1/std::pow(dist[i],3);}
        if (frac[i] > max){max = frac[i];}
    }
    frac[rank] = 1.1*max;
    // normalization factor
    const double norm = std::accumulate(frac.begin(),frac.end(),0.0); 
    // compute the fraction of connections for each rank
    for (size_t i = 0; i < frac.size(); i++){frac[i] = frac[i]/norm;}
    // compute the number of connections per rank
    for (size_t iRank = 0; iRank < frac.size(); iRank++){
        result[iRank] = static_cast<uint64_t>(std::floor(frac[iRank] * numConnections));
    }
    uint64_t remainingConns = numConnections - std::accumulate(result.begin(),result.end(),0);
    int i = 0;
    while (remainingConns > 0){
        result[i]++;
        remainingConns--;
        i++;
    }


    // check the number of connections
    const uint64_t test = std::accumulate(result.begin(),result.end(),0);
    if (test != numConnections){std::cout << rank << ": ERROR: number of connections doesn't match " << test << "/" << numConnections << std::endl;}
    
    for (size_t iRank = 0; iRank < numRanks; iRank++){
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == iRank){
            std::cout << rank << ": ";
            for (size_t i= 0; i < numRanks; i++){
                std::cout << dist[i] << "-" << result[i] << "  ;  ";
            }
            std::cout << std::endl;
        }
    }
    
    return result;
}

double selectionProbability(const double dist, const double hMean)
{   
    if (dist == 0 || dist > 0.5){
        return 0;
    } else {
        return 1;
    }
//     else if (dist < hMean){
//         return 1;
//     } else {
//         return std::pow(hMean,2)/std::pow(dist,2);
//     }
}

// find out, if a given particle should be selected or not
// input:       dist    : distance between local & test particle
//              hMean   : mean smoothing length in the simulation
//              norm    : factor to adjust selection probabilities. > 1 -> more selections; < 1 -> less selections
// output:      bool, was the particle selected? (true: selected; false: not selected)
bool selection(const double dist, const double hMean, const double norm){
    // selection probability
    const double prob = selectionProbability(dist, hMean);
    // random number generator
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    std::random_device rd;
    std::mt19937_64 gen(rd());
    const double hit = distribution(gen);
    // check if the particle was selected
    if (hit/norm < prob){
        return true;
    } else {
        return false;
    }
}

// select a single random particle
uint64_t selectRandomParticle(const uint64_t numLocalParticles){
    // random distribution range
    std::uniform_int_distribution<uint64_t> distribution(0,numLocalParticles-1);
    // generate random seed
    std::random_device rd;
    //generate Index
    std::mt19937_64 gen(rd());
    const uint64_t index = distribution(gen);
    return index;
}

// function to randomly select a given number of particles from a local data vector
// input:           numParts        : number of particles you want to select
//                  numLocalParts   : size of the data vector
// output:          a vector containing the indizes of the selected particles
std::vector<uint64_t> selectParts(const uint64_t numParts, const uint64_t numLocalParts)
{
    // check input
    assert(("in selectParts, numParts must be smaller or equal to numLocalParts", numParts <= numLocalParts));
    // result container
    std::vector<uint64_t> result(numParts, 0);
    // number of selected particles
    uint64_t numSelected = 0;
    // random distribution range
    std::uniform_int_distribution<uint64_t> dist(0,numLocalParts);
    // select the particles
    while (numSelected < numParts){
        // generate random index
        std::random_device rd;
        std::mt19937_64 gen(rd());
        uint64_t index = dist(gen);
        // check if this index was already selected
        bool notSelected = true;
        for (size_t iTest = 0; iTest < numSelected; iTest++){if (result[iTest] == index){notSelected=false;break;}}
        // if it was not yet selected, select it now
        if (notSelected){
            result[numSelected] = index;
            numSelected++;
        }
    }
    return result;
}


// function to store SF results to the vectors
void store(std::vector<double>& xSF, std::vector<double>& ySF, std::vector<uint64_t>& nSF,
            std::vector<double>& xSFbuffer, std::vector<double>& ySFbuffer,
            const double dist, const double SF, const double binWidth, const double numeric_epsilon, const double n=10000)
{
    // find out in which bin the new data belongs
    const int iBin = static_cast<int>(std::floor(dist/binWidth));
    if (iBin < xSFbuffer.size()){
        // check SF
        const int expSF = std::ilogb(ySFbuffer[iBin]);
        if (SF > n*std::ldexp(numeric_epsilon,expSF)){
            ySFbuffer[iBin] += SF;
        } else {
            ySF[iBin]       += ySFbuffer[iBin];
            ySFbuffer[iBin]  = SF;
        }
        // check dist
        const int expDist = std::ilogb(xSFbuffer[iBin]);
        if (dist > n*std::ldexp(numeric_epsilon,expDist)){
            xSFbuffer[iBin] += dist;
        } else {
            xSF[iBin]       += xSFbuffer[iBin];
            xSFbuffer[iBin]  = dist;
        }
        // increase nSF
        nSF[iBin]++;
    }
    return;
}


// function to compute the structure function of a given order 
// input:       vx, vy, vz      : velocities of the simulation data
//              order           : order of the SF
//              limConnections  : total number of connections that should be evaluated
//              numBins         : number of bins for the result
//              rank, numRanks  : local rank & total number of ranks
//              numGridPoints   : number of grid points per dimension (only needed for the volume-weighted version)
//              x, y, z         : SPH-particle positions (only needed for the mass-weighted version)
std::tuple<std::vector<double>,std::vector<double>,std::vector<uint64_t>> computeSF(
                const std::vector<double>& vx, const std::vector<double>& vy, const std::vector<double>& vz, 
                const int order, const uint64_t limConnections, const double hMean,
                const int numBins, const int rank, const int numRanks, const double norm = 1,
                const int numGridPoints = 0,
                const std::vector<double>& x = {}, const std::vector<double>& y = {}, const std::vector<double>& z = {})
{   
    if (rank==0){std::cout << "hMean:" << hMean << std::endl;}
    const bool useSPH = (x.size() > 0) && (x.size() == y.size()) && (x.size() == z.size());
    const bool useGrid = numGridPoints > 0;
    assert(("in computeSF, either numGridPoints (volume-weighted) or x,y,z (mass-weighted) must be supplied", useSPH || useGrid));
    assert(("in computeSF, you cannot provide numGridPoints (volume-weighted) AND x,y,z (mass-weighted) at the same time. pick one mode", not (useSPH && useGrid)));
    // check inputs
    assert(("in computeSF, order must be larger than zero", order > 0));
    assert(("in computeSF, vx & vy must have the same size", vx.size() == vy.size()));
    assert(("in computeSF, vx & vz must have the same size", vx.size() == vz.size()));
    assert(("in computeSF, x & y must have the same size", x.size() == y.size()));
    assert(("in computeSF, x & z must have the same size", x.size() == z.size()));
    assert(("in computeSF, rank must be at least zero", rank >= 0));
    assert(("in computeSF, rank must be smaller than numRanks", rank < numRanks));
    // use the grid-based version of this function
    const bool volumeWeighted = x.size() > 0;
    // maximum distance that should be considered
    const double maxDist = 0.5;
    // width of the individual bins
    const double binWidth = maxDist / static_cast<double>(numBins);
    // step size of the grid (only relevant for the volume-weighted version)
    double gridStep;
    const double boxSize = 1.0;
    if (useGrid){gridStep = boxSize/numGridPoints;}

    // result containers

    // x-values of the SF
    std::vector<double>   xSF(numBins, 0.0);
    // y-values of the SF
    std::vector<double>   ySF(numBins, 0.0);
    // number of connections per bin
    std::vector<uint64_t> nSF(numBins, 0.0);
    // number of connections evaluated by this rank so far
    uint64_t evalConnections = 0;

    // buffers to prevent precision issues for large number of connections
    // results are computed to the buffers and then periodically moved to the actual result containers

    // (BUFFER) x-values of the SF
    std::vector<double>   xSFbuffer(numBins, 0.0);
    // (BUFFER) y-values of the SF
    std::vector<double>   ySFbuffer(numBins, 0.0);


    // get the local grid section, if the grid-based version is used
    int ixmin,ixmax, iymin,iymax, izmin,izmax = -1;
    if (useGrid){
        heffte::box3d<> AllIndizes({0,0,0},{numGridPoints-1,numGridPoints-1,numGridPoints-1});
        std::array<int,3> ProcessorGrid = heffte::proc_setup_min_surface(AllIndizes, numRanks);
        std::vector<heffte::box3d<>> AllBoxes = heffte::split_world(AllIndizes,ProcessorGrid);
        ixmin = AllBoxes[rank].low[0];
        iymin = AllBoxes[rank].low[1];
        izmin = AllBoxes[rank].low[2];
        ixmax = AllBoxes[rank].high[0];
        iymax = AllBoxes[rank].high[1];
        izmax = AllBoxes[rank].high[2];
    }
    // decide how many connections this rank has with other ranks
    std::vector<uint64_t> numConns = distributeConnections(limConnections,rank,numRanks, x,y,z, ixmin,ixmax,iymin,iymax,izmin,izmax,numGridPoints);

    // -----------------------------------------------------------------------------------------------------------------------------------------------------
    // SF computation --------------------------------------------------------------------------------------------------------------------------------------
    // -----------------------------------------------------------------------------------------------------------------------------------------------------

    // numeric epsilon to avoid numerical problems
    const double numeric_epsilon = std::numeric_limits<double>::epsilon();
    MPI_Barrier(MPI_COMM_WORLD);

    // communication plan to avoid load imbalance
    auto [recvPlan,sendPlan] = communicationPlan(numConns,rank,numRanks);
    MPI_Barrier(MPI_COMM_WORLD);

    // declaring variables for the computation loop
    std::vector<double>  xSend;
    std::vector<double>  ySend;
    std::vector<double>  zSend;
    std::vector<double> vxSend;
    std::vector<double> vySend;
    std::vector<double> vzSend;
    std::vector<double>  xTest;
    std::vector<double>  yTest;
    std::vector<double>  zTest;
    std::vector<double> vxTest;
    std::vector<double> vyTest;
    std::vector<double> vzTest;

    for (size_t iStep = 0; iStep < sendPlan.size(); iStep++){
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0){std::cout << "\t" << iStep << "/" << sendPlan.size() << std::endl;}
        // number of connections evaluated during the current step
        uint64_t numStepEval = 0;
        // rank to which this rank sends test particles
        int sendRank = sendPlan[iStep];
        // rank from which this rank recieves test particles
        int recvRank = recvPlan[iStep];
        // decide if this rank is sending / recieving in the current step
        bool isSending   = sendRank >= 0;
        bool isRecieving = recvRank >= 0;
        // how many test particles do we want from the other rank?
        uint64_t numTest = 10000;
        // if (isRecieving){numTest = static_cast<uint64_t>(std::ceil(numConns[recvRank]/100));}
        // number of test particles that will be generated on this rank 
        uint64_t numSend = 0;
        // communicate the number of requested test particles
        if (rank == sendRank){
            numSend = numTest;
        } else {
            if (isSending && isRecieving){
                MPI_Sendrecv((void *)&numTest, 1, MPI_UNSIGNED_LONG_LONG, recvRank, 1, 
                             (void *)&numSend, 1, MPI_UNSIGNED_LONG_LONG, sendRank, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (isSending){
                MPI_Recv((void *)&numSend, 1, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            } else if (isRecieving){
                MPI_Send((void *)&numTest, 1, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD);
            }
        }
        //if (rank == 0){std::cout << "\t\te numTest:" << numTest << " numSend:" << numSend << std::endl;}

        // generate a number of local test particle indizes to send
        std::vector<uint64_t> sendIndizes(numSend);
        if (isSending){sendIndizes = selectParts(numSend,vx.size());}
        
        if (rank == sendRank && rank == recvRank){
            std::cout << "selfcomm " << rank << "/" << numRanks << " " << iStep << std::endl;
            // computation loop
            if (useGrid){
                while (numStepEval < numConns[recvRank]){
                    // select a random local particle
                    uint64_t indLocal = selectRandomParticle(vx.size());
                    // select a random test particle
                    uint64_t indTest  = sendIndizes[selectRandomParticle(sendIndizes.size())];
                    // get local grid position
                    auto [ix,iy,iz] = decomposeGridIndex(indLocal, ixmin,ixmax,iymin,iymax,izmin,izmax);
                    double xLocal = getGridPosition(ix,gridStep);
                    double yLocal = getGridPosition(iy,gridStep);
                    double zLocal = getGridPosition(iz,gridStep);
                    // get test grid position
                    auto [ixTest,iyTest,izTest] = decomposeGridIndex(indTest, ixmin,ixmax,iymin,iymax,izmin,izmax);
                    double xTest = getGridPosition(ixTest,gridStep);
                    double yTest = getGridPosition(iyTest,gridStep);
                    double zTest = getGridPosition(izTest,gridStep);
                    // compute distance
                    double dx = correct_periodic_cond(xTest - xLocal);
                    double dy = correct_periodic_cond(yTest - yLocal);
                    double dz = correct_periodic_cond(zTest - zLocal);
                    double dist = std::sqrt(std::pow(dx,2) + std::pow(dy,2) + std::pow(dz,2));
                    // decide, if the connection should be used or not
                    bool selected = selection(dist,hMean,norm);
                    if (selected){
                        // compute SF contribution
                        double temp = std::pow(std::sqrt(std::pow(vx[indLocal]-vx[indTest],2) 
                                                        + std::pow(vy[indLocal]-vy[indTest],2) 
                                                        + std::pow(vz[indLocal]-vz[indTest],2)),order);
                        // store result
                        store(xSF,ySF,nSF, xSFbuffer,ySFbuffer, dist,temp, binWidth,numeric_epsilon);
                        // increase number of evaluated connections
                        numStepEval++;
                    }
                }
            } else if (useSPH){
                while (numStepEval < numConns[recvRank]){
                    // select a random local particle
                    uint64_t indLocal = selectRandomParticle(vx.size());
                    // select a random test particle
                    uint64_t indTest  = sendIndizes[selectRandomParticle(numTest)];
                    // compute distance
                    if (indLocal >= vx.size()){std::cout << "ERR1: " << indLocal << "/" << x.size() << std::endl;}
                    if (indTest  >= vx.size()){std::cout << "ERR2: " << indTest  << "/" << x.size() << std::endl;}
                    double dx = correct_periodic_cond(x[indTest] - x[indLocal]);
                    double dy = correct_periodic_cond(y[indTest] - y[indLocal]);
                    double dz = correct_periodic_cond(z[indTest] - z[indLocal]);
                    double dist = std::sqrt(std::pow(dx,2) + std::pow(dy,2) + std::pow(dz,2));
                    // decide, if the connection should be used or not
                    bool selected = selection(dist,hMean,norm);
                    if (selected){
                        if (indLocal >= vx.size()){std::cout << "ERR1: " << indLocal << "/" << vx.size() << std::endl;}
                        if (indTest  >= vx.size()){std::cout << "ERR2: " << indTest  << "/" << vx.size() << std::endl;}
                        // compute SF contribution
                        double temp = std::pow(std::sqrt(std::pow(vx[indLocal]-vx[indTest],2) 
                                                        + std::pow(vy[indLocal]-vy[indTest],2) 
                                                        + std::pow(vz[indLocal]-vz[indTest],2)),order);
                        // store result
                        store(xSF,ySF,nSF, xSFbuffer,ySFbuffer, dist,temp, binWidth,numeric_epsilon);
                        // increase number of evaluated connections
                        numStepEval++;
                    }
                }
            }
        } else {
            std::cout << "elsecomm " << recvRank << "->" << rank << "->" << sendRank << "\t" << iStep << " " << numTest << "/" << numSend << std::endl;
            // test particle data for sending
            if (isSending){
                xSend.resize(numSend);
                ySend.resize(numSend);
                zSend.resize(numSend);
                vxSend.resize(numSend);
                vySend.resize(numSend);
                vzSend.resize(numSend);
                for (size_t i = 0; i < sendIndizes.size(); i++){
                    if (useSPH){
                        xSend[i] = x[sendIndizes[i]];
                        ySend[i] = y[sendIndizes[i]];
                        zSend[i] = z[sendIndizes[i]];
                    } else if (useGrid){
                        auto [ix,iy,iz] = decomposeGridIndex(sendIndizes[i], ixmin,ixmax,iymin,iymax,izmin,izmax);
                        xSend[i] = getGridPosition(ix,gridStep);
                        ySend[i] = getGridPosition(iy,gridStep);
                        zSend[i] = getGridPosition(iz,gridStep);
                    }
                    vxSend[i] = vx[sendIndizes[i]];
                    vySend[i] = vy[sendIndizes[i]];
                    vzSend[i] = vz[sendIndizes[i]];
                }
            }
            // test particle data used by this rank
            if (isRecieving){
                xTest.resize(numTest);
                yTest.resize(numTest);
                zTest.resize(numTest);
                vxTest.resize(numTest);
                vyTest.resize(numTest);
                vzTest.resize(numTest);
            }
            // communicate test particles
            if (isSending && isRecieving){
                MPI_Sendrecv((void *)&xSend[0], numSend, MPI_DOUBLE, sendRank, 1, 
                             (void *)&xTest[0], numTest, MPI_DOUBLE, recvRank, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv((void *)&ySend[0], numSend, MPI_DOUBLE, sendRank, 1, 
                             (void *)&yTest[0], numTest, MPI_DOUBLE, recvRank, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv((void *)&zSend[0], numSend, MPI_DOUBLE, sendRank, 1, 
                             (void *)&zTest[0], numTest, MPI_DOUBLE, recvRank, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                MPI_Sendrecv((void *)&vxSend[0], numSend, MPI_DOUBLE, sendRank, 1, 
                             (void *)&vxTest[0], numTest, MPI_DOUBLE, recvRank, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv((void *)&vySend[0], numSend, MPI_DOUBLE, sendRank, 1, 
                             (void *)&vyTest[0], numTest, MPI_DOUBLE, recvRank, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv((void *)&vzSend[0], numSend, MPI_DOUBLE, sendRank, 1, 
                             (void *)&vzTest[0], numTest, MPI_DOUBLE, recvRank, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (isSending){
                MPI_Send((void *)&xSend[0], numSend, MPI_DOUBLE, sendRank, 1, MPI_COMM_WORLD);
                MPI_Send((void *)&ySend[0], numSend, MPI_DOUBLE, sendRank, 1, MPI_COMM_WORLD);
                MPI_Send((void *)&zSend[0], numSend, MPI_DOUBLE, sendRank, 1, MPI_COMM_WORLD);

                MPI_Send((void *)&vxSend[0], numSend, MPI_DOUBLE, sendRank, 1, MPI_COMM_WORLD);
                MPI_Send((void *)&vySend[0], numSend, MPI_DOUBLE, sendRank, 1, MPI_COMM_WORLD);
                MPI_Send((void *)&vzSend[0], numSend, MPI_DOUBLE, sendRank, 1, MPI_COMM_WORLD);
            } else if (isRecieving){
                MPI_Recv((void *)&xTest[0], numTest, MPI_DOUBLE, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Recv((void *)&xTest[0], numTest, MPI_DOUBLE, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Recv((void *)&xTest[0], numTest, MPI_DOUBLE, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                
                MPI_Recv((void *)&vxTest[0], numTest, MPI_DOUBLE, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Recv((void *)&vyTest[0], numTest, MPI_DOUBLE, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Recv((void *)&vzTest[0], numTest, MPI_DOUBLE, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            // computation loop
            if (useGrid && isRecieving){
                while (numStepEval < numConns[recvRank]){
                    // select a random local particle
                    uint64_t indLocal = selectRandomParticle(vx.size());
                    // select a random test particle
                    uint64_t indTest  = selectRandomParticle(numTest);
                    // get local grid position
                    auto [ix,iy,iz] = decomposeGridIndex(indLocal, ixmin,ixmax,iymin,iymax,izmin,izmax);
                    double xLocal = getGridPosition(ix,gridStep);
                    double yLocal = getGridPosition(iy,gridStep);
                    double zLocal = getGridPosition(iz,gridStep);
                    // compute distance
                    double dx = correct_periodic_cond(xTest[indTest] - xLocal);
                    double dy = correct_periodic_cond(yTest[indTest] - yLocal);
                    double dz = correct_periodic_cond(zTest[indTest] - zLocal);
                    double dist = std::sqrt(std::pow(dx,2) + std::pow(dy,2) + std::pow(dz,2));
                    // decide, if the connection should be used or not
                    bool selected = selection(dist,hMean,norm);
                    if (selected){
                        // compute SF contribution
                        double temp = std::pow(std::sqrt(std::pow(vx[indLocal]-vxTest[indTest],2) 
                                                        + std::pow(vy[indLocal]-vyTest[indTest],2) 
                                                        + std::pow(vz[indLocal]-vzTest[indTest],2)),order);
                        // store result
                        store(xSF,ySF,nSF, xSFbuffer,ySFbuffer, dist,temp, binWidth,numeric_epsilon);
                        // increase number of evaluated connections
                        numStepEval++;
                    }
                }
            } else if (useSPH && isRecieving){
                while (numStepEval < numConns[recvRank]){
                    // select a random local particle
                    uint64_t indLocal = selectRandomParticle(vx.size());
                    // select a random test particle
                    uint64_t indTest  = selectRandomParticle(numTest);
                    // compute distance
                    double dx = correct_periodic_cond(xTest[indTest] - x[indLocal]);
                    double dy = correct_periodic_cond(yTest[indTest] - y[indLocal]);
                    double dz = correct_periodic_cond(zTest[indTest] - z[indLocal]);
                    double dist = std::sqrt(std::pow(dx,2) + std::pow(dy,2) + std::pow(dz,2));
                    // decide, if the connection should be used or not
                    bool selected = selection(dist,hMean,norm);
                    if (selected){
                        // compute SF contribution
                        double temp = std::pow(std::sqrt(std::pow(vx[indLocal]-vxTest[indTest],2) 
                                                        + std::pow(vy[indLocal]-vyTest[indTest],2) 
                                                        + std::pow(vz[indLocal]-vzTest[indTest],2)),order);
                        // store result
                        store(xSF,ySF,nSF, xSFbuffer,ySFbuffer, dist,temp, binWidth,numeric_epsilon);
                        // increase number of evaluated connections
                        numStepEval++;
                    }
                }
            }
        }
    } // end of step loop
    if (rank == 0){std::cout << "computation done" << std::endl;}
    // collapse the remaining buffer on the result
    for (size_t iBin = 0; iBin < xSF.size(); iBin++){
        xSF[iBin] += xSFbuffer[iBin];
        ySF[iBin] += ySFbuffer[iBin];
    }
    // communicate results to rank 0
    collapse(xSF,'d');
    collapse(ySF,'d');
    collapse(nSF,'u');
    // normalize results
    if (rank == 0){
        for (size_t iBin = 0; iBin < xSF.size(); iBin++){
            if (nSF[iBin] > 0){
                xSF[iBin] = xSF[iBin] / static_cast<double>(nSF[iBin]);
                ySF[iBin] = ySF[iBin] / static_cast<double>(nSF[iBin]);
            } else {
                xSF[iBin] = (static_cast<double>(iBin) + 0.5) * binWidth;
                ySF[iBin] = 0.0;
            }
        }
    }
    // return results
    return std::tie(xSF, ySF, nSF);
}


// function to save structure function results
void saveSF(const std::vector<double>& xSF, const std::vector<double>& ySF, const std::vector<uint64_t>& nSF, const int order, 
            const statisticsCls vStats,
            const std::string simFile, const int stepNo, const double time, const bool massWeighted, filePathsCls filePaths, const uint64_t numPoints,
            const bool nearestNeighbor = true, std::string filename = "")
{
    // get the filename
    if (filename.length() == 0){
        if (massWeighted){
            filename = "mass-SF-" + std::to_string(stepNo) + "-" + std::to_string(order) + ".txt";
        } else {
            filename = "volume-SF-" + std::to_string(stepNo) + "-" + std::to_string(order) + ".txt";
        }
    }
    // get the path to the file
    std::string path = filePaths.getSFPath(massWeighted,filename);

    // write the file
    std::ofstream outFile(path);
    if (outFile.is_open())
    {   
        // header
        outFile << "## structure function" << std::endl;
        outFile << "# order     : " << order << std::endl;
        outFile << "# simfile   : " << simFile << std::endl;
        outFile << "# stepNo    : " << stepNo << std::endl;
        outFile << "# time      : " << time << std::endl;
        outFile << "# numPoints : " << numPoints << std::endl;
        outFile << "# vRMS      : " << vStats.getRMS() << std::endl;
        outFile << "# vMEAN     : " << vStats.getMEAN() << std::endl;
        outFile << "# vSTD      : " << vStats.getSTD() << std::endl;
        outFile << "# mass-weighted : " << massWeighted << std::endl;
        if (not massWeighted){outFile << "# nearest-neighbor : " << nearestNeighbor << std::endl;}
        outFile << "### x     y     #N" << std::endl;
     
        // loop over the bins 
        for (size_t iBin = 0; iBin < xSF.size(); iBin++){
            //mean data in each bin
            outFile << xSF[iBin] << " ";
            // number of grid Points n each bin
            outFile << ySF[iBin] << " ";
            // total number of ceonnections per bin
            outFile << nSF[iBin] << std::endl;
        }
        // close the file
        outFile.close();
    }
    return;
}