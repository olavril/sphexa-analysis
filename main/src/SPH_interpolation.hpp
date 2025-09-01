#pragma once

#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

#include "kernel.hpp"
#include "periodic_conditions.hpp"
#include "vector-operations.hpp"
#include "find_neighbors.hpp"

#include "cstone/domain/domain.hpp"

using Realtype = double;
using KeyType = uint64_t;
using Domain  = cstone::Domain<uint64_t, double, cstone::CpuTag>;

// function to remove the domain synchronization of a given data vector
// this function removes the halo and only leaves the actual particles native to this rank
// input:   vector in which the halo should be removed
//          first & last Index of the native Interval that should remain. Last index NOT included: [firstIndex,lastIndex)
// output:  None, the input vector is modified to only contain the entries from firstIndex to lastIndex [firstIndex,lastIndex)
template<typename T>
void remHalo(std::vector<T>& vector, const uint64_t firstIndex, const uint64_t lastIndex)
{   
    assert(("when removing the domain halo (remHalo), lastIndex must be larger than firstIndex", lastIndex > firstIndex));
    assert(("when removing the domain halo (remHalo), lastIndex must be smaller than the size of the vector", lastIndex < vector.size()));
    // compute the size of the cut vector
    const uint64_t vecSize = lastIndex - firstIndex;
    // create buffer
    std::vector<T> buffer(vecSize);
    #pragma omp parallel for
    for (size_t i = firstIndex; i < lastIndex; i++){buffer[i-firstIndex] = vector[i];}
    // resize the original vector
    vector.resize(vecSize);
    vector.shrink_to_fit();
    // return data to the original vector
    #pragma omp parallel for
    for (size_t i = 0; i < vecSize; i++){vector[i] = buffer[i];}
    // memory cleanup
    buffer.clear();
    buffer.shrink_to_fit();
    return;
}

// function to interpolate quantities at the positions of the SPH particles using the SPH-inpterpolation scheme
// input:   x, y, z         : positon of the SPH-particles
//          h               : smoothing lengths of the SPH-particles
//          rho             : densities of the SPH-particles (if density should be interpolated, pass a vector of the correct size)
//          ftle X, Y, Z    : position of the SPH-particles at timestep 2 (for computing FTLEs)
//          IDs             : SPH-particle IDs (dragged along with the domain)
//          vx, vy, vz      : velocities of the SPH-particles at timestep 1 (only dragged along to keep consistency with positions)
//          rhoBool         : do you want to interpolate the density? (true: interpolate density, false: input rho is already a valid density)
//          ftleBool        : do you want to interpolate FTLE values? (true: yes, false: no)
//          numParticles    : total number of particles in the simulation
//          rank, numRanks  : local rank ID & total number of ranks
// output:  ?
std::tuple<std::vector<double>,std::vector<double>> SPHinterpolation(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z, 
                    std::vector<double>& h, std::vector<double>& rho, std::vector<double>& rho2,
                    std::vector<double>& ftleX, std::vector<double>& ftleY, std::vector<double>& ftleZ,
                    std::vector<uint64_t>& IDs, std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
                    const bool rhoBool, const bool ftleBool,
                    const uint64_t numParticles, const int rank, const int numRanks)
{   
    // check inputs
    assert(("x.size() must be equal to y.size() in SPHinterpolation",       x.size() == y.size()));
    assert(("x.size() must be equal to z.size() in SPHinterpolation",       x.size() == z.size()));
    assert(("x.size() must be equal to h.size() in SPHinterpolation",       x.size() == h.size()));
    assert(("x.size() must be equal to rho.size() in SPHinterpolation",     x.size() == rho.size()));
    assert(("x.size() must be equal to ftleX.size() in SPHinterpolation",   x.size() == ftleX.size()));
    assert(("x.size() must be equal to ftleY.size() in SPHinterpolation",   x.size() == ftleY.size()));
    assert(("x.size() must be equal to ftleZ.size() in SPHinterpolation",   x.size() == ftleZ.size()));
    assert(("x.size() must be equal to vx.size() in SPHinterpolation",      x.size() == vx.size()));
    assert(("x.size() must be equal to vy.size() in SPHinterpolation",      x.size() == vy.size()));
    assert(("x.size() must be equal to vz.size() in SPHinterpolation",      x.size() == vz.size()));
    assert(("rank must be smaller than numRanks in SPHinterpolation",       rank < numRanks));
    // mass of a single SPH particle
    const double m = 1.0 / static_cast<double>(numParticles);
    if (rank == 0){std::cout << "\tparticle mass: " << m << std::endl;}

    // --------------------------------------------------------------------------------------------------------------------------
    // synchronize the domain
    // --------------------------------------------------------------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tsynchrionizing the domain;";}
    auto domainRuntimeStart = std::chrono::high_resolution_clock::now();

    // creating buffers
    std::vector<uint64_t> scratch1(x.size());
    std::vector<uint64_t> scratch2(x.size());
    std::vector<double>   scratch3(x.size());
    std::vector<double>   scratch4(x.size());
    std::vector<double>   scratch5(x.size());

    // domain synchronization
    std::vector<uint64_t> keys(x.size());
    size_t                bucketSizeFocus = 64;
    size_t                bucketSize      = std::max(bucketSizeFocus, numParticles / (100 * numRanks));
    float                 theta           = 1.0;
    cstone::Box<double>   box(-0.5, 0.5, cstone::BoundaryType::periodic); 
    Domain                domain(rank, numRanks, bucketSize, bucketSizeFocus, theta, box);
    if (ftleBool){
        auto drag = std::tie(IDs,rho, ftleX,ftleY,ftleZ, vx,vy,vz, rho2);
        auto ScratchBuffer = std::tie(scratch1, scratch2, scratch3, scratch4, scratch5);
        domain.sync(keys, x, y, z, h, drag, ScratchBuffer);
        domain.exchangeHalos(std::tie(ftleX,ftleY,ftleZ,rho), scratch3, scratch4);
    } else {
        auto drag = std::tie(IDs,rho, vx,vy,vz, rho2);
        auto ScratchBuffer = std::tie(scratch1, scratch2, scratch3, scratch4, scratch5);
        domain.sync(keys, x, y, z, h, drag, ScratchBuffer);
        domain.exchangeHalos(std::tie(rho), scratch3, scratch4);
    }
    // memory cleanup
    scratch1.clear();
    scratch2.clear();
    scratch3.clear();
    scratch4.clear();
    scratch5.clear();
    scratch1.shrink_to_fit();
    scratch2.shrink_to_fit();
    scratch3.shrink_to_fit();
    scratch4.shrink_to_fit();
    scratch5.shrink_to_fit();

    MPI_Barrier(MPI_COMM_WORLD);
    auto domainRuntimeStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> domainRuntime = domainRuntimeStop - domainRuntimeStart;
    if (rank == 0){std::cout << " runtime: " << domainRuntime.count() << " sec" << std::endl;}

    // --------------------------------------------------------------------------------------------------------------------------
    // find neighbors
    // --------------------------------------------------------------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tfinding neighbors;";}
    auto neighborsRuntimeStart = std::chrono::high_resolution_clock::now();

    // expected / maximum number of neighbors
    const unsigned ng0   = 100;
    const unsigned ngmax = 150;
    // storage for the neighbor indizes
    std::vector<cstone::LocalIndex> neighbors;
    // storage for the number of neighbors 
    std::vector<unsigned> nc;
    // finding the neighbors
    nc.resize(domain.nParticles(), 0);
    resizeNeighbors(neighbors, domain.nParticles() * ngmax);
    findNeighborsSph(x.data(), y.data(), z.data(), h.data(), domain.startIndex(), domain.endIndex(), domain.box(),
                    domain.octreeProperties().nsView(), ng0, ngmax, neighbors.data(), nc.data());

    MPI_Barrier(MPI_COMM_WORLD);
    auto neighborsRuntimeStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> neighborsRuntime = neighborsRuntimeStop - neighborsRuntimeStart;
    if (rank == 0){std::cout << " runtime: " << neighborsRuntime.count() << " sec" << std::endl;}





    // --------------------------------------------------------------------------------------------------------------------------
    // interpolate generalized volume elements
    // --------------------------------------------------------------------------------------------------------------------------

    std::vector<double> ncU(nc.size());
    MPI_Barrier(MPI_COMM_WORLD);
    auto gveRuntimeStart = std::chrono::high_resolution_clock::now();
    if (rhoBool){
        if (rank == 0){std::cout << "\tinterpolating generalized volume elements;";}
        // make sure all rho entries are zero
        #pragma omp parallel for
        for (size_t i = 0; i < rho.size(); i++){rho[i] = 0.0;}
        // create buffer for X = m/rho0
        std::vector<double> X(x.size(), 0.0);

        // interpolation of sum(W)
        // loop over the local particles
        #pragma omp parallel for
        for (size_t iPart = domain.startIndex(); iPart < domain.endIndex(); iPart++){
            // loop over the neighbors
            for (size_t iNeighbor = 0; iNeighbor < nc[iPart - domain.startIndex()]; iNeighbor++){
                // index for accessing the neighbor index in neighbors
                uint64_t iN = static_cast<uint64_t>((iPart-domain.startIndex()) * ngmax + iNeighbor);
                // index for accessing the neighbor data
                uint64_t neighborID;
                if (iNeighbor < nc[iPart - domain.startIndex()]-1){
                    neighborID = neighbors[iN];
                } else {
                    neighborID = iPart;
                }
                // compute the distance between particle & neighbor
                double dist = std::sqrt(std::pow(correct_periodic_cond(x[iPart] - x[neighborID]),2)
                                        + std::pow(correct_periodic_cond(y[iPart] - y[neighborID]),2)
                                        + std::pow(correct_periodic_cond(z[iPart] - z[neighborID]),2));
                // sum up the kernel contribution
                double kern = kernel1(dist,h[iPart]);
                // double kern = kernel2(dist,h[neighborID]);
                if (kern > 0){
                    X[iPart] += kern;
                    ncU[iPart - domain.startIndex()]++;
                }
            }
        }
        // so far we computed 1/X. Now we take the reciprocal to get X = m/rho0
        // note that m is not needed because all SPH-particles have the same mass and rho0 = sum(mW) = m sum(W) -> X = m/(m sum(W)) = 1/sum(W)
        #pragma omp parallel for 
        for (size_t iPart = domain.startIndex(); iPart < domain.endIndex(); iPart++){
            assert(("to compute the reciprocal of X in SPHinterpolation, X must be > 0", X[iPart] > 0));
            X[iPart] = 1.0 / X[iPart];
            // rho[iPart] = m * X[iPart];
        }
        // synchronize X
        scratch3.resize(x.size());
        scratch4.resize(x.size());
        domain.exchangeHalos(std::tie(X), scratch3, scratch4);
        scratch3.clear();
        scratch4.clear();
        scratch3.shrink_to_fit();
        scratch4.shrink_to_fit();

        MPI_Barrier(MPI_COMM_WORLD);
        for (size_t iTest = 0; iTest < X.size(); iTest++){
            if (X[iTest] <= 0.0){std::cout << "ERROR in X domain halo exchange" << std::endl;}
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // now that we have X, we can compute the generalized volume elements V = Xa / (sum(Xb Wab))
        // loop over the local particles
        #pragma omp parallel for
        for (size_t iPart = domain.startIndex(); iPart < domain.endIndex(); iPart++){
            // loop over the neighbors
            for (size_t iNeighbor = 0; iNeighbor < nc[iPart - domain.startIndex()]; iNeighbor++){
                // index for accessing the neighbor index in neighbors
                uint64_t iN = (iPart-domain.startIndex()) * static_cast<uint64_t>(ngmax) + iNeighbor;
                // index for accessing the neighbor data
                uint64_t neighborID;
                if (iNeighbor < nc[iPart - domain.startIndex()]-1){
                    neighborID = neighbors[iN];
                } else {
                    neighborID = iPart;
                }
                // compute the distance between particle & neighbor
                double dist = std::sqrt(std::pow(correct_periodic_cond(x[iPart] - x[neighborID]),2)
                                        + std::pow(correct_periodic_cond(y[iPart] - y[neighborID]),2)
                                        + std::pow(correct_periodic_cond(z[iPart] - z[neighborID]),2));
                // compute the gve contribution
                double kern = kernel1(dist,h[iPart]);
                // double kern = kernel2(dist,h[neighborID]);
                rho[iPart] += X[neighborID] * kern;
            }
        }
        // so far we have sum(Xb Wab), now we need to finish the computation
        #pragma omp parallel for
        for (size_t iPart = domain.startIndex(); iPart < domain.endIndex(); iPart++){
            assert(("to compute the reciprocal of rho in SPHinterpolation, rho must be > 0", rho[iPart] > 0));
            rho[iPart] = X[iPart] / rho[iPart];
        }
        // memory cleanup
        X.clear();
        X.shrink_to_fit();
        // synchronize the GVEs if needed for computing the FTLEs
        if (ftleBool){
            scratch3.resize(x.size());
            scratch4.resize(x.size());
            domain.exchangeHalos(std::tie(rho), scratch3, scratch4);
            scratch3.clear();
            scratch4.clear();
            scratch3.shrink_to_fit();
            scratch4.shrink_to_fit();
        }
    
    // if the density was read from file, we need to turn it into generalized volume elements for the FTLE computation
    } else {
        if (rank == 0){std::cout << "\tconverting densities to volume elements;";}
        #pragma omp parallel for
        for (size_t i = 0; i < rho.size(); i++){
            assert(("to compute the reciprocal of rho in SPHinterpolation(2), rho must be > 0", rho[i] > 0));
            rho[i] = m / rho[i];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto gveRuntimeStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gveRuntime = gveRuntimeStop - gveRuntimeStart;
    if (rank == 0){std::cout << " runtime: " << gveRuntime.count() << " sec" << std::endl;}




    // --------------------------------------------------------------------------------------------------------------------------
    // run FTLE interpolation if turned on
    // --------------------------------------------------------------------------------------------------------------------------
    std::vector<double> ftle;

    if (ftleBool){
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0){std::cout << "\tcomputing FTLEs;";}
        auto ftleRuntimeStart = std::chrono::high_resolution_clock::now();
        


        MPI_Barrier(MPI_COMM_WORLD);
        auto ftleRuntimeStop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> ftleRuntime = ftleRuntimeStop - ftleRuntimeStart;
        if (rank == 0){std::cout << " runtime: " << ftleRuntime.count() << " sec" << std::endl;}
    }






    // remove the domain synchronization
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tremoving domain halos;";}
    auto cleanupRuntimeStart = std::chrono::high_resolution_clock::now();

    remHalo(IDs, static_cast<uint64_t>(domain.startIndex()),static_cast<uint64_t>(domain.endIndex()));
    remHalo(x,   static_cast<uint64_t>(domain.startIndex()),static_cast<uint64_t>(domain.endIndex()));
    remHalo(y,   static_cast<uint64_t>(domain.startIndex()),static_cast<uint64_t>(domain.endIndex()));
    remHalo(z,   static_cast<uint64_t>(domain.startIndex()),static_cast<uint64_t>(domain.endIndex()));
    remHalo(h,   static_cast<uint64_t>(domain.startIndex()),static_cast<uint64_t>(domain.endIndex()));
    remHalo(vx,  static_cast<uint64_t>(domain.startIndex()),static_cast<uint64_t>(domain.endIndex()));
    remHalo(vy,  static_cast<uint64_t>(domain.startIndex()),static_cast<uint64_t>(domain.endIndex()));
    remHalo(vz,  static_cast<uint64_t>(domain.startIndex()),static_cast<uint64_t>(domain.endIndex()));
    remHalo(rho,  static_cast<uint64_t>(domain.startIndex()),static_cast<uint64_t>(domain.endIndex()));
    remHalo(rho2,  static_cast<uint64_t>(domain.startIndex()),static_cast<uint64_t>(domain.endIndex()));

    MPI_Barrier(MPI_COMM_WORLD);
    auto cleanupRuntimeStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cleanupRuntime = cleanupRuntimeStop - cleanupRuntimeStart;
    if (rank == 0){std::cout << " runtime: " << cleanupRuntime.count() << " sec" << std::endl;}

    // convert generalized volume elements back to densities
    #pragma omp parallel for 
    for (size_t i = 0; i < rho.size(); i++){
        rho[i] = m / rho[i];
    }
    // return to main
    return std::make_tuple(ncU,ftle);
}

