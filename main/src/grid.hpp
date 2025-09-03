#pragma once

#include <string>
#include <vector>
#include <stdint.h>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <mpi.h>

#include <cassert>

#include "paths.hpp"
#include "ifile_io_impl.h"
#include "periodic_conditions.hpp"
#include "kernel.hpp"
#include "MPI_comms.hpp"

#include "heffte.h"


bool IsContributing(const double xPart, const double yPart, const double zPart, const double hPart, const double BoxSizeHalf, const double GridStep, 
                    const int ixmin, const int ixmax, const int iymin, const int iymax, const int izmin, const int izmax, const int numGridPoints);
std::tuple<int,int> contribution_range(const double pos, const double h2, const double ConsumedDist2, const double BoxSizeHalf, 
                                    const double GridStep);
double getGridPosition(const int Index, const double LayerDist);
uint64_t getGridIndex(const int ix,const int iy,const int iz, 
                    const int ixmin,const int ixmax, 
                    const int iymin,const int iymax, 
                    const int izmin,const int izmax);
std::tuple<int,int,int> decomposeGridIndex(const uint64_t GridIndex, 
                    const int ixmin,const int ixmax, 
                    const int iymin,const int iymax, 
                    const int izmin,const int izmax);

using namespace sphexa;

std::tuple<std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>> MapToGrid(
            std::vector<double> xSim, std::vector<double> ySim, std::vector<double> zSim, 
            std::vector<double> vxSim, std::vector<double> vySim, std::vector<double> vzSim,
            std::vector<double> hSim, std::vector<double> rhoSim, std::vector<double> ftleSim, const int NumGridPoints, const uint64_t numParticles,
            const int kernelChoice, const int numProcessors, const int Rank, const int numRanks, const bool NearestNeighbor)
{   
    // should ftle be interpolated?
    const bool useFTLE = ftleSim.size() > 0;
    // number of particles per processor
    uint64_t NumParticlesPerProc = static_cast<uint64_t>(std::ceil(static_cast<double>(xSim.size()) / static_cast<double>(numProcessors)));

    // size of the Box in 1 dimension
    const double BoxSize = 1.0;
    // step size of the grid in 1 dimension
    const double GridStep = BoxSize / static_cast<double>(NumGridPoints);


    // ------------------------------------------------------------------------------------------------------------------------
    // identify the local index box -------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------

    // box of the full domain
    heffte::box3d<> AllIndizes({0,0,0},{NumGridPoints-1,NumGridPoints-1,NumGridPoints-1});

    // find a processor grid that minimizes the total surface area of all boxes
    std::array<int,3> ProcessorGrid = heffte::proc_setup_min_surface(AllIndizes, numRanks);

    // get all boxes by splitting the Indizes onto the Processor Grid
    // local Box stored at entry Rank
    std::vector<heffte::box3d<>> AllBoxes = heffte::split_world(AllIndizes,ProcessorGrid);
    // std::cout << Rank << ": orders" << AllBoxes[Rank].order[0] << " " << AllBoxes[Rank].order[1] << " " << AllBoxes[Rank].order[2] << std::endl;

    // size of the local Gird-Box
    uint64_t localBoxSize = static_cast<uint64_t>(AllBoxes[Rank].size[0]) 
                            * static_cast<uint64_t>(AllBoxes[Rank].size[1]) 
                            * static_cast<uint64_t>(AllBoxes[Rank].size[2]);


    // communication is only neccessary if there is more than one rank
    if (numRanks > 1){

        // ------------------------------------------------------------------------------------------------------------------------
        // count how many of the local SPH-particles contribute to which box ------------------------------------------------------
        // ------------------------------------------------------------------------------------------------------------------------

        // number of local particles that contribute to each rank
        std::vector<uint64_t> NumContributingParticles(numRanks, 0);
        
        // loop over the local particles
        for (size_t iParticle = 0; iParticle < xSim.size(); iParticle++){
            // loop over the ranks to check whether this particle contributes to each rank or not
            //#pragma omp parallel for
            for (size_t iRank = 0; iRank < numRanks; iRank++){
                bool Contributing  = IsContributing(xSim[iParticle],ySim[iParticle],zSim[iParticle],hSim[iParticle],BoxSize/2,GridStep,
                                                    AllBoxes[iRank].low[0],AllBoxes[iRank].high[0],
                                                    AllBoxes[iRank].low[1],AllBoxes[iRank].high[1],
                                                    AllBoxes[iRank].low[2],AllBoxes[iRank].high[2], NumGridPoints);
                if (Contributing){
                    NumContributingParticles[iRank] += static_cast<uint64_t>(1);
                }
            }
        }


        // ------------------------------------------------------------------------------------------------------------------------
        // communicate how many contributing particles there are for each rank ----------------------------------------------------
        // ------------------------------------------------------------------------------------------------------------------------

        // total number of particles contributing to THIS rank
        uint64_t localNumContrParticles = NumContributingParticles[Rank];
        // number of particles that contribute to THIS rank, but come from a different rank
        std::vector<uint64_t> NumRecv(numRanks);
        NumRecv[Rank] = NumContributingParticles[Rank];
        // send Buffer
        uint64_t SendBufferUINT;
        // recieve Buffer
        uint64_t RecvBufferUINT;

        // communicate
        for (size_t iRank = 1; iRank < numRanks; iRank++){
            // rank to which this rank sends data
            int SendRank = correct_periodic_cond(Rank + iRank, numRanks);
            // rank fromo which this rank recieves data
            int RecvRank = correct_periodic_cond(Rank - iRank, numRanks);

            // get data for sending
            SendBufferUINT = NumContributingParticles[SendRank];
            // send / recieve communication
            MPI_Sendrecv((void *)&SendBufferUINT, 1, MPI_UNSIGNED_LONG, SendRank, 1, 
                        (void *)&RecvBufferUINT, 1, MPI_UNSIGNED_LONG, RecvRank, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // sort in the recieved data
            localNumContrParticles += RecvBufferUINT;
            NumRecv[RecvRank] = RecvBufferUINT;
        }


        // ------------------------------------------------------------------------------------------------------------------------
        // communicate such that each rank has only the particles that contribute to it -------------------------------------------
        // ------------------------------------------------------------------------------------------------------------------------

        // Buffer for sending data
        std::vector<double> SendDataBuffer;
        // Buffer for resieving data
        std::vector<double> RecvDataBuffer;
        // Buffer for new data vector
        std::vector<double> NewData(localNumContrParticles);
        uint64_t AssignedParticles;
        // vector containing the indizes of the data that needs to be sent to each rank
        std::vector<std::vector<uint64_t>> SendIndizes(numRanks);
        for (size_t iRank = 0; iRank < numRanks; iRank++){
            SendIndizes[iRank].resize(NumContributingParticles[iRank]);
        }


        for (size_t iRank = 0; iRank < numRanks; iRank++){
            //reset the number of Indizes that have been found
            AssignedParticles = 0;
            // loop over the particles
            if (NumContributingParticles[iRank] > 0){
                for (size_t iParticle = 0; iParticle < xSim.size(); iParticle++){
                    bool Contributing  = IsContributing(xSim[iParticle],ySim[iParticle],zSim[iParticle],hSim[iParticle],BoxSize/2,GridStep,
                                                        AllBoxes[iRank].low[0],AllBoxes[iRank].high[0],
                                                        AllBoxes[iRank].low[1],AllBoxes[iRank].high[1],
                                                        AllBoxes[iRank].low[2],AllBoxes[iRank].high[2], NumGridPoints);
                    if (Contributing){
                        SendIndizes[iRank][(size_t)(AssignedParticles)] = (uint64_t)(iParticle);
                        AssignedParticles++;
                    }
                } 
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (Rank == 0){std::cout << "\tindex lists done." << std::endl;}



        // communicating x
        MPI_Barrier(MPI_COMM_WORLD);
        if (Rank == 0){std::cout << "\tcommunicating x values." << std::endl;}
        AssignedParticles = 0;
        for (size_t iRank = 0; iRank < numRanks; iRank++){
            int SendRank = correct_periodic_cond(Rank + iRank, numRanks);
            int RecvRank = correct_periodic_cond(Rank - iRank, numRanks);
            SendDataBuffer.resize(NumContributingParticles[SendRank]);
            RecvDataBuffer.resize(NumRecv[RecvRank]);
            // send data
            for (size_t iParticle = 0; iParticle < SendDataBuffer.size(); iParticle++){SendDataBuffer[iParticle] = xSim[SendIndizes[SendRank][iParticle]];}
            MPI_Sendrecv((void *)&SendDataBuffer[0], NumContributingParticles[SendRank], MPI_DOUBLE, SendRank, 1, 
                        (void *)&RecvDataBuffer[0], NumRecv[RecvRank], MPI_DOUBLE, RecvRank, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t iParticle = 0; iParticle < RecvDataBuffer.size(); iParticle++){
                NewData[AssignedParticles] = RecvDataBuffer[iParticle];
                AssignedParticles++;
            }
        }
        xSim.resize(localNumContrParticles);
        #pragma omp parallel for
        for (size_t iParticle = 0; iParticle < xSim.size(); iParticle++){xSim[iParticle] = NewData[iParticle];}

        // communicating y
        MPI_Barrier(MPI_COMM_WORLD);
        if (Rank == 0){std::cout << "\tcommunicating y values." << std::endl;}
        AssignedParticles = 0;
        for (size_t iRank = 0; iRank < numRanks; iRank++){
            int SendRank = correct_periodic_cond(Rank + iRank, numRanks);
            int RecvRank = correct_periodic_cond(Rank - iRank, numRanks);
            SendDataBuffer.resize(NumContributingParticles[SendRank]);
            RecvDataBuffer.resize(NumRecv[RecvRank]);
            // send data
            for (size_t iParticle = 0; iParticle < SendDataBuffer.size(); iParticle++){SendDataBuffer[iParticle] = ySim[SendIndizes[SendRank][iParticle]];}
            MPI_Sendrecv((void *)&SendDataBuffer[0], NumContributingParticles[SendRank], MPI_DOUBLE, SendRank, 1, 
                        (void *)&RecvDataBuffer[0], NumRecv[RecvRank], MPI_DOUBLE, RecvRank, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t iParticle = 0; iParticle < RecvDataBuffer.size(); iParticle++){
                NewData[AssignedParticles] = RecvDataBuffer[iParticle];
                AssignedParticles++;
            }
        }
        ySim.resize(localNumContrParticles);
        #pragma omp parallel for
        for (size_t iParticle = 0; iParticle < ySim.size(); iParticle++){ySim[iParticle] = NewData[iParticle];}

        // communicating z
        MPI_Barrier(MPI_COMM_WORLD);
        if (Rank == 0){std::cout << "\tcommunicating z values." << std::endl;}
        AssignedParticles = 0;
        for (size_t iRank = 0; iRank < numRanks; iRank++){
            int SendRank = correct_periodic_cond(Rank + iRank, numRanks);
            int RecvRank = correct_periodic_cond(Rank - iRank, numRanks);
            SendDataBuffer.resize(NumContributingParticles[SendRank]);
            RecvDataBuffer.resize(NumRecv[RecvRank]);
            // send data
            for (size_t iParticle = 0; iParticle < SendDataBuffer.size(); iParticle++){SendDataBuffer[iParticle] = zSim[SendIndizes[SendRank][iParticle]];}
            MPI_Sendrecv((void *)&SendDataBuffer[0], NumContributingParticles[SendRank], MPI_DOUBLE, SendRank, 1, 
                        (void *)&RecvDataBuffer[0], NumRecv[RecvRank], MPI_DOUBLE, RecvRank, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t iParticle = 0; iParticle < RecvDataBuffer.size(); iParticle++){
                NewData[AssignedParticles] = RecvDataBuffer[iParticle];
                AssignedParticles++;
            }
        }
        zSim.resize(localNumContrParticles);
        #pragma omp parallel for
        for (size_t iParticle = 0; iParticle < zSim.size(); iParticle++){zSim[iParticle] = NewData[iParticle];}


        // communicating vx
        MPI_Barrier(MPI_COMM_WORLD);
        if (Rank == 0){std::cout << "\tcommunicating vx values." << std::endl;}
        AssignedParticles = 0;
        for (size_t iRank = 0; iRank < numRanks; iRank++){
            int SendRank = correct_periodic_cond(Rank + iRank, numRanks);
            int RecvRank = correct_periodic_cond(Rank - iRank, numRanks);
            SendDataBuffer.resize(NumContributingParticles[SendRank]);
            RecvDataBuffer.resize(NumRecv[RecvRank]);
            // send data
            for (size_t iParticle = 0; iParticle < SendDataBuffer.size(); iParticle++){SendDataBuffer[iParticle] = vxSim[SendIndizes[SendRank][iParticle]];}
            MPI_Sendrecv((void *)&SendDataBuffer[0], NumContributingParticles[SendRank], MPI_DOUBLE, SendRank, 1, 
                        (void *)&RecvDataBuffer[0], NumRecv[RecvRank], MPI_DOUBLE, RecvRank, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t iParticle = 0; iParticle < RecvDataBuffer.size(); iParticle++){
                NewData[AssignedParticles] = RecvDataBuffer[iParticle];
                AssignedParticles++;
            }
        }
        vxSim.resize(localNumContrParticles);
        #pragma omp parallel for
        for (size_t iParticle = 0; iParticle < vxSim.size(); iParticle++){vxSim[iParticle] = NewData[iParticle];}


        // communicating vy
        MPI_Barrier(MPI_COMM_WORLD);
        if (Rank == 0){std::cout << "\tcommunicating vy values." << std::endl;}
        AssignedParticles = 0;
        for (size_t iRank = 0; iRank < numRanks; iRank++){
            int SendRank = correct_periodic_cond(Rank + iRank, numRanks);
            int RecvRank = correct_periodic_cond(Rank - iRank, numRanks);
            SendDataBuffer.resize(NumContributingParticles[SendRank]);
            RecvDataBuffer.resize(NumRecv[RecvRank]);
            // send data
            for (size_t iParticle = 0; iParticle < SendDataBuffer.size(); iParticle++){SendDataBuffer[iParticle] = vySim[SendIndizes[SendRank][iParticle]];}
            MPI_Sendrecv((void *)&SendDataBuffer[0], NumContributingParticles[SendRank], MPI_DOUBLE, SendRank, 1, 
                        (void *)&RecvDataBuffer[0], NumRecv[RecvRank], MPI_DOUBLE, RecvRank, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t iParticle = 0; iParticle < RecvDataBuffer.size(); iParticle++){
                NewData[AssignedParticles] = RecvDataBuffer[iParticle];
                AssignedParticles++;
            }
        }
        vySim.resize(localNumContrParticles);
        #pragma omp parallel for
        for (size_t iParticle = 0; iParticle < vySim.size(); iParticle++){vySim[iParticle] = NewData[iParticle];}


        // communicating vz
        MPI_Barrier(MPI_COMM_WORLD);
        if (Rank == 0){std::cout << "\tcommunicating vz values." << std::endl;}
        AssignedParticles = 0;
        for (size_t iRank = 0; iRank < numRanks; iRank++){
            int SendRank = correct_periodic_cond(Rank + iRank, numRanks);
            int RecvRank = correct_periodic_cond(Rank - iRank, numRanks);
            SendDataBuffer.resize(NumContributingParticles[SendRank]);
            RecvDataBuffer.resize(NumRecv[RecvRank]);
            // send data
            for (size_t iParticle = 0; iParticle < SendDataBuffer.size(); iParticle++){SendDataBuffer[iParticle] = vzSim[SendIndizes[SendRank][iParticle]];}
            MPI_Sendrecv((void *)&SendDataBuffer[0], NumContributingParticles[SendRank], MPI_DOUBLE, SendRank, 1, 
                        (void *)&RecvDataBuffer[0], NumRecv[RecvRank], MPI_DOUBLE, RecvRank, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t iParticle = 0; iParticle < RecvDataBuffer.size(); iParticle++){
                NewData[AssignedParticles] = RecvDataBuffer[iParticle];
                AssignedParticles++;
            }
        }
        vzSim.resize(localNumContrParticles);
        #pragma omp parallel for
        for (size_t iParticle = 0; iParticle < vzSim.size(); iParticle++){vzSim[iParticle] = NewData[iParticle];}



        // communicating h
        MPI_Barrier(MPI_COMM_WORLD);
        if (Rank == 0){std::cout << "\tcommunicating h values." << std::endl;}
        AssignedParticles = 0;
        for (size_t iRank = 0; iRank < numRanks; iRank++){
            int SendRank = correct_periodic_cond(Rank + iRank, numRanks);
            int RecvRank = correct_periodic_cond(Rank - iRank, numRanks);
            SendDataBuffer.resize(NumContributingParticles[SendRank]);
            RecvDataBuffer.resize(NumRecv[RecvRank]);
            // send data
            for (size_t iParticle = 0; iParticle < SendDataBuffer.size(); iParticle++){SendDataBuffer[iParticle] = hSim[SendIndizes[SendRank][iParticle]];}
            MPI_Sendrecv((void *)&SendDataBuffer[0], NumContributingParticles[SendRank], MPI_DOUBLE, SendRank, 1, 
                        (void *)&RecvDataBuffer[0], NumRecv[RecvRank], MPI_DOUBLE, RecvRank, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t iParticle = 0; iParticle < RecvDataBuffer.size(); iParticle++){
                NewData[AssignedParticles] = RecvDataBuffer[iParticle];
                AssignedParticles++;
            }
        }
        hSim.resize(localNumContrParticles);
        #pragma omp parallel for
        for (size_t iParticle = 0; iParticle < hSim.size(); iParticle++){hSim[iParticle] = NewData[iParticle];}


        // communicating rho
        MPI_Barrier(MPI_COMM_WORLD);
        if (Rank == 0){std::cout << "\tcommunicating rho values." << std::endl;}
        AssignedParticles = 0;
        for (size_t iRank = 0; iRank < numRanks; iRank++){
            int SendRank = correct_periodic_cond(Rank + iRank, numRanks);
            int RecvRank = correct_periodic_cond(Rank - iRank, numRanks);
            SendDataBuffer.resize(NumContributingParticles[SendRank]);
            RecvDataBuffer.resize(NumRecv[RecvRank]);
            // send data
            for (size_t iParticle = 0; iParticle < SendDataBuffer.size(); iParticle++){SendDataBuffer[iParticle] = rhoSim[SendIndizes[SendRank][iParticle]];}
            MPI_Sendrecv((void *)&SendDataBuffer[0], NumContributingParticles[SendRank], MPI_DOUBLE, SendRank, 1, 
                        (void *)&RecvDataBuffer[0], NumRecv[RecvRank], MPI_DOUBLE, RecvRank, 1,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t iParticle = 0; iParticle < RecvDataBuffer.size(); iParticle++){
                NewData[AssignedParticles] = RecvDataBuffer[iParticle];
                AssignedParticles++;
            }
        }
        rhoSim.resize(localNumContrParticles);
        #pragma omp parallel for
        for (size_t iParticle = 0; iParticle < rhoSim.size(); iParticle++){rhoSim[iParticle] = NewData[iParticle];}


        if (useFTLE){
            // communicating rho
            MPI_Barrier(MPI_COMM_WORLD);
            if (Rank == 0){std::cout << "\tcommunicating ftle values." << std::endl;}
            AssignedParticles = 0;
            for (size_t iRank = 0; iRank < numRanks; iRank++){
                int SendRank = correct_periodic_cond(Rank + iRank, numRanks);
                int RecvRank = correct_periodic_cond(Rank - iRank, numRanks);
                SendDataBuffer.resize(NumContributingParticles[SendRank]);
                RecvDataBuffer.resize(NumRecv[RecvRank]);
                // send data
                for (size_t iParticle = 0; iParticle < SendDataBuffer.size(); iParticle++){SendDataBuffer[iParticle] = ftleSim[SendIndizes[SendRank][iParticle]];}
                MPI_Sendrecv((void *)&SendDataBuffer[0], NumContributingParticles[SendRank], MPI_DOUBLE, SendRank, 1, 
                            (void *)&RecvDataBuffer[0], NumRecv[RecvRank], MPI_DOUBLE, RecvRank, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (size_t iParticle = 0; iParticle < RecvDataBuffer.size(); iParticle++){
                    NewData[AssignedParticles] = RecvDataBuffer[iParticle];
                    AssignedParticles++;
                }
            }
            ftleSim.resize(localNumContrParticles);
            #pragma omp parallel for
            for (size_t iParticle = 0; iParticle < ftleSim.size(); iParticle++){ftleSim[iParticle] = NewData[iParticle];}
        }



        if (Rank == 0){std::cout << "\tsimulation data rearranged properly." << std::endl;}
        MPI_Barrier(MPI_COMM_WORLD);

    } else {
        std::cout << "\tonly one rank in play -> no rearrangement necessary" << std::endl;
    }



    // ------------------------------------------------------------------------------------------------------------------------
    // create fft geometry ----------------------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------

    // define transformation geometry
    heffte::fft3d<heffte::backend::fftw> fft(AllBoxes[Rank],AllBoxes[Rank], MPI_COMM_WORLD);

    // // resize the input boxes
    std::vector<double> vxGrid(fft.size_inbox(),0.0);
    std::vector<double> vyGrid(fft.size_inbox(),0.0);
    std::vector<double> vzGrid(fft.size_inbox(),0.0);
    std::vector<double> rhoGrid(fft.size_inbox(),0.0);
    std::vector<double> ftleGrid;
    std::vector<double> ncGrid(fft.size_inbox(),0.0);
    if (useFTLE){
        ftleGrid.resize(fft.size_inbox());
        for (size_t i = 0; i < ftleGrid.size(); i++){ftleGrid[i] = 0.0;}
    }

    // // reset grid values
    // #pragma omp parallel for
    // for (size_t iGrid = 0; iGrid < vxSim.size(); iGrid++){
    //     vxGrid[iGrid]  = 0.0;
    //     vyGrid[iGrid]  = 0.0;
    //     vzGrid[iGrid]  = 0.0;
    //     rhoGrid[iGrid] = 0.0;
    //     ncGrid[iGrid]  = 0.0;
    // }
    // std::cout << Rank << ": grid data geometry fixed." << std::endl;




    // ------------------------------------------------------------------------------------------------------------------------
    // interpolate the simulation data onto the grid --------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------------------------------------------

    // initial value for nc for nearest neighbor mapping
    const double initialNcValue = 100;
    // if you use the nearest neighbor approach, the nc vector is used for the distance to 
    // the nearest SPH-particle instead. We initially set it very high such that all particles 
    // that are encountered first are set as the first nearest neighbor 
    if (NearestNeighbor){
        #pragma omp parallel for
        for (size_t i = 0; i < ncGrid.size(); i++){
            ncGrid[i] = initialNcValue;
        }
    }

    bool newkernel = (kernelChoice == 1);
    if (Rank == 0){std::cout << "\tnew kernel used: " << newkernel << std::endl;}
    int ixmax;
    int ixmin;
    for (size_t iParticle = 0; iParticle < xSim.size(); iParticle++){
        if ((Rank == 0) && (iParticle%100000 == 0)){std::cout << iParticle << "/" << xSim.size() << std::endl;}
        // twice the smoothing length
        double h2 = 2*hSim[iParticle];
        // find grid points close to the SPH-particle
        ixmin = std::ceil( (xSim[iParticle]-h2+0.5*(1-GridStep)) / GridStep);
        ixmax = std::floor((xSim[iParticle]+h2+0.5*(1-GridStep)) / GridStep);

        // loop over these grid point to compute contributions from this SPH-particle
        #pragma omp parallel for
        for (int ix = ixmin; ix <= ixmax; ix++){
            int ixCorr = correct_periodic_cond(ix,NumGridPoints);
            if ((ixCorr < AllBoxes[Rank].low[0]) || (ixCorr > AllBoxes[Rank].high[0])){continue;}
            // compute the distance in x-direction
            double dx = correct_periodic_cond(xSim[iParticle] - (GridStep*((double)(correct_periodic_cond(ix,NumGridPoints))+0.5)-0.5));
            double reducedSearchRange = std::sqrt(std::pow(h2,2) - std::pow(dx,2));
            // minimum y-index on the grid to which the current SPH-particle can contribute
            int iymin = std::ceil( (ySim[iParticle]-h2+0.5*(1-GridStep)) / GridStep);
            // int iymin = std::ceil( (ySim[iParticle]-reducedSearchRange+0.5*(1-GridStep)) / GridStep);
            // maximum y-index on the grid to which the current SPH-particle can contribute
            int iymax = std::floor((ySim[iParticle]+h2+0.5*(1-GridStep)) / GridStep);
            // int iymax = std::floor((ySim[iParticle]+reducedSearchRange+0.5*(1-GridStep)) / GridStep);
            // loop over the y-indizes in that range
            for (int iy = iymin; iy <= iymax; iy++){
                int iyCorr = correct_periodic_cond(iy,NumGridPoints);
                if ((iyCorr < AllBoxes[Rank].low[1]) || (iyCorr > AllBoxes[Rank].high[1])){continue;}
                // compute the distance in the y-direction
                double dy = correct_periodic_cond(ySim[iParticle] - (GridStep*((double)(correct_periodic_cond(iy,NumGridPoints))+0.5)-0.5));
                reducedSearchRange = std::sqrt(std::pow(reducedSearchRange,2) - std::pow(dx,2));
                // minimum z-index on the grid to which the current SPH-particle can contribute
                int izmin = std::ceil( (zSim[iParticle]-h2+0.5*(1-GridStep)) / GridStep);
                // int izmin = std::ceil( (zSim[iParticle]-reducedSearchRange+0.5*(1-GridStep)) / GridStep);
                // maximum z-index on the grid to which the current SPH-particle can contribute
                int izmax = std::floor((zSim[iParticle]+h2+0.5*(1-GridStep)) / GridStep);
                // int izmax = std::floor((zSim[iParticle]+reducedSearchRange+0.5*(1-GridStep)) / GridStep);
                // loop over the z-indizes in that range
                for (int iz = izmin; iz <= izmax; iz++){
                    int izCorr = correct_periodic_cond(iz,NumGridPoints);
                    if ((izCorr < AllBoxes[Rank].low[2]) || (izCorr > AllBoxes[Rank].high[2])){continue;}
                    // compute the distance in the z-direction
                    double dz = correct_periodic_cond(zSim[iParticle] - (GridStep*((double)(correct_periodic_cond(iz,NumGridPoints))+0.5)-0.5));
                    // compute the total distance between the SPH-particle and the grid-point
                    double dist = std::sqrt(std::pow(dx,2) + std::pow(dy,2) + std::pow(dz,2));

                    if (NearestNeighbor){
                        // identify position of the current grid point in the result vectors
                        uint64_t GridIndex = getGridIndex(ixCorr,iyCorr,izCorr, 
                                                        AllBoxes[Rank].low[0],AllBoxes[Rank].high[0],
                                                        AllBoxes[Rank].low[1],AllBoxes[Rank].high[1],
                                                        AllBoxes[Rank].low[2],AllBoxes[Rank].high[2]);
                        // check if this SPH-particle is closer that the current nearest neighbor
                        if (dist < ncGrid[GridIndex]){
                            vxGrid[GridIndex]  = vxSim[iParticle];
                            vyGrid[GridIndex]  = vySim[iParticle];
                            vzGrid[GridIndex]  = vzSim[iParticle];
                            rhoGrid[GridIndex] = rhoSim[iParticle];
                            ncGrid[GridIndex]  = dist;
                        }
                    } else {                                                  
                        // compute the kernel for this distance
                        double kern;
                        if (newkernel){
                            kern = kernel2(dist, hSim[iParticle]);
                        } else {
                            kern = kernel1(dist, hSim[iParticle]);
                        }

                        // check if the kernel is larger than zero
                        // if not, the distance is too great and the SPH-particle does not contribute
                        if (kern > 0){
                            // identify position of the current grid point in the result vectors
                            uint64_t GridIndex = getGridIndex(ixCorr,iyCorr,izCorr, 
                                                            AllBoxes[Rank].low[0],AllBoxes[Rank].high[0],
                                                            AllBoxes[Rank].low[1],AllBoxes[Rank].high[1],
                                                            AllBoxes[Rank].low[2],AllBoxes[Rank].high[2]);
                            // interpolate data to this Grid Index
                            vxGrid[GridIndex]  += vxSim[iParticle]*kern/rhoSim[iParticle];
                            vyGrid[GridIndex]  += vySim[iParticle]*kern/rhoSim[iParticle];
                            vzGrid[GridIndex]  += vzSim[iParticle]*kern/rhoSim[iParticle];
                            rhoGrid[GridIndex] += kern;
                            ncGrid[GridIndex]  += 1.0;
                        }
                    } // end of SPH-interpolation routine
                }   
            }
        }
    }










    if (not NearestNeighbor){
        // mass of a single SPH particle
        const double m = 1/static_cast<double>(numParticles);
        if (Rank == 0){ std::cout << "\tSPH-particle mass:" << m << std::endl;}
        // muliply all data with the mass (last part of the interpolation)
        #pragma omp parallel for
        for (size_t iGrid = 0; iGrid < vxGrid.size(); iGrid++){
            vxGrid[iGrid]  = m*vxGrid[iGrid];
            vyGrid[iGrid]  = m*vyGrid[iGrid];
            vzGrid[iGrid]  = m*vzGrid[iGrid];
            rhoGrid[iGrid] = m*rhoGrid[iGrid];
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } else {
        // if nearest-neighbor was used, check how many grdpoints remain unassigned
        int numUnassigned = 0;
        for (size_t iGrid = 0; iGrid < ncGrid.size(); iGrid++){
            if (ncGrid[iGrid] == initialNcValue){numUnassigned++;}
        }
        if (numUnassigned > 0){ 
            std::cout << "\n" << numUnassigned << " grid points remain unassigned because no nearest neighbor could be identified\n" << std::endl;
        }
    }

    // quick nc analytics
    double meanNC = 0.0;
    double minNC = 100000;
    double maxNC = 0;
    for (size_t i = 0; i < ncGrid.size(); i++){
        meanNC += ncGrid[i];
        
        if (minNC > ncGrid[i]){
            minNC = ncGrid[i];
        }
        if (maxNC < ncGrid[i]){
            maxNC = ncGrid[i];
        }
    }
    meanNC = meanNC/(double)(ncGrid.size());
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << Rank << ":\tinterpolation finished.  meanNC=" << meanNC << "\tminNC=" << minNC << " \tmaxNC=" << maxNC << std::endl;


    // return the grid
    return std::make_tuple(vxGrid, vyGrid, vzGrid, rhoGrid, ftleGrid, ncGrid);
}

// check to intervals for overlap
// this function does take periodic conditions into account
// it is assumed, that the borders of the intervals are included [min,max] 
// inputs:  min,max of interval 1 (Grid Boxes)
//          min,max of interval 2 (SPH-particle)
// outputs: boolean, true if there is overlap
//                   false if not
bool DoesOverlap(const int Min1, const int Max1, const int Min2, const int Max2, const int numGridPoints)
{   
    bool Overlap = false;   
    if ((Min1 <= Min2) && (Min2 <= Max1)){
        Overlap = true;
    } else if ((Min1 <= Max2) && (Max2 <= Max1)){
        Overlap = true;
    } else if ((Min2 <= Min1) && (Max1 <= Max2)){
        Overlap = true;
    } else {
        const int CorrMin2 = correct_periodic_cond(Min2, numGridPoints);
        const int CorrMax2 = correct_periodic_cond(Max2, numGridPoints);
        if (CorrMin2 > CorrMax2){
            if (Min1 <= CorrMax2){
                Overlap = true;
            } else if (Max1 >= CorrMin2){
                Overlap = true;
            }
        }
    }


    return Overlap;
}

// function to find out whether a given particle is contributing to a given box
// the contribution sphere is approximated with a cube for computational reasons
// inputs:  x, y and z coordinates of the particle
//          h - smoothing length of the particle
//          BoxSize - HALF the size of the simulation box in one dimenstion (usually 1)
//          GridStep - step size between two grid points in one dimension
//          ixmin,ixmax, iymin,iymax, izmin,izmax border indizes of the box
// output:  boolean, true  if the particle can contribute
//                   false if not
bool IsContributing(const double xPart, const double yPart, const double zPart, const double hPart, const double BoxSizeHalf, const double GridStep, 
                    const int ixmin, const int ixmax, const int iymin, const int iymax, const int izmin, const int izmax, const int numGridPoints)
{   
    // borders of the particle contribution sphere
    int ixPartMin, ixPartMax, iyPartMin, iyPartMax, izPartMin, izPartMax;
    // compute the borders of the particle contribution sphere
    const double CorrectionTerm = BoxSizeHalf - 0.5*GridStep; // term to shift the positions
    ixPartMin = std::ceil(( xPart - 2*hPart + CorrectionTerm ) / GridStep);
    ixPartMax = std::floor((xPart + 2*hPart + CorrectionTerm ) / GridStep);
    iyPartMin = std::ceil(( yPart - 2*hPart + CorrectionTerm ) / GridStep);
    iyPartMax = std::floor((yPart + 2*hPart + CorrectionTerm ) / GridStep);
    izPartMin = std::ceil(( zPart - 2*hPart + CorrectionTerm ) / GridStep);
    izPartMax = std::floor((zPart + 2*hPart + CorrectionTerm ) / GridStep);

    // find out if the particle contribution sphere overlaps with the box
    bool xOverlap, yOverlap, zOverlap;
    // check overlaps
    xOverlap = DoesOverlap(ixmin,ixmax,ixPartMin,ixPartMax, numGridPoints);
    yOverlap = DoesOverlap(iymin,iymax,iyPartMin,iyPartMax, numGridPoints);
    zOverlap = DoesOverlap(izmin,izmax,izPartMin,izPartMax, numGridPoints);

    return ((xOverlap && yOverlap) && zOverlap);
}


// function to identify the 1D-range of grid indizes for which a given SPH-particle can contribute
// input:   position of the SPH-particle in the relevant direction
//          smoothing-length squared h² of the SPH-particle
//          square of the distance that was already consumed by previous steps (effectively reduces the remaining smoothing length)
//          half the size of the simulation box (in one dimension)
//          distance between two neighboring grid-points (GridStep)
// output:  tuuple contaning the minimum and maximum grid index in the given direction to which this SPH-particle can contribute
std::tuple<int,int> contribution_range(const double pos, const double h2, const double ConsumedDist2, const double BoxSizeHalf, 
                                    const double GridStep)
{
    // maximum possible distance in the given direction
    double maxDist = std::sqrt(h2 - ConsumedDist2);
    // term to shift the positions
    const double CorrectionTerm = BoxSizeHalf - 0.5*GridStep; 
    // compute the minimum grid index the SPH-particle can reach
    int minIndex =  std::ceil((pos - maxDist + CorrectionTerm) / GridStep);
    // compute the maximum grid index the SPH-particle can reach
    int maxIndex = std::floor((pos + maxDist + CorrectionTerm) / GridStep);

    return std::make_tuple(minIndex,maxIndex);
}

// function to compute the position of a grid-point in one dimension based on the corresponding global grid-Index
double getGridPosition(const int Index, const double GridStep){
    const double pos = ((double)(Index) + 0.5) * GridStep - 0.5;
    return pos;
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
    const uint64_t zSize = static_cast<uint64_t>(izmax - izmin + 1);

    // uint64_t GridIndex = ((localix * ySize + localiy) * zSize + localiz);
    uint64_t GridIndex = localix + localiy*xSize + localiz*xSize*ySize;
    return GridIndex;
}


// function to decompose a local GridIndex into global ix, iy and iz values
std::tuple<int,int,int> decomposeGridIndex(const uint64_t GridIndex, 
                    const int ixmin,const int ixmax, 
                    const int iymin,const int iymax, 
                    const int izmin,const int izmax)
{   
    // assert(("ERROR during decomposeGridIndex: GridIndex is larger than 10.000.000", GridIndex < 10000000));
    // assert(("ERROR during decomposeGridIndex: ixmin must be larger or equal to zero", ixmin >= 0));
    // assert(("ERROR during decomposeGridIndex: iymin must be larger or equal to zero", iymin >= 0));
    // assert(("ERROR during decomposeGridIndex: izmin must be larger or equal to zero", izmin >= 0));
    // assert(("ERROR during decomposeGridIndex: ixmax must be larger than ixmin", ixmax >= ixmin));
    // assert(("ERROR during decomposeGridIndex: iymax must be larger than iymin", iymax >= iymin));
    // assert(("ERROR during decomposeGridIndex: izmax must be larger than izmin", izmax >= izmin));
    int ix, iy, iz;

    const uint64_t xSize = static_cast<uint64_t>(ixmax - ixmin + 1);
    const uint64_t ySize = static_cast<uint64_t>(iymax - iymin + 1);
    const uint64_t zSize = static_cast<uint64_t>(izmax - izmin + 1);

    // for Osmans order
    iz = std::floor((double)(GridIndex) / (double)(xSize*ySize));
    // assert(("ERROR during decomposeGridIndex: GridIndex must be larger or equal to ix*ySize*zSize", GridIndex >= ix*ySize*zSize));
    iy = std::floor((double)(GridIndex - iz*xSize*ySize) / (double)(xSize));
    // assert(("ERROR during decomposeGridIndex: GridIndex must be larger or equal to ix*ySize*zSize + iy*zSize", GridIndex >= ix*ySize*zSize + iy*zSize));
    ix = static_cast<int>(GridIndex - iz*xSize*ySize - iy*xSize);

    ix += ixmin;
    iy += iymin;
    iz += izmin;

    return std::make_tuple(ix,iy,iz);
}

// function to compute the grid mass & the total kinetic energy on the grid
std::tuple<double,double> computeGridProperties(const std::vector<double> v, const std::vector<double> rho, 
                const int numGridPoints, const int numProcessors, const int Rank, const int numRanks)
{   
    // local grid-points per PRocessor
    const uint64_t NumGridPointsPerProc = std::ceil(static_cast<double>(rho.size()) / static_cast<double>(numProcessors));
    // total number of GridPoints
    const double totalNumGridPoints = std::pow(static_cast<double>(numGridPoints), 3);
    // volume per Grid Point
    const double volume = 1/totalNumGridPoints;


    // total mass on the grid
    double GridMass = 0.0;
    std::vector<double> GridDist(numProcessors, 0.0);
    // sum up the local grid point densities
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t i = iProc*NumGridPointsPerProc;(i < (iProc+1)*NumGridPointsPerProc) && (i < rho.size()); i++){
            GridDist[iProc] += rho[i];
        }
    }
    // sum up the local results
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        GridMass += GridDist[iProc];
        GridDist[iProc] = 0.0;
    }
    // collapse the results of the different ranks
    collapse(GridMass, 'd');
    // multiply the sum of densities with the volume per grid point
    GridMass = GridMass * volume;


    // total kinetic energy of the grid
    double GridEkin = 0.0;
    // sum up the local grid point densities
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t i = iProc*NumGridPointsPerProc;(i < (iProc+1)*NumGridPointsPerProc) && (i < rho.size()); i++){
            GridDist[iProc] += rho[i] * std::pow(v[i],2);
        }
    }
    // sum up the local results
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        GridEkin += GridDist[iProc];
        GridDist[iProc] = 0.0;
    }
    // collapse the results of the different ranks
    collapse(GridEkin, 'd');
    // multiply the sum of densities with the volume per grid point
    GridEkin = 0.5 * GridEkin * volume;


    // return the results
    return std::make_tuple(GridMass,GridEkin);
}


// function to determine the global index at which a gridpoint should be placed for storage
uint64_t getGlobalStorageID(const int ix, const int iy, const int iz, const int numGridPoints)
{
    return ix * std::pow(numGridPoints,2) + iy*numGridPoints + iz;
}


// function to save the grid to file in case it is needed for something else
void saveGrid(const std::vector<double>&  vx, const std::vector<double>& vy, const std::vector<double>& vz,
              const std::vector<double>& rho, const std::vector<double>& nc, const std::vector<double>& ftle,
              filePathsCls filePaths, const double time, const double ftleTime,
              const std::string simFile, const int stepNo, const std::string ftleFile, const int ftleStepNo,
              const bool nearestNeighbor, const int numGridPoints, const int rank, const int numRanks)
{   
    // decide if FTLE stuff needs to be saved
    const bool useFTLE = ftle.size() > 0;
    // size of the Box in 1 dimension
    const double boxSize = 1.0;
    // step size of the grid in 1 dimension
    const double gridStep = boxSize / static_cast<double>(numGridPoints); 
    // get local grid range
    heffte::box3d<> AllIndizes({0,0,0},{numGridPoints-1,numGridPoints-1,numGridPoints-1});
    std::array<int,3> ProcessorGrid = heffte::proc_setup_min_surface(AllIndizes, numRanks);
    std::vector<heffte::box3d<>> AllBoxes = heffte::split_world(AllIndizes,ProcessorGrid);
    const int ixmin = AllBoxes[rank].low[0];
    const int ixmax = AllBoxes[rank].high[0];
    const int iymin = AllBoxes[rank].low[1];
    const int iymax = AllBoxes[rank].high[1];
    const int izmin = AllBoxes[rank].low[2];
    const int izmax = AllBoxes[rank].high[2];

    // construct grid position vectors
    std::vector <double> x(vx.size(), 0.0), y(vx.size(), 0.0), z(vx.size(), 0.0);
    for (size_t i = 0; i < vx.size(); i++){
        auto [ix, iy, iz] = decomposeGridIndex(i, ixmin,ixmax,iymin,iymax,izmin,izmax);
        x[i] = getGridPosition(ix,gridStep);
        y[i] = getGridPosition(iy,gridStep);
        z[i] = getGridPosition(iz,gridStep);
    }

    // create the filepath where the grid should be stored
    const std::string path = filePaths.getResultPath("grid.h5");

    // nearest neighbor int
    int NN = 0;
    if (nearestNeighbor){NN = 1;}


    // h5 file writer
    auto writer = makeH5PartWriter(MPI_COMM_WORLD);
    // write the data to a new step
    writer->addStep(0, vx.size(), path);
    // write header information
    writer->stepAttribute("time",            &time, 1);
    writer->stepAttribute("nearestNeighbor", &NN, 1);
    writer->stepAttribute("stepNo",          &stepNo, 1);
    writer->stepAttribute("numGridPoints",   &numGridPoints, 1);
    writer->stepAttribute("gridStep",        &gridStep, 1);
    if (useFTLE){
        writer->stepAttribute("ftleTime",   &ftleTime, 1);
        writer->stepAttribute("ftleSTepNo", &ftleStepNo, 1);
    }
    // writing the data
    writer->writeField(  "x",   x.data(), 0);
    writer->writeField(  "y",   y.data(), 0);
    writer->writeField(  "z",   z.data(), 0);
    writer->writeField( "vx",  vx.data(), 0);
    writer->writeField( "vy",  vy.data(), 0);
    writer->writeField( "vz",  vz.data(), 0);
    writer->writeField("rho", rho.data(), 0);
    writer->writeField( "nc",  nc.data(), 0);
    if (useFTLE){
        writer->writeField("ftle", ftle.data(), 0);
    }
    // close the writer
    writer->closeStep();
}