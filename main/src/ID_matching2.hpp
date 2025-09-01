#pragma once

#include <chrono>
#include <cassert>
#include <cmath>
#include <tuple>
#include <vector>
#include <stdint.h> 
#include <omp.h>
#include <mpi.h>
#include <type_traits> // for type recognition
#include <typeinfo>

#include "MPI_comms.hpp"

// function to sort the data in timestep 1 by ID
// this function is used for matching the data of two timesteps (used for the FTLE computation & diffusion coefficients)
// input:   ID vector of the first timestep
//          vector containing the number of (sorted) particles that should be stored on each rank.
// output:  two vectors, the first contains the original rank and the second the original index on that rank. 
//          They are sorted by ID (ID=0 is at position 0 on rank 0, etc.) and can be used to send the sorted data of timestep 2 to the correct place
std::tuple<std::vector<int>,std::vector<uint64_t>> sortStep1(const std::vector<uint64_t> IDs, const std::vector<uint64_t> numPartsPerRank, 
                                                            const std::vector<uint64_t> firstLocalID,
                                                            const int rank, const int numRanks, const int numProcessors)
{
    // create the result vectors
    std::vector<int>        originalRank(numPartsPerRank[rank]);
    std::vector<uint64_t>   originalIndex(numPartsPerRank[rank]);

    // vector containing the rank on which each of the local particles belongs
    std::vector<int> trueRank(IDs.size());
    // vector containing the number of local particles on this rank that need to be send to each other rank
    std::vector<uint64_t> numSendParticles(numRanks, 0);
    std::vector<std::vector<uint64_t>> numSendParticlesDist(numProcessors);
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        numSendParticlesDist[iProc].resize(numRanks);
        for (size_t iRank = 0; iRank < numRanks; iRank++){numSendParticlesDist[iProc][iRank] = 0;}
    }
    uint64_t numPartsPerProcessor = static_cast<uint64_t>(std::ceil(static_cast<double>(IDs.size()) / static_cast<double>(numProcessors)));
    // find out how many particles on this rank need to be send to each other rank
    #pragma omp parallel for 
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t i = iProc*numPartsPerProcessor; (i < (iProc+1)*numPartsPerProcessor) && (i < IDs.size()); i++){
            // find out on which rank this particle belongs
            int temp = static_cast<int>(std::floor(static_cast<double>(IDs[i]) / static_cast<double>(numPartsPerRank[0])));
            // increse the number of particles sent to this rank
            numSendParticlesDist[iProc][temp]++;
            trueRank[i] = temp;
        }
    }
    // add up the distributed results
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t iRank = 0; iRank < numRanks; iRank++){numSendParticles[iRank] += numSendParticlesDist[iProc][iRank];}
    }
    // memory cleanup
    numSendParticlesDist.clear();
    numSendParticlesDist.shrink_to_fit();
    // check if all particles were checked
    uint64_t checksum = 0;
    for (size_t iRank = 0; iRank < numRanks; iRank++){checksum += numSendParticles[iRank];}
    assert(("in sortStep1 not all particles were checked for where they should be sent", checksum == IDs.size()));

    // create index lists for sending
    std::vector<std::vector<uint64_t>> indexLists(numRanks);
    for (size_t iRank = 0; iRank < numRanks; iRank++){indexLists[iRank].resize(numSendParticles[iRank]);}
    std::vector<uint64_t> assignedParticles(numRanks,0);
    //#pragma omp parallel for
    for (size_t iPart = 0; iPart < IDs.size(); iPart++){
        indexLists[trueRank[iPart]][assignedParticles[trueRank[iPart]]] = iPart;
        assignedParticles[trueRank[iPart]]++;
    }
    // memory cleanup
    trueRank.clear();
    assignedParticles.clear();
    trueRank.shrink_to_fit();
    assignedParticles.shrink_to_fit();




    // create a communication plan
    if (rank == 0){std::cout << "\tcreating a comm plan" << std::endl;}
    auto [sendPlan, recvPlan] = communicationPlan(numSendParticles, rank,numRanks);
    MPI_Barrier(MPI_COMM_WORLD);


    
    // buffer for sending
    std::vector<uint64_t> sendBufferIDs;
    // buffer for sending
    std::vector<uint64_t> sendBuffer;
    // buffer for recieving
    std::vector<uint64_t> recvBufferIDs;
    // buffer for recieving
    std::vector<uint64_t> recvBuffer;
    // actual communication
    if (rank == 0){std::cout << "\tcommunicating the data" << std::endl;}
    for (size_t iStep = 0; iStep < sendPlan.size(); iStep++){
        auto stepRuntimeStart = std::chrono::high_resolution_clock::now();
        //if (rank == 0){std::cout << "\tstep " << iStep << std::endl;}
        // rank to which this rank will send its data
        int sendRank = sendPlan[iStep];
        // rank from which this rank will recieve data
        int recvRank = recvPlan[iStep];
        // is this rank sending in this step of communication?
        bool isSending   = sendRank >= 0;
        // is this rank recieving in this step of communication?
        bool isRecieving = recvRank >= 0;
        // number of particles that will be sent
        uint64_t numSend = 0;
        // number of particles that will be recieved
        uint64_t numRecv = 0;
        if (isSending){numSend = numSendParticles[sendRank];}
        //synchronize the ranks
        MPI_Barrier(MPI_COMM_WORLD);
        if (sendRank == rank){
            numRecv = numSend;
            recvBuffer.resize(numRecv);
            recvBufferIDs.resize(numRecv);
            #pragma omp parallel for
            for (size_t i = 0; i < numSend; i++){
                recvBuffer[i] = indexLists[sendRank][i];
                recvBufferIDs[i] = IDs[recvBuffer[i]];
            }
        } else {
            // excange the number of particles that will be sent / recieved
            if (isSending && isRecieving){
                MPI_Sendrecv((void *)&numSend, 1, MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                            (void *)&numRecv, 1, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (isSending) {
                MPI_Send((void *)&numSend, 1, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
            } else if (isRecieving) {
                MPI_Recv((void *)&numRecv, 1, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            // assemble the data that should be sent
            if (isSending){
                sendBufferIDs.resize(numSend);
                sendBuffer.resize(numSend);
                sendBuffer = indexLists[sendRank];
                #pragma omp parallel for
                for (size_t i = 0; i < numSend; i++){sendBufferIDs[i] = IDs[sendBuffer[i]];}
            }
            // resize the recieve buffer
            if (isRecieving){
                recvBufferIDs.resize(numRecv);
                recvBuffer.resize(numRecv);
            }
            // communication
            if (isSending && isRecieving){
                MPI_Sendrecv((void *)&sendBufferIDs[0], numSend, MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                            (void *)&recvBufferIDs[0], numRecv, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Sendrecv((void *)&sendBuffer[0], numSend, MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                            (void *)&recvBuffer[0], numRecv, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (isSending) {
                MPI_Send((void *)&sendBufferIDs[0], numSend, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
                MPI_Send((void *)&sendBuffer[0], numSend, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
            } else if (isRecieving) {
                MPI_Recv((void *)&recvBufferIDs[0], numRecv, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                MPI_Recv((void *)&recvBuffer[0], numRecv, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }
        // sort the data into the originalRank and originalIndex vectors
        if (isRecieving){
            #pragma omp parallel for 
            for (size_t i = 0; i < numRecv; i++){
                uint64_t index = recvBufferIDs[i] - firstLocalID[rank];
                originalRank[index]  = recvRank;
                originalIndex[index] = recvBuffer[i];
            }
        }
        auto stepRuntimeStop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> stepRuntime = stepRuntimeStop - stepRuntimeStart;
        //std::cout << "\t\trank " << rank << " done with step " << iStep << ". runtime:" << stepRuntime.count() << " sec / " << stepRuntime.count()/60 << "min" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
    } // end of communication
    // memory cleanup
    sendBufferIDs.clear();
    sendBuffer.clear();
    recvBufferIDs.clear();
    recvBuffer.clear();

    return std::make_tuple(originalRank,originalIndex);
}


// function to create sendIndexLists, a 2D vector that contains the indizes for creating sendLists (sorting version)
// input:       IDs:                particle IDs based on which the function decides to which rank each particle belongs
//              numPartsPerRank:    number of particles per Rank
//              numSendParticles:   number of local particles that need to be sent to each rank
//              sendPlan:           the send plan for the communication
//              numRanks:           number of ranks used for this process
// output:      sendIndexLists 2D vector
std::vector<std::vector<uint64_t>> createSendIndexLists(const std::vector<uint64_t> IDs, const uint64_t numPartsPerRank, 
                                                        const std::vector<uint64_t> numSendParticles, const std::vector<int> sendPlan, const int numRanks)
{   
    // a map than maps a rank to the associated communication step, where this local rank send to the selected one (step = rankStep[rank])
    // -1 means that no communication is scheduled to that rank. if this is accessed, there is an error
    std::vector<int> rankSteps(numRanks, -1);
    for (size_t step = 0; step < sendPlan.size(); step++){
        if (sendPlan[step] >= 0){rankSteps[sendPlan[step]] = static_cast<int>(step);}
    }

    // result container
    std::vector<std::vector<uint64_t>> sendIndexLists(sendPlan.size());
    for (size_t iStep = 0; iStep < sendPlan.size(); iStep++){
        int sendRank = sendPlan[iStep];
        if (sendRank >= 0){
            sendIndexLists[iStep].resize(numSendParticles[sendRank]);
        } else {
            sendIndexLists[iStep].resize(0);
        }
    }
    // vector to keep track where the next free position in each sendIndexList vector is
    std::vector<uint64_t> ind(sendPlan.size(), 0);

    // fill in the data
    for (size_t iPart = 0; iPart < IDs.size(); iPart++){
        int targetRank = static_cast<int>(std::floor(static_cast<double>(IDs[iPart]) / static_cast<double>(numPartsPerRank)));
        if (targetRank < rankSteps.size()){
            int targetStep = rankSteps[targetRank];
            if ((targetStep >= 0) && (targetStep < sendPlan.size())){
                sendIndexLists[targetStep][ind[targetStep]] = static_cast<uint64_t>(iPart);
                ind[targetStep]++;
            } else {
                std::cout << "ERROR in createSendIndexLists: ID:" << IDs[iPart] << " targetStep:" << targetStep << " out of bounds [0;" << sendPlan.size() << ")   targetRank:" << targetRank << std::endl;
            }
        } else {
            std::cout << "ERROR in createSendIndexLists: ID:" << IDs[iPart] << " targetRank:" << targetRank << " >= rankSteps.size():" << rankSteps.size() << std::endl;
        }
    }

    // clear memory
    ind.clear();
    ind.shrink_to_fit();
    // return the result
    return sendIndexLists;
}

// function to create sendIndexLists, a 2D vector that contains the indizes for creating sendLists (return version)
// input:       originalRank:       original Ranks of the particles based on which the function decides to which rank each particle belongs
//              numSendParticles:   number of local particles that need to be sent to each rank
//              sendPlan:           the send plan for the communication
//              numRanks:           number of ranks used for this process
// output:      sendIndexLists 2D vector
std::vector<std::vector<uint64_t>> createSendIndexLists(const std::vector<int> originalRank,  
                                                        const std::vector<uint64_t> numSendParticles, const std::vector<int> sendPlan, const int numRanks)
{   
    // a map than maps a rank to the associated communication step, where this local rank send to the selected one (step = rankStep[rank])
    // -1 means that no communication is scheduled to that rank. if this is accessed, there is an error
    std::vector<int> rankSteps(numRanks, -1);
    for (size_t step = 0; step < sendPlan.size(); step++){
        if (sendPlan[step] >= 0){rankSteps[sendPlan[step]] = static_cast<int>(step);}
    }

    // result container
    std::vector<std::vector<uint64_t>> sendIndexLists(sendPlan.size());
    for (size_t iStep = 0; iStep < sendPlan.size(); iStep++){
        int sendRank = sendPlan[iStep];
        if (sendRank >= 0){
            sendIndexLists[iStep].resize(numSendParticles[sendRank]);
        } else {
            sendIndexLists[iStep].resize(0);
        }
    }
    // vector to keep track where the next free position in each sendIndexList vector is
    std::vector<uint64_t> ind(sendPlan.size(), 0);

    // fill in the data
    for (size_t iPart = 0; iPart < originalRank.size(); iPart++){
        int targetRank = originalRank[iPart];
        if (targetRank < rankSteps.size()){
            int targetStep = rankSteps[targetRank];
            if ((targetStep >= 0) && (targetStep < sendPlan.size())){
                sendIndexLists[targetStep][ind[targetStep]] = static_cast<uint64_t>(iPart);
                ind[targetStep]++;
            } else {
                std::cout << "ERROR in createSendIndexLists: originalRank:" << originalRank[iPart] << " targetStep:" << targetStep << " out of bounds [0;" << sendPlan.size() << ")   targetRank:" << targetRank << std::endl;
            }
        } else {
            std::cout << "ERROR in createSendIndexLists: originalRank:" << originalRank[iPart] << " targetRank:" << targetRank << " >= rankSteps.size():" << rankSteps.size() << std::endl;
        }
    }

    // clear memory
    ind.clear();
    ind.shrink_to_fit();
    // return the result
    return sendIndexLists;
}


// function to create recvIndexLists, a 2D vector that contains the indizes for sorting recieved particles into the local vectors (sorting version)
// input:       IDs of the local particles
//              sendIndexLists created with the createSendIndexLists function
//              firstPart, the ID of the first particle on this rank (sorted)
//              sendPlan & recvPlan for communication
//              rank on which this process lives
// output:      recvIndexLists 2D vector
std::vector<std::vector<uint64_t>> createRecvIndexLists(const std::vector<uint64_t> IDs, const std::vector<std::vector<uint64_t>> sendIndexLists, const uint64_t firstPart,
                                                        const std::vector<int> sendPlan, const std::vector<int> recvPlan, const int rank)
{   
    // check inputs
    assert(("sendPlan & recvPlan in createRecvIndexList must have the same size", sendPlan.size() == recvPlan.size()));

    // number of particles that will be recieved during each comm step
    if (rank==0){std::cout << "\tdetermining how many particles will be recieved from each rank" << std::endl;}
    std::vector<uint64_t> numRecvParticles(recvPlan.size(), 0);
    uint64_t scalarSendBuffer = 0;
    uint64_t scalarRecvBuffer = 0;
    for (size_t iStep = 0; iStep < recvPlan.size(); iStep++){
        MPI_Barrier(MPI_COMM_WORLD);
        // ranks with which this rank communicates during the current step
        int sendRank = sendPlan[iStep];
        int recvRank = recvPlan[iStep];
        // is this rank sending / recieving data during the current step?
        bool isSending   = sendRank >= 0;
        bool isRecieving = recvRank >= 0;
        // creating send buffer
        if (isSending){scalarSendBuffer = static_cast<uint64_t>(sendIndexLists[iStep].size());}
        // communication
        if (rank == sendRank){
            scalarRecvBuffer = scalarSendBuffer;
        } else {
            if (isSending && isRecieving){
                MPI_Sendrecv((void *)&scalarSendBuffer, 1, MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                            (void *)&scalarRecvBuffer, 1, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (isSending){
                MPI_Send((void *)&scalarSendBuffer, 1, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
            } else if (isRecieving){
                MPI_Recv((void *)&scalarRecvBuffer, 1, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }
        // sorting in recieved data
        if (isRecieving){numRecvParticles[iStep] = scalarRecvBuffer;}
        else {numRecvParticles[iStep] = 0;}
    }

    // create the result buffer for recvIndexLists
    std::vector<std::vector<uint64_t>> recvIndexLists(recvPlan.size());
    for (size_t iStep = 0; iStep < recvPlan.size(); iStep++){
        int recvRank = recvPlan[iStep];
        if (recvRank >= 0){
            recvIndexLists[iStep].resize(numRecvParticles[iStep]);
        } else {
            recvIndexLists[iStep].resize(0);
        }
    }

    // communicate the indizes
    std::vector<uint64_t> vectorSendBuffer;
    std::vector<uint64_t> vectorRecvBuffer;
    for (size_t iStep = 0; iStep < recvPlan.size(); iStep++){
        MPI_Barrier(MPI_COMM_WORLD);
        // ranks with which this rank communicates during the current step
        int sendRank = sendPlan[iStep];
        int recvRank = recvPlan[iStep];
        // is this rank sending / recieving data during the current step?
        bool isSending   = sendRank >= 0;
        bool isRecieving = recvRank >= 0;
        // number of particles that will be sent / recieved
        int numSend = 0;
        int numRecv = 0;
        // creating send buffer
        if (isSending){
            numSend = sendIndexLists[iStep].size();
            vectorSendBuffer.resize(numSend);
            for (size_t iPart = 0; iPart < numSend; iPart++){vectorSendBuffer[iPart] = IDs[sendIndexLists[iStep][iPart]];}
        }
        if (isRecieving){
            numRecv = recvIndexLists[iStep].size();
            vectorRecvBuffer.resize(numRecv);
        }
        // communication
        if (rank == sendRank){
            vectorRecvBuffer = vectorSendBuffer;
        } else {
            if (isSending && isRecieving){
                MPI_Sendrecv((void *)&vectorSendBuffer[0], numSend, MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                            (void *)&vectorRecvBuffer[0], numRecv, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (isSending){
                MPI_Send((void *)&vectorSendBuffer[0], numSend, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
            } else if (isRecieving){
                MPI_Recv((void *)&vectorRecvBuffer[0], numRecv, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }
        // sorting the recieved data into the local storage
        if (isRecieving){
            for (size_t iPart = 0; iPart < numRecv; iPart++){
                if (firstPart > vectorRecvBuffer[iPart]){std::cout << "ERROR(" << rank << "): recieved Indizes are not local to this rank" << std::endl;}
                recvIndexLists[iStep][iPart] = vectorRecvBuffer[iPart] - firstPart;
            }
        }
    }

    // memory cleanup
    numRecvParticles.clear();
    vectorSendBuffer.clear();
    vectorRecvBuffer.clear();
    numRecvParticles.shrink_to_fit();
    vectorSendBuffer.shrink_to_fit();
    vectorRecvBuffer.shrink_to_fit();
    // return results
    return recvIndexLists;
}

// function to create recvIndexLists, a 2D vector that contains the indizes for sorting recieved particles into the local vectors (return version)
// input:       originalIndex, the original indizes of the local particles
//              sendIndexLists created with the createSendIndexLists function
//              sendPlan & recvPlan for communication
//              rank on which this process lives
// output:      recvIndexLists 2D vector
std::vector<std::vector<uint64_t>> createRecvIndexLists(const std::vector<uint64_t> originalIndex, const std::vector<std::vector<uint64_t>> sendIndexLists,
                                                        const std::vector<int> sendPlan, const std::vector<int> recvPlan, const int rank)
{   
    // check inputs
    assert(("sendPlan & recvPlan in createRecvIndexList must have the same size", sendPlan.size() == recvPlan.size()));

    // number of particles that will be recieved during each comm step
    if (rank==0){std::cout << "\tdetermining how many particles will be recieved from each rank" << std::endl;}
    std::vector<uint64_t> numRecvParticles(recvPlan.size(), 0);
    uint64_t scalarSendBuffer = 0;
    uint64_t scalarRecvBuffer = 0;
    for (size_t iStep = 0; iStep < recvPlan.size(); iStep++){
        MPI_Barrier(MPI_COMM_WORLD);
        // ranks with which this rank communicates during the current step
        int sendRank = sendPlan[iStep];
        int recvRank = recvPlan[iStep];
        // is this rank sending / recieving data during the current step?
        bool isSending   = sendRank >= 0;
        bool isRecieving = recvRank >= 0;
        // creating send buffer
        if (isSending){scalarSendBuffer = static_cast<uint64_t>(sendIndexLists[iStep].size());}
        // communication
        if (rank == sendRank){
            scalarRecvBuffer = scalarSendBuffer;
        } else {
            if (isSending && isRecieving){
                MPI_Sendrecv((void *)&scalarSendBuffer, 1, MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                            (void *)&scalarRecvBuffer, 1, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (isSending){
                MPI_Send((void *)&scalarSendBuffer, 1, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
            } else if (isRecieving){
                MPI_Recv((void *)&scalarRecvBuffer, 1, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }
        // sorting in recieved data
        if (isRecieving){numRecvParticles[iStep] = scalarRecvBuffer;}
    }

    // create the result buffer for recvIndexLists
    std::vector<std::vector<uint64_t>> recvIndexLists(recvPlan.size());
    for (size_t iStep = 0; iStep < recvPlan.size(); iStep++){
        int recvRank = recvPlan[iStep];
        if (recvRank >= 0){
            recvIndexLists[iStep].resize(numRecvParticles[iStep]);
        } else {
            recvIndexLists[iStep].resize(0);
        }
    }

    // communicate the indizes
    std::vector<uint64_t> vectorSendBuffer;
    std::vector<uint64_t> vectorRecvBuffer;
    for (size_t iStep = 0; iStep < recvPlan.size(); iStep++){
        MPI_Barrier(MPI_COMM_WORLD);
        // ranks with which this rank communicates during the current step
        int sendRank = sendPlan[iStep];
        int recvRank = recvPlan[iStep];
        // is this rank sending / recieving data during the current step?
        bool isSending   = sendRank >= 0;
        bool isRecieving = recvRank >= 0;
        // number of particles that will be sent / recieved
        int numSend = 0;
        int numRecv = 0;
        // creating send buffer
        if (isSending){
            numSend = sendIndexLists[iStep].size();
            vectorSendBuffer.resize(numSend);
            for (size_t iPart = 0; iPart < numSend; iPart++){vectorSendBuffer[iPart] = originalIndex[sendIndexLists[iStep][iPart]];}
        }
        if (isRecieving){
            numRecv = recvIndexLists[iStep].size();
            vectorRecvBuffer.resize(numRecv);
        }
        // communication
        if (rank == sendRank){
            vectorRecvBuffer = vectorSendBuffer;
        } else {
            if (isSending && isRecieving){
                MPI_Sendrecv((void *)&vectorSendBuffer[0], numSend, MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                            (void *)&vectorRecvBuffer[0], numRecv, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (isSending){
                MPI_Send((void *)&vectorSendBuffer[0], numSend, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
            } else if (isRecieving){
                MPI_Recv((void *)&vectorRecvBuffer[0], numRecv, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }
        // sorting the recieved data into the local storage
        if (isRecieving){
            for (size_t iPart = 0; iPart < numRecv; iPart++){
                recvIndexLists[iStep][iPart] = vectorRecvBuffer[iPart];
            }
        }
    }

    // memory cleanup
    numRecvParticles.clear();
    vectorSendBuffer.clear();
    vectorRecvBuffer.clear();
    numRecvParticles.shrink_to_fit();
    vectorSendBuffer.shrink_to_fit();
    vectorRecvBuffer.shrink_to_fit();
    // return results
    return recvIndexLists;
}



// function to communicate the data for step 2
template <typename T>
void dataComm(std::vector<T>& data, const std::vector<std::vector<uint64_t>> sendIndexLists, const std::vector<std::vector<uint64_t>> recvIndexLists,
              const std::vector<int> sendPlan, const std::vector<int> recvPlan, const uint64_t newSize, const char commType, const int rank)
{
    // check inputs
    assert(("commType must be 'd', 'u' or 'i' in dataComm", ((commType == 'd') || (commType == 'u') || (commType == 'i'))));
    assert(("sendIndexLists & recvIndexLists must have the same size in dataComm", sendIndexLists.size() == recvIndexLists.size()));
    assert(("sendPlan & recvPlan must have the same size in dataComm", sendPlan.size() == recvPlan.size()));
    assert(("sendIndexLists & sendPlan must have the same size in dataComm", sendIndexLists.size() == sendPlan.size()));

    // create send buffers
    std::vector<std::vector<T>> sendBuffers(sendPlan.size());
    for (size_t iStep = 0; iStep < sendPlan.size(); iStep++){sendBuffers[iStep].resize(sendIndexLists[iStep].size());}
    #pragma omp parallel for
    for (size_t iStep = 0; iStep < sendPlan.size(); iStep++){
        for (size_t iPart = 0; iPart < sendIndexLists[iStep].size(); iPart++){
            uint64_t dataIndex = sendIndexLists[iStep][iPart];
            sendBuffers[iStep][iPart] = data[dataIndex];
        }
    }

    // resize the local data vector
    data.resize(newSize);

    // communication
    std::vector<T> vectorRecvBuffer;
    for (size_t iStep = 0; iStep < sendPlan.size(); iStep++){
        MPI_Barrier(MPI_COMM_WORLD);
        // ranks with which this rank communicates during the current step
        int sendRank = sendPlan[iStep];
        int recvRank = recvPlan[iStep];
        // is this rank sending / recieving data during the current step?
        bool isSending   = sendRank >= 0;
        bool isRecieving = recvRank >= 0;
        // number of particles that will be sent / recieved
        int numSend = 0;
        int numRecv = 0;
        // resizing the recieve buffer if necessary
        if (isSending){numSend = sendIndexLists[iStep].size();}
        if (isRecieving){
            numRecv = recvIndexLists[iStep].size();
            vectorRecvBuffer.resize(numRecv);
        }
        // communication
        if (rank == sendRank){
            vectorRecvBuffer = sendBuffers[iStep];
        } else {
            if (commType == 'd'){
                if (isSending && isRecieving){
                    MPI_Sendrecv((void *)&sendBuffers[iStep][0], numSend, MPI_DOUBLE, sendRank, 1, 
                                (void *)&vectorRecvBuffer[0], numRecv, MPI_DOUBLE, recvRank, 1,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (isSending){
                    MPI_Send((void *)&sendBuffers[iStep][0], numSend, MPI_DOUBLE, sendRank, 1, MPI_COMM_WORLD);
                } else if (isRecieving){
                    MPI_Recv((void *)&vectorRecvBuffer[0], numRecv, MPI_DOUBLE, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                }
            } else if (commType == 'u'){
                if (isSending && isRecieving){
                    MPI_Sendrecv((void *)&sendBuffers[iStep][0], numSend, MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                                (void *)&vectorRecvBuffer[0], numRecv, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (isSending){
                    MPI_Send((void *)&sendBuffers[iStep][0], numSend, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
                } else if (isRecieving){
                    MPI_Recv((void *)&vectorRecvBuffer[0], numRecv, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                }
            } else if (commType == 'i'){
                if (isSending && isRecieving){
                    MPI_Sendrecv((void *)&sendBuffers[iStep][0], numSend, MPI_INT, sendRank, 1, 
                                (void *)&vectorRecvBuffer[0], numRecv, MPI_INT, recvRank, 1,
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (isSending){
                    MPI_Send((void *)&sendBuffers[iStep][0], numSend, MPI_INT, sendRank, 1, MPI_COMM_WORLD);
                } else if (isRecieving){
                    MPI_Recv((void *)&vectorRecvBuffer[0], numRecv, MPI_INT, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                }
            }
        }
        // sort the recieved data into the local storage
        if (isRecieving){
            for (size_t iPart = 0; iPart < numRecv; iPart++){
                uint64_t dataIndex = recvIndexLists[iStep][iPart];
                data[dataIndex] = vectorRecvBuffer[iPart];
            }
        }
    }

    // memory cleanup
    sendBuffers.clear();
    vectorRecvBuffer.clear();
    vectorRecvBuffer.shrink_to_fit();
    sendBuffers.shrink_to_fit();
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}



// function to sort timestep 2 by ID
// input:       IDs of the local particles (timestep 2)
//              ftleX, ftleY & ftleZ data for communicating
//              numPartsPerRank, number of sorted particles on each rank
//              rank, numRanks & numProcessors
// output:      NONE, ftleX, ftleY & ftleZ are modified accordingly
void sortStep2(const std::vector<uint64_t> IDs, 
               std::vector<uint64_t>& IDsTest, std::vector<double>& ftleX, std::vector<double>& ftleY, std::vector<double>& ftleZ,
               const std::vector<uint64_t> numPartsPerRank, 
               const int rank, const int numRanks, const int numProcessors)
{   
    // number of particles this rank should contain (sorted)
    const uint64_t numSortedParts = numPartsPerRank[rank];
    // index of the first (sorted) particles on this rank
    uint64_t firstIndex = 0;
    for (size_t i = 0; i < rank; i++){firstIndex += numPartsPerRank[i];}

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // find out how many local particles need to be sent where
    // ------------------------------------------------------------------------------------------------------------------------------------------

    // vector containing the number of local particles on this rank that need to be send to each other rank
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcounting how many particles need to be sent to which rank" << std::endl;}
    std::vector<uint64_t> numSendParticles(numRanks, 0);
    std::vector<std::vector<uint64_t>> numSendParticlesDist(numProcessors);
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        numSendParticlesDist[iProc].resize(numRanks);
        for (size_t iRank = 0; iRank < numRanks; iRank++){numSendParticlesDist[iProc][iRank] = 0;}
    }
    uint64_t numPartsPerProcessor = static_cast<uint64_t>(std::ceil(static_cast<double>(IDs.size()) / static_cast<double>(numProcessors)));
    // find out how many particles on this rank need to be send to each other rank
    #pragma omp parallel for 
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t i = iProc*numPartsPerProcessor; (i < (iProc+1)*numPartsPerProcessor) && (i < IDs.size()); i++){
            // find out on which rank this particle belongs
            int targetRank = static_cast<int>(std::floor(static_cast<double>(IDs[i]) / static_cast<double>(numPartsPerRank[0])));
            // increse the number of particles sent to this rank
            numSendParticlesDist[iProc][targetRank]++;
        }
    }
    // add up the distributed results
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t iRank = 0; iRank < numRanks; iRank++){numSendParticles[iRank] += numSendParticlesDist[iProc][iRank];}
    }
    // memory cleanup
    numSendParticlesDist.clear();
    numSendParticlesDist.shrink_to_fit();

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // creating the communication plan
    // ------------------------------------------------------------------------------------------------------------------------------------------

    if (rank == 0){std::cout << "\tcreating communication plan for sorting" << std::endl;}
    auto [sendPlan, recvPlan] = communicationPlan(numSendParticles, rank,numRanks);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcomm plan created" << std::endl;}

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // creating sendIndexLists
    // ------------------------------------------------------------------------------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcreating sendIndexLists" << std::endl;}
    std::vector<std::vector<uint64_t>> sendIndexLists = createSendIndexLists(IDs, numPartsPerRank[0], numSendParticles, sendPlan, numRanks);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tsendIndexLists created" << std::endl;}

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // creating recvIndexLists
    // ------------------------------------------------------------------------------------------------------------------------------------------
    // index lists to sort recieved data on this rank into the correct local positions

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcreating recvIndexLists" << std::endl;}
    std::vector<std::vector<uint64_t>> recvIndexLists = createRecvIndexLists(IDs, sendIndexLists,firstIndex, sendPlan,recvPlan,rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\trecvIndexLists created" << std::endl;}

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // communicating the data
    // ------------------------------------------------------------------------------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcommunicating the data" << std::endl;}
    dataComm(IDsTest, sendIndexLists,recvIndexLists, sendPlan,recvPlan, numSortedParts,'u', rank);
    dataComm(ftleX, sendIndexLists,recvIndexLists, sendPlan,recvPlan, numSortedParts,'d', rank);
    dataComm(ftleY, sendIndexLists,recvIndexLists, sendPlan,recvPlan, numSortedParts,'d', rank);
    dataComm(ftleZ, sendIndexLists,recvIndexLists, sendPlan,recvPlan, numSortedParts,'d', rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tdata communicated" << std::endl;}

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // memory cleanup
    // ------------------------------------------------------------------------------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tmemory cleanup" << std::endl;}
    numSendParticles.clear();
    sendPlan.clear();
    recvPlan.clear();
    sendIndexLists.clear();
    recvIndexLists.clear();
    numSendParticles.shrink_to_fit();
    sendPlan.shrink_to_fit();
    recvPlan.shrink_to_fit();
    sendIndexLists.shrink_to_fit();
    recvIndexLists.shrink_to_fit();
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tdone" << std::endl;}
    return;
}


// function to return timestep 2 data to its correct place
// input:       originalRank, the original rank to which each particle should be sent
//              originalIndex, the original Index on the originalRank, where each particle should be placed
//              ftleX, ftleY & ftleZ data vectors that will be communicated
//              numLocalParticles, number of local particles after communication
//              rank, numRanks & numProcessors
// output:      NONE, the ftleX, ftleY and ftleZ vectors are modified
void returnStep2(const std::vector<int> originalRank, const std::vector<uint64_t> originalIndex,
                 std::vector<uint64_t>& IDsTest, std::vector<double>& ftleX, std::vector<double>& ftleY, std::vector<double>& ftleZ,
                 const uint64_t numLocalParticles, const int rank, const int numRanks, const int numProcessors)
{   
    assert(("originalRank & originalIndex must have the same size in returnStep2", originalRank.size() == originalIndex.size()));
    // ------------------------------------------------------------------------------------------------------------------------------------------
    // find out how many local particles need to be sent where
    // ------------------------------------------------------------------------------------------------------------------------------------------

    // vector containing the number of local particles on this rank that need to be send to each other rank
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcounting how many particles need to be sent to which rank" << std::endl;}
    std::vector<uint64_t> numSendParticles(numRanks, 0);
    std::vector<std::vector<uint64_t>> numSendParticlesDist(numProcessors);
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        numSendParticlesDist[iProc].resize(numRanks);
        for (size_t iRank = 0; iRank < numRanks; iRank++){numSendParticlesDist[iProc][iRank] = 0;}
    }
    uint64_t numPartsPerProcessor = static_cast<uint64_t>(std::ceil(static_cast<double>(originalIndex.size()) / static_cast<double>(numProcessors)));
    // find out how many particles on this rank need to be send to each other rank
    #pragma omp parallel for 
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t i = iProc*numPartsPerProcessor; (i < (iProc+1)*numPartsPerProcessor) && (i < originalIndex.size()); i++){
            // find out on which rank this particle belongs
            int targetRank = originalRank[i];
            // increse the number of particles sent to this rank
            numSendParticlesDist[iProc][targetRank]++;
        }
    }
    // add up the distributed results
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t iRank = 0; iRank < numRanks; iRank++){numSendParticles[iRank] += numSendParticlesDist[iProc][iRank];}
    }
    // memory cleanup
    numSendParticlesDist.clear();
    numSendParticlesDist.shrink_to_fit();

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // creating the communication plan
    // ------------------------------------------------------------------------------------------------------------------------------------------

    if (rank == 0){std::cout << "\tcreating communication plan for sorting" << std::endl;}
    auto [sendPlan, recvPlan] = communicationPlan(numSendParticles, rank,numRanks);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcomm plan created" << std::endl;}

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // creating sendIndexLists
    // ------------------------------------------------------------------------------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcreating sendIndexLists" << std::endl;}
    std::vector<std::vector<uint64_t>> sendIndexLists = createSendIndexLists(originalRank, numSendParticles, sendPlan, numRanks);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tsendIndexLists created" << std::endl;}

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // creating recvIndexLists
    // ------------------------------------------------------------------------------------------------------------------------------------------
    // index lists to sort recieved data on this rank into the correct local positions

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcreating recvIndexLists" << std::endl;}
    std::vector<std::vector<uint64_t>> recvIndexLists = createRecvIndexLists(originalIndex, sendIndexLists, sendPlan,recvPlan,rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\trecvIndexLists created" << std::endl;}

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // communicating the data
    // ------------------------------------------------------------------------------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcommunicating the data" << std::endl;}
    dataComm(IDsTest, sendIndexLists,recvIndexLists, sendPlan,recvPlan, numLocalParticles,'u', rank);
    dataComm(ftleX, sendIndexLists,recvIndexLists, sendPlan,recvPlan, numLocalParticles,'d', rank);
    dataComm(ftleY, sendIndexLists,recvIndexLists, sendPlan,recvPlan, numLocalParticles,'d', rank);
    dataComm(ftleZ, sendIndexLists,recvIndexLists, sendPlan,recvPlan, numLocalParticles,'d', rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tdata communicated" << std::endl;}

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // memory cleanup
    // ------------------------------------------------------------------------------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tmemory cleanup" << std::endl;}
    numSendParticles.clear();
    sendPlan.clear();
    recvPlan.clear();
    sendIndexLists.clear();
    recvIndexLists.clear();
    numSendParticles.shrink_to_fit();
    sendPlan.shrink_to_fit();
    recvPlan.shrink_to_fit();
    sendIndexLists.shrink_to_fit();
    recvIndexLists.shrink_to_fit();
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tdone" << std::endl;}

    return;
}


// function to match the particles from two timesteps by ID
// timestep 2 will be sorted to match timestep 1
void IDMatching(std::vector<uint64_t> IDs1, std::vector<uint64_t> IDs2, 
                std::vector<uint64_t>& IDsTest, std::vector<double>& ftleX, std::vector<double>& ftleY, std::vector<double>& ftleZ, 
                const uint64_t totalNumParts,
                const int rank, const int numRanks, const int numProcessors)
{
    if (rank == 0){std::cout << "\nmatching the particle IDs:" << std::endl;}
    // number of particles local on this rank (timestep 1)
    uint64_t numLocalParts = static_cast<uint64_t>(IDs1.size());

    // number of sorted particles per rank
    std::vector<uint64_t> numPartsPerRank(numRanks);
    uint64_t defaultNum = static_cast<uint64_t>(std::ceil(static_cast<double>(totalNumParts)/static_cast<double>(numRanks)));
    for (size_t iRank = 0; iRank < numRanks; iRank++){
        if (iRank == numRanks-1){numPartsPerRank[iRank] = totalNumParts - iRank*defaultNum;} 
        else {                   numPartsPerRank[iRank] = defaultNum;}
    }

    // compute the first local ID on each rank
    std::vector<uint64_t> firstLocalID(numRanks);
    firstLocalID[0] = 0;
    for (size_t iRank = 1; iRank < numRanks; iRank++){
        firstLocalID[iRank] = firstLocalID[iRank-1] + numPartsPerRank[iRank-1];
    }

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // sorting timestep 1
    // ------------------------------------------------------------------------------------------------------------------------------------------

    if (rank == 0){std::cout << "sorting timestep 1" << std::endl;}
    auto Runtime1Start = std::chrono::high_resolution_clock::now();
    auto [originalRank,originalIndex] = sortStep1(IDs1,numPartsPerRank,firstLocalID, rank,numRanks,numProcessors);
    auto Runtime1Stop  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> Runtime1 = Runtime1Stop - Runtime1Start;
    if (rank == 0){std::cout << "runtime for sorting timestep 1: " << Runtime1.count() << std::endl;}

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // sorting timestep 2
    // ------------------------------------------------------------------------------------------------------------------------------------------

    if (rank == 0){std::cout << "\nsorting timestep 2" << std::endl;}
    auto Runtime2Start = std::chrono::high_resolution_clock::now();
    sortStep2(IDs2, IDsTest,ftleX,ftleY,ftleZ, numPartsPerRank, rank,numRanks,numProcessors);
    returnStep2(originalRank,originalIndex, IDsTest,ftleX,ftleY,ftleZ, numLocalParts, rank,numRanks,numProcessors);
    auto Runtime2Stop  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> Runtime2 = Runtime2Stop - Runtime2Start;
    if (rank == 0){std::cout << "runtime for sorting timestep 2: " << Runtime2.count() << std::endl;}

    // ------------------------------------------------------------------------------------------------------------------------------------------
    // memory cleanup
    // ------------------------------------------------------------------------------------------------------------------------------------------
    if (rank == 0){std::cout << "memory cleanup" << std::endl;}
    numPartsPerRank.clear();
    firstLocalID.clear();
    originalRank.clear();
    originalIndex.clear();
    numPartsPerRank.shrink_to_fit();
    firstLocalID.shrink_to_fit();
    originalRank.shrink_to_fit();
    originalIndex.shrink_to_fit();
    return;
}