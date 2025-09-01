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

#include "sort.hpp"

// communicate a data field given the comm meta data
// inputs:  the vector that should be communicated
//          a 2D vector containing the index lists to create the sendbuffers from the data vector ([iStep][iParticle])
//          a 2D vector containing the positions at which recieved data values should be placed (pos[iStep][iParticle] = index)
//          the sendPlan & recievePlan for the communication
//          size of the vector after communication / how many particles the final vector should hold
//          the rank on which this process lives
// output:  None, the input vector is modified accordingly
template <typename T>
void sortComm2(std::vector<T>& vec, 
               const std::vector<std::vector<uint64_t>> sendIndexLists, const std::vector<std::vector<uint64_t>> recvIndexLists,
               const std::vector<int> sendPlan, const std::vector<int> recvPlan,
               const uint64_t newSize, const int rank)
{   
    //if (rank == 0){std::cout << "communicating data vector " << vec.size() << std::endl;}
    //if (rank == 0){std::cout << "sendPlan.size()=" << sendPlan.size() << std::endl;}
    // check the inputs
    assert(("sendPlan and recvPlan in sortComm2 must have the same size", sendPlan.size() == recvPlan.size()));
    assert(("send/recvPlan and sendIndexLists in sortComm2 must have the same size", sendPlan.size() == sendIndexLists.size()));
    assert(("send/recvPlan and recvIndexLists in sortComm2 must have the same size", sendPlan.size() == recvIndexLists.size()));
    uint64_t recvIndexSize = 0;
    for (size_t i = 0; i < recvIndexLists.size(); i++){recvIndexSize += static_cast<uint64_t>(recvIndexLists[i].size());}
    assert(("size of the recvIndexLists and newSize must be equal in sortComm2", recvIndexSize == newSize));
    uint64_t sendIndexSize = 0;
    for (size_t i = 0; i < sendIndexLists.size(); i++){sendIndexSize += static_cast<uint64_t>(sendIndexLists[i].size());}
    assert(("size of the sendIndexLists and vec must be equal in sortComm2", sendIndexSize == vec.size()));

    //if (rank == 15){std::cout << "\tcreating send buffers" << std::endl;}
    // create the send buffers
    std::vector<std::vector<T>> sendBuffers(sendPlan.size());
    // resize the buffers
    //if (rank == 15){std::cout << "\t\tresizing the sendBuffers" << std::endl;}
    for (size_t step = 0; step < sendPlan.size(); step++){
        //if (rank == 2){std::cout << "\t\tstep: " << step << " newsize: " << sendIndexLists[step].size() << " currentsize: " << sendBuffers[step].size() << std::endl;}
        uint64_t newSize = static_cast<uint64_t>(sendIndexLists[step].size());
        sendBuffers[step].resize(newSize);
    }
    // fill in the data
    //if (rank == 15){std::cout << "\t\tfilling in the data" << std::endl;}
    for (size_t step = 0; step < sendPlan.size(); step++){
        for (size_t iPart = 0; iPart < sendIndexLists[step].size(); iPart++){
            assert(("sendIndexLists[step][iPart] >= vec.size() in sortComm2", sendIndexLists[step][iPart] < vec.size()));
            assert(("iPart >= sendBuffers[step].size() in sortComm2", iPart < sendBuffers[step].size()));
            sendBuffers[step][iPart] = vec[sendIndexLists[step][iPart]];
        }
    }
    //if (rank == 2){std::cout << "\tsend buffers created" << std::endl;}

    // resize the original vector
    vec.resize(newSize);

    // recieve buffer
    std::vector<T> sendBuffer;
    std::vector<T> recvBuffer;
    // communication
    for (size_t step = 0; step < sendPlan.size(); step++){
        MPI_Barrier(MPI_COMM_WORLD);
        //if (rank == 15){std::cout << "\tstep: " << step << std::endl;}
        // rank to which trhis rank sends / from which this rank recieves data
        int sendRank = sendPlan[step];
        int recvRank = recvPlan[step];
        // is this rank actively sending / recieving at the current comm step
        bool isSending   = sendRank >= 0;
        bool isRecieving = recvRank >= 0;
        // number of particles that will be sent / recieved
        uint64_t numSendParts = static_cast<uint64_t>(sendBuffers[step].size());
        uint64_t numRecvParts = static_cast<uint64_t>(recvIndexLists[step].size());
        
        // resize the recieve buffer
        recvBuffer.resize(numRecvParts);
        sendBuffer.resize(numSendParts);
        for (size_t i = 0; i < numSendParts; i++){sendBuffer[i] = sendBuffers[step][i];}

        // communicate the data
        // MPI_Barrier(MPI_COMM_WORLD);
        // if (isSending && isRecieving){
        //     std::cout << "\trank " << rank << "("<<step<<"): communicating " << recvRank << "-> (" << numRecvParts << ") " << rank << " (" << numSendParts << ") -> " << sendRank << std::endl;
        // } else if (isSending){
        //     std::cout << "\trank " << rank << "("<<step<<"): communicating " << rank << " (" << numSendParts << ") -> " << sendRank << std::endl;
        // } else if (isRecieving){
        //     std::cout << "\trank " << rank << "("<<step<<"): communicating " << recvRank << " -> (" << numRecvParts << ") " << rank << std::endl;
        // } else {
        //     std::cout << "\trank " << rank << "("<<step<<"): noComm" << std::endl;
        // }
        MPI_Barrier(MPI_COMM_WORLD);
        if (isSending && (sendRank == rank)){
            //std::cout << "\trank " << rank << "("<<step<<"): using selfcomm step " << step << std::endl;
            recvBuffer = sendBuffers[step];
        } else {
            using vecType = decltype(vec);
            if (isSending && isRecieving){
                if ((std::is_same<vecType, std::vector<double>>::value) || (std::is_same<vecType, std::vector<double>&>::value)){
                    //std::cout << "\trank " << rank << "("<<step<<"): using double at step " << step << " " << sendBuffers[step].size() << "/" << numSendParts << " " << recvBuffer.size() << "/" << numRecvParts << std::endl;
                    MPI_Sendrecv((void *)&sendBuffer[0], numSendParts, MPI_DOUBLE, sendRank, 1, 
                                 (void *)&recvBuffer[0], numRecvParts, MPI_DOUBLE, recvRank, 1,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if ((std::is_same<vecType, std::vector<uint64_t>>::value) || (std::is_same<vecType, std::vector<uint64_t>&>::value)){
                    //std::cout << "\trank " << rank << "("<<step<<"): using uint64_t at step " << step << " " << sendBuffers[step].size() << "/" << numSendParts << " " << recvBuffer.size() << "/" << numRecvParts << std::endl;
                    MPI_Sendrecv((void *)&sendBuffer[0], numSendParts, MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                                 (void *)&recvBuffer[0], numRecvParts, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if ((std::is_same<vecType, std::vector<int>>::value) || (std::is_same<vecType, std::vector<int>&>::value)){
                    //std::cout << "\trank " << rank << "("<<step<<"): using int at step " << " " << sendBuffers[step].size() << "/" << numSendParts << " " << recvBuffer.size() << "/" << numRecvParts << std::endl;
                    MPI_Sendrecv((void *)&sendBuffer[0], numSendParts, MPI_INT, sendRank, 1, 
                                 (void *)&recvBuffer[0], numRecvParts, MPI_INT, recvRank, 1,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {std::cout << "\trank " << rank << ": ERROR: unidentified type in sortComm2 sendrecv" << std::endl;}
            } else if (isSending) {
                if ((std::is_same<vecType, std::vector<double>>::value) || (std::is_same<vecType, std::vector<double>&>::value)){
                    //std::cout << "\trank " << rank << "("<<step<<"): using double at step " << step << std::endl;
                    MPI_Send((void *)&sendBuffer[0], numSendParts, MPI_DOUBLE, sendRank, 1, MPI_COMM_WORLD);
                } else if ((std::is_same<vecType, std::vector<uint64_t>>::value) || (std::is_same<vecType, std::vector<uint64_t>&>::value)){
                    //std::cout << "\trank " << rank << "("<<step<<"): using uint64_t at step " << step << std::endl;
                    MPI_Send((void *)&sendBuffer[0], numSendParts, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
                } else if ((std::is_same<vecType, std::vector<int>>::value) || (std::is_same<vecType, std::vector<int>&>::value)){
                    //std::cout << "\trank " << rank << "("<<step<<"): using int at step " << step << std::endl;
                    MPI_Send((void *)&sendBuffer[0], numSendParts, MPI_INT, sendRank, 1, MPI_COMM_WORLD);
                } else {std::cout << "\trank " << rank << ": ERROR: unidentified type in sortComm2 send" << std::endl;}
            } else if (isRecieving) {
                if ((std::is_same<vecType, std::vector<double>>::value) || (std::is_same<vecType, std::vector<double>&>::value)){
                    //std::cout << "\trank " << rank << "("<<step<<"): using double at step " << step << std::endl;
                    MPI_Recv((void *)&recvBuffer[0], numRecvParts, MPI_DOUBLE, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                } else if ((std::is_same<vecType, std::vector<uint64_t>>::value) || (std::is_same<vecType, std::vector<uint64_t>&>::value)){
                    //std::cout << "\trank " << rank << "("<<step<<"): using uint64_t at step " << step << std::endl;
                    MPI_Recv((void *)&recvBuffer[0], numRecvParts, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                } else if ((std::is_same<vecType, std::vector<int>>::value) || (std::is_same<vecType, std::vector<int>&>::value)){
                    //std::cout << "\trank " << rank << "("<<step<<"): using int at step " << step << std::endl;
                    MPI_Recv((void *)&recvBuffer[0], numRecvParts, MPI_INT, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                } else {std::cout << "\trank " << rank << ": ERROR: unidentified type in sortComm2 recv" << std::endl;}
            }
        }
        //if (rank == 0){std::cout << "\tsorting in the data" << std::endl;}
        // sort the data into the original vector
        if (isRecieving){
            for (size_t iPart = 0; iPart < numRecvParts; iPart++){
                vec[recvIndexLists[step][iPart]] = recvBuffer[iPart];
            }
        }
    }
    // memory cleanup
    sendBuffers.clear();
    sendBuffer.clear();
    recvBuffer.clear();
    sendBuffers.shrink_to_fit();
    sendBuffer.shrink_to_fit();
    recvBuffer.shrink_to_fit();
    MPI_Barrier(MPI_COMM_WORLD);
    return;
}

// sort the communications by number of particles that need to be communicated
// also cuts communications where no particles are communicated
void sortComms(std::vector<uint64_t>& numParts, std::vector<int>& Sender, std::vector<int>& Reciever)
{
    // sort the communication by the number of particles
    sort(numParts,std::tie(Sender,Reciever),1,true);

    // cut the communications where no particles will be communicated
    uint64_t numCut = 0;
    for (size_t iComm = 1; iComm <= numParts.size(); iComm++){
        if (numParts[numParts.size()-iComm] == 0){numCut++;} 
        else {break;}
    }
    Sender.resize(    Sender.size() - numCut);
    Reciever.resize(Reciever.size() - numCut);
    Sender.shrink_to_fit();
    Reciever.shrink_to_fit();
    return;
}

// this function creates a communication plan to reduce load imbalance
// input:   vector containing the number of local particles on this rank that need to be sent to each other rank
// output:  vector containing the rank numbers to which this rank should send its data (in the correct order)
//          vector containing the rank numbers from which this rank will recieve data (in the correct order)
std::tuple<std::vector<int>,std::vector<int>> communicationPlan(const std::vector<uint64_t> numSendParticles, const int rank, const int numRanks)
{   
    // checking inputs
    assert(("invalid rank input in communicationPlan(), rank must be smaller than numRanks", rank < numRanks));
    assert(("to create a communicationPlan, numSendParticles.size() and numRanks must be equal", numRanks == numSendParticles));
    // total number of communication processes
    uint64_t totalComms = static_cast<uint64_t>(numRanks) * static_cast<uint64_t>(numRanks);
    // collection of all numSendParticles vectors on rank 0
    std::vector<int>        Sender;
    std::vector<int>        Reciever;
    std::vector<uint64_t>   numParts;
    // number of communication steps required
    int numCommSteps = 0;
    // individual send plan for each rank
    // sendPlan[iRank] conatins the vector with the ranks to which iRank will send data in the correct order
    std::vector<std::vector<int>> sendPlan(numRanks);
    // individual recieve plan for each rank
    // recvPlan[iRank] conatins the vector with the ranks from which iRank will recieve data in the correct order
    std::vector<std::vector<int>> recvPlan(numRanks);
    // buffer for reieving data
    std::vector<uint64_t> buffer1;
    if (rank == 0){
        buffer1.resize(numRanks);
        Sender.resize(totalComms);
        Reciever.resize(totalComms);
        numParts.resize(totalComms);
    }

    // sending all numSendParticles vectors to rank 0 so that it can create the communication plans
    if (rank == 0){std::cout << "\tsending all data to rank 0" << std::endl;}
    uint64_t ind = 0;
    for (size_t iRank = 0; iRank < numRanks; iRank++){
        MPI_Barrier(MPI_COMM_WORLD);
        // communication
        if (iRank == 0){
            if (rank == 0){
                buffer1 = numSendParticles;
            }
        } else {
            // sending the vector
            if (rank == iRank){
                MPI_Send((void *)&numSendParticles[0], numRanks, MPI_UNSIGNED_LONG_LONG, 0, 1, MPI_COMM_WORLD);
            // recieving the data
            } else if (rank == 0){
                MPI_Recv((void *)&buffer1[0], numRanks, MPI_UNSIGNED_LONG_LONG, iRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }
        // assigning the data
        if (rank == 0){
            for (size_t i = 0; i < numRanks; i++){
                Sender[ind]     = static_cast<int>(iRank);
                Reciever[ind]   = static_cast<int>(i);
                numParts[ind]   = static_cast<uint64_t>(buffer1[i]);
                ind++;
            }
        }
    }
    // memory cleanup
    buffer1.clear();
    buffer1.shrink_to_fit();



    // rank 0 creates the communication plan
    if (rank == 0){
        assert(("totalComms must be equal to the numParts.size() in communicationPlan()", totalComms == numParts.size()));
        assert(("totalComms must be equal to the Sender.size() in communicationPlan()", totalComms == Sender.size()));
        assert(("totalComms must be equal to the Reciever.size() in communicationPlan()", totalComms == Reciever.size()));

        // std::cout << "\tSender / Reciever (sorted):" << std::endl;
        // for (size_t i = 0; i < Sender.size(); i++){std::cout << "\t\t" << Sender[i] << "/" << Reciever[i] << std::endl;}

        std::cout << "\tsorting the communications" << std::endl;
        // sort the communication processes by number of communicated particles (numParts)
        sortComms(numParts,Sender,Reciever);
        totalComms = Sender.size();
        std::cout << "\tcommunications sorted, new size:" << numParts.size() << " " << Sender.size()<< " " << Reciever.size() << std::endl;

        // std::cout << "\tSender / Reciever (sorted):" << std::endl;
        // for (size_t i = 0; i < Sender.size(); i++){std::cout << "\t\t" << Sender[i] << "/" << Reciever[i] << std::endl;}

        // clear memeory
        // numParts.clear();
        // numParts.shrink_to_fit();

        // position of the first unscheduled communication
        uint64_t firstUnscheduledComm = 0;
        // vector to keep track of which communications are not yet scheduled (true: not scheduled; false: scheduled)
        std::vector<bool> notScheduled(totalComms, true);
        // vector to keep track of which ranks are available for sending in the current comm step and which are already busy
        // true: available; false: busy
        std::vector<bool> sendAvailable(numRanks, true);
        // vector to keep track of which ranks are available for recieving in the current comm step and which are already busy
        // true: available; false: busy
        std::vector<bool> recvAvailable(numRanks, true);
        // vector to keep track of which ranks still need to send data
        // true: this rank still needs to send data; false: this rank is done sending
        std::vector<bool> rankActive(numRanks, true);
        // communication plan
        // contains the index of the reciever rank for each communication step at the position of the sender
        // example: in step 3 of the communication, rank 2 sends to rank 5 --- commPlan[3][2] = 5
        // a -1 means that there is no comm scheduled for this step
        std::vector<std::vector<int>> commPlan(numRanks);
        for (size_t i = 0; i < numRanks; i++){
            commPlan[i].resize(numRanks);
            for (size_t j = 0; j < numRanks; j++){commPlan[i][j] = -1;}
        }

        // vector to keep track for which ranks we still see activity (need to send data) in the current comm step 
        // for this it does not matter wether the rank CAN send in the current step, the question is wether it NEEDS to
        // true: the rank is active; false: the rank is done sending
        std::vector<bool> rankActivity(numRanks,false);


        // creating the plan
        std::cout << "\tcreating the plan step by step" << std::endl;
        for (size_t iStep = 0; iStep < numRanks; iStep++){
            //std::cout << "\n\t\tstep " << iStep << std::endl;
            // increase the number of communication steps
            numCommSteps++;
            // reset the availabilities
            for (size_t iRank = 0; iRank < numRanks; iRank++){
                sendAvailable[iRank] = true;
                recvAvailable[iRank] = true;
                rankActivity[iRank] = false;
            }
            // is this comm step done?
            bool stepDone = false;
            // was a new firstUnscheduledComm set during this step? (true: no, westill need to set one; false: yes, we already have a new one)
            bool setNewFirst = true;
            // loop over the communications
            for (size_t iComm = firstUnscheduledComm; iComm < totalComms; iComm++){
                //std::cout << "\t\tiComm " << iComm << "/" << totalComms << " " << numParts[iComm] << " " << Sender[iComm] << " " << Reciever[iComm] << " - " << notScheduled[iComm];
                // check if this communication is already scheduled
                if (notScheduled[iComm]){
                    // check if both sender & reciever are available
                    if ((sendAvailable[Sender[iComm]]) && (recvAvailable[Reciever[iComm]])){
                        // schedule the communication
                        rankActivity[Sender[iComm]] = true;
                        commPlan[iStep][Sender[iComm]] = Reciever[iComm];
                        notScheduled[iComm] = false;
                        sendAvailable[Sender[iComm]]   = false;
                        recvAvailable[Reciever[iComm]] = false;
                        // check if this comm step is done
                        stepDone = true;
                        for (size_t iRank = 0; iRank < numRanks; iRank++){
                            if ((commPlan[iStep][iRank] == -1) && rankActive[iRank]){
                                stepDone = false;
                                break;
                            }
                        }
                        if (iComm == totalComms-1){stepDone = true;}
                        if (stepDone){
                            // set a new firstUnscheduledComm if we don't have one
                            if (setNewFirst){
                                firstUnscheduledComm = iComm;
                                while (setNewFirst){
                                    firstUnscheduledComm += 1;
                                    if (firstUnscheduledComm == totalComms){        setNewFirst = false;}
                                    else if (notScheduled[firstUnscheduledComm]){   setNewFirst = false;}
                                }
                            }
                            // end the Comm loop
                            break;
                        }
                    } else {
                        rankActivity[Sender[iComm]] = true;
                        // set a new firstUnscheduledComm if none was set already
                        if (setNewFirst){
                            firstUnscheduledComm = static_cast<uint64_t>(iComm);
                            setNewFirst = false;
                        }
                    }
                }
                //std::cout << "/" << notScheduled[iComm] << std::endl;
            } // end of the Comm loop
            // check if there are unscheduled Communications left
            if (firstUnscheduledComm >= totalComms){break;}
            // check which ranks still need to send data
            rankActive = rankActivity;
        } // end of the Comm step loop
        // memory cleanup
        notScheduled.clear();
        sendAvailable.clear();
        recvAvailable.clear();
        rankActive.clear();
        rankActivity.clear();
        notScheduled.shrink_to_fit();
        sendAvailable.shrink_to_fit();
        recvAvailable.shrink_to_fit();
        rankActive.shrink_to_fit();
        rankActivity.shrink_to_fit();
        // end of creating the communication plan



        // creating the individual plans for each rank
        if (rank == 0){std::cout << "\tcreating the individual plans" << std::endl;}
        // resizing the vectors
        for (size_t iRank = 0; iRank < numRanks; iRank++){
            sendPlan[iRank].resize(numCommSteps);
            recvPlan[iRank].resize(numCommSteps);
            for (size_t iStep = 0; iStep < numCommSteps; iStep++){
                recvPlan[iRank][iStep] = -1;
            }
        }
        //creating the individual plans
        #pragma omp parallel for 
        for (size_t iRank = 0; iRank < numRanks; iRank++){
            for (size_t iStep = 0; iStep < numCommSteps; iStep++){
                sendPlan[iRank][iStep] = commPlan[iStep][iRank];
                for (size_t ind = 0; ind < numRanks; ind++){
                    if (iRank == commPlan[iStep][ind]){
                        recvPlan[iRank][iStep] = static_cast<int>(ind);
                    }
                }
            }
        }
        // memory cleanup
        commPlan.clear();
        commPlan.shrink_to_fit();
    } // end of rank 0 specific code 
    
    
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcommunicating the individual plans" << std::endl;}
    // broadcast the number of communication steps to all ranks
    MPI_Bcast((void *)&numCommSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // plan for sending for THIS rank
    std::vector<int> sendList(numCommSteps);
    // plan for recieving for THIS rank
    std::vector<int> recvList(numCommSteps);

    // distributing the individual plans to the ranks
    for (size_t iRank = 1; iRank < numRanks; iRank++){
        if (rank == 0){
            MPI_Send((void *)&sendPlan[iRank][0], numCommSteps, MPI_INT, iRank, 1, MPI_COMM_WORLD);
            MPI_Send((void *)&recvPlan[iRank][0], numCommSteps, MPI_INT, iRank, 1, MPI_COMM_WORLD);
        } else if (rank == iRank){
            MPI_Recv((void *)&sendList[0], numRanks, MPI_INT, 0, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Recv((void *)&recvList[0], numRanks, MPI_INT, 0, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
    }
    if (rank == 0){
        sendList = sendPlan[0];
        recvList = recvPlan[0];
    }
    // memory cleanup
    sendPlan.clear();
    recvPlan.clear();
    sendPlan.shrink_to_fit();
    recvPlan.shrink_to_fit();
    // return results
    MPI_Barrier(MPI_COMM_WORLD);
    return std::make_tuple(sendList,recvList);
}




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

// function to sort the data in timestep 2 by ID
// this function is used for matching the data of two timesteps (used for the FTLE computation & diffusion coefficients)
// intput:  ID vector of the second timestep
//          tuple of data vectors that will be sorted by ID (usually x, y and z)
//          vector containing the number of (sorted) particles per Rank
//          number of particles that will be local on this rank at the end of the routine
//          vectors containing the original rank & index on that rank for sending the sorted data back to where it belongs
// output:  None, the drag vectors will be modified inplace
template<class... dragVector_t>
void sortStep2(std::vector<uint64_t> IDs, std::vector<uint64_t>& IDsTest, std::vector<double>& ftleX, std::vector<double>& ftleY, std::vector<double>& ftleZ,
                const std::vector<uint64_t> numPartsPerRank, const uint64_t numLocalParts, const std::vector<uint64_t> firstLocalID,
                const std::vector<int> originalRank, const std::vector<uint64_t> originalIndex, const int rank, const int numRanks, const int numProcessors)
{   
    std::cout << rank << ": numParticles = " << IDs.size() << std::endl;
    // make sure that the originalRank & originalIndex input match the numPartsPerRank input
    assert(("the size of originalRank must be equal to the numPartsPerRank input (sortStep2)"  originalRank.size()  == numPartsPerRank[rank]));
    assert(("the size of originalIndex must be equal to the numPartsPerRank input (sortStep2)", originalIndex.size() == numPartsPerRank[rank]));
    assert(("the size of firstLocalID must match the numRanks input (sortStep2)", firstLocalID.size() == numRanks));

    // number of particles this rank should contain (sorted)
    const uint64_t numSortedParts = numPartsPerRank[rank];
    // index of the first (sorted) particles on this rank
    uint64_t firstIndex = 0;
    for (size_t i = 0; i < rank; i++){firstIndex += numPartsPerRank[i];}

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
    assert(("in sortStep2 not all particles were checked for where they should be sent", checksum == IDs.size()));



    // create a communication plan
    if (rank == 0){std::cout << "\tcreating communication plan for sorting" << std::endl;}
    auto [sendPlan, recvPlan] = communicationPlan(numSendParticles, rank,numRanks);
    assert(("ERROR: send & recvPlan must have the same size!", sendPlan.size() == recvPlan.size()));
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcomm plan created" << std::endl;}


    if (rank == 0){std::cout << "\tcreating sendIndexLists" << std::endl;}
    // create a table rank -> commStep to speed up cpmputation (table is for sending)
    std::vector<int> rankSteps(sendPlan.size(), -1);
    for (size_t step = 0; step < sendPlan.size(); step++){
        if (sendPlan[step] >= 0){rankSteps[sendPlan[step]] = static_cast<int>(step);}
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\ta" << std::endl;}
    // create index lists for finding the particles to send in each step more easily [step] retuirns the list for a step ind sendPlan
    std::vector<std::vector<uint64_t>> sendIndexLists(sendPlan.size());
    std::vector<uint64_t> ind(sendPlan.size(), 0);
    for (size_t step = 0; step < sendPlan.size(); step++){sendIndexLists[step].resize(numSendParticles[sendPlan[step]]);}
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tb" << std::endl;}
    for (size_t iPart = 0; iPart < IDs.size(); iPart++){
        int trueRank = static_cast<int>(std::floor(static_cast<double>(IDs[iPart]) / static_cast<double>(numPartsPerRank[0])));
        sendIndexLists[rankSteps[trueRank]][ind[rankSteps[trueRank]]] = static_cast<uint64_t>(iPart);
        ind[rankSteps[trueRank]]++;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tc" << std::endl;}
    // memory cleanup
    ind.clear();
    rankSteps.clear();
    ind.shrink_to_fit();
    rankSteps.shrink_to_fit();
    if (rank == 0){std::cout << "\tsendIndexLists created" << std::endl;}


    if (rank == 0){std::cout << "\tcreating recvIndexLists" << std::endl;}
    // create index lists for sorting the communicated particles into the local data
    std::vector<std::vector<uint64_t>> recvIndexLists(recvPlan.size());
    // send buffer for the IDs
    std::vector<uint64_t> IDsBuffer;
    // communication
    for (size_t step = 0; step < sendPlan.size(); step++){
        // ranks with which this rank will communicate
        int sendRank = sendPlan[step];
        int recvRank = recvPlan[step];
        // check if this rank is active sending / recieving data in this step
        bool isSending   = sendRank >= 0;
        bool isRecieving = recvRank >= 0;
        // number of particles that will be sent
        uint64_t numSendParts = static_cast<uint64_t>(sendIndexLists[step].size());
        uint64_t numRecvParts = 0;

        if (sendRank == rank){
            recvIndexLists[step].resize(numSendParts);
            for (size_t i = 0; i < numSendParts; i++){recvIndexLists[step][i] = IDs[sendIndexLists[step][i]];}
        } else {
            // communicate the number of particles that will be sent/recieved
            if (isSending && isRecieving){
                MPI_Sendrecv((void *)&numSendParts, 1, MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                            (void *)&numRecvParts, 1, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (isSending) {
                MPI_Send((void *)&numSendParts, 1, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
            } else if (isRecieving) {
                MPI_Recv((void *)&numRecvParts, 1, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }

            // resize the section of recvIndexLists
            recvIndexLists[step].resize(numRecvParts);

            // assemble the send buffer
            if (isSending){
                IDsBuffer.resize(numSendParts);
                #pragma omp parallel for
                for (size_t i = 0; i < numSendParts; i++){IDsBuffer[i] = IDs[sendIndexLists[step][i]];}
            }
            // communicate the indizes
            if (isSending && isRecieving){
                MPI_Sendrecv((void *)&IDsBuffer[0], numSendParts, MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                            (void *)&recvIndexLists[step][0], numRecvParts, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (isSending) {
                MPI_Send((void *)&IDsBuffer[0], numSendParts, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
            } else if (isRecieving) {
                MPI_Recv((void *)&recvIndexLists[step][0], numRecvParts, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
        }
        // reduce the recieved indizes to adjust for the firstindex and get usable vector indizes
        if (isRecieving){
            for (size_t i = 0; i < numRecvParts; i++){
                if (firstIndex > recvIndexLists[step][i]){std::cout << "ERROR " << rank << ": there must be a problem in assembling the recvIndexLists for sorting in sortStep2" << std::endl;}
                recvIndexLists[step][i] -= firstIndex;
            }
        }
    } // end of index communication
    // memory cleanup
    IDsBuffer.clear();
    IDsBuffer.shrink_to_fit();
    if (rank == 0){std::cout << "\trecvIndexLists created" << std::endl;}



    if (rank == 0){std::cout << "\tcommunicating the data" << std::endl;}
    // communicate the data (sorting)
    sortComm2(IDsTest, sendIndexLists,recvIndexLists, sendPlan,recvPlan, numSortedParts,rank);
    sortComm2(ftleX, sendIndexLists,recvIndexLists, sendPlan,recvPlan, numSortedParts,rank);
    sortComm2(ftleY, sendIndexLists,recvIndexLists, sendPlan,recvPlan, numSortedParts,rank);
    sortComm2(ftleZ, sendIndexLists,recvIndexLists, sendPlan,recvPlan, numSortedParts,rank);
    // memory cleanup
    recvPlan.clear();
    sendPlan.clear();
    recvPlan.shrink_to_fit();
    sendPlan.shrink_to_fit();

    sendIndexLists.clear();
    recvIndexLists.clear();
    sendIndexLists.shrink_to_fit();
    recvIndexLists.shrink_to_fit();
    if (rank == 0){std::cout << "\tdata communicated" << std::endl;}


    // return the sorted data to its true position -------------------------------------------------------------------------------------------------------------

    // find out how many particles need to be sent where
    for (size_t i = 0; i < numRanks; i++){numSendParticles[i] = 0;}
    std::vector<std::vector<uint64_t>> numSendParticlesDist2(numProcessors);
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        numSendParticlesDist2[iProc].resize(numRanks);
        for (size_t iRank = 0; iRank < numRanks; iRank++){numSendParticlesDist2[iProc][iRank] = 0;}
    }
    numPartsPerProcessor = static_cast<uint64_t>(std::ceil(static_cast<double>(numSortedParts) / static_cast<double>(numProcessors)));
    // find out how many particles on this rank need to be send to each other rank
    #pragma omp parallel for 
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t i = iProc*numPartsPerProcessor; (i < (iProc+1)*numPartsPerProcessor) && (i < IDs.size()); i++){
            // increse the number of particles sent to this rank
            numSendParticlesDist2[iProc][originalRank[i]]++;
        }
    }
    // add up the distributed results
    for (size_t iProc = 0; iProc < numProcessors; iProc++){
        for (size_t iRank = 0; iRank < numRanks; iRank++){numSendParticles[iRank] += numSendParticlesDist2[iProc][iRank];}
    }
    // memory cleanup
    numSendParticlesDist2.clear();
    numSendParticlesDist2.shrink_to_fit();
    // check if all particles were checked
    checksum = 0;
    for (size_t iRank = 0; iRank < numRanks; iRank++){checksum += numSendParticles[iRank];}
    assert(("in sortStep2 not all particles were checked for where they should be sent", checksum == numSortedParts));



    // create a communication plan
    if (rank == 0){std::cout << "\tcreating communication plan for returning the data" << std::endl;}
    auto [sendPlan2, recvPlan2] = communicationPlan(numSendParticles, rank,numRanks);
    assert(("ERROR: send & recvPlan must have the same size!", sendPlan2.size() == recvPlan2.size()));
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcomm plan created" << std::endl;}



    // create sendIndexLists for faster access in assembling sendBuffers
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tclear sendIndexLists for reusage" << std::endl;}
    rankSteps.resize(sendPlan2.size());
    for (size_t i = 0; i < sendPlan2.size(); i++){rankSteps[i] = -1;}
    for (size_t step = 0; step < sendPlan2.size(); step++){
        if (sendPlan2[step] >= 0){ rankSteps[sendPlan2[step]] = static_cast<int>(step);}
    }
    // create index lists for finding the particles to send in each step more easily [step] retuirns the list for a step ind sendPlan
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcreating sendIndexLists for returning" << std::endl;}
    std::vector<std::vector<uint64_t>> sendIndexLists2(sendPlan2.size());
    std::vector<uint64_t> ind2(sendPlan2.size(), 0);
    for (size_t step = 0; step < sendPlan2.size(); step++){sendIndexLists2[step].resize(numSendParticles[sendPlan2[step]]);}
    for (size_t iPart = 0; iPart < IDs.size(); iPart++){
        int trueRank = static_cast<int>(std::floor(static_cast<double>(IDs[iPart]) / static_cast<double>(numPartsPerRank[0])));
        sendIndexLists2[rankSteps[trueRank]][ind2[rankSteps[trueRank]]] = static_cast<uint64_t>(iPart);
        ind2[rankSteps[trueRank]]++;
    }
    // memory cleanup
    ind2.clear();
    rankSteps.clear();
    ind2.shrink_to_fit();
    rankSteps.shrink_to_fit();


    // create recvIndexists for sorting communicated data into the local storage
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tcreating recvIndexLists for returning" << std::endl;}
    std::vector<std::vector<uint64_t>> recvIndexLists2(sendPlan2.size());
    // communication
    for (size_t step = 0; step < sendPlan2.size(); step++){
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0){std::cout << "\tstep " << step << std::endl;}
        // ranks with which this rank will communicate
        int sendRank = sendPlan2[step];
        int recvRank = recvPlan2[step];
        // check if this rank is active sending / recieving data in this step
        bool isSending   = sendRank >= 0;
        bool isRecieving = recvRank >= 0;
        // number of particles that will be sent
        uint64_t numSendParts = static_cast<uint64_t>(sendIndexLists2[step].size());
        uint64_t numRecvParts = 0;
        
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0){std::cout << "\ta "<< std::endl;}

        if (sendRank == rank){
            std::cout << "\tselfcomm a " << numSendParts << "/" << recvIndexLists2[step].size() << " " << step << "/" << recvIndexLists2.size() << std::endl;
            recvIndexLists2[step].resize(numSendParts);
            std::cout << "\tselfcomm b"<< std::endl;
            for (size_t i = 0; i < numSendParts; i++){recvIndexLists2[step][i] = originalIndex[sendIndexLists2[step][i]];}
            std::cout << "\tselfcomm c"<< std::endl;
        } else {
            if (rank == 0){std::cout << "\tb "<< std::endl;}
            // communicate the number of particles that will be sent/recieved
            if (isSending && isRecieving){
                MPI_Sendrecv((void *)&numSendParts, 1, MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                             (void *)&numRecvParts, 1, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (isSending) {
                MPI_Send((void *)&numSendParts, 1, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
            } else if (isRecieving) {
                MPI_Recv((void *)&numRecvParts, 1, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            
            if (rank == 0){std::cout << "\tc "<< std::endl;}
            // resize the section of recvIndexLists
            recvIndexLists2[step].resize(numRecvParts);
            if (rank == 0){std::cout << "\td "<< std::endl;}

            // assemble the send buffer
            if (isSending){
                IDsBuffer.resize(numSendParts);
                #pragma omp parallel for
                for (size_t i = 0; i < numSendParts; i++){IDsBuffer[i] = originalIndex[sendIndexLists2[step][i]];}
            }
            // communicate the indizes
            if (isSending && isRecieving){
                MPI_Sendrecv((void *)&IDsBuffer[0], numSendParts, MPI_UNSIGNED_LONG_LONG, sendRank, 1, 
                            (void *)&recvIndexLists2[step][0], numRecvParts, MPI_UNSIGNED_LONG_LONG, recvRank, 1,
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (isSending) {
                MPI_Send((void *)&IDsBuffer[0], numSendParts, MPI_UNSIGNED_LONG_LONG, sendRank, 1, MPI_COMM_WORLD);
            } else if (isRecieving) {
                MPI_Recv((void *)&recvIndexLists2[step][0], numRecvParts, MPI_UNSIGNED_LONG_LONG, recvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }
            if (rank == 0){std::cout << "\te "<< std::endl;}
        }
    } // end of index communication
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tf "<< std::endl;}
    // memory cleanup
    IDsBuffer.clear();
    IDsBuffer.shrink_to_fit();

    // communicate the data (return)
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\treturning the data" << std::endl;}
    sortComm2(IDsTest, sendIndexLists2,recvIndexLists2,sendPlan2,recvPlan2,numLocalParts,rank);
    sortComm2(ftleX,   sendIndexLists2,recvIndexLists2,sendPlan2,recvPlan2,numLocalParts,rank);
    sortComm2(ftleY,   sendIndexLists2,recvIndexLists2,sendPlan2,recvPlan2,numLocalParts,rank);
    sortComm2(ftleZ,   sendIndexLists2,recvIndexLists2,sendPlan2,recvPlan2,numLocalParts,rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tdata returned" << std::endl;}

    // memory cleanup
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tmemory cleanup" << std::endl;}
    sendIndexLists2.clear();
    recvIndexLists2.clear();
    sendPlan2.clear();
    recvPlan2.clear();
    sendIndexLists2.shrink_to_fit();
    recvIndexLists2.shrink_to_fit();
    sendPlan2.shrink_to_fit();
    recvPlan2.shrink_to_fit();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\tfunction done" << std::endl;}
    return;
}


template<class... dragVector_t>
void IDMatching(std::vector<uint64_t> IDs1, std::vector<uint64_t> IDs2, 
                std::vector<uint64_t>& IDsTest, std::vector<double>& ftleX, std::vector<double>& ftleY, std::vector<double>& ftleZ, 
                const uint64_t totalNumParts,
                const int rank, const int numRanks, const int numProcessors)
{   
    if (rank == 0){std::cout << "\nmatching the particle IDs:" << std::endl;}
    // number of particles local on this rank (timestep 1)
    uint64_t numLocalParts = static_cast<uint64_t>(IDs1.size());

    // number of sorted particles per rank
    if (rank == 0){std::cout << "number of sorted particles per rank:" << std::endl;}
    std::vector<uint64_t> numPartsPerRank(numRanks);
    uint64_t defaultNum = static_cast<uint64_t>(std::ceil(static_cast<double>(totalNumParts)/static_cast<double>(numRanks)));
    for (size_t iRank = 0; iRank < numRanks; iRank++){
        if (iRank == numRanks-1){numPartsPerRank[iRank] = totalNumParts - iRank*defaultNum;} 
        else {                   numPartsPerRank[iRank] = defaultNum;}
        if (rank == 0){std::cout << "\t" << iRank << "\t" << numPartsPerRank[iRank] << std::endl;}
    }

    // compute the first local ID on each rank
    std::vector<uint64_t> firstLocalID(numRanks);
    firstLocalID[0] = 0;
    for (size_t iRank = 1; iRank < numRanks; iRank++){
        firstLocalID[iRank] = firstLocalID[iRank-1] + numPartsPerRank[iRank-1];
    }

    // sort timestep 1
    if (rank == 0){std::cout << "sorting timestep 1" << std::endl;}
    auto Runtime1Start = std::chrono::high_resolution_clock::now();
    auto [originalRank,originalIndex] = sortStep1(IDs1,numPartsPerRank,firstLocalID, rank,numRanks,numProcessors);
    auto Runtime1Stop  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> Runtime1 = Runtime1Stop - Runtime1Start;
    if (rank == 0){std::cout << "runtime for sorting timestep 1: " << Runtime1.count() << std::endl;}

    // sort timestep 2
    if (rank == 0){std::cout << "\nsorting timestep 2" << std::endl;}
    auto Runtime2Start = std::chrono::high_resolution_clock::now();
    sortStep2(IDs2, IDsTest,ftleX,ftleY,ftleZ, numPartsPerRank,numLocalParts,firstLocalID, originalRank,originalIndex, rank,numRanks,numProcessors);
    auto Runtime2Stop  = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> Runtime2 = Runtime2Stop - Runtime2Start;
    if (rank == 0){std::cout << "runtime for sorting timestep 2: " << Runtime2.count() << std::endl;}

    // memory cleanup
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