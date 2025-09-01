#pragma once

#include <cassert>
#include <stdint.h>
#include <omp.h>
#include <mpi.h>
#include <iostream>

#include "sort.hpp"

// function to print a communication plan for diagnostic purposes
void printCommPlan(const std::vector<int> sendPlan, const std::vector<int> recvPlan, const int rank, const int numRanks){
    assert(("ERROR: to print the comm plan, sendPlan & recvPlan must have the same size"));
    if (rank == 0){std::cout << "\n\ncommunication plan:" << std::endl;}
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t iRank = 0; iRank < numRanks; iRank++){
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == iRank){
            std::cout << "\trank " << rank << ": " << std::endl;
            for (size_t iStep = 0; iStep < sendPlan.size(); iStep++){
                std::cout << "\t   " << iStep << ":\t";
                if (recvPlan[iStep] >= 0){
                    std::cout << recvPlan[iStep] << "\t->\t";
                } else {
                    std::cout << "\t  \t";
                }
                std::cout << rank;
                if (sendPlan[iStep] >= 0){
                    std::cout << "\t->\t" << sendPlan[iStep];
                }
                std::cout << std::endl;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\n" << std::endl;}
    return;
}

// function to print a communication plan for diagnostic purposes
void printCommPlan(const std::vector<int> sendPlan, const std::vector<int> recvPlan, 
                   const std::vector<std::vector<uint64_t>> sendIndexLists, const std::vector<std::vector<uint64_t>> recvIndexLists,
                   const int rank, const int numRanks){
    assert(("ERROR: to print the comm plan, sendPlan & recvPlan must have the same size"));
    if (rank == 0){std::cout << "\n\ncommunication plan:" << std::endl;}
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t iRank = 0; iRank < numRanks; iRank++){
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == iRank){
            std::cout << "\trank " << rank << ": " << std::endl;
            for (size_t iStep = 0; iStep < sendPlan.size(); iStep++){
                std::cout << "\t   " << iStep << ":\t";
                if (recvPlan[iStep] >= 0){
                    std::cout << recvPlan[iStep] << " (" << recvIndexLists[iStep].size() << ")" "\t->\t";
                } else {
                    std::cout << "\t\t  \t";
                }
                std::cout << rank;
                if (sendPlan[iStep] >= 0){
                    std::cout << "\t->\t" << sendPlan[iStep] << " (" << sendIndexLists[iStep].size() << ")";
                }
                std::cout << std::endl;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0){std::cout << "\n" << std::endl;}
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
    // maximum umber of communication steps
    int numSteps = numRanks + 3;
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
    std::vector<std::vector<int>> sendPlan(numSteps);
    // individual recieve plan for each rank
    // recvPlan[iRank] conatins the vector with the ranks from which iRank will recieve data in the correct order
    std::vector<std::vector<int>> recvPlan(numSteps);
    // buffer for reieving data
    std::vector<uint64_t> buffer1;
    if (rank == 0){
        buffer1.resize(numRanks);
        Sender.resize(totalComms);
        Reciever.resize(totalComms);
        numParts.resize(totalComms);
    }

    // sending all numSendParticles vectors to rank 0 so that it can create the communication plans
    //if (rank == 0){std::cout << "\tsending all data to rank 0" << std::endl;}
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

        //std::cout << "\tsorting the communications" << std::endl;
        // sort the communication processes by number of communicated particles (numParts)
        sortComms(numParts,Sender,Reciever);
        totalComms = Sender.size();
        //std::cout << "\tcommunications sorted, new size:" << numParts.size() << " " << Sender.size()<< " " << Reciever.size() << std::endl;

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
        std::vector<std::vector<int>> commPlan(numSteps);
        for (size_t i = 0; i < commPlan.size(); i++){
            commPlan[i].resize(numRanks);
            for (size_t j = 0; j < commPlan[i].size(); j++){commPlan[i][j] = -1;}
        }

        // vector to keep track for which ranks we still see activity (need to send data) in the current comm step 
        // for this it does not matter wether the rank CAN send in the current step, the question is wether it NEEDS to
        // true: the rank is active; false: the rank is done sending
        std::vector<bool> rankActivity(numRanks,false);

        // creating the plan
        //std::cout << "\tcreating the plan step by step" << std::endl;
        for (size_t iStep = 0; iStep < numSteps; iStep++){
            // std::cout << "\tstep " << iStep << std::endl;
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
                        rankActivity[Sender[iComm]]     = true;
                        commPlan[iStep][Sender[iComm]]  = Reciever[iComm];
                        notScheduled[iComm]             = false;
                        sendAvailable[Sender[iComm]]    = false;
                        recvAvailable[Reciever[iComm]]  = false;
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
        //if (rank == 0){std::cout << "\tcreating the individual plans" << std::endl;}
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
    //if (rank == 0){std::cout << "\tcommunicating the individual plans" << std::endl;}
    // broadcast the number of communication steps to all ranks
    MPI_Bcast((void *)&numCommSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // plan for sending for THIS rank
    std::vector<int> sendList(numCommSteps);
    // plan for recieving for THIS rank
    std::vector<int> recvList(numCommSteps);

    // broadcasting the number of comm steps
    MPI_Bcast((void *)&numCommSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // distributing the individual plans to the ranks
    for (size_t iRank = 1; iRank < numRanks; iRank++){
        if (rank == 0){
            MPI_Send((void *)&sendPlan[iRank][0], numCommSteps, MPI_INT, iRank, 1, MPI_COMM_WORLD);
            MPI_Send((void *)&recvPlan[iRank][0], numCommSteps, MPI_INT, iRank, 1, MPI_COMM_WORLD);
        } else if (rank == iRank){
            MPI_Recv((void *)&sendList[0], numCommSteps, MPI_INT, 0, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Recv((void *)&recvList[0], numCommSteps, MPI_INT, 0, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
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
    if (sendList.size() != recvList.size()){std::cout << "ERROR: sendPlan & recvPlan of the communication plan have different sizes! ---------------" << std::endl;}
    return std::make_tuple(sendList,recvList);
}


// function to collapse a given vector to rank 0
// this function assumes that each rank holds a version of the full vector and adds up the values 
// such that rank 0 later holds the total sum 
// CommMode must be specified to get the correct communicator for the input vector 
// use 'd' for double, 'i' for int, 'u' for uint64_t
template <typename T>
void collapse(std::vector<T>& vec, const char CommMode)
{   
    assert(("in collapse() the CommMode must be 'd', 'i' or 'u'", ((CommMode == 'd') || (CommMode == 'i')) || (CommMode == 'u')));
    // local rank
    int rank;
    // total number of ranks
    int numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // number of ranks containing relevant data
    int numRemaining     = numRanks;
    // number of ranks containing relevant data after the next collapse
    int numRemainingNext;
    // length of the vector that will be communicated
    const uint64_t vectorLength = vec.size();
    // communication buffer
    std::vector<T> buffer(vectorLength, 0);
    
    // continue to collapse until only 1 rank with relevant information is left (rank 0)
    while (numRemaining > 1){
        MPI_Barrier(MPI_COMM_WORLD);
        // compute how many ranks will remain after this collpse
        numRemainingNext = std::ceil(static_cast<double>(numRemaining) / 2.0);
        // if this rank is in the [numRemainingNext,numRemaining) interval, it sends data
        if ((numRemainingNext <= rank) && (rank < numRemaining)){
            // rank to which this ranks sends its data
            int SendRank = rank - numRemainingNext;
            if (CommMode == 'd'){
                MPI_Send((void *)&vec[0], vectorLength, MPI_DOUBLE, SendRank, 1, MPI_COMM_WORLD);
            } else if (CommMode == 'i'){
                MPI_Send((void *)&vec[0], vectorLength, MPI_INT, SendRank, 1, MPI_COMM_WORLD);
            } else if (CommMode == 'u'){
                MPI_Send((void *)&vec[0], vectorLength, MPI_UNSIGNED_LONG_LONG, SendRank, 1, MPI_COMM_WORLD);
            } 
        } 
        // if this rank is in the [0,numRemainingNext) interval it recieves data or remains unchanged (if odd number of ranks)
        else if (rank < numRemainingNext) {
            // rank from which this rank recieves its data
            int RecvRank = rank + numRemainingNext;
            if (RecvRank < numRemaining){
                // recieve the data
                if (CommMode == 'd'){
                    MPI_Recv((void *)&buffer[0], vectorLength, MPI_DOUBLE, RecvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                } else if (CommMode == 'i'){
                    MPI_Recv((void *)&buffer[0], vectorLength, MPI_INT, RecvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                } else if (CommMode == 'u'){
                    MPI_Recv((void *)&buffer[0], vectorLength, MPI_UNSIGNED_LONG_LONG, RecvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                } 
                // sum it to the local data
                #pragma omp parallel for
                for (size_t i = 0; i < vectorLength; i++){
                    vec[i] += buffer[i];
                    buffer[i] = 0;
                }
            }
        }
        // reduce the number of remaining ranks
        numRemaining = numRemainingNext;
    } // end of the collapse loop
    // clear memory
    buffer.clear();
    buffer.shrink_to_fit();
    // return result
    return;
}

// function to collapse a given data value to rank 0
// this function assumes that each rank holds a version of the data value and adds up the values 
// such that rank 0 later holds the total sum 
// CommMode must be specified to get the correct communicator for the input vector 
// use 'd' for double, 'i' for int, 'u' for uint64_t
template <typename T>
void collapse(T& value, const char CommMode)
{   
    assert(("in collapse() the CommMode must be 'd', 'i' or 'u'", ((CommMode == 'd') || (CommMode == 'i')) || (CommMode == 'u')));
    // local rank
    int rank;
    // total number of ranks
    int numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // number of ranks containing relevant data
    int numRemaining     = numRanks;
    // number of ranks containing relevant data after the next collapse
    int numRemainingNext;
    // communication buffer
    T buffer = 0;
    
    // continue to collapse until only 1 rank with relevant information is left (rank 0)
    while (numRemaining > 1){
        MPI_Barrier(MPI_COMM_WORLD);
        // compute how many ranks will remain after this collpse
        numRemainingNext = std::ceil(static_cast<double>(numRemaining) / 2.0);
        // if this rank is in the [numRemainingNext,numRemaining) interval, it sends data
        if ((numRemainingNext <= rank) && (rank < numRemaining)){
            // rank to which this ranks sends its data
            int SendRank = rank - numRemainingNext;
            if (CommMode == 'd'){
                MPI_Send((void *)&value, 1, MPI_DOUBLE, SendRank, 1, MPI_COMM_WORLD);
            } else if (CommMode == 'i'){
                MPI_Send((void *)&value, 1, MPI_INT, SendRank, 1, MPI_COMM_WORLD);
            } else if (CommMode == 'u'){
                MPI_Send((void *)&value, 1, MPI_UNSIGNED_LONG_LONG, SendRank, 1, MPI_COMM_WORLD);
            } 
        } 
        // if this rank is in the [0,numRemainingNext) interval it recieves data or remains unchanged (if odd number of ranks)
        else if (rank < numRemainingNext) {
            // rank from which this rank recieves its data
            int RecvRank = rank + numRemainingNext;
            if (RecvRank < numRemaining){
                // recieve the data
                if (CommMode == 'd'){
                    MPI_Recv((void *)&buffer, 1, MPI_DOUBLE, RecvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                } else if (CommMode == 'i'){
                    MPI_Recv((void *)&buffer, 1, MPI_INT, RecvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                } else if (CommMode == 'u'){
                    MPI_Recv((void *)&buffer, 1, MPI_UNSIGNED_LONG_LONG, RecvRank, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                } 
                value += buffer;
            }
        }
        // reduce the number of remaining ranks
        numRemaining = numRemainingNext;
    } // end of the collapse loop
    // return result
    return;
}


// function to combine a distributed vector
// this function assumes that each rank holds a section of the full vector and combines them
// such that each rank holds the full vector
// CommMode must be specified to get the correct communicator for the input vector 
// use 'd' for double, 'i' for int, 'u' for uint64_t
template <typename T>
std::vector<T> combine(const std::vector<T> vec, const char CommMode)
{   
    assert(("in combine() the CommMode must be 'd', 'i' or 'u'", ((CommMode == 'd') || (CommMode == 'i')) || (CommMode == 'u')));
    // local rank
    int rank;
    // total number of ranks
    int numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // length of the combined vector
    uint64_t combinedLength = 0;
    // communication buffer
    uint64_t buffer1 = 0;
    // how many values does each rank hold?
    std::vector<uint64_t> numElements(numRanks);
    // communicate how many values each rank holds
    for (size_t iRank = 0; iRank < numRanks; iRank++){
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == iRank){buffer1 = static_cast<uint64_t>(vec.size());}
        MPI_Bcast((void *)&buffer1, 1, MPI_UNSIGNED_LONG_LONG, iRank, MPI_COMM_WORLD);
        numElements[iRank] = buffer1;
        combinedLength    += buffer1;
        buffer1 = 0;
    }

    // result vector
    std::vector<T> result(combinedLength, 0);
    // global index for moving through the result vector
    uint64_t ind = 0;
    // buffer for communicating the vectors
    std::vector<T> buffer2;
    // communicate the vectors
    for (size_t iRank = 0; iRank < numRanks; iRank++){
        MPI_Barrier(MPI_COMM_WORLD);
        // number of elements that will be communicated
        buffer1 = numElements[iRank];
        buffer2.resize(buffer1);
        if (rank == iRank){buffer2 = vec;}
        if (CommMode == 'd'){
            MPI_Bcast((void *)&buffer2, buffer1, MPI_DOUBLE, iRank, MPI_COMM_WORLD);
        } else if (CommMode == 'i'){
            MPI_Bcast((void *)&buffer2, buffer1, MPI_INT, iRank, MPI_COMM_WORLD);
        } else if (CommMode == 'u'){
            MPI_Bcast((void *)&buffer2, buffer1, MPI_UNSIGNED_LONG_LONG, iRank, MPI_COMM_WORLD);
        } 
        // fill the data from the buffer into the result vector
        #pragma omp parallel for
        for (size_t i = 0; i < buffer2.size(); i++){
            result[ind] = buffer2[i];
            ind++;
        }
    }
    // clear up memory
    numElements.clear();
    buffer2.clear();
    numElements.shrink_to_fit();
    buffer2.shrink_to_fit();
    // return result
    return result;
}
