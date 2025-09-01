#pragma once

#include <tuple>
#include <omp.h>
#include <mpi.h>

// initialize MPI environment
// returns the local rank number & the total number of ranks
std::tuple<int,int> initMPI()
{
    // initialize Rank & number of ranks containers
    int Rank;
    int numRanks;

    // start up MPI envronment
    MPI_Init(NULL,NULL);

    // get the local rank number
    MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
    // get total number of ranks
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // return local ranks number and total number of ranks
    return std::make_tuple(Rank,numRanks);
}

// stop MPI environment
int MPIfinal()
{
    MPI_Finalize();
    return 0;
}