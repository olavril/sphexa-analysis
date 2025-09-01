#pragma once

#include <iostream>
#include <vector>
#include <limits>
#include "MPI_comms.hpp"



class statisticsCls{
private:
    // RMS value of the given dataset
    double RMS;
    // mean value of the given dataset
    double MEAN;
    // std value of the gven dataset
    double STD;

    // function to compute the RMS value of a given distributed data set
    // input:       data            : local data vector for which to compute the RMS
    //              numData         : total number of data points across all ranks
    //              rank, numRanks  : local rank ID & total number of ranks
    // output:      RMS value (double) of the dataset
    template <typename T>
    double computeRMS(const std::vector<T>& data, const uint64_t numData, const int rank, const int numRanks)
    {   
        // result container
        double RMS = 0.0;
        // buffer to prevent precision issues
        double RMSbuffer = 0.0;
        // numeric epsilon value to check for precision issues
        const double numeric_epsilon = std::numeric_limits<double>::epsilon();
        // threshold for finding precision issues
        const double n = 1000;
        // add up local RMS contribution
        for (size_t i = 0; i < data.size(); i++){
            double temp = std::pow(static_cast<double>(data[i]),2);
            const int exp = std::ilogb(temp);
            if (temp > n*std::ldexp(numeric_epsilon,exp)){
                RMSbuffer += temp;
            } else {
                RMS += RMSbuffer;
                RMSbuffer = temp;
            }
        }
        // add up remaining buffer value
        RMS += RMSbuffer;
        // collapse result to rank 0
        collapse(RMS,'d');
        // rank 0 completes the computation
        if (rank == 0){
            RMS = std::sqrt(RMS/static_cast<double>(numData));
        }
        // distribute results
        MPI_Bcast((void *)&RMS, 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        // return results
        return RMS;
    }

    // function to compute the mean value of a given distributed data set
    // input:       data            : local data vector for which to compute the RMS
    //              numData         : total number of data points across all ranks
    //              rank, numRanks  : local rank ID & total number of ranks
    // output:      mean value (double) of the dataset
    template <typename T>
    double computeMEAN(const std::vector<T>& data, const uint64_t numData, const int rank, const int numRanks)
    {   
        // result container
        double MEAN = 0.0;
        // buffer to prevent precision issues
        double MEANbuffer = 0.0;
        // numeric epsilon value to check for precision issues
        const double numeric_epsilon = std::numeric_limits<double>::epsilon();
        // threshold for finding precision issues
        const double n = 1000;
        // add up local mean contribution
        for (size_t i = 0; i < data.size(); i++){
            double temp = static_cast<double>(data[i]);
            const int exp = std::ilogb(temp);
            if (temp > n*std::ldexp(numeric_epsilon,exp)){
                MEANbuffer += temp;
            } else {
                MEAN += MEANbuffer;
                MEANbuffer = temp;
            }
        }
        // add up remaining buffer value
        MEAN += MEANbuffer;
        // collapse result to rank 0
        collapse(MEAN,'d');
        // rank 0 completes the computation
        if (rank == 0){
            MEAN = MEAN/static_cast<double>(numData);
        }
        // distribute results
        MPI_Bcast((void *)&MEAN, 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        // return results
        return MEAN;
    }

    // function to compute the std value of a given distributed data set
    // input:       data            : local data vector for which to compute the RMS
    //              mean            : mean value of the given dataset
    //              numData         : total number of data points across all ranks
    //              rank, numRanks  : local rank ID & total number of ranks
    // output:      std value (double) of the dataset
    template <typename T, typename S>
    double computeSTD(const std::vector<T>& data, const S mean, const uint64_t numData, const int rank, const int numRanks)
    {   
        // result container
        double STD = 0.0;
        // buffer to prevent precision issues
        double STDbuffer = 0.0;
        // numeric epsilon value to check for precision issues
        const double numeric_epsilon = std::numeric_limits<double>::epsilon();
        // threshold for finding precision issues
        const double n = 1000;
        // add up local std contribution
        for (size_t i = 0; i < data.size(); i++){
            double temp = std::pow(static_cast<double>(data[i]) - static_cast<double>(mean),2);
            const int exp = std::ilogb(temp);
            if (temp > n*std::ldexp(numeric_epsilon,exp)){
                STDbuffer += temp;
            } else {
                STD += STDbuffer;
                STDbuffer = temp;
            }
        }
        // add up remaining buffer value
        STD += STDbuffer;
        // collapse result to rank 0
        collapse(STD,'d');
        // rank 0 completes the computation
        if (rank == 0){
            STD = std::sqrt(STD/static_cast<double>(numData));
        }
        // distribute results
        MPI_Bcast((void *)&STD, 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        // return results
        return STD;
    }


public:
    // constructor
    template <typename T>
    statisticsCls(const std::vector<T>& data, const uint64_t numData, const int rank, const int numRanks){
        // compute statistics for the given data vector
        RMS     = computeRMS(data, numData, rank,numRanks);
        if (rank == 0){std::cout << "rms: " << RMS << std::endl;}
        MEAN    = computeMEAN(data, numData, rank,numRanks);
        if (rank == 0){std::cout << "mean:" << MEAN << std::endl;}
        STD     = computeSTD(data,MEAN, numData, rank,numRanks);
        if (rank == 0){std::cout << "std: " << STD << std::endl;}
        return;
    }

    // function to return the RMS value
    double getRMS() const{return RMS;}
    // function to return the mean value
    double getMEAN() const{return MEAN;}
    // function to return the mean value
    double getSTD() const{return STD;}
};