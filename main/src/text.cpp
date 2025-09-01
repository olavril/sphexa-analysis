#include <iostream>
#include <vector>
#include <limits>
#include <stdint.h>
#include <type_traits>

//#include "MPI_utils.hpp"
//#include "MPI_comms.hpp"
#include "vector-operations.hpp"

template <typename T>
void typeTest(std::vector<T> vec)
{
    //type recognition test
    using vecType1 = decltype(vec);
    if (std::is_same<vecType1, std::vector<uint64_t>>::value){
        std::cout << "Type recognition successfull" << std::endl;
    } else {
        std::cout << "Type recognition failed" << vecType1 << std::endl;
    }
}

int main()
{
    // testing vector operations
    std::cout << "Testing vector adding:" << std::endl;

    std::vector<double> aDouble   {1, 5.0,6.3,2.5, 7,33,900.5, 100,3.5, 5.66,289,    5.827636};
    std::vector<double> bDouble   {2, 6.2,3.4,2.5,18,44,  6.37, 40,3.5,71,    10.43, 5.3647};
    std::vector<double> refDouble {3,11.2,9.7,5.0,25,77,906.87,140,7.0,76.66,299.43,11.192336};

    std::vector<double> resDouble = vector_add(aDouble,bDouble);
    std::cout << "\tdouble test: " << vector_comparison(refDouble,resDouble) << std::endl;

    std::vector<int> aInt   { 1,  4, 7, 8, 33,40,100,35, 28,90,16};
    std::vector<int> bInt   {63,130, 7,12, 67,23,111,12, 87, 5,16};
    std::vector<int> refInt {64,134,14,20,100,63,211,47,115,95,32};

    std::vector<int> resInt = vector_add(aInt,bInt);
    std::cout << "\tint test:    " << vector_comparison(refInt,resInt) << std::endl;

    std::vector<uint64_t> aUInt   { 1,  4, 7, 8, 33,40,100,35, 28,90,16};
    std::vector<uint64_t> bUInt   {63,130, 7,12, 67,23,111,12, 87, 5,16};
    std::vector<uint64_t> refUInt {64,134,14,20,100,63,211,47,115,95,32};

    std::vector<uint64_t> resUInt = vector_add(aUInt,bUInt);
    std::cout << "\tuint64 test: " << vector_comparison(refUInt,resUInt) << std::endl;

    //type recognition test
    typeTest(aUInt);


    // testing vector multiplication
    std::cout << "Testing vector multiplying:" << std::endl;

    refDouble = {2,31,21.42,6.25,126,1452,5736.185,4000,12.25,401.86,3014.27,31.263518849};
    resDouble = vector_multiply(aDouble,bDouble);
    std::cout << "\tdouble test: " << vector_comparison(refDouble,resDouble) << std::endl;

    refInt = {63,520,49,96,2211,920,11100,420,2436,450,256};
    resInt = vector_multiply(aInt,bInt);
    std::cout << "\tint test:    " << vector_comparison(refInt,resInt) << std::endl;

    refUInt = {63,520,49,96,2211,920,11100,420,2436,450,256};
    resUInt = vector_multiply(aUInt,bUInt);
    std::cout << "\tuint64 test: " << vector_comparison(refUInt,resUInt) << std::endl;


    // testing vector-scalar multiplication
    std::cout << "Testing vector-scalar multiplying:" << std::endl;

    refDouble = {2.3,11.5,14.49,5.75,16.1,75.9,2071.15,230,8.05,13.018,664.7,13.4035628};
    resDouble = vector_multiply(aDouble,2.3);
    std::cout << "\tdouble test: " << vector_comparison(refDouble,resDouble) << std::endl;

    refInt = {2,8,14,16,66,80,200,70,56,180,32};
    resInt = vector_multiply(aInt,2);
    std::cout << "\tint test:    " << vector_comparison(refInt,resInt) << std::endl;

    refUInt = {2,8,14,16,66,80,200,70,56,180,32};
    resUInt = vector_multiply(aUInt,static_cast<uint64_t>(2));
    std::cout << "\tuint64 test: " << vector_comparison(refUInt,resUInt) << std::endl;


    // testing vector power
    std::cout << "Testing vector power:" << std::endl;

    refDouble = {1,25,39.69,6.25,49,1089,810900.25,10000,12.25,32.0356,83.521,33.961341348496};
    resDouble = vector_pow(aDouble,2.0);
    std::cout << "\tdouble test: " << vector_comparison(refDouble,resDouble) << std::endl;

    refInt = {1,16,49,64,1089,1600,10000,1225,784,8100,256};
    resInt = vector_pow(aInt,2);
    std::cout << "\tint test:    " << vector_comparison(refInt,resInt) << std::endl;

    refUInt = {1,16,49,64,1089,1600,10000,1225,784,8100,256};
    resUInt = vector_pow(aUInt,static_cast<uint64_t>(2));
    std::cout << "\tuint64 test: " << vector_comparison(refUInt,resUInt) << std::endl;



    // communication test
    // auto [rank,numRanks] = initMPI();
    
    // aDouble = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
    // aInt    = {1,1,1,1,1,1,1,1,1,1};
    // aUInt   = {1,1,1,1,1,1,1,1,1,1};
    // double   cDouble = static_cast<double>(rank);
    // int      cInt    = static_cast<int>(rank);
    // uint64_t cUInt   = static_cast<uint64_t>(rank);
    // int resultValue = 0;
    // for (int iRank = 0; iRank < numRanks; iRank++){resultValue += iRank;}

    // // collapse vector test 
    // std::cout << "collapse test vector:" << std::endl;
    // resDouble = collapse(aDouble,'d');
    // resInt    = collapse(aInt,'i');
    // resUInt   = collapse(aUInt,'u');
    // std::cout << "\tdouble test: " << vector_comparison(vector_multiply(aDouble,static_cast<double>(numRanks)),resDouble) << std::endl;
    // std::cout << "\tint test:    " << vector_comparison(vector_multiply(aInt,static_cast<int>(numRanks)),resInt) << std::endl;
    // std::cout << "\tuint64 test: " << vector_comparison(vector_multiply(aUInt,static_cast<uint64_t>(numRanks)),resUInt) << std::endl;

    // // collapse value test
    // std::cout << "collapse test value:" << std::endl;
    // double   rescDouble = collapse(cDouble,'d');
    // int      rescInt    = collapse(cInt,'i');
    // uint64_t rescUInt   = collapse(cUInt,'u');
    // std::cout << "\tdouble test: " << (static_cast<double>(resultValue)   == rescDouble) << std::endl;
    // std::cout << "\tint test:    " << (static_cast<int>(resultValue)      == rescInt)    << std::endl;
    // std::cout << "\tuint64 test: " << (static_cast<uint64_t>(resultValue) == rescUInt)   << std::endl;

    // // combine test
    // std::cout << "combine test:" << std::endl;
    // resDouble = combine(aDouble,'d');
    // resInt    = combine(aInt,'i');
    // resUInt   = combine(aUInt,'u');
    // std::vector<double>   referenceDouble(numRanks*aDouble.size(), 1.0); 
    // std::vector<int>      referenceInt(numRanks*aInt.size(),   1); 
    // std::vector<uint64_t> referenceUInt(numRanks*aUInt.size(), 1); 
    // std::cout << "\tdouble test: " << vector_comparison(referenceDouble,resDouble) << std::endl;
    // std::cout << "\tint test:    " << vector_comparison(referenceInt,resInt) << std::endl;
    // std::cout << "\tuint64 test: " << vector_comparison(referenceUInt,resUInt) << std::endl;

    // return MPIfinal();
    return 0;
}