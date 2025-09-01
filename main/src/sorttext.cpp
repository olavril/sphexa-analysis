#include <vector>
#include <iostream>
#include <stdint.h>
#include <tuple>
#include <random>
#include <chrono>

#include "sort.hpp"

template<typename T>
bool isSorted(std::vector<T> vec){
    bool result = true;
    for (size_t i = 0; i < vec.size()-1; i++){
        if (vec[i] > vec[i+1]){
            result = false;
            std::cout << vec[i] << "(" << i << ") > " << vec[i+1] << "(" << i+1 << ")" << std::endl;
            break;
        }
    }
    return result;
}

template<typename T> 
void test(std::vector<T>& vec, const int a){
    for (size_t i = 0; i < vec.size(); i++){
        vec[i] += static_cast<T>(a);
    }
    return;
}

int main()
{
    // std::vector<uint64_t> IDs = {0,5,2,8,7,1,9,3,4,6};
    // std::vector<double>     x = {10,15,12,18,17,11,19,13,14,16};
    // std::vector<double>     y = {20,25,22,28,27,21,29,23,24,26};

    // std::cout << "unsorted: " << IDs.size() << " " << x.size() << " " << y.size() << std::endl;
    // for (size_t i = 0; i < IDs.size(); i++){
    //     std::cout << IDs[i] << " ";
    //     std::cout << x[i] << " ";
    //     std::cout << y[i] << " ";
    //     std::cout << std::endl;
    // }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0,10000000);
    uint64_t numParts = 10;
    std::vector<uint64_t> IDs(numParts);
    std::vector<double> x(numParts);
    std::vector<double> y(numParts);
    for (size_t i = 0; i < numParts; i++){
        uint64_t temp = static_cast<uint64_t>(dist(gen));
        IDs[i] = temp;
        x[i] = static_cast<double>(temp);
        y[i] = static_cast<double>(temp);
    }
    std::cout << "data generated: ";
    for (size_t i = 0; i < 10; i++){
        std::cout << IDs[i] << " ";
    }
    std::cout << std::endl;
    
    auto RuntimeStart = std::chrono::high_resolution_clock::now();
    sort(IDs, std::tie(x,y), 1, false);
    auto RuntimeStop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> Runtime = RuntimeStop - RuntimeStart;

    std::cout << "test result: " << isSorted(IDs) << " " << isSorted(x) << " " << isSorted(y) << " " << true << std::endl;
    std::cout << "runtime: " << Runtime.count() << " sec" << std::endl;

    // std::cout << "\nsorted: " << IDs.size() << " " << x.size() << " " << y.size() << std::endl;
    // for (size_t i = 0; i < IDs.size(); i++){
    //     std::cout << IDs[i] << " ";
    //     std::cout << x[i] << " ";
    //     std::cout << y[i] << " ";
    //     std::cout << std::endl;
    // }

    auto inp = std::tie(IDs, x);
    int a = 5;
    std::apply([a](auto&&... vectors) {(test(vectors,a), ...);}, inp);

    std::cout << "\nsorted: " << IDs.size() << " " << x.size() << " " << y.size() << std::endl;
    for (size_t i = 0; i < IDs.size(); i++){
        std::cout << IDs[i] << " ";
        std::cout << x[i] << " ";
        std::cout << y[i] << " ";
        std::cout << std::endl;
    }

    return 0;
}

