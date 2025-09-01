#pragma once

#include <complex>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <cassert>
#include <omp.h>

// function to print a vector element-by-element
template <typename T>
void vector_print(const std::vector<T> vec)
{
    for (size_t i = 0; i < vec.size(); i++){
        std::cout << std::setprecision(12) << vec[i] << " ";
    }
    std::cout << std::endl;
}

// function to compare two vectors of equal length element-by-element
// returns true if both vectors are identical in all elements
// standard tolerance to consider elements equal is 1e-10
template <typename T>
bool vector_comparison(const std::vector<T> vec1, const std::vector<T> vec2, const double tolerance = 1e-10)
{
    // check that both vectors have the same length
    assert(("trying to compare two vectors of unequal length will not work", vec1.size() == vec2.size()));
    // result variable
    bool result = true;
    // comparison
    for (size_t i = 0; i < vec1.size(); i++){
        if (vec1[i] - vec2[i] > tolerance){
            std::cout << "comparison fails at i = " << i << " " << vec1[i] << " " << vec2[i] << std::endl;
            result = false;
            break;
        }
    }
    return result;
}

// function to add up two vectors of equal length element-by-element
template <typename T>
std::vector<T> vector_add(const std::vector<T> vec1, const std::vector<T> vec2)
{
    // check that both vectors have the same length
    assert(("trying to add two vectors of unequal length will not work", vec1.size() == vec2.size()));
    // create a result vector
    std::vector<T> result(vec1.size());
    // adding up the vectors
    #pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); i++){
        result[i] = vec1[i] + vec2[i];
    }
    return result;
}

// function to subtract two vectors of equal length element-by-element
template <typename T>
std::vector<T> vector_subtract(const std::vector<T> vec1, const std::vector<T> vec2)
{
    // check that both vectors have the same length
    assert(("trying to subtract two vectors of unequal length will not work", vec1.size() == vec2.size()));
    // create a result vector
    std::vector<T> result(vec1.size());
    // subtracting the vectors
    #pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); i++){
        result[i] = vec1[i] - vec2[i];
    }
    return result;
}

// function to multiply two vectors of equal length element-by-element
template <typename T>
std::vector<T> vector_multiply(const std::vector<T> vec1, const std::vector<T> vec2)
{
    // check that both vectors have the same length
    assert(("trying to multiply two vectors of unequal length will not work", vec1.size() == vec2.size()));
    // create a result vector
    std::vector<T> result(vec1.size());
    // adding up the vectors
    #pragma omp parallel for
    for (size_t i = 0; i < vec1.size(); i++){
        result[i] = vec1[i] * vec2[i];
    }
    return result;
}

// function to multiply a vector with a scalar element-by-element
template <typename T>
std::vector<T> vector_multiply(const std::vector<T> vec, const T fac)
{
    // create a result vector
    std::vector<T> result(vec.size());
    // adding up the vectors
    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); i++){
        result[i] = vec[i] * fac;
    }
    return result;
}

// function to compute the power of a vector element-by-element
template <typename T, typename S>
std::vector<T> vector_pow(const std::vector<T> vec, const S power)
{
    // create a result vector
    std::vector<T> result(vec.size());
    // adding up the vectors
    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); i++){
        result[i] = std::pow(vec[i],static_cast<T>(power));
    }
    return result;
}

// function to compute the square-root of a vector element-by-element
template <typename T>
std::vector<T> vector_sqrt(const std::vector<T> vec)
{
    // create a result vector
    std::vector<T> result(vec.size());
    // adding up the vectors
    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); i++){
        result[i] = std::sqrt(vec[i]);
    }
    return result;
}

// function to compute the natural logarithm of a vector element-by-element
template <typename T>
std::vector<T> vector_ln(const std::vector<T> vec)
{
    // create a result vector
    std::vector<T> result(vec.size());
    // adding up the vectors
    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); i++){
        result[i] = std::log(vec[i]);
    }
    return result;
}

// function to compute the base-10 logarithm of a vector element-by-element
template <typename T>
std::vector<T> vector_log(const std::vector<T> vec)
{
    // create a result vector
    std::vector<T> result(vec.size());
    // adding up the vectors
    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); i++){
        result[i] = std::log10(vec[i]);
    }
    return result;
}

// function to compute the reciprocal of a vector (X -> 1/X)
// modifies the input vector
template <typename T>
void vector_reciprocal(const std::vector<T>& vec)
{   
    #pragma omp parallel for
    for (size_t i = 0; i < vec.size(); i++){vec[i] = static_cast<T>(1) / vec[i];}
    return;
}

// function to compute the (element-wise) square of a complex vector
std::vector<std::complex<double>> vector_square(const std::vector<std::complex<double>> data)
{
    std::vector<std::complex<double>> result(data.size());
    #pragma omp parallel for
    for(size_t i = 0; i < data.size(); i++){
        result[i] = data[i] * data[i];
    }
    return result;
}

// function to compute the (element-wise) sum of three (complex double) vectors
std::vector<std::complex<double>> vector_sum(const std::vector<std::complex<double>> data1, 
                                            const std::vector<std::complex<double>> data2, 
                                            const std::vector<std::complex<double>> data3)
{
    assert(("the vectors that should be summed up (element wise) need to have the same size (1/2 mismatch)", data1.size() == data2.size()));
    assert(("the vectors that should be summed up (element wise) need to have the same size (2/3 mismatch)", data2.size() == data3.size()));
    std::vector<std::complex<double>> result(data1.size());
    #pragma omp parallel for
    for(size_t i = 0; i < data1.size(); i++){
        result[i] = data1[i] + data2[i] + data3[i];
    }
    return result;
}

// function to compute the base-10 logarithm of a vector element-by-element
template <typename T>
std::vector<T> vector_abs(const std::vector<std::complex<T>> vec)
{
    // create a result vector
    std::vector<T> result(vec.size());
    // adding up the vectors
    //#pragma omp parallel for
    for (size_t i = 0; i < vec.size(); i++){
        result[i] = std::abs(vec[i]);
    }
    return result;
}