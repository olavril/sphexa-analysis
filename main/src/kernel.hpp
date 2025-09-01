#pragma once

#include <iostream>
#include <cmath>

// function to compute sinc(x) = sin(pi*x)/(pi*x)
double sinc(const double x)
{
    const double xpi = M_PI * x;
    double result;
    if (x == 0) 
    { 
        result = 1;
    } else {
        result = std::sin(xpi) / xpi;
    }
    return result;
}

// new combined kernel function for SPH-interpolation
double kernel2(const double dist, const double h){
    // normalisation constant
    // const double K = 0.491881; // original value
    const double K = 0.470625;
    // ratio between distance and smoothing length
    const double nu = dist/h;
    // kernel value
    double kern;
    if (nu > 2){
        kern = 0.0;
    } else if (0 < nu){
        kern = 0.9 * std::pow(sinc(nu/2),4) + 0.1 * std::pow(sinc(nu/2),9);
    } else if (nu == 0){
        kern = 1;
    } else {
        std::cout << "ERROR: trying to compute kernel for a negative distance (d=" << dist << "; h=" << h << ")" << std::endl;
    }
    // normalize the kernel
    kern = K * kern / std::pow(h,3);
    return kern;
}

// old sinc6 kernel function for SPH-interpolation
double kernel1(const double dist, const double h)
{
    const double K6 = 0.790450;
    double kern; //result container

    if (dist > 2*h) {
        kern = 0.0;
    } else {
        kern = K6/(std::pow(h,3)) * std::pow( sinc(dist/(2*h)) ,6);
    } 
    
    return kern;
}