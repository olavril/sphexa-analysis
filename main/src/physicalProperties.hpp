#pragma once

#include <cassert>
#include <iostream>
#include <vector>
#include <limits>
#include "MPI_comms.hpp"

class physPropertiesCls{
private:
    // kinetic energy 
    double Ekin = 0.0;

    // function to compute the kinetic energy from density & velocity
    void computeEkin(const std::vector<double>& vx, const std::vector<double>& vy, const std::vector<double>& vz, 
                     const uint64_t numParticles, const std::vector<double>& rho = {})
    {   
        // check inputs
        assert(("in physPropertiesCls instance, when using computeEkin, numParticles must be larger than 0", numParticles > 0));
        assert(("in physPropertiesCls instance, when using computeEkin, vx & vy must have the same size", vx.size() == vy.size()));
        assert(("in physPropertiesCls instance, when using computeEkin, vy & vz must have the same size", vx.size() == vz.size()));
        // use the SPh-version (true) or the grid-version (false)
        const bool useSPH = rho.size() == vx.size();
        // reset Ekin value in case it was modified before
        Ekin = 0.0;
        // SPH-particle mass (SPH-version) or grid cell volume (grid-version)
        const double m = 1 / static_cast<double>(numParticles);
        // loop over the local particles to add up Ekin contributions
        if (useSPH){
            // SPH-version
            for (size_t i = 0; i < rho.size(); i++){
                Ekin += std::pow(vx[i],2) + std::pow(vy[i],2) + std::pow(vz[i],2);
            }
        } else {
            // grid version
            for (size_t i = 0; i < rho.size(); i++){
                Ekin += rho[i] * std::pow(vx[i],2) + std::pow(vy[i],2) + std::pow(vz[i],2);
            }
        }
        Ekin = 0.5 * m * Ekin;
        // communications 
        collapse(Ekin,'d');
        MPI_Bcast((void *)&Ekin, 1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        return;
    }

public:
    // constructor
    physPropertiesCls(const std::vector<double>& vx, const std::vector<double>& vy, const std::vector<double>& vz,
                      const uint64_t numParticles, const std::vector<double>& rho = {})
    {   
        computeEkin(vx,vy,vz,numParticles,rho);
        return;
    }

    // method to get the Ekin value
    double getEkin() const{return Ekin;}
};