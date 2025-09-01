#pragma once

#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include "paths.hpp"

class runtimeCls{
private:
    // total runtime of the entire script
    std::chrono::duration<double> totalRuntime;
    // runtime for loading the checkpoint
    std::chrono::duration<double> loadingRuntime;
    // runtime for matching the main & FTLE timesteps
    std::chrono::duration<double> matchingRuntime;
    // runtime for he SPH-interpolation (rho + FTLE)
    std::chrono::duration<double> SPHRuntime;
    // runtime for computing basic mass-weighted statistics
    std::chrono::duration<double> massStatsRuntime;
    // runtime for computing mass-weighted PDFs
    std::chrono::duration<double> massPDFRuntime;
    // runtime for computing mass-weighted structure functions
    std::chrono::duration<double> massSFRuntime;
    // runtime for mapping the SPH-data to the grid
    std::chrono::duration<double> gridMappingRuntime;
    // runtime for computing basic volume-weighted statistics
    std::chrono::duration<double> volStatsRuntime;
    // runtime for plotting box slices
    std::chrono::duration<double> slicesRuntime;
    // runtime for computing volume-weighted PDFs
    std::chrono::duration<double> volumePDFRuntime;
    // runtime for computing power-spectra & Fourier-relaed stuff
    std::chrono::duration<double> powerSpecRuntime;
    // runtime for computing volume-weighted structure functions
    std::chrono::duration<double> volumeSFRuntime;

public:
    // set methods
    void setTotalRuntime    (const std::chrono::_V2::system_clock::time_point start, const std::chrono::_V2::system_clock::time_point stop){totalRuntime    = stop - start;}
    void setLoadingRuntime  (const std::chrono::_V2::system_clock::time_point start, const std::chrono::_V2::system_clock::time_point stop){loadingRuntime  = stop - start;}
    void setMatchingRuntime (const std::chrono::_V2::system_clock::time_point start, const std::chrono::_V2::system_clock::time_point stop){matchingRuntime = stop - start;}
    void setSPHRuntime      (const std::chrono::_V2::system_clock::time_point start, const std::chrono::_V2::system_clock::time_point stop){SPHRuntime      = stop - start;}
    void setMassStatsRuntime(const std::chrono::_V2::system_clock::time_point start, const std::chrono::_V2::system_clock::time_point stop){massStatsRuntime= stop - start;}
    void setMassPDFRuntime  (const std::chrono::_V2::system_clock::time_point start, const std::chrono::_V2::system_clock::time_point stop){massPDFRuntime  = stop - start;}
    void setMassSFRuntime   (const std::chrono::_V2::system_clock::time_point start, const std::chrono::_V2::system_clock::time_point stop){massSFRuntime   = stop - start;}
    void setGridRuntime     (const std::chrono::_V2::system_clock::time_point start, const std::chrono::_V2::system_clock::time_point stop){gridMappingRuntime= stop - start;}
    void setVolStatsRuntime (const std::chrono::_V2::system_clock::time_point start, const std::chrono::_V2::system_clock::time_point stop){volStatsRuntime = stop - start;}
    void setSlicesRuntime   (const std::chrono::_V2::system_clock::time_point start, const std::chrono::_V2::system_clock::time_point stop){slicesRuntime   = stop - start;}
    void setVolumePDFRuntime(const std::chrono::_V2::system_clock::time_point start, const std::chrono::_V2::system_clock::time_point stop){volumePDFRuntime= stop - start;}
    void setPSRuntime       (const std::chrono::_V2::system_clock::time_point start, const std::chrono::_V2::system_clock::time_point stop){powerSpecRuntime= stop - start;}
    void setVolumeSFRuntime (const std::chrono::_V2::system_clock::time_point start, const std::chrono::_V2::system_clock::time_point stop){volumeSFRuntime = stop - start;}

    // get methods
    double getLoadingRuntime    () const{return static_cast<double>(loadingRuntime.count());}
    double getMatchingRuntime   () const{return static_cast<double>(matchingRuntime.count());}
    double getSPHRuntime        () const{return static_cast<double>(SPHRuntime.count());}
    double getMassStatsRuntime  () const{return static_cast<double>(massStatsRuntime.count());}
    double getMassPDFRuntime    () const{return static_cast<double>(massPDFRuntime.count());}
    double getMassSFRuntime     () const{return static_cast<double>(massSFRuntime.count());}
    double getGridRuntime       () const{return static_cast<double>(gridMappingRuntime.count());}
    double getVolStatsRuntime   () const{return static_cast<double>(volStatsRuntime.count());}
    double getSlicesRuntime     () const{return static_cast<double>(slicesRuntime.count());}
    double getVolumePDFRuntime  () const{return static_cast<double>(volumePDFRuntime.count());}
    double getPSRuntime         () const{return static_cast<double>(powerSpecRuntime.count());}
    double getVolumeSFRuntime   () const{return static_cast<double>(volumeSFRuntime.count());}

    // print runtime information
    void printInfo() const{
        const double threshold = 0.000001;
        std::cout << "\ntotal runtime: " << totalRuntime.count() << " sec" << std::endl;
        std::cout <<                                    "\tloading runtime:          \t" << loadingRuntime.count() <<       " sec\t(" << 100*(loadingRuntime.count()    /totalRuntime.count()) << "%)" << std::endl;
        if (matchingRuntime.count() > threshold){   std::cout << "\tparticle matching runtime:\t" << matchingRuntime.count() <<    " sec\t(" << 100*(matchingRuntime.count()   /totalRuntime.count()) << "%)" << std::endl;}
        if (SPHRuntime.count() > threshold){        std::cout << "\tSPH interpolation runtime:\t" << SPHRuntime.count() <<         " sec\t(" << 100*(SPHRuntime.count()        /totalRuntime.count()) << "%)" << std::endl;}
        if (massStatsRuntime.count() > threshold){  std::cout << "\tmass Stats runtime:       \t" << massStatsRuntime.count() <<   " sec\t(" << 100*(massStatsRuntime.count()  /totalRuntime.count()) << "%)" << std::endl;}
        if (massPDFRuntime.count() > threshold){    std::cout << "\tmass PDF runtime:         \t" << massPDFRuntime.count() <<     " sec\t(" << 100*(massPDFRuntime.count()    /totalRuntime.count()) << "%)" << std::endl;}
        if (massSFRuntime.count() > threshold){     std::cout << "\tmass SF runtime:          \t" << massSFRuntime.count() <<      " sec\t(" << 100*(massSFRuntime.count()     /totalRuntime.count()) << "%)" << std::endl;}
        if (gridMappingRuntime.count() > threshold){std::cout << "\tgrid mapping runtime:     \t" << gridMappingRuntime.count() << " sec\t(" << 100*(gridMappingRuntime.count()/totalRuntime.count()) << "%)" << std::endl;}
        if (volStatsRuntime.count() > threshold){   std::cout << "\tvolume Stats runtime:     \t" << volStatsRuntime.count() <<    " sec\t(" << 100*(volStatsRuntime.count()   /totalRuntime.count()) << "%)" << std::endl;}
        if (slicesRuntime.count() > threshold){     std::cout << "\tgrid slices runtime:      \t" << slicesRuntime.count() <<      " sec\t(" << 100*(slicesRuntime.count()     /totalRuntime.count()) << "%)" << std::endl;}
        if (volumePDFRuntime.count() > threshold){  std::cout << "\tvolume PDF runtime:       \t" << volumePDFRuntime.count() <<   " sec\t(" << 100*(volumePDFRuntime.count()  /totalRuntime.count()) << "%)" << std::endl;}
        if (powerSpecRuntime.count() > threshold){  std::cout << "\tpower spectrum runtime:   \t" << powerSpecRuntime.count() <<   " sec\t(" << 100*(powerSpecRuntime.count()  /totalRuntime.count()) << "%)" << std::endl;}
        if (volumeSFRuntime.count() > threshold){   std::cout << "\tvolume SF runtime:        \t" << volumeSFRuntime.count() <<    " sec\t(" << 100*(volumeSFRuntime.count()   /totalRuntime.count()) << "%)" << std::endl;}
        return;
    }

    // save runtime info to file
    void saveInfo(const std::string simFile, const int stepNo, const int numRanks, const bool nearestNeighbor, filePathsCls filePaths) const{
        const double threshold = 0.000001;
        // filename to which the runtime info should be saved
        const std::string filename = "runtimes-" + std::to_string(stepNo) + ".txt";
        // path to the file
        const std::string path = filePaths.getRuntimePath(filename);
        // saving the runtimes to file
        std::ofstream outFile(path);
        if (outFile.is_open()){
            // header
            outFile << "## runtimes" << std::endl;
            outFile << "# simFile         : " << simFile << std::endl;
            outFile << "# stepNo          : " << stepNo << std::endl;
            outFile << "# numRanks        : " << numRanks << std::endl;
            if (gridMappingRuntime.count() > threshold){outFile << "# nearestNeighbor : " << nearestNeighbor << std::endl;}
            outFile << "# totalRuntime          : " << totalRuntime.count()         << std::endl;
            if (matchingRuntime.count() > threshold){   outFile << "# particle-matching     : " << matchingRuntime.count()    << std::endl;}
            if (SPHRuntime.count() > threshold){        outFile << "# SPH-Interpolation     : " << SPHRuntime.count()         << std::endl;}
            if (massStatsRuntime.count() > threshold){  outFile << "# mass-Stats            : " << massStatsRuntime.count()   << std::endl;}
            if (massPDFRuntime.count() > threshold){    outFile << "# mass-PDF              : " << massPDFRuntime.count()     << std::endl;}
            if (massSFRuntime.count() > threshold){     outFile << "# mass-SF               : " << massSFRuntime.count()      << std::endl;}
            if (gridMappingRuntime.count() > threshold){outFile << "# grid-mapping          : " << gridMappingRuntime.count() << std::endl;}
            if (volStatsRuntime.count() > threshold){   outFile << "# volume-Stats          : " << volStatsRuntime.count()    << std::endl;}
            if (slicesRuntime.count() > threshold){     outFile << "# grid-slices           : " << slicesRuntime.count()      << std::endl;}
            if (volumePDFRuntime.count() > threshold){  outFile << "# volume-PDF            : " << volumePDFRuntime.count()   << std::endl;}
            if (powerSpecRuntime.count() > threshold){  outFile << "# power-spectrum        : " << powerSpecRuntime.count()   << std::endl;}
            if (volumeSFRuntime.count() > threshold){   outFile << "# volume-SF             : " << volumeSFRuntime.count()    << std::endl;}
        }   
        return;
    }
};