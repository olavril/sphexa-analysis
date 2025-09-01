#include <iostream>
#include <vector>
#include <tuple>
#include <cassert>
#include <chrono>

#include "runtimes.hpp"
#include "paths.hpp"
#include "statistics.hpp"
#include "physicalProperties.hpp"

#include "MPI_utils.hpp"
#include "ID_matching2.hpp"
#include "SPH_interpolation.hpp"
#include "PDF.hpp"
#include "grid.hpp"
#include "plot.hpp"
#include "PS.hpp"
#include "SF.hpp"

#include "ifile_io_impl.h"
#include "arg_parser.hpp"


void printHelp(char* binName, int rank);

using Realtype = double;
using namespace sphexa;


int main(int argc, char** argv){
    // instance to store runtime information
    runtimeCls runtimes;
    // start the total runtime measurement
    auto totalRuntimeStart = std::chrono::high_resolution_clock::now();
    // initiallize the MPI environment
    auto [rank,numRanks] = initMPI();

    // variables to control which part of the code will be executed

    // does the density need to be interpolated? (true: interpolate density; false: read density from file)
    bool rhoBool = false;
    // run the FTLE computation
    bool ftleBool = false;
    // run the mass-weighted PDF computation
    bool massPDFBool = false;
    // run the mass-weighted structure function computation on the grid
    bool massSFBool = false;
    // map the simulation data to a grid
    bool gridMappingBool = false;
    bool nearestNeighbor = false;
    // plot slices of the grid
    bool plotSlicesBool = false;
    // run the volume-weighted PDF computation
    bool volumePDFBool = false;
    // run the power-spectrum computation on the grid
    bool powerSpectrumBool = false;
    // run the volume-weighted structure function computation on the grid
    bool volumeSFBool = false;
    // control variable to see if custom input was provided or if the default should be used
    bool customSetup = false;

    // --------------------------------------------------------------------------------------------------------------------------------------------------------------
    // getting the arguments

    // get arguments that were passed to the code
    const ArgParser parser(argc, (const char**)argv);
    if (parser.exists("-h") || parser.exists("--h") || parser.exists("-help") || parser.exists("--help")){
        printHelp(argv[0], rank);
        return 0;
    }
    // setup inputs
    if (parser.exists("-ftle") || parser.exists("--ftle") || parser.exists("-FTLE") || parser.exists("--FTLE")){ftleBool = true; customSetup=true;}
    if (parser.exists("--massPDF") || parser.exists("--masspdf") || parser.exists("--MassPdf") || parser.exists("--Masspdf")){massPDFBool = true; customSetup=true;}
    if (parser.exists("--plotSlices") || parser.exists("--plotslices") || parser.exists("--PlotSlices") || parser.exists("--Plotslices")){plotSlicesBool = true; customSetup=true;}
    if (parser.exists("--volumePDF") || parser.exists("--volumepdf") || parser.exists("--VolumePdf") || parser.exists("--Volumepdf")){volumePDFBool = true; customSetup=true;}
    if (parser.exists("--ps") || parser.exists("--PS") || parser.exists("--powerspectrum") || parser.exists("--powerSpectrum")){powerSpectrumBool = true; customSetup=true;}
    if (parser.exists("--massSF") || parser.exists("--MassSF") || parser.exists("--masssf") || parser.exists("--Masssf")){massSFBool = true; customSetup=true;}
    if (parser.exists("--volSF") || parser.exists("--volumeSF") || parser.exists("--volumesf") || parser.exists("--Volsf")){volumeSFBool = true; customSetup=true;}

    // label to name the result folder. If none is provided, "results" is used as the name b default
    const std::string defaultLabel = "";
    const std::string outputLabel = parser.get("--label", defaultLabel);
    // file containing the simulation data of the checkpoint that should be analyzed
    const std::string simFile = parser.get("--sim");
    // file containing the simulation data of the additional checkpoint that should be used for FTLE computation
    const std::string ftleFile = parser.get("--simFTLE", simFile);
    // step number of the checkpoint that should be analyzed in simFile 
    const int stepNo = parser.get("--stepNo", 0);
    // step number of the checkpoint that should be used for FTLE computation in ftleFile
    const int ftleStepNo = parser.get("--stepNoFTLE", 0);

    // number of result bins for the structure functions
    const int sfNumBins = parser.get("--sfBins", 1000);
    // norm value used in SF particle selection
    double sfNorm = 1.0;
    // order of the structure function you want to compute
    const int sfOrder = parser.get("--sfOrder", 2);
    // total number of connections that should be computed for the structure functions
    const uint64_t numSfConnections = parser.get("--sfConns",10000000);
    // number of processors per rank
    const int numProcessors = 1;

    // check inputs
    assert(("stepNo must be non-negative",     stepNo >= 0));
    assert(("stepNoFTLE must be non-negative", ftleStepNo >= 0));
    assert(("sfOrder must be larger than zero", sfOrder > 0));

    // set the default setup if no custom setup was provided
    if (customSetup){
        if (plotSlicesBool || volumePDFBool || powerSpectrumBool || volumeSFBool){gridMappingBool=true;}
    } else {
        ftleBool            = false;
        massPDFBool         = true;
        massSFBool          = true;
        gridMappingBool     = true;
        plotSlicesBool      = true;
        volumePDFBool       = true;
        powerSpectrumBool   = true;
        volumeSFBool        = true;
    }

    // create file paths & result folders
    filePathsCls filePaths(simFile,outputLabel);

    // --------------------------------------------------------------------------------------------------------------------------------------------------------------
    // loading the simulation data
    // --------------------------------------------------------------------------------------------------------------------------------------------------------------

    // time at the start of the loading routine
    auto loadingRuntimeStart = std::chrono::high_resolution_clock::now();
    if (rank == 0){std::cout << "\nloading data" << std::endl;}

    // total number of SPH-particles in the simulation
    uint64_t numParticles;
    // local number of SPH-particles on this rank
    uint64_t localNumParticles;
    // physical time of the timestep that should be analyzed
    double time;
    // physical time of the timestep for the FTLE computation
    double ftleTime;
    // SPH-particle IDs of the timestep that should be analyzed
    std::vector<uint64_t> IDs;
    // x SPH-particle positions of the timestep that should be analyzed
    std::vector<double> x;
    // y SPH-particle positions of the timestep that should be analyzed
    std::vector<double> y;
    // z SPH-particle positions of the timestep that should be analyzed
    std::vector<double> z;
    // x SPH-particle velocities of the timestep that should be analyzed
    std::vector<double> vx;
    // y SPH-particle velocities of the timestep that should be analyzed
    std::vector<double> vy;
    // z SPH-particle velocities of the timestep that should be analyzed
    std::vector<double> vz;
    // SPH-particle densities of the timestep that should be analyzed
    std::vector<double> rho;
    // smoothing lengths of the SPH-particles of the timestep that should be analyzed
    std::vector<double> h;
    // SPH-particle IDs of the timestep for the FTLE computation
    std::vector<uint64_t> ftleIDs;
    // x SPH-particle positions for the FTLE computation
    std::vector<double> ftleX;
    // y SPH-particle positions for the FTLE computation
    std::vector<double> ftleY;
    // z SPH-particle positions for the FTLE computation
    std::vector<double> ftleZ;

    // HDF5 reader
    auto reader = makeH5PartReader(MPI_COMM_WORLD);

    // loading the main checkpoint that should be analyzed
    reader->setStep(simFile, stepNo, FileMode::collective);
    numParticles  = reader->globalNumParticles();
    localNumParticles = reader->localNumParticles(); 
    std::cout << "\t" << rank << ": loading " << localNumParticles << "/" << numParticles << " for step " << stepNo << std::endl;
    // resizing
    IDs.resize(localNumParticles);
    x.resize(localNumParticles);
    y.resize(localNumParticles);
    z.resize(localNumParticles);
    vx.resize(localNumParticles);
    vy.resize(localNumParticles);
    vz.resize(localNumParticles);
    rho.resize(localNumParticles);
    h.resize(localNumParticles);
    // loading
    reader->stepAttribute("time", &time, 1);
    // reader->readField("id",  IDs.data()); // deactivated for testing
    reader->readField("x",   x.data());
    reader->readField("y",   y.data());
    reader->readField("z",   z.data());
    reader->readField("vx",  vx.data());
    reader->readField("vy",  vy.data());
    reader->readField("vz",  vz.data());
    reader->readField("h",   h.data());
    // TODO: add density loading
    reader->readField("rho", rho.data());
    reader->closeStep();
    MPI_Barrier(MPI_COMM_WORLD);

    // loading the second timestep for the FTLE computation
    if (ftleBool){
        std::cout << "\t" << rank << ": loading " << localNumParticles << "/" << numParticles << " for FTLE-step " << stepNo << std::endl;
        reader->setStep(ftleFile, ftleStepNo, FileMode::collective);
        // resizing
        ftleIDs.resize(localNumParticles);
        ftleX.resize(localNumParticles);
        ftleY.resize(localNumParticles);
        ftleZ.resize(localNumParticles);
        // loading
        reader->readField("id",  ftleIDs.data());
        reader->readField("x",   ftleX.data());
        reader->readField("y",   ftleY.data());
        reader->readField("z",   ftleZ.data());
        reader->closeStep();
    }


    MPI_Barrier(MPI_COMM_WORLD);
    // time at the end of the loading routine
    auto loadingRuntimeStop = std::chrono::high_resolution_clock::now();
    runtimes.setLoadingRuntime(loadingRuntimeStart,loadingRuntimeStop);
    if (rank == 0){std::cout << "\truntime: " << runtimes.getLoadingRuntime() << " sec\n" << std::endl;}

    // --------------------------------------------------------------------------------------------------------------------------------------------------------------
    // matching the data (only needed for FTLE)
    // --------------------------------------------------------------------------------------------------------------------------------------------------------------

    if (ftleBool){
        // time at the start of the particle matching routine
        auto matchingRuntimeStart = std::chrono::high_resolution_clock::now();
        if (rank == 0){std::cout << "\nmatching analysis data & FTLE data by particle ID" << std::endl;}

        std::vector<uint64_t> IDs2(ftleIDs.size());
        std::copy(ftleIDs.begin(),ftleIDs.end(), IDs2.begin());
        IDMatching(IDs, ftleIDs, IDs2,ftleX,ftleY,ftleZ, numParticles, rank,numRanks,numProcessors);

        for ( size_t i = 0; i < IDs.size(); i++){
            if (IDs[i] != IDs2[i]){
                std::cout << "ERROR on rank " << rank << ": mismatch in matched IDs at position " << i << "/" << IDs.size() << std::endl;
                break;
            }
        }

        // memory cleanup
        ftleIDs.clear();
        IDs2.clear();
        ftleIDs.shrink_to_fit();
        IDs2.shrink_to_fit();

        // time at the end of the particle matching routine
        auto matchingRuntimeStop = std::chrono::high_resolution_clock::now();
        runtimes.setMatchingRuntime(matchingRuntimeStart,matchingRuntimeStop);
        if (rank == 0){std::cout << "\truntime: " << runtimes.getMatchingRuntime() << " sec\n" << std::endl;}
    }

    // --------------------------------------------------------------------------------------------------------------------------------------------------------------
    // interpolate density if needed & compute FTLEs
    // --------------------------------------------------------------------------------------------------------------------------------------------------------------
    std::vector<double> ftle;
    std::vector<double> nc;
    std::vector<double> rho2(rho.size());
    std::copy(rho.begin(),rho.end(), rho2.begin());
    if (rhoBool || ftleBool){
        MPI_Barrier(MPI_COMM_WORLD);
        auto SPHRuntimeStart = std::chrono::high_resolution_clock::now();
        if (rank == 0){std::cout << "\nSPH-interpolation\trho: " << rhoBool << "; FTLE: " << ftleBool << " numParticles:" << numParticles << std::endl;}
        std::tuple<std::vector<double>,std::vector<double>> temp = SPHinterpolation(x,y,z, h,rho,rho2, ftleX,ftleY,ftleZ, IDs,vx,vy,vz, rhoBool,ftleBool, numParticles, rank,numRanks);
        nc      = std::get<0>(temp);
        ftle    = std::get<1>(temp);

        auto [ncPDFx,ncPDFy,ncPDFn]    = PDF(nc, -0.5,200.5,201, numParticles,rank);
        if (rank==0){savePDF("nc", ncPDFx,ncPDFy,ncPDFn, filePaths,true,simFile,stepNo,time,numParticles);}

        MPI_Barrier(MPI_COMM_WORLD);
        auto SPHRuntimeStop = std::chrono::high_resolution_clock::now();
        runtimes.setSPHRuntime(SPHRuntimeStart,SPHRuntimeStop);
        if (rank == 0){std::cout << "\truntime: " << runtimes.getSPHRuntime() << " sec\n" << std::endl;}
    }


    // --------------------------------------------------------------------------------------------------------------------------------------------------------------
    // compute mass-weighted statistics
    // --------------------------------------------------------------------------------------------------------------------------------------------------------------

    MPI_Barrier(MPI_COMM_WORLD);
    auto massStatsRuntimeStart = std::chrono::high_resolution_clock::now();
    if (rank == 0){std::cout << "\ncomputing mass-weighted statistics" << std::endl;}
    statisticsCls   hMassStats( h, numParticles,rank,numRanks);
    statisticsCls  vxMassStats(vx, numParticles,rank,numRanks);
    statisticsCls  vyMassStats(vy, numParticles,rank,numRanks);
    statisticsCls  vzMassStats(vz, numParticles,rank,numRanks);
    statisticsCls   vMassStats(vector_sqrt(vector_add(vector_pow(vx,2),vector_add(vector_pow(vy,2),vector_pow(vz,2)))), numParticles,rank,numRanks);
    statisticsCls rhoMassStats(rho, numParticles,rank,numRanks);
    statisticsCls   sMassStats(vector_log(rho), numParticles,rank,numRanks);
    MPI_Barrier(MPI_COMM_WORLD);
    auto massStatsRuntimeStop = std::chrono::high_resolution_clock::now();
    runtimes.setMassStatsRuntime(massStatsRuntimeStart,massStatsRuntimeStop);
    if (rank == 0){std::cout << "\truntime: " << runtimes.getMassStatsRuntime() << " sec\n" << std::endl;}

    // --------------------------------------------------------------------------------------------------------------------------------------------------------------
    // mass-weighted PDFs
    // --------------------------------------------------------------------------------------------------------------------------------------------------------------

    if (massPDFBool){
        MPI_Barrier(MPI_COMM_WORLD);
        // time at the start of the mass-weighted PDF routine
        auto massPDFRuntimeStart = std::chrono::high_resolution_clock::now();
        if (rank == 0){std::cout << "\ncomputing mass-weighted PDFs" << std::endl;}

        if (rank == 0){std::cout << "\tv" << std::endl;}
        auto [vPDFx,vPDFy,vPDFn]         = PDF(vector_sqrt(vector_add(vector_pow(vx,2),vector_add(vector_pow(vy,2),vector_pow(vz,2)))), 0.0,0.6,300, numParticles,rank);
        if (rank == 0){std::cout << "\tvx" << std::endl;}
        auto [vxPDFx,vxPDFy,vxPDFn]      = PDF(vx, -0.6,0.6,200, numParticles,rank);
        if (rank == 0){std::cout << "\tvy" << std::endl;}
        auto [vyPDFx,vyPDFy,vyPDFn]      = PDF(vy, -0.6,0.6,200, numParticles,rank);
        if (rank == 0){std::cout << "\tvz" << std::endl;}
        auto [vzPDFx,vzPDFy,vzPDFn]      = PDF(vz, -0.6,0.6,200, numParticles,rank);
        if (rank == 0){std::cout << "\trho" << std::endl;}
        auto [rhoPDFx,rhoPDFy,rhoPDFn]   = PDF(rho,  0.6,1.4,200, numParticles,rank);
        if (rank == 0){std::cout << "\ts = log(rho)" << std::endl;}
        auto [sPDFx,sPDFy,sPDFn]         = PDF(vector_log(rho),  -2.0,2.0,400, numParticles,rank);
        if (rank == 0){std::cout << "\trho2" << std::endl;}
        auto [rho2PDFx,rho2PDFy,rho2PDFn]   = PDF(rho2, 0.6,1.4,200, numParticles,rank);
        if (rank == 0){std::cout << "\tdiff" << std::endl;}
        std::vector<double> difference = vector_subtract(rho,rho2);
        auto [diffx,diffy,diffn] = PDF(difference, -0.01,0.01,400, numParticles,rank);

        if (rank == 0){
            std::cout << "\tsaving PDFs" << std::endl;
            savePDF(  "v",    vPDFx,   vPDFy,   vPDFn,  filePaths,true,simFile,stepNo,time,numParticles,  vMassStats);
            savePDF( "vx",   vxPDFx,  vxPDFy,  vxPDFn,  filePaths,true,simFile,stepNo,time,numParticles, vxMassStats);
            savePDF( "vy",   vyPDFx,  vyPDFy,  vyPDFn,  filePaths,true,simFile,stepNo,time,numParticles, vyMassStats);
            savePDF( "vz",   vzPDFx,  vzPDFy,  vzPDFn,  filePaths,true,simFile,stepNo,time,numParticles, vzMassStats);
            savePDF("rho",  rhoPDFx, rhoPDFy, rhoPDFn,  filePaths,true,simFile,stepNo,time,numParticles,rhoMassStats);
            savePDF(  "s",    sPDFx,   sPDFy,   sPDFn,  filePaths,true,simFile,stepNo,time,numParticles,  sMassStats);
            std::cout << "test" << std::endl;
            savePDF("rho2", rho2PDFx,rho2PDFy,rho2PDFn, filePaths,true,simFile,stepNo,time,numParticles);
            savePDF("diff", diffx,diffy,diffn, filePaths,true,simFile,stepNo,time,numParticles);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto massPDFRuntimeStop = std::chrono::high_resolution_clock::now();
        runtimes.setMassPDFRuntime(massPDFRuntimeStart,massPDFRuntimeStop);
        if (rank == 0){std::cout << "\truntime: " << runtimes.getMassPDFRuntime() << " sec\n" << std::endl;}
    }

    // --------------------------------------------------------------------------------------------------------------------------------------------------------------
    // mass-weighted SF
    // --------------------------------------------------------------------------------------------------------------------------------------------------------------

    if (massSFBool){
        MPI_Barrier(MPI_COMM_WORLD);
        auto massSFRuntimeStart = std::chrono::high_resolution_clock::now();
        if (rank == 0){std::cout << "\ncomputing mass-weighted structure functions" << std::endl;}

        auto [xSF,ySF,nSF] = computeSF(vx,vy,vz, sfOrder,numSfConnections,2*hMassStats.getMEAN(),sfNumBins,rank,numRanks,sfNorm,0,x,y,z);

        if (rank == 0){
            saveSF(xSF,ySF,nSF,sfOrder,vMassStats, simFile,stepNo,time,true,filePaths,numParticles,nearestNeighbor);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto massSFRuntimeStop = std::chrono::high_resolution_clock::now();
        runtimes.setMassSFRuntime(massSFRuntimeStart,massSFRuntimeStop);
        if (rank == 0){std::cout << "\truntime: " << runtimes.getMassSFRuntime() << " sec\n" << std::endl;}
    }
    
    // --------------------------------------------------------------------------------------------------------------------------------------------------------------
    // grid-mapping
    // --------------------------------------------------------------------------------------------------------------------------------------------------------------
    int kernelChoice = 1;
    int numGridPoints=200;

    std::vector<double> vxGrid,vyGrid,vzGrid,rhoGrid,ftleGrid,ncGrid;
    if (gridMappingBool){
        MPI_Barrier(MPI_COMM_WORLD);
        auto gridMappingRuntimeStart = std::chrono::high_resolution_clock::now();
        if (rank == 0){
            std::cout << "\nmapping the simulation to the grid using ";
            if (nearestNeighbor){std::cout << "nearest-neighbor mapping" << std::endl;}
            else {std::cout << "SPH-interpolation" << std::endl;}
            std::cout << "\tnumGridPoints: " << numGridPoints << std::endl;
        }

        // total number of grid points
        const uint64_t totalNumGridPoints = std::pow(static_cast<uint64_t>(numGridPoints),2);

        // auto temp = mapToGrid(x,y,z,vx,vy,vz,h,rho,ftle,numGridPoints,numParticles,kernelChoice,nearestNeighbor,rank,numRanks); 
        auto temp = MapToGrid(x,y,z,vx,vy,vz,h,rho,ftle, numGridPoints,numParticles,kernelChoice,numProcessors,rank,numRanks,nearestNeighbor);
        // memory cleanup
        if (rank == 0){std::cout << "\tmemory cleanup" << std::endl;}
        x.clear();
        y.clear();
        z.clear();
        vx.clear();
        vy.clear();
        vz.clear();
        // h.clear();
        // rho.clear();
        // ftle.clear();
        x.shrink_to_fit();
        y.shrink_to_fit();
        z.shrink_to_fit();
        vx.shrink_to_fit();
        vy.shrink_to_fit();
        vz.shrink_to_fit();
        // h.shrink_to_fit();
        // rho.shrink_to_fit();
        // ftle.shrink_to_fit();
        // extract results
        if (rank == 0){std::cout << "\textracting results" << std::endl;}
        vxGrid      = std::get<0>(temp);
        vyGrid      = std::get<1>(temp);
        vzGrid      = std::get<2>(temp);
        rhoGrid     = std::get<3>(temp);
        ftleGrid    = std::get<4>(temp);
        ncGrid      = std::get<5>(temp);
        std::vector<double> vGrid = vector_sqrt(vector_add(vector_pow(vxGrid,2),vector_add(vector_pow(vyGrid,2),vector_pow(vzGrid,2))));

        MPI_Barrier(MPI_COMM_WORLD);
        auto gridMappingRuntimeStop = std::chrono::high_resolution_clock::now();
        runtimes.setGridRuntime(gridMappingRuntimeStart,gridMappingRuntimeStop);
        if (rank == 0){std::cout << "\truntime: " << runtimes.getGridRuntime() << " sec\n" << std::endl;}


        // --------------------------------------------------------------------------------------------------------------------------------------------------------------
        // plotting grid slices
        // --------------------------------------------------------------------------------------------------------------------------------------------------------------

        if (plotSlicesBool){
            MPI_Barrier(MPI_COMM_WORLD);
            auto slicesRuntimeStart = std::chrono::high_resolution_clock::now();
            if (rank == 0){std::cout << "\nplotting grid slices" << std::endl;}

            plotBoxSlice(ncGrid,  'z',  50, "nc",numGridPoints,time,rank,numRanks,simFile,stepNo,filePaths,nearestNeighbor);
            plotBoxSlice(rhoGrid, 'z',  50,"rho",numGridPoints,time,rank,numRanks,simFile,stepNo,filePaths,nearestNeighbor);
            plotBoxSlice(ncGrid,  'z', 100, "nc",numGridPoints,time,rank,numRanks,simFile,stepNo,filePaths,nearestNeighbor);
            plotBoxSlice(rhoGrid, 'z', 100,"rho",numGridPoints,time,rank,numRanks,simFile,stepNo,filePaths,nearestNeighbor);
            plotBoxSlice(ncGrid,  'z', 150, "nc",numGridPoints,time,rank,numRanks,simFile,stepNo,filePaths,nearestNeighbor);
            plotBoxSlice(rhoGrid, 'z', 150,"rho",numGridPoints,time,rank,numRanks,simFile,stepNo,filePaths,nearestNeighbor);

            MPI_Barrier(MPI_COMM_WORLD);
            auto slicesRuntimeStop = std::chrono::high_resolution_clock::now();
            runtimes.setSlicesRuntime(slicesRuntimeStart,slicesRuntimeStop);
            if (rank == 0){std::cout << "\truntime: " << runtimes.getSlicesRuntime() << " sec\n" << std::endl;}
        }


        // --------------------------------------------------------------------------------------------------------------------------------------------------------------
        // compute volume-weighted statistics
        // --------------------------------------------------------------------------------------------------------------------------------------------------------------

        MPI_Barrier(MPI_COMM_WORLD);
        auto volStatsRuntimeStart = std::chrono::high_resolution_clock::now();
        if (rank == 0){std::cout << "\ncomputing volume-weighted statistics" << std::endl;}
        statisticsCls  ncVolumeStats( ncGrid, numParticles,rank,numRanks);
        statisticsCls   vVolumeStats(  vGrid, numParticles,rank,numRanks);
        statisticsCls  vxVolumeStats( vxGrid, numParticles,rank,numRanks);
        statisticsCls  vyVolumeStats( vyGrid, numParticles,rank,numRanks);
        statisticsCls  vzVolumeStats( vzGrid, numParticles,rank,numRanks);
        statisticsCls rhoVolumeStats(rhoGrid, numParticles,rank,numRanks);
        statisticsCls   sVolumeStats(vector_log(rhoGrid), numParticles,rank,numRanks);
        MPI_Barrier(MPI_COMM_WORLD);
        auto volStatsRuntimeStop = std::chrono::high_resolution_clock::now();
        runtimes.setVolStatsRuntime(volStatsRuntimeStart,volStatsRuntimeStop);
        if (rank == 0){std::cout << "\truntime: " << runtimes.getVolStatsRuntime() << " sec\n" << std::endl;}

        // --------------------------------------------------------------------------------------------------------------------------------------------------------------
        // volume-weighted PDFs
        // --------------------------------------------------------------------------------------------------------------------------------------------------------------
        
        if (volumePDFBool){
            MPI_Barrier(MPI_COMM_WORLD);
            auto volumePDFRuntimeStart = std::chrono::high_resolution_clock::now();
            if (rank == 0){std::cout << "\ncomputing volume-weighted PDFs" << std::endl;}

            if (rank == 0){std::cout << "\tv" << std::endl;}
            auto [vPDFx,vPDFy,vPDFn]       = PDF(  vGrid, 0.0,10,1000, numParticles,rank);
            if (rank == 0){std::cout << "\tnc" << std::endl;}
            auto [ncPDFx,ncPDFy,ncPDFn]    = PDF( ncGrid, -0.5,200.5,201, numParticles,rank);
            if (rank == 0){std::cout << "\tvx" << std::endl;}
            auto [vxPDFx,vxPDFy,vxPDFn]    = PDF( vxGrid, -0.6,0.6,200, numParticles,rank);
            if (rank == 0){std::cout << "\tvy" << std::endl;}
            auto [vyPDFx,vyPDFy,vyPDFn]    = PDF( vyGrid, -0.6,0.6,200, numParticles,rank);
            if (rank == 0){std::cout << "\tvz" << std::endl;}
            auto [vzPDFx,vzPDFy,vzPDFn]    = PDF( vzGrid, -0.6,0.6,200, numParticles,rank);
            if (rank == 0){std::cout << "\trho" << std::endl;}
            auto [rhoPDFx,rhoPDFy,rhoPDFn] = PDF(rhoGrid,  0.6,1.4,200, numParticles,rank);
            if (rank == 0){std::cout << "\ts = log(rho)" << std::endl;}
            auto [sPDFx,sPDFy,sPDFn] = PDF(vector_log(rhoGrid),  -2,2,400, numParticles,rank);

            if (rank == 0){
                std::cout << "\tsaving PDFs" << std::endl;
                savePDF(  "v",   vPDFx,   vPDFy,   vPDFn,  filePaths,false,simFile,stepNo,time,totalNumGridPoints,  vVolumeStats,nearestNeighbor);
                savePDF( "nc",  ncPDFx,  ncPDFy,  ncPDFn,  filePaths,false,simFile,stepNo,time,totalNumGridPoints, ncVolumeStats,nearestNeighbor);
                savePDF( "vx",  vxPDFx,  vxPDFy,  vxPDFn,  filePaths,false,simFile,stepNo,time,totalNumGridPoints, vxVolumeStats,nearestNeighbor);
                savePDF( "vy",  vyPDFx,  vyPDFy,  vyPDFn,  filePaths,false,simFile,stepNo,time,totalNumGridPoints, vyVolumeStats,nearestNeighbor);
                savePDF( "vz",  vzPDFx,  vzPDFy,  vzPDFn,  filePaths,false,simFile,stepNo,time,totalNumGridPoints, vzVolumeStats,nearestNeighbor);
                savePDF("rho", rhoPDFx, rhoPDFy, rhoPDFn,  filePaths,false,simFile,stepNo,time,totalNumGridPoints,rhoVolumeStats,nearestNeighbor);
                savePDF(  "s",   sPDFx,   sPDFy,   sPDFn,  filePaths,false,simFile,stepNo,time,totalNumGridPoints,  sVolumeStats,nearestNeighbor);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto volumePDFRuntimeStop = std::chrono::high_resolution_clock::now();
            runtimes.setVolumePDFRuntime(volumePDFRuntimeStart,volumePDFRuntimeStop);
            if (rank == 0){std::cout << "\truntime: " << runtimes.getVolumePDFRuntime() << " sec\n" << std::endl;}
        }

        // --------------------------------------------------------------------------------------------------------------------------------------------------------------
        // power spectra
        // --------------------------------------------------------------------------------------------------------------------------------------------------------------
        double cutRadius = 40;
        int numBins = 100;
        double simEkin = 1.0;
        double gridEkin = 1.0;

        if (powerSpectrumBool){
            MPI_Barrier(MPI_COMM_WORLD);
            auto psRuntimeStart = std::chrono::high_resolution_clock::now();
            if (rank == 0){std::cout << "\ncomputing power spectra" << std::endl;}

            if(rank==0){std::cout << "\tcomputing frequencies k" << std::endl;}  
            std::vector<double> k1D = computeK(numGridPoints);

            if(rank==0){std::cout << "\tcomputing power spectra" << std::endl;}  
            std::vector<std::complex<double>>   sPS = computePS(vector_log(rhoGrid), numGridPoints, rank,numRanks);
            std::vector<std::complex<double>> rhoPS = computePS(rhoGrid, numGridPoints, rank,numRanks);
            std::vector<std::complex<double>>  vxPS = computePS( vxGrid, numGridPoints, rank,numRanks);
            std::vector<std::complex<double>>  vyPS = computePS( vyGrid, numGridPoints, rank,numRanks);
            std::vector<std::complex<double>>  vzPS = computePS( vzGrid, numGridPoints, rank,numRanks);

            if(rank==0){std::cout << "\tbinning rho power spectrum" << std::endl;}  
            auto [krho,PSrho,Numrho] = PSbinning(k1D,rhoPS,numGridPoints,numBins,rank,numRanks);
            auto [ks,PSs,Nums]       = PSbinning(k1D,  sPS,numGridPoints,numBins,rank,numRanks);
            auto [kv,PSv,Numv]       = PSbinning(k1D,vector_sum(vector_square(vxPS),vector_square(vyPS),vector_square(vzPS)),numGridPoints,numBins,rank,numRanks);

            if (rank == 0){
                std::cout << "\tsaving power spectra to file" << std::endl;
                save_PS(  ks,  PSs,  Nums, simFile, simEkin,gridEkin, time,  "s",stepNo,nearestNeighbor,filePaths,numGridPoints);
                save_PS(krho,PSrho,Numrho, simFile, simEkin,gridEkin, time,"rho",stepNo,nearestNeighbor,filePaths,numGridPoints);
                save_PS(  kv,  PSv,  Numv, simFile, simEkin,gridEkin, time,  "v",stepNo,nearestNeighbor,filePaths,numGridPoints);
            }

            if(rank==0){std::cout << "\tdensity reconstruction test" << std::endl;}
            std::vector<double> sInverse   = inverseTransform(sPS, numGridPoints,rank,numRanks);
            std::vector<double> rhoInverse = inverseTransform(rhoPS, numGridPoints,rank,numRanks);
            if(rank==0){std::cout << "\tplotting reconstructed box slices" << std::endl;}
            plotBoxSlice(vector_abs(rhoPS),   'z',  0,   "rhoPS",numGridPoints,time, rank,numRanks,simFile,stepNo,filePaths,nearestNeighbor);
            plotBoxSlice(vector_abs(sPS),     'z',  0,     "sPS",numGridPoints,time, rank,numRanks,simFile,stepNo,filePaths,nearestNeighbor);
            plotBoxSlice(vector_log(rhoGrid), 'z',100,       "s",numGridPoints,time, rank,numRanks,simFile,stepNo,filePaths,nearestNeighbor);
            plotBoxSlice(sInverse,            'z',100,  "srecon",numGridPoints,time, rank,numRanks,simFile,stepNo,filePaths,nearestNeighbor);
            plotBoxSlice(rhoInverse,          'z',100,"rhorecon",numGridPoints,time, rank,numRanks,simFile,stepNo,filePaths,nearestNeighbor);
            sInverse.clear();
            sInverse.shrink_to_fit();


            MPI_Barrier(MPI_COMM_WORLD);
            auto psRuntimeStop = std::chrono::high_resolution_clock::now();
            runtimes.setPSRuntime(psRuntimeStart,psRuntimeStop);
            if (rank == 0){std::cout << "\truntime: " << runtimes.getPSRuntime() << " sec\n" << std::endl;}
        }

        // --------------------------------------------------------------------------------------------------------------------------------------------------------------
        // volume-weighted SF
        // --------------------------------------------------------------------------------------------------------------------------------------------------------------

        if (volumeSFBool){
            MPI_Barrier(MPI_COMM_WORLD);
            auto volumeSFRuntimeStart = std::chrono::high_resolution_clock::now();
            if (rank == 0){std::cout << "\ncomputing volume-weighted structure functions" << std::endl;}

            auto [xSF,ySF,nSF] = computeSF(vxGrid,vyGrid,vzGrid, sfOrder,numSfConnections,2*hMassStats.getMEAN(),sfNumBins,rank,numRanks,sfNorm,numGridPoints);

            if (rank == 0){
                saveSF(xSF,ySF,nSF,sfOrder,vVolumeStats, simFile,stepNo,time,false,filePaths,totalNumGridPoints,nearestNeighbor);
            }

            MPI_Barrier(MPI_COMM_WORLD);
            auto volumeSFRuntimeStop = std::chrono::high_resolution_clock::now();
            runtimes.setVolumeSFRuntime(volumeSFRuntimeStart,volumeSFRuntimeStop);
            if (rank == 0){std::cout << "\truntime: " << runtimes.getVolumeSFRuntime() << " sec\n" << std::endl;}
        }
    } // end of the volume-weighted part







    // stop the total runtime measurement
    auto totalRuntimeStop = std::chrono::high_resolution_clock::now();
    runtimes.setTotalRuntime(totalRuntimeStart,totalRuntimeStop);
    // print runtime information
    if (rank == 0){
        runtimes.printInfo();
        runtimes.saveInfo(simFile,stepNo,numRanks,nearestNeighbor,filePaths);
    }

    // stop the MPI environment
    return MPIfinal();
}

void printHelp(char* name, int rank)
{
    if (rank == 0)
    {
        printf("\nUsage:\n\n");
        printf("%s [OPTIONS]\n", name);
        printf("\nWhere possible options are:\n\n");

        printf("\t--simfile \t\t HDF5 checkpoint file with simulation data. Needed for computng the grid if none is available\n\n");
        printf("\t--stepNo \t\t step number of the data in the SimFile. Default is 0.\n\n");
        printf("\t--gridfile \t\t HDF5 file containing the girid data. If a simfile is provided, the computed grid will be stored here.\n\n");
        printf("\t--numGridPoints \t\t number of grid points in 1 dimension. Default is 5000.\n\n");
        printf("\t--numProcs \t\t number of processors available to each rank. Default is 1.\n\n");
        printf("\t--numOut \t\t number that is used to identify the outputs from this run. Default is 0.\n\n");
    }
}