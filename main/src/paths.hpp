#pragma once

#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace fs;

class filePathsCls {
private:
    // path to the folder that will contain the analysis results
    fs::path resultFolder;
    fs::path pdfFolder;
    fs::path pdfMassFolder;
    fs::path pdfVolumeFolder;
    fs::path sfFolder;
    fs::path sfMassFolder;
    fs::path sfVolumeFolder;
    fs::path psFolder;
    fs::path slicesFolder;
    fs::path runtimeFolder;

public:
    //constuctor
    // the label input can be used for custom result folder names. If an empty string is provided, the folder is names "results"
    filePathsCls(const std::string simFile, const std::string label = ""){
        // get the path to the simulation file
        fs::path simPath(simFile);
        // extract the path to the folder
        fs::path baseFolder = simPath.parent_path();

        // check if the base directory exists
        if (fs::exists(baseFolder)){
            std::string resultFolderName;
            if (label.length() > 0){resultFolderName = label;}
            else {resultFolderName = "results";}
            // create the results folder
            resultFolder = baseFolder.append(resultFolderName);
            if (not fs::exists(resultFolder)){fs::create_directory(resultFolder);}
            // create the PDF folder
            pdfFolder = resultFolder;
            pdfFolder.append("PDFs");
            if (not fs::exists(pdfFolder)){fs::create_directory(pdfFolder);}
            pdfMassFolder = pdfFolder;
            pdfMassFolder.append("mass");
            if (not fs::exists(pdfMassFolder)){fs::create_directory(pdfMassFolder);}
            pdfVolumeFolder = pdfFolder;
            pdfVolumeFolder.append("volume");
            if (not fs::exists(pdfVolumeFolder)){fs::create_directory(pdfVolumeFolder);}
            // create the SF folder
            sfFolder = resultFolder;
            sfFolder.append("SFs");
            if (not fs::exists(sfFolder)){fs::create_directory(sfFolder);}
            sfMassFolder = sfFolder;
            sfMassFolder.append("mass");
            if (not fs::exists(sfMassFolder)){fs::create_directory(sfMassFolder);}
            sfVolumeFolder = sfFolder;
            sfVolumeFolder.append("volume");
            if (not fs::exists(sfVolumeFolder)){fs::create_directory(sfVolumeFolder);}
            // create the PS folder
            psFolder = resultFolder;
            psFolder.append("PS");
            if (not fs::exists(psFolder)){fs::create_directory(psFolder);}
            // create the box-slices folder
            slicesFolder = resultFolder;
            slicesFolder.append("box-slices");
            if (not fs::exists(slicesFolder)){fs::create_directory(slicesFolder);}
            runtimeFolder = resultFolder;
            runtimeFolder.append("runtimes");
            if (not fs::exists(runtimeFolder)){fs::create_directory(runtimeFolder);}
        } else {
            std::cout << "ERROR: the folder " << baseFolder << " does not exist." << std::endl;
        }
        return;
    }

    // function to get the PDF folder path
    std::string getPDFPath(const bool massWeighted, const std::string filename){
        // path to the result folder
        fs::path path;
        if (massWeighted){path = pdfMassFolder;}
        else {path = pdfVolumeFolder;}
        // add the filename
        path = path.append(filename);
        // return the filepath
        return path.string();
    }

    // function to get the SF folder path
    std::string getSFPath(const bool massWeighted, const std::string filename){
        // path to the result folder
        fs::path path;
        if (massWeighted){path = sfMassFolder;}
        else {path = sfVolumeFolder;}
        // add the filename
        path = path.append(filename);
        // return the filepath
        return path.string();
    }

    // function to get the PS folder path
    std::string getPSPath(const std::string filename){
        // path to the result folder
        fs::path path = psFolder;
        // add the filename
        path = path.append(filename);
        // return the filepath
        return path.string();
    }

    // function to get the box-slices folder path
    std::string getSlicesPath(const std::string filename){
        // path to the result folder
        fs::path path = slicesFolder;
        // add the filename
        path = path.append(filename);
        // return the filepath
        return path.string();
    }

    // function to get the main results folder path
    std::string getRuntimePath(const std::string filename){
        // path to the result folder
        fs::path path = runtimeFolder;
        // add the filename
        path = path.append(filename);
        // return the filepath
        return path.string();
    }

    // function to get the main results folder path
    std::string getResultPath(const std::string filename){
        // path to the result folder
        fs::path path = resultFolder;
        // add the filename
        path = path.append(filename);
        // return the filepath
        return path.string();
    }
};