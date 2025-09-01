#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 13:24:10 2025

@author: Oliver Avril

this file creates a run.sh file to start the analysis according to your settings
"""

import os

def createRunFile(settings,checkpoints,pipeline,SFs):
    ### delete the ile if it already exists
    if (os.path.exists(settings["runFileName"])):
        os.remove(settings["runFileName"])
    
    ### write the run file
    with open(settings["runFileName"],"w") as file:
        ### header
        file.write("#!/bin/bash\n")
        file.write("#SBATCH --job-name={}\n".format(settings["jobName"]))
        file.write("#SBATCH --time={}\n".format(settings["timelimit"]))
        file.write("#SBATCH --nodes={}\n".format(settings["#nodes"]))
        file.write("#SBATCH --ntasks-per-node={}\n".format(settings["tasksPerNode"]))
        file.write("#SBATCH --partition={}\n".format(settings["partition"]))
        file.write("#SBATCH --output={}\n".format(settings["outputFile"]))
        file.write("#SBATCH --exclude={}\n\n".format(settings["excludedNodes"]))

        ###create the run command
        runComm = "srun {} --sim {} --stepNo {}".format(settings["executablePath"], checkpoints["simFile"], checkpoints["stepNo"])
        if (len(settings["outputLabel"]) > 0):
            runComm += " --label {}".format(settings["outputLabel"])
        if (pipeline["FTLE"]):
            runComm += " --ftle --simFTLE {} --stepNoFTLE {}".format(checkpoints["ftleFile"], checkpoints["ftleStepNo"])
        if (pipeline["massPDF"]):
            runComm += " --massPDF"
        if (pipeline["volumePDF"]):
            runComm += " --volumePDF"
        if (pipeline["plotSlices"]):
            runComm += " --plotSlices"
        if (pipeline["PS"]):
            runComm += " --PS"
        if (pipeline["massSF"]):
            runComm += " --massSF"
        if (pipeline["volumeSF"]):
            runComm += " --volumeSF"
        if (pipeline["massSF"] or pipeline["volumeSF"]):
            runComm += " --sfBins {} --sfOrder {} --sfConns {}".format(SFs["numBins"],SFs["order"],SFs["#Connections"])
        ### write the run command to file 
        file.write(runComm)
    return

### general settings
settings = {}
settings["runFileName"]     = "run-analysis.sh"
settings["jobName"]         = "sphexa-analysis"
settings["#nodes"]          = 2
settings["timelimit"]       = "00:02:00"
settings["tasksPerNode"]    = 4                     # should always be 4 on DAINT
settings["partition"]       = "normal"
settings["outputFile"]      = "out"
settings["excludedNodes"]   = "nid005676,nid005684"
settings["executablePath"]  = "../../executables/sphexa_analysis"
settings["outputLabel"]     = "test"                # this is the folder name in which the results will be stored

### checkpoint information
checkpoints = {}
checkpoints["simFile"]      = ""
checkpoints["stepNo"]       = 0
checkpoints["ftleFile"]     = ""
checkpoints["ftleStepNo"]   = 0

### setup of the pipeline
pipeline = {}
pipeline["FTLE"]            = False
pipeline["massPDF"]         = True
pipeline["volumePDF"]       = True
pipeline["plotSlices"]      = True
pipeline["PS"]              = True
pipeline["massSF"]          = True
pipeline["volumeSF"]        = True

### structure functions settings
SFs = {}
SFs["order"]                = 2
SFs["numBins"]              = 1000
SFs["#Connections"]         = 2000000


createRunFile(settings,checkpoints,pipeline,SFs)

