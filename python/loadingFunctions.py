#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:48:00 2025

@author: Oliver Avril

This file contains a number of load functions to load output of the c++ code 
into an appropriate dictionary
"""

import numpy as np

### function to load PDF output
def loadPDF(path):
    ### result container
    pdf = {"filepath": path}
    ### load header
    with open(path, "r") as file:
        for line in file:
            segments = line.split(" ")
            if (segments[0] == "###"):
                break;
            elif (segments[0] == "#"):
                if (segments[1] in ["time","ftleTime","RMS","MEAN","STD","numPoints"]):
                    pdf[segments[1]] = float(segments[-1])
                elif (segments[1] in ["order","stepNo","ftleStepNo"]):
                    pdf[segments[1]] = int(segments[-1])
                elif (segments[1] in ["mass-weighted","nearest-neighbor"]):
                    pdf[segments[1]] = bool(segments[-1])
                else:
                    pdf[segments[1]] = segments[-1]        
    ### load data
    pdf["x"], pdf["y"], pdf["#N"] = np.loadtxt(path, usecols=(0,1,2), unpack=True)
    return pdf



### function to load structure function output
def loadSF(path):
    ### result container
    sf = {"filepath": path}
    ### load header
    with open(path, "r") as file:
        for line in file:
            segments = line.split(" ")
            if (segments[0] == "###"):
                break;
            elif (segments[0] == "#"):
                if (segments[1] in ["time","vRMS","vMEAN","vSTD","numPoints"]):
                    sf[segments[1]] = float(segments[-1])
                elif (segments[1] in ["stepNo"]):
                    sf[segments[1]] = int(segments[-1])
                elif (segments[1] in ["mass-weighted","nearest-neighbor"]):
                    sf[segments[1]] = bool(segments[-1])
                else:
                    sf[segments[1]] = segments[-1]        
    ### load data
    sf["x"], sf["y"], sf["#N"] = np.loadtxt(path, usecols=(0,1,2), unpack=True)
    return sf



### function to load power spectrum output
### in this function "numGridPoints" refers to the number of grid points per dimension
def loadPS(path):
    ### result container
    ps = {"filepath": path}
    ### load header
    with open(path, "r") as file:
        for line in file:
            segments = line.split(" ")
            if (segments[0] == "###"):
                break;
            elif (segments[0] == "#"):
                if (segments[1] in ["time","SimEkin","GridEkin"]):
                    ps[segments[1]] = float(segments[-1])
                elif (segments[1] in ["stepNo","numGridPoints"]):
                    ps[segments[1]] = int(segments[-1])
                elif (segments[1] in ["nearest-neighbor"]):
                    ps[segments[1]] = bool(segments[-1])
                else:
                    ps[segments[1]] = segments[-1]        
    ### load data
    ps["k"], ps["ps"], ps["#N"] = np.loadtxt(path, usecols=(0,1,2), unpack=True)
    return ps



### function to load runtime output
def loadRuntime(path):
    ### result container
    runtimes = {"filepath": path}
    ### load header
    with open(path, "r") as file:
        for line in file:
            segments = line.split(" ")
            if (segments[0] == "###"):
                break;
            elif (segments[0] == "#"):
                if (segments[1] in ["simFile"]):
                    runtimes[segments[1]] = segments[-1]
                elif (segments[1] in ["stepNo","numRanks"]):
                    runtimes[segments[1]] = int(segments[-1])
                elif (segments[1] in ["nearestNeighbor"]):
                    runtimes[segments[1]] = bool(segments[-1])
                else:
                    runtimes[segments[1]] = float(segments[-1])  
    return runtimes



### structure to load box slices from the c++ output
class boxSlice:
    def __init__(self,path):
        self.filepath = path
        ### load header
        with open(path, "r") as file:
            for line in file:
                segments = line.split(" ")
                if (segments[0] == "###"):
                    break;
                elif (segments[0] == "#"):
                    if (segments[1] == "datakey"):
                        self.datakey        = segments[-1]
                    elif (segments[1] == "simFile"):
                        self.simFile        = segments[-1]
                    elif (segments[1] == "stepNo"):
                        self.stepNo         = int(segments[-1])
                    elif (segments[1] == "time"):
                        self.time           = float(segments[-1])
                    elif (segments[1] == "orthogonal"):
                        self.orthogonal     = segments[-1]
                    elif (segments[1] == "layer"):
                        self.height         = int(segments[-1])
                    elif (segments[1] == "height"):
                        self.height         = float(segments[-1])
                    elif (segments[1] == "numGridPoints"):
                        self.numGridPoints  = int(segments[-1])
        ### load data
        self.data = np.loadtxt(path)
        return
    
    def getPositions(self):
        boxSize  = 1.0
        gridStep = boxSize/float(self.numGridPoints)
        ### position containers
        x = np.zeros(len(self.data))
        y = np.zeros(len(self.data))
        ### create positions
        for i1 in range(0,self.numGridPoints):
            for i2 in range(0,self.numGridPoints):
                resIndex = i1*self.numGridPoints + i2
                x[resIndex] = gridStep*(float(i1)+0.5) - boxSize/2
                y[resIndex] = gridStep*(float(i2)+0.5) - boxSize/2
        return x, y, self.data
    
    def getData(self):
        result = np.zeros(shape=(self.numGridPoints,self.numGridPoints))
        ### fill in the data
        for i1 in range(0,self.numGridPoints):
            for i2 in range(0,self.numGridPoints):
                index = i1*self.numGridPoints + i2
                result[i1,i2] = self.data[index]
        return result
        
        
        
        
        
        
        
        