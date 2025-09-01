#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 13:43:10 2025

@author: Oliver Avril
"""

import numpy as np
import matplotlib.pyplot as plt

from loadingFunctions import *

step        = 9
weights     = ["mass","volume"]
basePath    = "/home/hpotter/Documents/simulations/Ruben-subsonic/avc/new-results/results"
colors      = ["red","blue","green","purple"]


#%% PDF

key     = "rho"

for i in range(0,len(weights)):
    weight = weights[i]
    path = f"{basePath}/PDFs/{weight}/{weight}-PDF-{key}-{step}.txt"
    PDF = loadPDF(path)
    ### plot the data
    plt.plot(PDF["x"],PDF["y"], color=colors[i],label=weight)
plt.title(f"{key}-PDF")
plt.legend()
plt.show()


#%% structure function

order = 2

for i in range(0,len(weights)):
    weight = weights[i]
    path = f"{basePath}/SFs/{weight}/{weight}-SF-{step}-{order}.txt"
    SF = loadSF(path)
    ### plot the data
    plt.plot(SF["x"],SF["y"], color=colors[i],label=weight)
    plt.hlines(y=2*SF["vRMS"]**2,xmin=0,xmax=0.5, linestyle="--",color=colors[i],label="2vRMS")
plt.title(f"structure functions")
plt.xlim(0,0.5)
plt.legend()
plt.show()


#%% power spectra

key = "v"

path = f"{basePath}/PS/PS-{key}-{step}.txt"
PS = loadPS(path)
plt.plot(PS["k"],PS["ps"], color=colors[0],label="SPH-EXA")
plt.title("power spectrum")
plt.xscale("log")
plt.yscale("log")
plt.xlim(1,max(PS["k"]))

### plot references
# refpath = "../../../references/"
# k_arepo256, P_arepo256 = np.loadtxt(refpath+"arepo256.txt" ,usecols=(0,1),unpack=True)
# k_arepo256 = k_arepo256/(2*np.pi)
# P_arepo256 = P_arepo256/k_arepo256
# plt.plot(k_arepo256,P_arepo256, color=colors[1],label="AREPO 256")
# k_arepo512, P_arepo512 = np.loadtxt(refpath+"arepo512.txt", usecols=(0,1),unpack=True)
# k_arepo512 = k_arepo512/(2*np.pi)
# P_arepo512 = P_arepo512/k_arepo512
# plt.plot(k_arepo512,P_arepo512, color=colors[2],label="AREPO 512")
# k_gizmo, P_gizmo = np.loadtxt(refpath+"gizmoMFV256.txt", usecols=(0,1),unpack=True)
# k_gizmo = k_gizmo/(2*np.pi)
# P_gizmo = P_gizmo/(k_gizmo**(5/3))
# plt.plot(k_gizmo,P_gizmo, color=colors[3],label="Gizmo 256")


plt.legend()
plt.show()