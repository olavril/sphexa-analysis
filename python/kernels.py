#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 20:20:55 2025

@author: Oliver Avril
"""
import numpy as np
import matplotlib.pyplot as plt

def kernel2(dist, h):
    nu = dist / h
    K = 0.470626
    kern = K/h**3 * (0.9 * np.sinc(nu/2)**4 + 0.1 * np.sinc(nu/2)**9);
    
    return kern

def kernel1(dist, h):
    
    K = 0.790450
    kern = K/h**3 * (np.sinc(dist/(2*h)))**6
    
    return kern

h = 1.0
x = np.linspace(0.0,3.0,1000000)
y = np.zeros(len(x))
for i in range(0,len(x)):
    y[i] = kernel2(x[i],h)
    
mask = y > 0.000088
plt.plot(x[mask],y[mask])
