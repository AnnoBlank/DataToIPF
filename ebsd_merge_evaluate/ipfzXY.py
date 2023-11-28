#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:36:16 2017

@author: karsten, altered by Robin Barth on 21.09.2023
"""

import numpy as np


# =============================================================================
# Returns the normalized rotation matrix (hkl)[uvw]
# =============================================================================

def rotation(phi1, phi, phi2):
    
   phi1 = np.deg2rad(phi1);
   phi = np.deg2rad(phi);
   phi2 = np.deg2rad(phi2);
   
   return np.array([[np.cos(phi1)*np.cos(phi2)-np.cos(phi)*np.sin(phi1)*np.sin(phi2),
            -np.cos(phi)*np.cos(phi2)*np.sin(phi1)-np.cos(phi1)*
            np.sin(phi2),np.sin(phi)*np.sin(phi1)],[np.cos(phi2)*np.sin(phi1)
            +np.cos(phi)*np.cos(phi1)*np.sin(phi2),np.cos(phi)*np.cos(phi1)
            *np.cos(phi2)-np.sin(phi1)*np.sin(phi2), -np.cos(phi1)*np.sin(phi)],
            [np.sin(phi)*np.sin(phi2), np.cos(phi2)*np.sin(phi), np.cos(phi)]],float)

                                                    
# =============================================================================
# projection algorithms (not yet understood)                                                    
# =============================================================================
                                                    
def proj(x, y, z): 
  
    if z == 1: 
        X = Y = 0
    elif z < -0.000001:
        X = X = 250
    else: 
            
        X = x/(1+z)
        Y = y/(1+z)
    
    return np.array([X, Y], float) 

 
def poleA2(pole1, pole2, pole3):
    
    D, Dstar, V = cristalStruc()
    MA = rotation(0,0,0)    
    
    pole1, pole2, pole3 = pole22(pole1, pole2, pole3)

    Gs = np.array([pole1,pole2,pole3], float)
    Gsh = np.dot(Dstar, Gs)/np.linalg.norm(np.dot(Dstar, Gs))
    S = np.dot(MA, Gsh)
    
    if S[2]<0:
        S = -S


    return proj(S[0], S[1], S[2])*600/2


# =============================================================================
# Sorting of angles according to scheme 2 < 1 < 3
# old sorting was inefficient, shorter algorithm was implemented
# =============================================================================

def pole22(pole1, pole2, pole3):
    
    poles = np.sort([np.abs(pole1), np.abs(pole2), np.abs(pole3)])
    
    return poles[1], poles[0], poles[2]


# =============================================================================
# Calculation of characteristic crystal matrix, inverse and volume (standard 
# set to tungsten)
# =============================================================================

def cristalStruc(a = 3.1648, b = 3.1648, c = 3.1648, alp = 90, bet = 90, gam = 90):
    
    alp = np.deg2rad(alp);
    bet = np.deg2rad(bet);
    gam = np.deg2rad(gam);
    
    V = a*b*c*np.sqrt(1-(np.cos(alp)**2)-(np.cos(bet))**2-(np.cos(gam))**2+2*b*c*np.cos(alp)*np.cos(bet)*np.cos(gam))
    D = np.array([[a,b*np.cos(gam),c*np.cos(bet)],[0,b*np.sin(gam),  c*(np.cos(alp)-np.cos(bet)*np.cos(gam))/np.sin(gam)],[0,0,V/(a*b*np.sin(gam))]])
    Dstar = np.transpose(np.linalg.inv(D))
            
    return D, Dstar, V