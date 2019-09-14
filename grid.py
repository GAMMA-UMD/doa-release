# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 10:40:58 2018

@author: WSJF7149
"""
import numpy as np

# %%
def deg2rad(angle):
    return angle/180*np.pi

# %%
def rad2deg(angle):
    return angle*180/np.pi

#%%
def angularDistance(el0, az0, el1, az1):
    '''
    Computes the angle difference between two directions.
    
    Parameters
    ----------
    el0: float
    Elevation of the first direction, in degrees
    
    az0: float
    Azimuth of the first direction, in degrees
    
    el1: float
    Elevation of the second direction, in degrees
    
    az1: float
    Azimuth of the second direction, in degrees
    
    Returns
    -------
    delta: float
    The angle difference, in degrees
    
    Reference
    -----
    https://en.wikipedia.org/wiki/Angular_distance
    
    Notes
    -----
    Using just the formula without clipping can create rounding errors (try el0=12, az0=25, el1=12, az1=25)
    
    '''
    
    appearingCos = np.sin(deg2rad(el0))*np.sin(deg2rad(el1))+ \
                   np.cos(deg2rad(el0))*np.cos(deg2rad(el1)) \
                   *np.cos(abs(deg2rad(az0)-deg2rad(az1)))
    appearingCos = np.clip(appearingCos, -1, 1)
    delta = np.arccos(appearingCos)
    return rad2deg(delta)

# %%
def makeDoaGrid(stepDeg, fig=True):
    '''
    Make an elevation and an azimuth grid on a sphere with a uniform spacing for both.
    
    Parameters
    ----------
    stepDeg: float
    The spacing, in degrees. 
    
    Returns
    -------
    EL: 1d array, float
    The elevation grid, in degrees
    
    AZ: 1d array, float
    The azimuth grid, in degrees
    
    '''
    nEl = int(np.round(180/stepDeg)+1)
    el_l = np.linspace(-90, 90, num=nEl, endpoint=True)
    az_l = []
    nPos = 0
    for iEl in range(nEl):
        nAz = int(np.round(360*np.cos(deg2rad(el_l[iEl]))/stepDeg)+1)
        azIEl = np.linspace(-180, 180, num=nAz, endpoint=False)
        az_l.append(azIEl)
        nPos += len(azIEl)
    
    el_grid = np.empty(nPos,)
    az_grid = np.empty(nPos,)
    iPos = 0
    for iEl in range(nEl):
        nPosIEl = len(az_l[iEl])
        el_grid[iPos:iPos+nPosIEl] = el_l[iEl]*np.ones((nPosIEl,))
        az_grid[iPos:iPos+nPosIEl] = az_l[iEl]
        iPos += nPosIEl
    
    if fig:
        x = np.cos(deg2rad(el_grid)) * np.cos(deg2rad(az_grid))
        y = np.cos(deg2rad(el_grid)) * np.sin(deg2rad(az_grid))
        z = np.sin(deg2rad(el_grid))
        
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        fontsize = 20
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')       
        
        ax.scatter(x, y, z, s=8)
        ax.set_xticks([], []); ax.set_yticks([], []); ax.set_zticks([], []);
        ax.set_xlabel('x', fontsize=fontsize); ax.set_ylabel('y', fontsize=fontsize); ax.set_zlabel('z', fontsize=fontsize); 
        
    return el_grid, az_grid

#%%        
def neighboursOnGrid(el_grid, az_grid):
    """
    The angular distance between any pair of points of the grid.
    
    Parameters
    ----------
    el_grid: 1d-array
    az_grid: 1d_array
        Length: nPos
    
    Returns
    -------
    neighbour_grid: nd-array
        Shape: (nPos, nPos)
    """
    nPos = len(el_grid)
    neighbour_grid = np.zeros((nPos,nPos))

    # Speedup batch version
    for iPos in range(nPos):
        neighbour_grid[iPos, :] = angularDistance(el_grid[iPos], az_grid[iPos], el_grid, az_grid)
    return neighbour_grid

def peaksOnGrid(array, el_grid, az_grid, neighbour_tol):
    """
    Find the main peaks of an array corresponding to a spherical grid.
    
    Parameters
    ----------
    array: 1d-array
        Length: nPos
    
    el_grid: 1d-array
    az_grid: 1d_array
        Length: nPos
    
    neighbour_tol: int
        Angular tolerance to define the neighbourhood of a point.
    
    Returns
    -------
    peaks: list
        Values of the peaks.
    
    idxPeaks: list of int
        Corresponding indexes in array.
    """
    nPos = len(array)
    neighbour_grid = neighboursOnGrid(el_grid, az_grid)
    neighbour_bool = neighbour_grid<=neighbour_tol
    array_sm = smoothOnGrid(array, neighbour_grid, neighbour_tol)
    
    peaks = []
    idxPeaks = []
    for iPos in range(nPos):
        if abs(array_sm[iPos]) == np.max(abs(array_sm[neighbour_bool[iPos]])):
            peaks.append(array[iPos])
            idxPeaks.append(iPos)
            
    return peaks, idxPeaks
        
            
def smoothOnGrid(array, neighbour_grid, neighbour_tol):
    """
    Each entry of the input array is replaced by the weighted mean of its neighbours.
    Neighbours are defined by points closer than neighbour_tol in the neighbour_grid.
    Weights decrease when the point gets closer to the considered position.
    
    Parameters
    ----------
    array: 1d-array
        Length: nPos
    
    neighbour_grid: nd-array
        Angular distance in degrees between each pair of point on the grid.
        Shape: (nPos, nPos)
    
    neighbour_tol: int
        Angular tolerance to define the neighbourhood of a point.
    
    Returns
    -------
    array_sm: nd-array
        Smoothed version of array.
    
    """
    nPos = len(array)
    array_sm = np.zeros_like(array)
    for iPos in range(nPos):
        weights = (neighbour_tol-neighbour_grid[iPos].copy())/neighbour_tol
        weights = weights.clip(0)
        array_sm[iPos] = np.average(array, weights=weights)
    return array_sm
