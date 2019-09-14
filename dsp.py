# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:20:34 2017

@author: WSJF7149
"""

import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft


#%% Root mean square
def rms(s):
    '''
    Computes the RMS value of a signal
    
    Parameters
    ----------
    s: nd array
    The temporal signal or its spectrogram
    
    Returns
    -------
    rms: float
    
    '''
    rms = np.sqrt(np.mean(abs(s)**2))
    return rms

#%% Buffering
def frame(s, lWindow=1024, window='sin', nOverlap=None):
    '''
    Cut the input signal into windowed frames.
    Remember to use windows that satisfy the COLA constraint
    
    Parameters
    ----------
    s: array_like, 1-D
        input signal to be framed
    lWindow: int, optional
        length of the frames; default is 1024.
        If window is specified as an array, lWindow needs to be its length
    window: string or 1-D array_like, optional
        if string, either 'sin' or a window handled by scipy.signal.get_window()
        Defaults to sinusoidal window
    nOverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``noverlap = lWindow // 2``. Defaults to `None`.
        
    Returns
    -------
    sFr: ndarray
        shape: (nFrame, lWindow)

    '''
    
    # Check inputs
    s = np.asarray(s)
    if s.ndim != 1:
        raise ValueError('input signal must be 1D')
    if window is 'sin':
        window = np.sin(np.arange(0.5, lWindow+0.5)/lWindow*np.pi)
    elif isinstance(window, str):
        window = signal.get_window(window, lWindow)
    else:
        window = np.asarray(window)
        if window.ndim != 1:
            raise ValueError('window must be 1-D')
        if len(s) < window.shape[-1]:
            raise ValueError('window is longer than input signal')
        if lWindow != window.shape[0]:
            raise ValueError('value specified for lWindow is different from length of window')
    if nOverlap is None:
        nOverlap = lWindow//2
    else:
        nOverlap = int(nOverlap)
        if nOverlap >= lWindow:
            raise ValueError('nOverlap must be less than lWindow.')
       
    
    # Zero-padding the last frame
    hop     = lWindow - nOverlap
    nFrame  = len(s)//hop
    sPad    = np.concatenate((s, np.zeros(nFrame*hop+nOverlap-len(s),)))
    
    # Filling sFr
    sFr     = np.array([ sPad[iFrame*hop:iFrame*hop+lWindow]*window  
                        for iFrame in range(nFrame)])
    return sFr


def ola(sFr, window='sin', nOverlap=None):
    '''
    Reconstruct a framed signal with the overlapp-add method.
    Remember to use windows that satisfy the COLA constraint
    
    Parameters
    ----------
    sFr : array_like
        input signal to be framed.
        Shape : (nFrame, lWindow)
    window : string or 1-D array_like, optional
        if string, either 'sin' or a window handled by scipy.signal.get_window()
        Defaults to sinusoidal window
    nOverlap : int, optional
        Number of points to overlap between segments.
        If `None`, ``nOverlap = lWindow // 2``. Defaults to `None`.
        
    Returns
    -------
    s : 1-D array
        Reconstructed signal
    '''
    
    sFr = np.asarray(sFr)
    # Check inputs
    nFrame, lWindow = sFr.shape
    if window is 'sin':
        window = np.sin(np.arange(0.5, lWindow+0.5)/lWindow*np.pi)
    elif isinstance(window, str):
        window = signal.get_window(window, lWindow)
    else:
        window = np.asarray(window)
        if window.ndim != 1:
            raise ValueError('window must be 1-D')
        if lWindow != window.shape[0]:
            raise ValueError("value specified for lWindow is different from length of window")
    if nOverlap is None:
        nOverlap = lWindow//2
    else:
        nOverlap = int(nOverlap)
        if nOverlap >= lWindow:
            raise ValueError('nOverlap must be less than lWindow.')
    
    # overlap - add
    hop     = lWindow - nOverlap
    nSmp    = (nFrame+1)*hop
    s       = np.zeros((nSmp,), dtype=np.float32)
    for iFrame in range(nFrame):
        s[iFrame*hop:iFrame*hop+lWindow] += sFr[iFrame,:]*window
    
    return s

#%% STFT
def stft(s, lWindow=1024, window='sin', nOverlap=None, nfft=None, isReal=True):
    '''
    Computes the short-time Fourier transform of the input signal.
    Remember to use windows that satisfy the COLA constraint
    
    Parameters
    ----------
    s: array_like, 1-D
        input signal to be framed
    lWindow: int, optional
        length of the frames; default is 1024.
        If window is specified as an array, lWindow needs to be its length
    window: string or 1-D array_like, optional
        if string, either 'sin' or a window handled by scipy.signal.get_window()
        Defaults to sinusoidal window
    nOverlap: int, optional
        Number of points to overlap between segments. 
        If `None`, ``nOverlap = lWindow // 2``. Defaults to `None`.
    nfft: int, optional
        Number of points on which to perform the fourier transform.
        If `None`, ``nfft = lWindow``. Defaults to `None`.
    isReal: bool, optional
        Specify whether the signal is real. In this case, only nfft/2+1 frequency
        points will be returned. 
        Defaults to ``True``.
        
    Returns
    -------
    sf: ndarray
        shape:  (nFrame, nfft//2+1)  if ``isReal=True``
                (nFrame, nfft)      if ``isReal=False``
                

    '''
    
    # Check inputs 
    if nfft is None:
        nfft = lWindow
        
    # Short-Time Fourier Transform
    sFr = frame(s, lWindow, window, nOverlap)
    sf  = fft(sFr, n=nfft)
    if isReal:
        sf = sf[:,0:nfft//2+1]
    
    return sf


def istft(sf, lWindow=1024, window='sin', nOverlap=None, nfft=None, isSym=True):
    '''
    Computes the inverse short-time Fourier transform of the input signal
    with the overlapp-add reconstruction method.
    Remember to use windows that satisfy the COLA constraint
    
    Parameters
    ----------
    sf: array_like
        input signal to be framed. 
        Shape : (nFrame, nfft)
    lWindow: int, optional
        length of the frames; default is 1024.
        If window is specified as an array, lWindow needs to be its length
    window: string or 1-D array_like, optional
        if string, either 'sin' or a window handled by scipy.signal.get_window()
        Defaults to sinusoidal window
    nOverlap: int, optional
        Number of points to overlap between segments. If `None`,
        ``nOverlap = lWindow // 2``. Defaults to `None`.
    nfft: int, optional
        Number of points on which to perform the fourier transform.
        If `None`, ``nfft = lWindow``. Defaults to `None`.
    isSym: bool, optional
        Specify whether the spectrogram is only half of a hermitian spectrogram,
        corresponding to a real time-signal. In this case, it will be symetrized
        before its ifft is taken.
        Defaults to ``True``.
        
    Returns
    -------
    s: 1-D array
        Time-series corresponding to the iSTFT of input.
    '''
    # Check inputs
    if nfft is None:
        nfft = lWindow
    
    # Inverse Short-Time Fourier Transform
    if isSym:
        sfSym = np.concatenate((sf, sf[:,-2:0:-1].conjugate()), axis=1)
    else:
        sfSym = sf.copy()
    sFr     = ifft(sfSym, n=nfft).real
    s       = ola(sFr, window=window, nOverlap=nOverlap)
    return s


#%%
def rmsNormByFrame(sFr, axis=-1):
    '''
    Normalizes each vector (supposed to be a frame of signal) of the array 
    with respect to its RMS power.
    
    Inputs
    ------
    sFr : array-like, float
    The framed signal
    axis : int, optional
    The dimension along which to normalize. Defaults to -1.
    
    Returns
    -------
    sFrNorm : np-array, float
    The normalized version of the input.
    '''
    
    # Shape management
    sFrNorm = np.float32(sFr)
    sFrNorm = np.rollaxis(sFrNorm, axis, len(sFr.shape))        # putting the axis of interest last
    
    shapeS          = sFrNorm.shape
    sFrNorm         = sFrNorm.flatten()
    sFrNorm.shape   = (-1,shapeS[-1])           # reshaping sFrNorm so that it is 2D with the axis of interest last
    
    # Normalization
    for iFrame,frame in enumerate(sFrNorm):
        rms = np.sqrt(np.mean(frame**2))
        sFrNorm[iFrame,:] = frame/rms     
    
    # Shape management
    sFrNorm.shape = shapeS                      # reshaping sFrNorm to N dimensions
    if axis < 0:                                # Specify as positive axis index
        axis = len(sFrNorm.shape)-1-axis
    sFrNorm = np.rollaxis(sFrNorm, -1, axis)    # Roll axis of interest back to where it comes from
    
    return sFrNorm
