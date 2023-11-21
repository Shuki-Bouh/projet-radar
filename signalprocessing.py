# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:44:35 2023

@author: amte7
"""
import numpy as np
import matplotlib.pyplot as plt
import parametre as prm

def range_calculation(cube):
    S = prm.S
    c = prm.c
    Fs = prm.fs
    """ prend en entré un cube correspondant  """
    try:    
        ff1 = np.fft.fft(cube, axis=0)
        freq = np.fft.fftfreq(cube.shape[0], d=1/Fs)
        rang = freq * c / (2 * S)
        ff2 = np.mean(ff1, axis=-1)  # Moyenne selon les antennes de réception
        ff3 = np.abs(np.mean(ff2, axis=-1))  # Moyenne selon les chirps et valeur absolue
        
        ff_norm = ff3 / np.max(ff3)
    except Exception as e:
        print(f"Erreur: {e}")
        return None

    plt.plot(rang, ff_norm)
    plt.title("Radar Range Profile")
    plt.xlabel("Range (meters)")
    plt.ylabel("Normalized Intensity")
    plt.show()

    return rang, ff_norm

        
        
    
    
    
    
    
    
    
    
    
    
    
    
    