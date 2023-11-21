# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:44:35 2023

@author: amte7
"""
import numpy as np
import matplotlib.pyplot as plt

def range_calculation(cube):#,param_radar):
#    B = param_radar.bandwith
#    c = param_radar.celerity
#    Fs = param_radar.sampling_rate
    """ prend en entré un cube corespondant  """
    B = 4
    c = 3
    Fs = 100
    try:    
        ff1 = np.fft.fft(cube, axis=1)
        freq = np.fft.fftfreq(cube.shape[0], d=1/Fs)
        rang = freq * c / (2 * B)
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

        
        
    
    
    
    
    
    
    
    
    
    
    
    
    