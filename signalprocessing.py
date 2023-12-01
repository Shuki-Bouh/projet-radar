# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:44:35 2023

@author: amte7
"""
import numpy as np
import matplotlib.pyplot as plt
import parametre as prm


def TDMA_antenna_2_channel(cube, N_t):
    """
    Transforme le cube radar en ajoutant une dimension pour les canaux d'antennes.

    Entrées:
    - N_t : nombre d'antennes d'émission actives
    - cube : np array de forme (Ns, N_t*Nc, Nr) - le cube radar obtenu avec TDMA
            (Nc : nombre de chirp émis par chaque antenne,
             Ns : nombre d'échantillons)
    Sortie:
    - np array de forme (Ns, Nc, Nr*Nt) - le cube radar avec les canaux dans la dernière dimension
    """
    cube = cube.reshape((cube.shape[0], -1, cube.shape[2]*N_t))
    return cube
    
    
def calculate_range_fft(cube):
    """Calcule la transformée de Fourier en distance du cube radar.

    Entrée:
    - cube : np array de forme (Ns, Nc, Nr*Nt) - le cube radar

    Sortie:
    - np array de forme (Ns, Nc, Nr*Nt) - le cube transformé de Fourier en distance
    """
    try:
        fft1 = np.fft.fft(cube, axis=0)  # FFT selon chaque colonne
    except Exception as e:
        print(f"Erreur: {e}")
        return None
    
    return fft1

def plot_range_spectrum(fft1):
    """Affiche le spectre linéaire et en dB de la transformée de Fourier en distance.

    Entrée:
    - fft1 : np array de forme (Ns, Nc, Nr*Nt) - le résultat de la transformée de Fourier en distance
    """
    S = prm.S 
    c = prm.c
    Tc = prm.Tc
    
    fft_abs = np.abs(fft1)
    fft_abs_norm = fft_abs / np.max(fft_abs, axis=0)
    fft_abs_norm_mean = np.mean(fft_abs_norm, axis=(1, 2))  # Moyenne selon les antennes de réception et les chirps
    rang = np.arange(fft1.shape[0]) * c / (2 * S * Tc)

    # Affichage des spectres
    fig, axs = plt.subplots(2, 1, figsize=(12, 4))
    # En linéaire
    axs[0].plot(rang, fft_abs_norm_mean)
    axs[0].set_ylabel("Spectre linéaire")
    axs[0].set_title("Spectre linéaire de la transformée de Fourier en distance dans le domaine des distances")

    # En dB
    axs[1].plot(rang, 20 * np.log10(fft_abs_norm_mean))
    axs[1].set_ylabel("Spectre de puissance (dB)")
    axs[1].set_xlabel("Distance (m)")

    plt.show()
    
def calculate_doppler_fft(fft1):
    """Calcule la transformée de Fourier en vitesse (Doppler) du cube radar deja transformé de fourrie en distance.

    Entrée:
    - fft1 : np array de forme (Ns, Nc, Nr*Nt) - le cube transformé de Fourier en distance

    Sortie:
    - np array de forme (Ns, Nc, Nr*Nt) - le cube transformé de Fourier en vitesse (Doppler)
    """
    try:
        fft2 = np.fft.fft(fft1, axis=1)  # FFT selon chaque ligne
    except Exception as e:
        print(f"Erreur: {e}")
        return None
    
    return fft2

def plot_range_doppler(fft2, mode= [None,None]):
    """Affiche le spectre en vitesse (Doppler) de la transformée de Fourier.

    Entrée:
    - fft2 : np array de forme (Ns, Nc, Nr*Nt) - le résultat de la transformée de Fourier en vitesse (Doppler)
    si tdm mode = ['TDMA', nombre de recepteur]
    si ddma, mode = ['DDMA', np.array des offsets (en radiant), de shape (nombre de Tx,)] pas encore implemente
    """
    lamb = prm.λ 
    Tc = prm.Tc
    c = prm.c
    S = prm.S
    nb_tdma_tx = 1
    if mode[0] == 'TDMA':
        nb_tdma_tx = mode[1]
        
    fft_abs = np.abs(fft2)
    fft_abs_norm = fft_abs /np.max(fft_abs, axis=(1,0), keepdims=True)
    fft_abs_norm_mean = np.mean(fft_abs_norm, axis=2) # Moyenne selon les antennes de réception
    fft_abs_norm_mean = np.fft.fftshift(fft_abs_norm_mean, axes=1)
    speed = np.arange(-fft2.shape[1]//2,fft2.shape[1]//2 )* lamb/(2*Tc*fft2.shape[1])/nb_tdma_tx  #le tdma divise la vitesse max par le nombre d antenne Tx
    rang = np.arange(fft2.shape[0]) * c / (2 * S * Tc)
    
    
    plt.figure(figsize=(8, 6))
    plt.imshow(fft_abs_norm_mean,extent=[np.min(speed),np.max(speed),np.min(rang),np.max(rang)], origin = 'bottom')
    plt.title("2d fft range velocity")
    plt.xlabel("Vitesse (m/s)")
    plt.ylabel("range (m)")
    plt.show()
