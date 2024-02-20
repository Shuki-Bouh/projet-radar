import numpy as np
import matplotlib.pyplot as plt
from parametre import *
from scipy.signal import convolve2d



class SignalProcessing:
    def __init__(self, mode="SIMO", *args):
        self.mode = mode
        if mode == 'DDMA':
            self.offset_phase = args
        return

    # def TDMA_antenna_2_channel(self, cube: np.array, nt: int) -> np.array:
    #     """
    #     Transforme le cube radar en ajoutant une dimension pour les canaux d'antennes.
    #
    #     :param np.array cube: np array de forme (Ns, nt*Nc, Nr) - le cube radar obtenu avec TDMA
    #             (Nc : nombre de chirp émis par chaque antenne,
    #              Ns : nombre d'échantillons)
    #     :param int nt: nombre d'antennes d'émission actives
    #
    #     :return: np array de forme (Ns, Nc, Nr*Nt) - le cube radar avec les canaux dans la dernière dimension
    #     """
    #     cube = cube.reshape((cube.shape[0], -1, cube.shape[2] * nt))
    #     return cube

    def calculate_range_fft(self, cube: np.array) -> np.array:
        """Calcule la transformée de Fourier en distance du cube radar.

        :param np.array cube: array de forme (Ns, Nc, Nr*Nt) - le cube radar

        :return: array de forme (Ns, Nc, Nr*Nt) - le cube transformé de Fourier en distance
        """
        fft_r = np.fft.fft(cube, axis = 0) # FFT selon chaque colonne
        return fft_r

    def calculate_doppler_fft(self, fft_r: np.array) -> np.array:
        """Calcule la transformée de Fourier en vitesse (Doppler) du cube radar deja transformé de fourrie en distance.

        :param np.array fft_r : array de forme (Ns, Nc, Nr*Nt) - le cube transformé de Fourier en distance

        :return: array de forme (Ns, Nc, Nr*Nt) - le cube transformé de Fourier en vitesse (Doppler)
        """
        fft_d = np.fft.fft(fft_r, axis=1)
        return fft_d

    def plot_range_spectrum(self, fft_r: np.array) -> None:
        """Affiche le spectre linéaire et en dB de la transformée de Fourier en distance.

        :param np.array fft_r : np array de forme (Ns, Nc, Nr*Nt) - le résultat de la transformée de Fourier en distance
        """

        fft_abs = np.abs(fft_r)
        fft_abs_norm = fft_abs / np.max(fft_abs, axis=0)
        fft_abs_norm_mean = np.mean(fft_abs_norm, axis=(1, 2))  # Moyenne selon les antennes de réception et les chirps
        rang = np.arange(fft_r.shape[0]) * c / (2 * S * Tc)

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



    # def DDMA_antenna2_channel(self, fft_d: np.array, offset_phase: np.array):
    #
    #     """Txi__1_Rxj_1....Txi_1__Rxj_n|Txi_2__Rxj_1....Txi_2__Rxj_n|..."""
    #
    #     res_chanel = np.zeros((fft_d.shape[0], fft_d.shape[1], offset_phase.shape[0] * fft_d.shape[2]), dtype=complex)
    #     deph = np.arange(0, fft_d.shape[1]) * np.pi * 2 / fft_d.shape[1]
    #     shilft_array = - np.argmin(np.abs(deph - offset_phase[:, None]), axis = 1)
    #
    #     for k in range(shilft_array.shape[0]):
    #         res_chanel[:, :, k * fft_d.shape[2]:(k+1) * fft_d.shape[2]] = np.roll(fft_d, shilft_array[k], axis=1)
    #     return res_chanel

    # def channel_Tx_2_channel_Rx(self, cube, N_t, N_r):
    #     """permet de passer de la respresetation Txi__1_Rxj_1....Txi_1__Rxj_n|Txi_2__Rxj_1....Txi_2__Rxj_n|...
    #       à Rxj_1__Txi_1....Rxj_1__Txi_m|Txi_2__Rxj_1....Txi_2__Rxj_n| dans le cube"""
    #     indices = np.arange(cube.shape[2]).reshape((N_t, N_r)).T.flatten()
    #     return cube[:,:,indices]

    def plot_doppler_spectrum(self, fft_d):
        """Affiche le spectre en vitesse (Doppler) de la transformée de Fourier.

        :param fft_d: array de forme (Ns, Nc, Nr*Nt) - le résultat de la transformée de Fourier en vitesse (Doppler)
        si tdm mode = ['TDMA', nombre de recepteur]
        si ddma, mode = ['DDMA', np.array des offsets (en radiant), de shape (nombre de Tx,)] pas encore implemente
        """

        nb_tdma_tx = 1
        if self.mode == 'DDMA':
            fft_d = self.DDMA_antenna2_channel(fft_d, self.offset_phase)

        fft_abs = np.abs(fft_d)
        if self.mode == 'DDMA': #conservation du min selon les channels de la meme antenne Tx, pour enlever les pics fantomes
            fft_abs = fft_abs.reshape((fft_abs.shape[0], fft_abs.shape[1],-1, Nr))
            fft_abs = np.min(fft_abs, axis=2).reshape((fft_abs.shape[0], fft_abs.shape[1], Nr))

        fft_abs_norm = fft_abs / np.max(fft_abs, axis=(1, 0), keepdims=True)
        fft_abs_norm_mean = np.mean(fft_abs_norm, axis=2) # Moyenne selon les antennes de réception
        fft_abs_norm_mean = np.fft.fftshift(fft_abs_norm_mean, axes=1)
        speed = np.arange(-fft_d.shape[1] // 2, fft_d.shape[1] // 2) * λ / (2 * Tc * fft_d.shape[1]) / nb_tdma_tx  #le tdma divise la vitesse max par le nombre d antenne Tx
        rang = np.arange(fft_d.shape[0]) * c / (2 * S * Tc)

        plt.figure(figsize=(8, 6))

        plt.imshow(fft_abs_norm_mean, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')
    #    plt.colorbar(label="Power Spectrum (dB)")
        if self.mode == "DDMA":
            plt.title("2d fft range velocity " + "DDMA " + str(len(self.offset_phase)) + " antennes TX")
        elif self.mode == "TDMA":
            plt.title("2d fft range velocity " + "TDMA " + str(Nt) + " antennes TX")
        else:
            plt.title("2d fft range velocity SIMO ou SISO")
        plt.xlabel("Vitesse (m/s)")
        plt.ylabel("range (m)")

        # Créer un tableau 2D de données
        data = fft_abs_norm_mean + np.random.rand(fft_abs_norm_mean.shape[0], fft_abs_norm_mean.shape[1]) / 6

        # Créer une grille pour les coordonnées x et y
        x = speed
        y = rang
        x, y = np.meshgrid(x, y)

        # Créer une figure en 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Afficher la surface
        ax.plot_surface(x, y, data, cmap='viridis')

        # Ajouter des étiquettes aux axes
        ax.set_xlabel('vitesse (m/s)')
        ax.set_ylabel('range (m)')
    #    ax.set_zlabel('Axe Z')

        # Afficher la figure
        plt.show()
        return data


    def cfar2D(self,pfa : float,nb_guard : int ,nb_reference : int,frame : np.array) -> np.array:
        return

    def music(self,x : int,y : int ,frame : np.array):
        return

    def DDMA_antenna_2_channels(self,frame : np.array) -> np.array :
        return

    def MIMO_artefact_processing(self,x : int, y : int, frame : np.array):
        return

    def active_plot(self,data : np.array):
        return

    def active_cfar_plot(self,data : np.array):
        return
if __name__ == '__main__':
    # tg1 = simp.Target(10,np.deg2rad(0),0,0,1)
    #
    # tg2 = simp.Target(10,np.deg2rad(45),0,7,1)
    #
    # tg3 = simp.Target(30,np.deg2rad(-15),0,0.5,0.5)
    #
    # tg4 = simp.Target(35,np.deg2rad(-60),0,0.2,0.5)
    #
    # tg5 = simp.Target(40,np.deg2rad(-30),0,-1,0.9)
    #
    # simu = simp.Simulateur()
    #
    # simu.run([1,2,3,4],[1,2,3],["DDMA",np.deg2rad(np.array([0,90,270]))],[tg1,tg2])
    pass
    
    
    
    
    
    
    
    
    
