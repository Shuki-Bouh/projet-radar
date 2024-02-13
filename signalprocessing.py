# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:44:35 2023

@author: amte7
"""
import numpy as np
import matplotlib.pyplot as plt
import parametre as prm
import simu_v2 as simp
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import convolve2d
from matplotlib import cm

tg1 = simp.Target(10, np.deg2rad(0), 0, 0, 1)

tg2 = simp.Target(10, np.deg2rad(45), 0, 7, 1)

tg3 = simp.Target(30, np.deg2rad(-15), 0, 0.5, 0.5)

tg4 = simp.Target(35, np.deg2rad(-60), 0, 0.2, 0.5)

tg5 = simp.Target(40, np.deg2rad(-30), 0, -1, 0.9)

simu = simp.Simulateur()

simu.run([1, 2, 3, 4], [1, 2, 3], ["DDMA", np.deg2rad(np.array([0, 90, 270]))], [tg1, tg2])


#


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
    cube = cube.reshape((cube.shape[0], -1, cube.shape[2] * N_t))
    return cube


# cube1 = TDMA_antenna_2_channel(cube1,2)
# cube1=simu.result_channel
# f2 = np.fft.fft2(cube1,axes = (1,0))
# f2 = np.fft.fftshift(f2, axes =1)
# deph = np.rad2deg(np.arange(0,f2.shape[1])*np.pi*2/f2.shape[1])


A = 0


# cubetd = TDMA_antenna_2_channel(cube1,2)
# ff1 = calculate_range_fft(cubetd)
# plot_range_spectrum(ff1)
def DDMA_antenna2_channel(ff_dop, offset_phase):
    """Txi__1_Rxj_1....Txi_1__Rxj_n|Txi_2__Rxj_1....Txi_2__Rxj_n|..."""

    res_chanel = np.zeros((ff_dop.shape[0], ff_dop.shape[1], offset_phase.shape[0] * ff_dop.shape[2]), dtype=complex)
    deph = np.arange(0, ff_dop.shape[1]) * np.pi * 2 / ff_dop.shape[1]
    shilft_array = -np.argmin(np.abs(deph - offset_phase[:, None]), axis=1)

    for k in range(shilft_array.shape[0]):
        res_chanel[:, :, k * ff_dop.shape[2]:(k + 1) * ff_dop.shape[2]] = np.roll(ff_dop, shilft_array[k], axis=1)
    return res_chanel


def channel_Tx_2_channel_Rx(cube, N_t, N_r):
    """permet de passer de la respresetation Txi__1_Rxj_1....Txi_1__Rxj_n|Txi_2__Rxj_1....Txi_2__Rxj_n|...
      à Rxj_1__Txi_1....Rxj_1__Txi_m|Txi_2__Rxj_1....Txi_2__Rxj_n| dans le cube"""
    indices = np.arange(cube.shape[2]).reshape((N_t, N_r)).T.flatten()
    return cube[:, :, indices]





# cube_channel = TDMA_antenna_2_channel(simu.result_true_antenna,3)
ff_range = calculate_range_fft(simu.result_true_antenna)
# plot_range_spectrum(ff_range)
ff_dop = calculate_doppler_fft(ff_range)
A = plot_doppler_spectrum(ff_dop, ["DDMA", np.deg2rad(np.array([0, 90, 270]))])


def circular_2d_cfar(falsealmar, nb_guard, nb_reference, ff2d_im):
    # circulaire
    mask = np.zeros(((nb_guard + nb_reference) * 2 + 1, (nb_guard + nb_reference) * 2 + 1))
    x, y = np.meshgrid(np.arange(0, nb_guard + nb_reference + 1), np.arange(0, nb_guard + nb_reference + 1))
    V = (x) ** 2 + (y) ** 2 <= (((nb_guard + nb_reference) * 2 + 1) / 2) ** 2
    mask[nb_guard + nb_reference:, nb_guard + nb_reference:] = V
    mask[:nb_guard + nb_reference, nb_guard + nb_reference:] = V[1:, :][::-1, :]
    mask[:, :nb_guard + nb_reference] = mask[:, nb_guard + nb_reference + 1:][:, ::-1]

    mask[nb_reference:-nb_reference, nb_reference:-nb_reference] = 0

    # carre
    #    mask = np.ones(((nb_guard+nb_reference)*2+1,(nb_guard+nb_reference)*2+1))
    #    mask[nb_reference:-nb_reference,nb_reference:-nb_reference] = 0

    N = np.sum(mask)  # number of reference cells
    alpha = N * (falsealmar ** (-1 / N) - 1)
    mask = mask / N
    conv = convolve2d(ff2d_im, mask, mode='same', boundary='wrap') * alpha
    result = conv < ff2d_im
    return result, conv


detec, conv = circular_2d_cfar(10 ** -4, 1, 2, A)
speed = np.arange(-conv.shape[1] // 2, conv.shape[1] // 2) * prm.λ / (2 * prm.Tc * conv.shape[1])
rang = np.arange(conv.shape[0]) * 3e8 / (2 * prm.S * prm.Tc)
x = speed
y = rang
x, y = np.meshgrid(x, y)
# Créer une figure en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Afficher la surface
ax.plot_surface(x, y, conv, cmap='viridis')

# Ajouter des étiquettes aux axes
ax.set_xlabel('vitesse (m/s)')
ax.set_ylabel('range (m)')
#    ax.set_zlabel('Axe Z')

# Afficher la figure
plt.show()


class Traitement_fft():

    def __init__(self, mode):
        self.mode = mode
        self.nbTx = 3
        return

    @property
    def fft_range(self) -> np.array:
        return self.__fft_range

    @fft_range.setter
    def fft_range(self, fft) -> None:
        self.__fft_range = fft

    @property
    def fft_doppler(self) -> np.array:
        return self.__fft_range

    @fft_doppler.setter
    def fft_doppler(self, fft) -> None:
        self.__fft_range = fft

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, new_mode):
        self.__mode = new_mode
        if new_mode == 'TDMA':
            self.nbTx = 13445


    def calculate_range_fft(self, cube: np.array) -> None:
        """Calcule la transformée de Fourier en distance du cube radar.

        :param cube : np array size (Ns, Nc, Nr*Nt) - le cube radar

        :return np array size (Ns, Nc, Nr*Nt) - le cube transformé de Fourier en distance
        """
        try:
            fft_range = np.fft.fft(cube, axis=0)  # FFT selon chaque colonne
            self.fft_range = fft_range
        except Exception as e:
            print(f"Erreur: {e}")

    def __str__(self):
        """Affiche le spectre linéaire et en dB de la transformée de Fourier en distance.

        :param fft_range : np array de forme (Ns, Nc, Nr*Nt) - le résultat de la transformée de Fourier en distance
        """

        S = prm.S
        c = prm.c
        Tc = prm.Tc

        fft_abs = np.abs(self.fft_range)

        fft_abs_norm = fft_abs / np.max(fft_abs, axis=0)
        fft_abs_norm_mean = np.mean(fft_abs_norm, axis=(1, 2))  # Moyenne selon les antennes de réception et les chirps

        rang = np.arange(np.shape(self.fft_range)[0]) * c / (2 * S * Tc)

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

    def calculate_doppler_fft(self):
        """Calcule la transformée de Fourier en vitesse (Doppler) du cube radar deja transformé de fourrie en distance.

        Entrée:
        - fft1 : np array de forme (Ns, Nc, Nr*Nt) - le cube transformé de Fourier en distance

        Sortie:
        - np array de forme (Ns, Nc, Nr*Nt) - le cube transformé de Fourier en vitesse (Doppler)
        """
        try:
            self.fft_doppler = np.fft.fft(self.fft_range, axis=1)
        except Exception as e:
            print(f"Erreur: {e}")

    def plot_doppler_spectrum(self,  mode=[None, None]):
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
                elif mode[0] == 'DDMA':

                    fft2 = DDMA_antenna2_channel(self.fft_doppler, mode[1])
                    Nt = len(mode[1])
                    Nr = fft2.shape[2] // Nt

                fft_abs = np.abs(fft2)
                if mode[
                    0] == 'DDMA':  # conservation du min selon les channels de la meme antenne Tx, pour enlever les pics fantomes
                    fft_abs = fft_abs.reshape((fft_abs.shape[0], fft_abs.shape[1], -1, Nr))
                    fft_abs = np.min(fft_abs, axis=2).reshape((fft_abs.shape[0], fft_abs.shape[1], Nr))

                fft_abs_norm = fft_abs / np.max(fft_abs, axis=(1, 0), keepdims=True)
                fft_abs_norm_mean = np.mean(fft_abs_norm, axis=2)  # Moyenne selon les antennes de réception
                fft_abs_norm_mean = np.fft.fftshift(fft_abs_norm_mean, axes=1)
                speed = np.arange(-fft2.shape[1] // 2, fft2.shape[1] // 2) * lamb / (
                        2 * Tc * fft2.shape[1]) / nb_tdma_tx  # le tdma divise la vitesse max par le nombre d antenne Tx
                rang = np.arange(fft2.shape[0]) * c / (2 * S * Tc)

                plt.figure(figsize=(8, 6))

                plt.imshow(fft_abs_norm_mean, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)],
                           origin='bottom')
                #    plt.colorbar(label="Power Spectrum (dB)")
                if mode[0] == "DDMA":
                    plt.title("2d fft range velocity" + " " + mode[0] + " " + str(len(mode[1])) + " " + "antennes TX")
                elif mode[0] == "TDMA":
                    plt.title("2d fft range velocity" + " " + mode[0] + " " + str(mode[1]) + " " + "antennes TX")
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
                plt.show()
                return data


if __name__ == '__main__':
    pass