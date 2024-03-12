import numpy as np
import matplotlib.pyplot as plt
from parametre import *
from scipy.signal import convolve2d
from collections import Counter



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

    def range_fft(self, cube: np.array) -> np.array:
        """Calcule la transformée de Fourier en distance du cube radar.

        :param np.array cube: array de forme (Ns, Nc, Nr*Nt) - le cube radar

        :return: array de forme (Ns, Nc, Nr*Nt) - le cube transformé de Fourier en distance
        """
        fft_r = np.fft.fft(cube, axis = 0) # FFT selon chaque colonne
        return fft_r

    def doppler_fft(self, fft_r: np.array) -> np.array:
        """Calcule la transformée de Fourier en vitesse (Doppler) du cube radar deja transformé de fourrie en distance.

        :param np.array fft_r : array de forme (Ns, Nc, Nr*Nt) - le cube transformé de Fourier en distance

        :return: array de forme (Ns, Nc, Nr*Nt) - le cube transformé de Fourier en vitesse (Doppler)
        """
        fft_d = np.fft.fft(fft_r, axis=1)
        fft_d = np.fft.fftshift(fft_d, axes=1)
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
        plt.figure()
        # En linéaire
        plt.subplot(211)
        plt.plot(rang, fft_abs_norm_mean)
        plt.ylabel("Spectre linéaire")
        plt.title("Spectre linéaire de la transformée de Fourier en distance dans le domaine des distances")

        # En dB
        plt.subplot(212)
        plt.plot(rang, 20 * np.log10(fft_abs_norm_mean))
        plt.ylabel("Spectre de puissance (dB)")
        plt.xlabel("Distance (m)")

        plt.show()
        return None


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

    def plot_doppler_spectrum(self, fft2D : np.array) -> None:
        """Affiche le spectre en vitesse (Doppler) de la transformée de Fourier.

        :param fft_d: array de forme (Ns, Nc, Nr*Nt) - le résultat de la transformée de Fourier en vitesse (Doppler)
        si tdm mode = ['TDMA', nombre de recepteur]
        si ddma, mode = ['DDMA', np.array des offsets (en radiant), de shape (nombre de Tx,)] pas encore implemente
        """

        fft2D_abs = np.abs(fft2D)
        fft2D_abs_norm = fft2D_abs / np.max(fft2D_abs, axis=(1, 0), keepdims=True)
        fft2D_abs_norm_mean = np.mean(fft2D_abs_norm, axis=2)  # Moyenne selon les antennes de réception
        # fft2D_abs_norm_mean_shift = np.fft.fftshift(fft2D_abs_norm_mean, axes=1)

        speed = np.arange(-fft2D.shape[1] // 2, fft2D.shape[1] // 2) * λ / (
                    2 * Tc * fft2D.shape[1])  # le tdma divise la vitesse max par le nombre d antenne Tx
        rang = np.arange(fft2D.shape[0]) * c / (2 * S * Tc)

        plt.figure()

        plt.imshow(fft2D_abs_norm_mean, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')

        if self.mode == "DDMA":
            plt.title("2d fft range velocity " + "DDMA " + str(len(self.offset_phase)) + " antennes TX")
        elif self.mode == "TDMA":
            plt.title("2d fft range velocity " + "TDMA " + str(Nt) + " antennes TX")
        else:
            plt.title("2d fft range velocity SIMO")
        plt.xlabel("Vitesse (m/s)")
        plt.ylabel("range (m)")
        plt.colorbar()

        return None


    def cfar2D(self,pfa : float,nb_guard : int ,nb_reference : int,fft2D : np.array, mean=True) -> np.array:
        """

        :param pfa:
        :param nb_guard:
        :param nb_reference:
        :param fft2D:
        :return:
        """

        mask = np.ones(((nb_guard + nb_reference) * 2 + 1, (nb_guard + nb_reference) * 2 + 1))
        mask[nb_reference:-nb_reference, nb_reference:-nb_reference] = 0
        N = np.sum(mask)
        mask = mask / N
        alpha = N * (pfa ** (-1 / N) - 1)

        fft2D_abs = np.abs(fft2D)
        fft2D_abs_norm = fft2D_abs / np.max(fft2D_abs, axis=(1, 0), keepdims=True)
        if mean :
            fft2D_abs_norm_mean = np.mean(fft2D_abs_norm, axis=2)  # Moyenne selon les antennes de réception
            mu = convolve2d(fft2D_abs_norm_mean, mask, mode='same', boundary='wrap')
        else :
            fft2D_abs_norm_mean = fft2D_abs_norm
            mu = np.zeros_like(fft2D_abs_norm_mean)
            for i in range(fft2D.shape[2]):
                mu[:, :, i] = convolve2d(fft2D_abs_norm_mean[:, :, i], mask, mode='same', boundary='wrap')

        T = mu * alpha
        result = T < fft2D_abs_norm_mean

        # pfa = [1e-10,0.00001,0.0001,0.001,0.01,0.1,0.5,0.9]
        # plt.figure()
        # for i in range(8):
        #     result, conv, mask = sgp.CFAR2D(pfa[i], 5, 9, fftd_abs_norm_mean)
        #     plt.subplot(2,4,i+1)
        #     plt.imshow(result, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')
        #
        # nb_guard = [1,3,5,7,9,11,13,15]
        # plt.figure()
        # for i in range(8):
        #     result, conv, mask = sgp.CFAR2D(1e-10, nb_guard[i], 1, fftd_abs_norm_mean)
        #     plt.subplot(2,4,i+1)
        #     plt.imshow(result, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')
        #
        # nb_ref = [1,3,5,7,9,11,13,15]
        # plt.figure()
        # for i in range(8):
        #     result, conv, mask = sgp.CFAR2D(1e-10, 1, nb_ref[i], fftd_abs_norm_mean)
        #     plt.subplot(2,4,i+1)
        #     plt.imshow(result, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')

        return result

    def plot_cfar2D(self,cfar : np.array) -> None:
        speed = np.arange(-cfar.shape[1] // 2, cfar.shape[1] // 2) * λ / (2 * Tc * cfar.shape[1])  # le tdma divise la vitesse max par le nombre d antenne Tx
        rang = np.arange(cfar.shape[0]) * c / (2 * S * Tc)

        plt.figure()
        plt.imshow(cfar, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')
        plt.title("2DCFAR")
        plt.xlabel("speed (m/s)")
        plt.ylabel("range (m)")
        plt.colorbar()

        return None

    def music(self, fft2D : np.array, cfar : np.array) -> list :

        targets = []
        M = fft2D.shape[2]  # nb de capteurs
        N = 180 * 4 + 1  # nb de points = précision
        d_norm = 1 / 8  # D/lambda

        thetas = np.linspace(-np.pi / 2, np.pi / 2, N)
        pos_sensors = np.arange(M)[:, None]

        # L= frame.shape[1]
        L = 1  # nb de snaps

        A = np.exp(-2 * 1j * np.pi * d_norm * pos_sensors * np.sin(thetas))

        indices = np.where(cfar == 1)
        # Afficher les coordonnées
        coordonnees = list(zip(indices[0], indices[1]))
        print(coordonnees)
        for i in range(len(coordonnees)):
            (x, y) = coordonnees[i]
            # fft2D = np.fft.fftshift(fft2D, axes=1)
            y_snap = fft2D[x, y, :].reshape((4, 1))

            Syy = y_snap @ y_snap.T.conj() / L

            U, lamda, _ = np.linalg.svd(Syy)
            lamdalog = 10 * np.log10(lamda)
            # T = np.min(lamdalog) + np.abs(np.max(lamdalog) - np.min(lamdalog)) / 2
            # Sc = sum(1 for valeur in lamdalog if valeur > T)
            Sc = 1


            P_music = 1 / np.diag((A.T.conj() @ U[:, Sc - M:] @ U[:, Sc - M:].T.conj() @ A)).real
            P_music = P_music / max(P_music)

            targets.append((x,y,thetas[np.argmax(P_music)]))

            # plt.figure()
            # plt.subplot(211)
            # plt.plot(thetas, P_music ** 0.5, label="MUSIC")
            # plt.grid()
            # plt.legend()

            # plt.subplot(212)
            # plt.plot(lamdalog, "xk", label="valeurs propres")
            # plt.axhline(y=T, color='red', linestyle='--', label='Threshold')
            # plt.plot(T)

        return np.array(targets)


    def DDMA_antenna_2_channels(self,fft2D : np.array, offset_phase : np.array) -> np.array :
        res_chanel = np.zeros((fft2D.shape[0], fft2D.shape[1], offset_phase.shape[0] * fft2D.shape[2]), dtype=complex)
        deph = np.arange(0, fft2D.shape[1]) * np.pi * 2 / fft2D.shape[1]
        shilft_array = - np.argmin(np.abs(deph - offset_phase[:, None]), axis = 1)
        print(shilft_array)

        for k in range(shilft_array.shape[0]):
            res_chanel[:, :, k * fft2D.shape[2]:(k+1) * fft2D.shape[2]] = np.roll(fft2D, shilft_array[k], axis=1)
        return res_chanel

    def MIMO_artefact_processing(self, frame : np.array, cfar3D) :
        indices = np.where(cfar3D == 1)
        targets = np.array(list(zip(indices[0], indices[1])))
        comptage = Counter(targets)
        targets_filtres = [tuple_element for tuple_element, count in comptage.items() if count == frame.shape[2]]
        return targets_filtres

    def hanning_windowing(self, frame : np.array) -> np.array:
        return np.apply_along_axis(np.hanning, axis=0, arr=(np.apply_along_axis(np.hanning, axis=1, arr=frame)))

    def plot_music(self, targets):
        rang = np.arange(Ns) * c / (2 * S * Tc)
        angles = targets[:,2]
        # angles= [valeur for valeur in angles if np.abs(valeur) < 1.57]
        distances = targets[:,0] * c / (2 * S * Tc)
        print(angles)
        print(distances)


        # Créer un graphique polaire avec un quart seulement
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(angles, distances, marker='x', linestyle='')

        # Limiter l'affichage à un quart
        ax.set_thetamin(-45)
        ax.set_thetamax(45)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Ajouter des annotations
        # for angle, distance in zip(angles, distances):
        #     ax.text(angle, distance, f'({np.degrees(angle):.0f}°, {round(distance,2)})', ha='right')

        # Ajuster les paramètres pour un meilleur affichage
        plt.title("Graphique Polaire - Un Quart du Graphique")
        plt.grid(True)
        return None

    def on_key(event):
        if event.key == 'q':
            plt.ioff()
            plt.close()

    def active_fft2D_plot(self,data : np.array):
        plt.ion()
        fig, ax = plt.subplots()

        plt.connect('key_press_event', self.on_key)

        for i in range(data.shape[3]):
            frame = data[:, :, :, i]
            fftr = self.range_fft(frame)
            fft2D = self.doppler_fft(fftr)

            fft2D_abs = np.abs(fft2D)
            fft2D_abs_norm = fft2D_abs / np.max(fft2D_abs, axis=(1, 0), keepdims=True)
            fft2D_abs_norm_mean = np.mean(fft2D_abs_norm, axis=2)  # Moyenne selon les antennes de réception
            fft2D_abs_norm_mean_shift = np.fft.fftshift(fft2D_abs_norm_mean, axes=1)
            speed = np.arange(-fft2D.shape[1] // 2, fft2D.shape[1] // 2) * λ / (2 * Tc * fft2D.shape[1])  # le tdma divise la vitesse max par le nombre d antenne Tx
            rang = np.arange(fft2D.shape[0]) * c / (2 * S * Tc)

            ax.clear()

            ax.imshow(fft2D_abs_norm_mean_shift, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)],
                       origin='lower')

            plt.draw()
            plt.pause(0.1)
        plt.ioff()
        plt.close()
        return

    def active_cfar_plot(self,data : np.array,pfa : float,nb_guard : int, nb_ref : int) -> None:
        plt.ion()
        fig, ax = plt.subplots()

        plt.connect('key_press_event', self.on_key)

        for i in range(data.shape[3]):
            frame = data[:, :, :, i]
            fftr = self.range_fft(frame)
            fft2D = self.doppler_fft(fftr)

            speed = np.arange(-fft2D.shape[1] // 2, fft2D.shape[1] // 2) * λ / (2 * Tc * fft2D.shape[1])  # le tdma divise la vitesse max par le nombre d antenne Tx
            rang = np.arange(fft2D.shape[0]) * c / (2 * S * Tc)

            result = self.cfar2D(pfa, nb_guard, nb_ref, fft2D)

            ax.clear()

            plt.imshow(result, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')
            plt.title("2DCFAR, pfa : {}, nb_guard : {}, nb_ref : {}".format(pfa, nb_guard, nb_ref))
            plt.xlabel("speed")
            plt.ylabel("range")

            plt.draw()
            plt.pause(0.1)
        plt.ioff()
        plt.close()
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
    
def CFAR2D(pfa, nb_guard, nb_reference, ff2d_im):

    mask = np.ones(((nb_guard+nb_reference)*2+1,(nb_guard+nb_reference)*2+1))
    mask[nb_reference:-nb_reference, nb_reference:-nb_reference] = 0

    N = np.sum(mask)
    mask = mask / N
    alpha = N*(pfa**(-1/N) - 1)

    mu = convolve2d(ff2d_im, mask, mode='same', boundary='wrap')
    T = mu*alpha
    result = T < ff2d_im
    return result, mu, mask


def Music(x,y,frame):
    # y_snap = frame[2,:,:].T.conj()
    print(frame[x,y,:].shape)
    y_snap = frame[x,y,:].T.conj().reshape((4,1))
    M = frame.shape[2] #nb de capteurs
    N = 180 * 4 + 1 #nb de points = précision
    d_norm = 1 / 8 #D/lambda

    thetas = np.linspace(-np.pi / 2, np.pi / 2, N)
    pos_sensors = np.arange(M)[:, None]

    # L= frame.shape[1]
    L = 1 #nb de snaps
    Syy = y_snap @ y_snap.T.conj() / L
    print(np.shape(Syy))

    A = np.exp(-2 * 1j * np.pi * d_norm * pos_sensors * np.sin(thetas))
    print(np.shape(A))

    U, lamda, _ = np.linalg.svd(Syy)
    lamdalog = 10 * np.log10(lamda)
    T = np.min(lamdalog) + np.abs(np.max(lamdalog) - np.min(lamdalog))/2
    Sc = sum(1 for valeur in lamdalog if valeur > T)
    print(Sc)
    P_music = 1 / np.diag((A.T.conj() @ U[:, Sc - M:] @ U[:, Sc - M:].T.conj() @ A)).real
    P_music = P_music / max(P_music)
    plt.figure()
    plt.subplot(211)
    plt.plot(thetas, P_music ** 0.5, label="MUSIC")
    plt.grid()
    plt.legend()
    plt.subplot(212)
    plt.plot(lamdalog, "xk", label="valeurs propres")
    plt.axhline(y=T, color='red', linestyle='--', label='Threshold')
    plt.plot(T)
    # plt.legend()

    return Syy

    
    
    
    
    
    
    
