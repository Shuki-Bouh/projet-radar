from parametre import *
from read_data import convertFile
import pickle as pk
from time import time
import signalprocessing as sgp
import numpy as np
from matplotlib import pyplot as plt
import simu_v2 as sml
import scipy

if __name__ == '__main__':
    # converter = convertFile(numADCSample=Ns, numRx=Nt*Nr, numChirps=Nc, numFrame=Nf)
    # data = converter.read('adc_data_200F_128C.bin', isReal=isReal)
    # data = converter.big_reshaper(data)
    # print(data[0,0,0,0])
    #
    # frame = data[:,:,:,100]
    # print(frame.shape)
    # print(data.shape)
    # # fft1 = sgp.calculate_range_fft(frame)
    # # sgp.plot_range_spectrum(fft1)
    #
    # plt.figure(1)
    # fftr = sgp.calculate_range_fft(frame)
    # fftr_abs = np.abs(fftr)
    # fftr_abs_norm = fftr_abs / np.max(fftr_abs, axis=0)
    # fftr_abs_norm_mean = np.mean(fftr_abs_norm, axis=(1, 2))
    # rang = np.arange(fftr.shape[0]) * c / (2 * S * Tc)
    # plt.plot(rang,fftr_abs_norm_mean)
    #
    # plt.figure(2)
    # fftd = sgp.calculate_doppler_fft(fftr)
    # fftd_abs = np.abs(fftd)
    # fftd_abs_norm = fftd_abs / np.max(fftd_abs, axis=(1, 0), keepdims=True)
    # fftd_abs_norm_mean = np.mean(fftd_abs_norm, axis=2)  # Moyenne selon les antennes de réception
    # # fftd_abs_norm_mean = fftd_abs_norm[:,:,0]
    # fftd_abs_norm_mean1 = np.fft.fftshift(fftd_abs_norm_mean, axes=1)
    # print(fftd_abs_norm_mean1.shape)
    # speed = np.arange(-fftd.shape[1] // 2, fftd.shape[1] // 2) * λ / (2 * Tc * fftd.shape[1]) # le tdma divise la vitesse max par le nombre d antenne Tx
    # rang = np.arange(fftd.shape[0]) * c / (2 * S * Tc)
    # plt.imshow(fftd_abs_norm_mean1, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')
    # plt.colorbar()

    tg1 = sml.Target(10, 0.5, 0, 0, 1)

    tg2 = sml.Target(10, np.deg2rad(45), 0, 7, 1)

    tg3 = sml.Target(30,-0.5, 0, 5, 0.5)

    tg4 = sml.Target(35, np.deg2rad(-60), 0, 0.2, 0.5)

    tg5 = sml.Target(40, np.deg2rad(-30), 0, -1, 0.9)

    simu = sml.Simulateur()

    data = simu.run([1, 2, 3, 4], [1], ["SIMO"], [tg1,tg3])

    print(data.shape)
    fft1 = sgp.calculate_range_fft(data)
    # sgp.plot_range_spectrum(fft1)
    plt.figure(3)
    fftr_abs = np.abs(fft1)
    fftr_abs_norm = fftr_abs / np.max(fftr_abs, axis=0)
    fftr_abs_norm_mean = np.mean(fftr_abs_norm, axis=(1, 2))
    rang = np.arange(fft1.shape[0]) * c / (2 * S * Tc)
    plt.plot(rang, fftr_abs_norm_mean)

    fft2 = sgp.calculate_doppler_fft(fft1)
    plt.figure(4)
    fftd_abs = np.abs(fft2)
    fftd_abs_norm = fftd_abs / np.max(fftd_abs, axis=(1, 0), keepdims=True)
    fftd_abs_norm_mean = np.mean(fftd_abs_norm, axis=2)  # Moyenne selon les antennes de réception
    fftd_abs_norm_mean = np.fft.fftshift(fftd_abs_norm_mean, axes=1)
    speed = np.arange(-fft2.shape[1] // 2, fft2.shape[1] // 2) * λ / (
                2 * Tc * fft2.shape[1])  # le tdma divise la vitesse max par le nombre d antenne Tx
    rang = np.arange(fft2.shape[0]) * c / (2 * S * Tc)
    plt.imshow(fftd_abs_norm_mean, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')


    pfa = [1e-10,0.00001,0.0001,0.001,0.01,0.1,0.5,0.9]
    plt.figure()
    for i in range(8):
        result, conv, mask = sgp.CFAR2D(pfa[i], 5, 9, fftd_abs_norm_mean)
        plt.subplot(2,4,i+1)
        plt.imshow(result, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')

    nb_guard = [1,3,5,7,9,11,13,15]
    plt.figure()
    for i in range(8):
        result, conv, mask = sgp.CFAR2D(1e-10, nb_guard[i], 1, fftd_abs_norm_mean)
        plt.subplot(2,4,i+1)
        plt.imshow(result, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')

    nb_ref = [1,3,5,7,9,11,13,15]
    plt.figure()
    for i in range(8):
        result, conv, mask = sgp.CFAR2D(1e-10, 1, nb_ref[i], fftd_abs_norm_mean)
        plt.subplot(2,4,i+1)
        plt.imshow(result, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')

    pfa = 1e-10
    nb_guard = 1
    nb_ref = 3
    plt.figure()
    result, conv, mask = sgp.CFAR2D(pfa, nb_guard, nb_ref, fftd_abs_norm_mean)
    plt.imshow(result, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')
    plt.title("2DCFAR, pfa : {}, nb_guard : {}, nb_ref : {}".format(pfa,nb_guard,nb_ref))
    plt.xlabel("speed")
    plt.ylabel("range")
    # plt.figure()
    # plt.imshow(conv, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(mask)
    # plt.colorbar()

    # plt.figure()
    # plt.imshow(moyenneur_convolution(M,7))
    # plt.colorbar()
    indices_1 = np.where(result == 1)
    # Afficher les coordonnées
    coordonnees_1 = list(zip(indices_1[0], indices_1[1]))
    print(coordonnees_1)
    for i in range(len(coordonnees_1)):
        (x, y) = coordonnees_1[i]
        Syy = sgp.Music(x,y,data)
    plt.show()