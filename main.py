from parametre import *
from read_data import convertFile
from time import time
from signalprocessing import SignalProcessing
import numpy as np
from matplotlib import pyplot as plt
import simu_v2 as simp
import signalprocessing as sgp

if __name__ == '__main__':
    a = time()
    converter = convertFile(numADCSample=Ns, numRx=Nt*Nr, numChirps=Nc, numFrame=Nf)
    data = converter.read('adc_data_200F_128C.bin', isReal=isReal)
    data = converter.big_reshaper(data)

    frame = data[:, :, :, 19]

#     # fft_r = sgp.calculate_range_fft(frame)
#     # sgp.plot_range_spectrum(fft_r)
#
#     sgp = SignalProcessing()
#
#     fftr = sgp.calculate_range_fft(frame)
#     sgp.plot_range_spectrum(fftr)
#     fftd = sgp.calculate_doppler_fft(fftr)
#     sgp.plot_doppler_spectrum(fftd)
#
#     fftr_abs = np.abs(fftr)
#     fftr_abs_norm = fftr_abs / np.max(fftr_abs, axis=0)
#     fftr_abs_norm_mean = np.mean(fftr_abs_norm, axis=(1, 2))
#     rang = np.arange(fftr.shape[0]) * c / (2 * S * Tc)
#     plt.plot(rang,fftr_abs_norm_mean)
#
#     plt.figure(2)
#     fftd = sgp.calculate_range_fft(fftr)
#     fftd_abs = np.abs(fftd)
#     fftd_abs_norm = fftd_abs / np.max(fftd_abs, axis=(1, 0), keepdims=True)
#     fftd_abs_norm_mean = np.mean(fftd_abs_norm, axis=2)  # Moyenne selon les antennes de réception
#     fftd_abs_norm_mean = np.fft.fftshift(fftd_abs_norm_mean, axes=1)
#     speed = np.arange(-fftd.shape[1] // 2, fftd.shape[1] // 2) * λ / (2 * Tc * fftd.shape[1]) # le tdma divise la vitesse max par le nombre d antenne Tx
#     rang = np.arange(fftd.shape[0]) * c / (2 * S * Tc)
#     plt.imshow(fftd_abs_norm_mean, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')
#
#     plt.show()
#     b = time()
#     print(b-a)

    tg1 = simp.Target(10,np.deg2rad(0),0,0,1)

    tg2 = simp.Target(10,np.deg2rad(45),0,7,1)

    tg3 = simp.Target(30,np.deg2rad(-15),0,0.5,0.5)

    tg4 = simp.Target(35,np.deg2rad(-60),0,0.2,0.5)

    tg5 = simp.Target(40,np.deg2rad(-30),0,-1,0.9)

    simu = simp.Simulateur()

    # frame = simu.run([1,2,3,4],[1],["SIMO"],[tg1,tg2])
    # np.deg2rad(np.array([0, 90, 270]))

    frame = simu.run([1,2,3,4],[1,2,3],["DDMA",np.deg2rad(np.array([0, 90, 270]))],[tg1])
    np.deg2rad(np.array([0, 90, 270]))

    sigproc = sgp.SignalProcessing()


    fftr = sigproc.range_fft(frame)
    fftd = sigproc.doppler_fft(fftr)
    print(fftd.shape)
    fftd = sigproc.DDMA_antenna_2_channels(fftd, np.deg2rad(np.array([0, 90, 270])))
    print(fftd.shape)
    cfar = sigproc.cfar2D(1e-10, 1, 3, fftd)

    # sigproc.plot_range_spectrum(fftr)
    sigproc.plot_doppler_spectrum(fftd)
    sigproc.plot_cfar2D(cfar)

    for i in range(12):
        plt.subplot(3,4,i+1)
        fft2D_abs = np.abs(fftd)
        fft2D_abs_norm = fft2D_abs / np.max(fft2D_abs, axis=(1, 0), keepdims=True)
        fft2D_abs_norm_mean = fft2D_abs_norm[:,:,i]  # Moyenne selon les antennes de réception
        fft2D_abs_norm_mean_shift = np.fft.fftshift(fft2D_abs_norm_mean, axes=1)
        speed = np.arange(-fftd.shape[1] // 2, fftd.shape[1] // 2) * λ / (
                2 * Tc * fftd.shape[1])  # le tdma divise la vitesse max par le nombre d antenne Tx
        rang = np.arange(fftd.shape[0]) * c / (2 * S * Tc)
        plt.imshow(fft2D_abs_norm_mean_shift, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)],
                   origin='lower')
        plt.title(str(i))

    plt.show()

    # sigproc.active_fft2D_plot(data)
    # sigproc.active_cfar_plot(data,1e-10,1,3)








