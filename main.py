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
    print(frame.shape)

    tg1 = simp.Target(30,np.deg2rad(0),0,0,1)

    tg2 = simp.Target(10,np.deg2rad(10),0,7,1)

    tg3 = simp.Target(30,np.deg2rad(-15),0,0.5,0.5)

    tg4 = simp.Target(35,np.deg2rad(-60),0,0.2,0.5)

    tg5 = simp.Target(40,np.deg2rad(-30),0,-1,0.9)

    simu = simp.Simulateur()

    # trg = [simp.Target(10 + 5*i,np.deg2rad(-20 + i*10),0,-30+15*i,1) for i in range(6)]
    # t = [-20 + 10*i for i in range(6)]
    # print(t)

    frame = simu.run([1,2,3,4],[1],["SIMO"],[tg1])

    # frame = simu.run([1,2,3,4],[1,2,3],["DDMA",np.deg2rad(np.array([0, 90, 270]))],[tg1])
    # np.deg2rad(np.array([0, 90, 270]))

    sigproc = sgp.SignalProcessing()

    fftr = sigproc.range_fft(frame)
    fftd = sigproc.doppler_fft(fftr)
    # fftd = sigproc.DDMA_antenna_2_channels(fftd, np.deg2rad(np.array([0, 90, 270])))
    # print(fftd.shape)
    cfar = sigproc.cfar2D(1e-10, 1, 3, fftd)

    # sigproc.plot_range_spectrum(fftr)
    sigproc.plot_doppler_spectrum(fftd)
    sigproc.plot_cfar2D(cfar)
    targets = sigproc.music(fftd, cfar)
    print(targets)
    # print(targets[:,2])
    # print(np.rad2deg(1.57079))
    # print(np.deg2rad(45))
    sigproc.plot_music(targets)
    plt.show()


    # for i in range(12):
    #     plt.subplot(3,4,i+1)
    #     fft2D_abs = np.abs(fftd)
    #     fft2D_abs_norm = fft2D_abs / np.max(fft2D_abs, axis=(1, 0), keepdims=True)
    #     fft2D_abs_norm_mean = fft2D_abs_norm[:,:,i]  # Moyenne selon les antennes de réception
    #     fft2D_abs_norm_mean_shift = np.fft.fftshift(fft2D_abs_norm_mean, axes=1)
    #     speed = np.arange(-fftd.shape[1] // 2, fftd.shape[1] // 2) * λ / (
    #             2 * Tc * fftd.shape[1])  # le tdma divise la vitesse max par le nombre d antenne Tx
    #     rang = np.arange(fftd.shape[0]) * c / (2 * S * Tc)
    #     plt.imshow(fft2D_abs_norm_mean_shift, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)],
    #                origin='lower')
    #     plt.title(str(i))
    # plt.show()

    #sigproc.active_fft2D_plot(data)
    #sigproc.active_cfar_plot(data,1e-3,1,3)

    print(dmax)
    print(dres)
    print(vmax)
    print(vres)
    print(thetares)









