from parametre import *
from read_data import convertFile
from time import time
from signalprocessing import SignalProcessing
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    a = time()
    converter = convertFile(numADCSample=Ns, numRx=Nt*Nr, numChirps=Nc, numFrame=Nf)
    data = converter.read('adc_data_200F_128C.bin', isReal=isReal)


    data = converter.big_reshaper(data)

    frame = data[:, :, :, 100]

    # fft_r = sgp.calculate_range_fft(frame)
    # sgp.plot_range_spectrum(fft_r)

    sgp = SignalProcessing()

    fftr = sgp.calculate_range_fft(frame)
    sgp.plot_range_spectrum(fftr)
    fftd = sgp.calculate_doppler_fft(fftr)
    sgp.plot_doppler_spectrum(fftd)

    fftr_abs = np.abs(fftr)
    fftr_abs_norm = fftr_abs / np.max(fftr_abs, axis=0)
    fftr_abs_norm_mean = np.mean(fftr_abs_norm, axis=(1, 2))
    rang = np.arange(fftr.shape[0]) * c / (2 * S * Tc)
    plt.plot(rang,fftr_abs_norm_mean)

    plt.figure(2)
    fftd = sgp.calculate_range_fft(fftr)
    fftd_abs = np.abs(fftd)
    fftd_abs_norm = fftd_abs / np.max(fftd_abs, axis=(1, 0), keepdims=True)
    fftd_abs_norm_mean = np.mean(fftd_abs_norm, axis=2)  # Moyenne selon les antennes de réception
    fftd_abs_norm_mean = np.fft.fftshift(fftd_abs_norm_mean, axes=1)
    speed = np.arange(-fftd.shape[1] // 2, fftd.shape[1] // 2) * λ / (2 * Tc * fftd.shape[1]) # le tdma divise la vitesse max par le nombre d antenne Tx
    rang = np.arange(fftd.shape[0]) * c / (2 * S * Tc)
    plt.imshow(fftd_abs_norm_mean, extent=[np.min(speed), np.max(speed), np.min(rang), np.max(rang)], origin='lower')

    plt.show()
    b = time()
    print(b-a)




