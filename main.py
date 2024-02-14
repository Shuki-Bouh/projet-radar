from parametre import *
from read_data import convertFile
import pickle as pk
from time import time
import signalprocessing as sgp
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    converter = convertFile(numADCSample=Ns, numRx=Nt*Nr, numChirps=Nc, numFrame=Nf)
    data = converter.read('adc_data_200F_128C.bin', isReal=isReal)
    data = converter.big_reshaper(data)
    print(data[0,0,0,0])

    frame = data[:,:,:,100]
    print(frame.shape)
    print(data.shape)
    fft1 = sgp.calculate_range_fft(frame)
    sgp.plot_range_spectrum(fft1)

    # range = frame[:,0,0]
    # fft = sgp.calculate_range_fft(np.abs(range))
    # fft_abs = np.abs(fft)
    # fft_abs_norm = fft_abs / np.max(fft_abs, axis=0)
    # plt.plot(fft_abs_norm)
    # plt.show()


