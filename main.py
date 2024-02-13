# import struct
#
# print(struct.unpack('f', b'\xdb\x0fI@'))

import simu_v2 as sml
import numpy as np
import signalprocessing as sgp
from read_data import *
from parametre import *


if __name__ == "__main__":

    # Tc = 10 ** -3
    # Te = Tc / 1000
    # T = 64 * Tc
    # fc = 77 * 10 ** 9
    # B = 4 * 10 ** 9
    # simulation = sml.Simulation(T, Tc, Te, fc, B)
    #
    # ltx = [[0, 0]]
    # lrx = [[-1, 0], [1, 0]]
    # simulation.board.add(ltx, lrx)
    #
    # ltargets = [[0, 5, 0, 0]]
    # simulation.addTrg(ltargets)
    # print(simulation.board.rx)
    # print(simulation.board.tx)
    # print(simulation.targets)
    #
    # ts = simulation.process()
    # print(ts)
    #
    # xif = simulation.simulation()
    # print(xif)

    # tg1 = sml.Target(10, np.deg2rad(0), 0, 0, 1)
    #
    # tg2 = sml.Target(10, np.deg2rad(45), 0, 7, 1)

    # tg3 = sml.Target(30, np.deg2rad(-15), 0, 0.5, 0.5)
    #
    # tg4 = sml.Target(35, np.deg2rad(-60), 0, 0.2, 0.5)
    #
    # tg5 = sml.Target(40, np.deg2rad(-30), 0, -1, 0.9)
    #
    # simu = sml.Simulateur()
    #
    # data = simu.run([1, 2, 3, 4], [1, 2, 3], ["DDMA", np.deg2rad(np.array([0, 90, 270]))], [tg1, tg3])
    #
    # print(data.shape)
    # fft1 = sgp.calculate_range_fft(data)
    # sgp.plot_range_spectrum(fft1)
    #
    # fft2 = sgp.calculate_doppler_fft(fft1)

    conver = convertFile(Nr * Nt)
    data = conver.read('adc_data.bin',isReal=False)
    data = conver.reshaper(data)
    print(data.shape)

    fft1 = sgp.calculate_range_fft(data)
    sgp.plot_range_spectrum(fft1)