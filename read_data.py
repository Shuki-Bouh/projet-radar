import numpy as np
from time import time
import multiprocessing



class convertFile:

    def __init__(self, numADCSample=256, numRx=4, numChirps=128, numFrame=8):
        self.numADCSamples = numADCSample
        self.numADCBits = 16
        self.numRX = numRx
        self.numChirps = numChirps
        self.numFrame = numFrame
        return


    def read(self, fileName, isReal=True):

        adcData = np.fromfile(fileName, dtype=np.int16)
        fileSize = len(adcData)

        if isReal:
            ldvs = adcData.copy()

        else:
            ldvs = np.zeros(fileSize // 2, dtype=complex)
            counter = 0

            nombre_de_coeurs = multiprocessing.cpu_count()

            pool = multiprocessing.Pool(processes=nombre_de_coeurs)

            nb_proc = self.numFrame
            sections = [adcData[t * fileSize // nb_proc: (t + 1) * fileSize // nb_proc] for t in
                        range(nb_proc)]

            res = pool.map(self.complex_threading, sections)

            pool.close()
            pool.join()

            ldvs = np.concatenate(res)

        return ldvs

    def dataFrameNo(self, ldvs, no):
        dPFrame1 = no * self.numADCSamples * self.numChirps * self.numRX
        dPFrame2 = (no + 1) * self.numADCSamples * self.numChirps * self.numRX
        ret = ldvs[dPFrame1:dPFrame2]

        return ret

    def little_reshaper(self, ldvs):
        """
                    Les données sont transmises de la façon suivante pour une frame :
                    Chirps 0 :
                        Rx0 : sample 0
                        Rx0 : sample 1
                            .
                            .
                            .
                        Rx0 : sample numADCSamples (nombres de données par chirps)
                        Rx1 : sample 0
                            .
                        Rx1 : sample numADCSamples
                            .
                        Rx2 : sample 0
                            .
                        Rx3 : sample numADCSamples

                    Chirps 1 :
                        Rx0 : sample 0
                            .
                        Rx3 : sample numADCSamples
                    ...
                    Chirps numChirps :
                        Rx0 : sample 0
                            .
                        Rx3 : sample numADCSamples

                    """

        ldvs = np.reshape(ldvs, (self.numADCSamples, self.numRX, self.numChirps), order="F")

        if type(ldvs[0,0,0]) is np.int16:
            ret = np.zeros((self.numADCSamples, self.numChirps, self.numRX))
        else:
            ret = np.zeros((self.numADCSamples, self.numChirps, self.numRX), dtype=complex)

        for i in range(self.numRX):
            ret[:, :, i] = ldvs[:, i, :]

        return ret

    def big_reshaper(self, ldvs):
        """On fait le little reshaper pour toutes les frames (on se place en dim 4)"""

        if type(ldvs[0]) is np.int16:
            ret = np.zeros((self.numADCSamples, self.numChirps, self.numRX, self.numFrame))
        else:
            ret = np.zeros((self.numADCSamples, self.numChirps, self.numRX, self.numFrame), dtype=complex)

        for i in range(self.numFrame):
            data_frame = self.dataFrameNo(ldvs, i)  # Les données du ie frame
            ret[:, :, :, i] = self.little_reshaper(data_frame)  # On reshape les données d'une frame qu'on place dans le ième frame

        return ret

    @staticmethod
    def complex_threading(data):
        counter = 0
        fileSize = len(data)
        ldvs_inter = np.zeros(fileSize//2, dtype=complex)
        for i in range(0, fileSize - 1, 4):
                ldvs_inter[counter] = data[i] + data[i + 2] * 1j
                ldvs_inter[counter + 1] = data[i + 1] + data[i + 3] * 1j
                counter += 2
        return ldvs_inter
