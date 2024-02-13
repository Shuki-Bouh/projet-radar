import matplotlib.pyplot
import numpy as np



class convertFile:

    def __init__(self, numRx):
        self.numADCSamples = 256
        self.numADCBits = 16
        self.numRX = numRx
        self.numChirps = 0
        return


    def read(self, fileName, isReal=True):


        with open(fileName, 'rb') as data:  # Il sagit de connaître la taille du fichier qu'on va lire
            fileSize = len(data.readlines())

        with open(fileName, 'rb') as data:  # On remplie ADC data des données (int16 donc on lit 2 bytes à la fois

            adcData = []
            for k in range(fileSize):
                c = int.from_bytes(data.readline(2), "big", signed=True)  # Le logiciel fourni des bytes en big endian
                adcData.append(c)

        if isReal:
            self.numChirps = fileSize // self.numADCSamples // self.numRX + 1  # On n'obtient pas l'arrondi, on obtient
            # systématiquement la valeur supérieure,
            # Cette astuce permet de remplir les données manquantes du dernier chirps par des 0
            for k in range(abs(self.numADCSamples * self.numRX * self.numChirps - fileSize)):
                adcData.append(0)

            ldvs = adcData.copy()


    
        else:  # Complexes
            self.numChirps = fileSize // 2 // self.numADCSamples // self.numRX + 1
            for k in range(abs(self.numADCSamples * self.numRX * self.numChirps - fileSize)):
                adcData.append(0)
            ldvs = np.zeros(fileSize / 2)
    
            counter = 0
            for i in range(fileSize - 1, 4):
                ldvs[counter] = adcData[i], adcData[i + 2] * 1j
                ldvs[counter + 1] = adcData[i+1], adcData[i + 3] * 1j
                counter += 2

        return ldvs

    def reshaper(self, ldvs):
        """
                    Les données sont transmises de la façon suivante :
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

        ret = np.zeros((self.numADCSamples, self.numChirps, self.numRX))

        for i in range(self.numRX):
            ret[:, :, i] = ldvs[:, i, :]

        return ret


if __name__ == '__main__':
    conver = convertFile()
    data = conver.read('adc_data.bin')
    data = conver.reshaper(data)
    print(data.shape)