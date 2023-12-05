import struct
import numpy
import numpy as np


def retVal(fileName, isReal):
    numADCSamples = 256
    numADCBits = 16
    numRX = 4
    numLanes = 2
    with open(fileName, 'rb') as data:
        adcData = []
        for line in data.readlines():  # Subtilité : il faut en réalité ajouter un nombre et non une ligne (chercher à savoir s'il y a plusieurs nombre sur une ligne)
            adcData.append(struct.unpack("h", line))
    fileSize = len(adcData)
    if isReal:
        numChirps = fileSize / numADCSamples / numRX
        LVDS = np.reshape(adcData, (numADCSamples * numRX, numChirps))

    else:  # Complexes
        numChirps = fileSize / 2 / numADCSamples / numRX
        LDVS = np.zeros(fileSize / 2)

        counter = 0
        for i in range(fileSize - 1, 4):
            LDVS[counter] = adcData[i], adcData[i + 2] * 1j
            LDVS[counter + 1] = adcData[i+1], adcData[i + 3] * 1j
            counter += 2

        LDVS = np.reshape(LDVS, (numADCSamples * numRX, numChirps))

    adcData = np.zeros((numRX, numChirps * numADCSamples))
    for j in range(numRX):
        for i in range(numChirps):
            adcData[j, (i - 1) * numADCSamples +1] = process_adc_data.m #Pas sur de comprendre là

    return adcData