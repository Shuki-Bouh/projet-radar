import numpy as np
from parametre import *
from read_data import *

nchirp = 3
nRx = 4
nSample = 8
nFrame = 2

A = np.zeros(nchirp * nRx * nSample * nFrame)
itt = 0

for n in range(1, nFrame + 1):
    for i in range(1, nchirp + 1):
        for j in range(1, nRx + 1):
            for k in range(1, nSample + 1):
                nb = i * 10 + j * 100 + k + 1000 * n
                A[itt] = nb
                itt += 1


converter = convertFile()
converter.numChirps = nchirp
converter.numRX = nRx
converter.numADCSamples = nSample
converter.numFrame = nFrame


A = converter.big_reshaper(A)
print(A[:,:,0,1])