import unittest
import numpy as np
from read_data import convertFile
from parametre import *



class TestConvertFile(unittest.TestCase):
    """Pour ce test, on crée une matrice de dimensions : Nsample * Nchirps * NRx * NFrame de la forme :
       M[a, b, c, d] = dcba -> 1234 correspond à la première frame, deuxième Rx, troisième chirp et quatrième sample
       Ainsi, si pour tout a, b, c et d M[a, b, c, d] = dcba, alors le big_reshape fonctionne
       (on test d'abord avec le little reshape)"""

    def test_little_reshape(self):
        nchirp = 3
        nRx = 4
        nSample = 8

        A = np.zeros(nchirp * nRx * nSample)
        itt = 0

        for i in range(1, nchirp + 1):
            for j in range(1, nRx + 1):
                for k in range(1, nSample + 1):
                    nb = i * 10 + j * 100 + k
                    A[itt] = nb
                    itt += 1

        converter = convertFile(numADCSample=nSample, numChirps=nchirp, numRx=nRx)

        A = converter.little_reshaper(A)
        for i in range(converter.numRX):
            for j in range(converter.numChirps):
                for k in range(converter.numADCSamples):
                    self.assertEqual(int(A[k, j, i]), (i + 1) * 100 + (j + 1) * 10 + k + 1)  # C'est un cas particulier
                    # qui permet de vérifier le bon fonctionnement de reshaper

    def test_big_reshape(self):
        nchirp = 3
        nRx = 4
        nSample = 8
        nFrame = 5

        A = np.zeros(nFrame * nchirp * nRx * nSample)
        itt = 0

        for n in range(1, nFrame + 1):
            for i in range(1, nchirp + 1):
                for j in range(1, nRx + 1):
                    for k in range(1, nSample + 1):
                        nb = n * 1000 + i * 10 + j * 100 + k
                        A[itt] = nb
                        itt += 1

        converter = convertFile(numADCSample=nSample, numChirps=nchirp, numRx=nRx, numFrame=nFrame)

        A = converter.big_reshaper(A)
        for n in range(converter.numFrame):
            for i in range(converter.numRX):
                for j in range(converter.numChirps):
                    for k in range(converter.numADCSamples):
                        self.assertEqual(int(A[k, j, i, n]),(n + 1) * 1000 + (i + 1) * 100 + (j + 1) * 10 + k + 1)  # C'est un cas particulier
                    # qui permet de vérifier le bon fonctionnement de reshaper

    def test_size_data(self):
        converter = convertFile(numADCSample=Ns, numRx=Nr, numChirps=Nc, numFrame=Nf)
        data_200 = converter.read("adc_data_200F_128C.bin", isReal=isReal)
        self.assertEqual(len(data_200), Ns * Nr * Nc * Nf)
        data_200 = converter.big_reshaper(data_200)
        self.assertEqual(len(data_200[0,0,0]), Nf)
        self.assertEqual(len(data_200[0,0]), Nr)
        self.assertEqual(len(data_200[0]), Nc)
        self.assertEqual(len(data_200), Ns)

if __name__ == '__main__':

    unittest.main()



