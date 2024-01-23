import unittest
import numpy as np
from read_data import convertFile


class TestReshaper(unittest.TestCase):

    def test_reshape(self):
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

        converter = convertFile()
        converter.numChirps = nchirp
        converter.numRX = nRx
        converter.numADCSamples = nSample

        A = converter.reshaper(A)

        for i in range(converter.numRX):
            for j in range(converter.numChirps):
                for k in range(converter.numADCSamples):
                    self.assertEqual(int(A[k, j, i]), (i + 1) * 100 + (j + 1) * 10 + k + 1)  # C'est un cas particulier
                    # qui permet de vérifier le bon fonctionnement de reshaper


if __name__ == '__main__':

    unittest.main()



