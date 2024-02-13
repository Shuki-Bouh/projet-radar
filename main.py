from parametre import *
from read_data import convertFile
import pickle as pk
from time import time

if __name__ == '__main__':
    converter = convertFile(numADCSample=Ns, numRx=Nt*Nr, numChirps=Nc, numFrame=Nf)
    data = converter.read('adc_data_200F_128C.bin', isReal=isReal)
    data = converter.big_reshaper(data)
    print(data[0,0,0,0])