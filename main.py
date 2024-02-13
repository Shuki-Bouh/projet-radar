from parametre import *
from read_data import convertFile
import pickle as pk

if __name__ == '__main__':
    converter = convertFile(numADCSample=Ns, numRx=Nt*Nr, numChirps=Nc, numFrame=Nf)
    data = converter.read('adc_data_200F_128C.bin')
    data = converter.big_reshaper(data)
    with open('matrix', 'wb') as saveMatrix:
        saveMatrix = pk.Pickler(saveMatrix)
        saveMatrix.dump(data)
    print(data[:,0,0,0])