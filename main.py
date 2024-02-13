from parametre import *
from read_data import convertFile

if __name__ == '__main__':
    converter = convertFile(numADCSample=Ns, numRx=Nt*Nr, numChirps=Nc, numFrame=Nf)
    data = converter.read('adc_data_30F_128C.bin')
    data = converter.big_reshaper(data)
    print(data[:,0,0,0])