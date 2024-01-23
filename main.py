from parametre import *
from read_data import convertFile

if __name__ == '__main__':
    conver = convertFile(Nr * Nt)
    data = conver.read('adc_data.bin')
    data = conver.reshaper(data)
    print(data.shape)