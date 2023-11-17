import serial
import time
import matplotlib.pyplot as plt
import numpy.fft as fft
import numpy as np
import struct



def serialConfig(configFile):
    # Open the serial ports for the configuration and the data ports

    # Raspberry pi
    # CLIport = serial.Serial('/dev/ttyACM0', 115200)
    # Dataport = serial.Serial('/dev/ttyACM1', 921600)

    # Windows
    CLIport = serial.Serial('COM6', 115200)
    Dataport = serial.Serial('COM7', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFile)]
    for i in config:
        CLIport.write((i + '\n').encode())
        print(i)
        time.sleep(0.01)
    x=[]
    for i in range(10000):
        data = Dataport.read()
        data += Dataport.read()
        data += Dataport.read()
        data += Dataport.read()
        x.append(struct.unpack('f', data))
        print(x[i])

    t = np.arange(10000)





    freq = fft.fftfreq(t.shape[-1])
    X = fft.fft(x)
    plt.figure()
    plt.plot(freq, np.abs(X))
    plt.figure()
    plt.plot(np.real(X), np.imag(X))

    plt.show()
    return CLIport #, Dataport










if __name__ == '__main__':

    configFile = "config.cfg"
    serialConfig(configFile)