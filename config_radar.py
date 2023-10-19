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
    Dataport = serial.Serial('COM5', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFile)]
    for i in config:
        CLIport.write((i + '\n').encode())
        print(i)
        time.sleep(0.01)
    x=[]
    t=[t for t in range(1000)]
    for i in range(1000):
        data = Dataport.read()
        data += Dataport.read()
        data += Dataport.read()
        data += Dataport.read()
        x.append(struct.unpack('f', data))

    t = np.arange(1000)

    freq = fft.fftfreq(t.shape[-1])
    X = fft.fft(x)
    plt.plot(t,X.real)

    plt.show()
    return CLIport #, Dataport










if __name__ == '__main__':

    configFile = "config.cfg"
    serialConfig(configFile)