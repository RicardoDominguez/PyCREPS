import serial
folder = "C:\pythonGPREPS\\"

import numpy as np
import matplotlib.pyplot as plt

def readRollOutData(nstates, H):
    '''
    rolloutDuration in seconds
    '''
    ser = serial.Serial('COM4', 921600)
    recordingData = False
    latent = np.empty([0, nstates])
    k = 0
    while k <= H:
        # Read line coming from serial
        line = ser.readline()
        print line

        # Only my strings start with $
        if line[0] == "$":
            # Extract numerical values into array l
            l = []
            for t in line.split():
                try:
                    l.append(float(t))
                except ValueError:
                    pass

            if l[0] == 1:
                latent = np.concatenate((latent, np.array(l[1:nstates+1]).reshape(1, nstates)))
                k += 1

    X = latent[0:-1, :]
    Y = latent[1:, :]
    return X, Y

if __name__ == "__main__":
    X, Y = readRollOutData(2, 500)
    plt.plot(Y)
    plt.show()
