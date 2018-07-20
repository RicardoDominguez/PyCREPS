import serial
folder = "C:\pythonGPREPS\\"

def readRollOutData():
    '''
    rolloutDuration in seconds
    '''
    f = open(folder + 'data_log.txt', 'w')
    try:
        ser = serial.Serial('COM4', 921600)
        recordingData = False
        rolloutFinished = False

        while True:
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

                if recordingData:
                    if l[0] == 0:
                        break
                    else:
                        # Write extracted values into file
                        format_str = ('{} ' * len(l))[:-1] + '\n'
                        out_str = format_str.format(*l)
                        f.write(out_str)
                else:
                    recordingData = l[0] == 1
        f.close()
    finally:
        f.close()

if __name__ == "__main__":
    readRollOutData()
