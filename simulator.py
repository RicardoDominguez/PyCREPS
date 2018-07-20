import numpy as np
import matplotlib.pyplot as plt
import pdb
import time

def robotModel(odoL, odoR, theta):
    '''
    Inputs:
        odoL  - odometry left wheel
        odoR  - odometry right wheel
        theta - robot angle (pi/2 is pointing forwards/upwards)

    Outputs:
        delta_x     - change in robot position (x-axis)
        delta_y     - change in robot position (y-axis)
        delta_theta - robot rotation
    '''
    TICKS_TO_MM = 12.0
    ROBOT_LENGTH = 280.0 # wheel to wheel in mm

    # Odometry to displacement
    delta_l = odoL / TICKS_TO_MM
    delta_r = odoR / TICKS_TO_MM

    if(delta_l == delta_r): # Robot moved forward (no rotation)
        delta_x = np.cos(theta) * delta_l
        delta_y = np.sin(theta) * delta_r
        delta_theta = 0

    else: # Includes rotation
        R = ROBOT_LENGTH * (delta_l + delta_r) / (2 * (delta_r - delta_l))
        wd = (delta_r - delta_l) / ROBOT_LENGTH
        delta_x = R * (np.sin(wd + theta) - np.sin(theta))
        delta_y = R * (np.cos(theta) - np.cos(wd + theta))
        delta_theta = wd

    return delta_x, delta_y, delta_theta

class Robot:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.sensor_theta = 18 * np.pi / 180 # 18 deg
        self. m = 0

    def resetPosition(self):
        # Global frame of reference is robot initial position, pointing to the
        # y-axis (forwards/upwards)
        self.x = 0
        self.y = 0
        self.theta = np.pi / 2

    def plot(self):
        rx = self.x
        ry = self.y
        plt.plot(rx, ry, 'bo', markersize = 10) # Sensor location
        rt = self.theta - np.pi / 2
        l = 280
        d = 300
        robo_pos = np.array([[0, -l, -l, 0, 0], [0, 0, -d, -d, 0]])
        trans_mat = np.array([[np.cos(rt), -np.sin(rt)], [np.sin(rt), np.cos(rt)]])
        robo_trans = np.matmul(trans_mat, robo_pos)
        plt.plot(robo_trans[0, :] + rx, robo_trans[1, :] + ry, 'b') # Robot body

class Wall:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0

    def setPosition(self, m, theta, theta_sensor):
        '''
        Inputs
            m     - distance from robot to wall as measured by TOF sensor
            theta - angle of the robot with respect to wall
        '''
        self.x = m * np.cos(theta_sensor)
        self.y = m * np.sin(theta_sensor)
        self.theta = np.pi / 2 + theta

    def plot(self):
        scale_wall = 1000
        x1 = self.x + scale_wall * np.cos(self.theta)
        y1 = self.y + scale_wall * np.sin(self.theta)
        x2 = self.x + scale_wall * np.cos(np.pi + self.theta)
        y2 = self.y + scale_wall * np.sin(np.pi + self.theta)
        plt.plot([x1, x2], [y1, y2], 'g')

class Scenario:
    def __init__(self, dt = 0.02):
        self.robot = Robot()
        self.wall = Wall()
        self.dt = dt;

    def initScenario(self, x):
        '''
        Inputs
            x[0] - distance from robot to wall as measured by TOF sensor
            x[1] - angle of the robot with respect to wall

        Outpus:
            None
        '''
        self.robot.resetPosition()
        self.robot.m = x[0] # Last TOF measurement
        self.wall.setPosition(x[0], x[1], self.robot.sensor_theta)

    def step(self, u):
        '''
        Simulates the scenario for self.dt seconds, with the specified control action

        Inputs
            u - Control action, speed of wheels in mm/s ([0] for left, [1] for right)

        Outputs
            y - The new state vector [d, w]
        '''
        odoL = u[0] * self.dt
        odoR = u[1] * self.dt
        self.stepMotors(odoL, odoR) # New robot position
        valid_sample, m2 = self.sampleTOFSensor() # New tof sample
        d_wall, theta_wall = self.computeDistanceAngle(odoL, odoR, self.robot.m, m2) # New state estimation
        self.robot.m = m2
        x = np.array([d_wall, theta_wall]).reshape(-1)
        return x

    def stepMotors(self, odoL, odoR):
        '''
        Updates the robot position

        Inputs:
            odoL - Displacement odometry (left wheel)
            odoR - Displacement odometry (right wheel)
        '''
        delta_x, delta_y, delta_theta = robotModel(odoL, odoR, self.robot.theta)
        self.robot.x, self.robot.y = self.detectCollision(delta_x, delta_y)
        self.robot.theta += delta_theta

    def detectCollision(self, delta_x, delta_y):
        '''
        Inputs
            delta_x - change in robot position (x-axis)
            delta_y - change in robot position (y-axis)

        Outputs
            Robot x, y position according to delta_x and delta_y and the wall position
        '''
        # Point of collision with the wall following the trajectory
        x_i = (delta_y*self.robot.x + delta_x * (self.wall.y - self.robot.y - self.wall.x*np.tan(self.wall.theta)))/(delta_y - delta_x*np.tan(self.wall.theta))
        y_i = self.robot.y + (delta_y*(x_i - self.robot.x))/delta_x

        # Check if point of collision is within robot displacement
        if delta_x >= 0:
            valid = x_i < self.robot.x or x_i > self.robot.x + delta_x
        else:
            valid = x_i < self.robot.x + delta_x or x_i > self.robot.x

        if valid:
            return self.robot.x + delta_x, self.robot.y + delta_y
        else:
            return x_i - 0.01, y_i - 0.01

    def sampleTOFSensor(self):
        '''
        Outpus:
            valid - whether the robot would have returned a valid measurement
            m     - distance measured by sensor (255 for invalid)
        '''
        self.robot.y = self.robot.y
        x_i = -(self.robot.y - self.wall.y - self.robot.x*np.tan(self.robot.theta - np.pi / 2 + self.robot.sensor_theta) + self.wall.x*np.tan(self.wall.theta))/(np.tan(self.robot.theta - np.pi / 2 + self.robot.sensor_theta) - np.tan(self.wall.theta))
        y_i = np.tan(self.robot.sensor_theta + self.robot.theta - np.pi / 2) * (x_i - self.robot.x) + self.robot.y

        o_abs = np.remainder(self.robot.sensor_theta + self.robot.theta - np.pi / 2 + 100000*np.pi, 2*np.pi)
        if (o_abs >= 0) and (o_abs <= np.pi / 2):
            valid = ((x_i - self.robot.x) >= 0) and ((y_i - self.robot.y) >= 0)
        elif (o_abs >= np.pi/2) and (o_abs <= np.pi):
            valid = ((x_i - self.robot.x) <= 0) and ((y_i - self.robot.y) >= 0)
        elif (o_abs >= np.pi) and (o_abs <= 3 * np.pi / 2):
            valid = ((x_i - self.robot.x) <= 0) and ((y_i - self.robot.y) <= 0)
        elif (o_abs >= 3*np.pi/2) and (o_abs <= 2 * np.pi):
            valid = ((x_i - self.robot.x) >= 0) and ((y_i - self.robot.y) <= 0)

        if valid:
            m = np.sqrt((self.robot.x - x_i)**2 + (self.robot.y - y_i)**2)
            if m > 255:
                valid = False
                m = 255
        else:
            m = 255

        return valid, m

    def computeDistanceAngle(self, odoL, odoR, m1, m2):
        '''
        Inputs
            odoL - Displacement odometry (left wheel)
            odoR - Displacement odometry (right wheel)
            m1   - TOF measurement prior to displacement
            m2   - TOF measurment after displacement

        Outputs
            d_wall  - Distance from robot to the wall
            d_theta - Angle of wall with respect to the robot
        '''
        delta_x, delta_y, delta_theta = robotModel(odoL, odoR, self.robot.theta)
        theta_s = self.robot.sensor_theta

        # Compute 4 known points
        x2 = m1 * np.cos(theta_s)
        y2 = m1 * np.sin(theta_s)
        x3 = delta_x
        y3 = delta_y
        x4 = x3 + m2 * np.cos(theta_s + delta_theta)
        y4 = y3 + m2 * np.sin(theta_s + delta_theta)

        if(y2 != y4 and x2 != x4): # Prevent division by 0
            # Calculate perpendicular to wall through current sensor position
            m = (y2 - y4) / (x2 - x4)
            x_perp = (y3 - y2 + x2*m + (x3 / m)) / (m + (1 / m))
            y_perp = m * (x_perp - x2) + y2

            # Calculate distances between calculated points
            a_2 = (x4 - x3)*(x4 - x3) + (y4 - y3)*(y4 - y3)
            b_2 = (x4 - x_perp)*(x4 - x_perp) + (y4 - y_perp)*(y4 - y_perp)
            c_2 = (x_perp - x3)*(x_perp - x3) + (y_perp - y3)*(y_perp - y3)

            # Calculate distance to wall
            d_wall = np.sqrt(c_2)

            # Calculate angle with wall
            if(x_perp < x4 and y_perp > y4):
                theta_wall = np.pi/2 - np.arccos((a_2 + b_2 - c_2) / (2 * np.sqrt(a_2) * np.sqrt(b_2))) + self.robot.sensor_theta;
            else:
                theta_wall = np.arccos((a_2 + b_2 - c_2) / (2 * np.sqrt(a_2) * np.sqrt(b_2))) + self.robot.sensor_theta - np.pi/2;

        else:
            d_wall = 0
            theta_wall = 0

        return d_wall, theta_wall

    def plot(self):
        self.robot.plot()
        self.wall.plot()

        # Plot sensor measurement
        valid_tof, m = self.sampleTOFSensor()
        if valid_tof:
            plt.plot([self.robot.x, self.robot.x + m * np.cos(self.robot.theta + self.robot.sensor_theta  - np.pi/2)], [self.robot.y, self.robot.y + m * np.sin(self.robot.theta + self.robot.sensor_theta - np.pi/2)], 'k')
        else:
            plt.plot([self.robot.x, self.robot.x + m * np.cos(self.robot.theta + self.robot.sensor_theta - np.pi/2)], [self.robot.y, self.robot.y + m * np.sin(self.robot.theta + self.robot.sensor_theta - np.pi/2)], 'k')

        plt.draw()
        plt.pause(0.001)
        raw_input("Press [enter] to continue.")

def simulate(scn):
    plt.ion()
    plt.show()

    T = 10
    speedL = 150
    speedR = 150
    scn.plot()
    #time.sleep(0.5)
    errors = np.zeros(T)
    for t in range(T):
        u = [speedL, speedR]
        x = scn.step(u)
        d_wall = x[0]
        theta_wall = x[1]

        if theta_wall >= 0:
            speedL = 50
            speedR = 150
        elif theta_wall < 0:
            speedL = 150
            speedR = 50

        scn.plot()
        errors[t] = -theta_wall
    print errors

if __name__ == '__main__':
    scn = Scenario(4)
    x0 = np.array([140, np.pi/6])
    scn.initScenario(x0)
    simulate(scn)
