import numpy as np
import matplotlib.pyplot as plt
import pdb
import time

from policy import Proportional

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

    def plot(self, size_dot = 1):
        rx = self.x
        ry = self.y
        plt.plot(rx, ry, 'bo', markersize = size_dot) # Sensor location
        rt = self.theta - np.pi / 2
        l = 280
        d = 300
        #robo_pos = np.array([[0, -l, -l, 0, 0], [0, 0, -d, -d, 0]])
        robo_pos = np.array([[0, 0, 0], [0, -d, 0]])
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
        self.dt = dt

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
        odoL = u[0] * self.dt * 12
        odoR = u[1] * self.dt * 12
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

        return valid, m # Measurements can only be integers

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
        ang0 = theta_s + self.robot.theta - np.pi/2

        # print 'm1 ' , m1
        # print 'm2 ' , m2
        # print 'x3 ' , delta_x
        # print 'y3 ' , delta_y
        # print 'dt ' , delta_theta
        # print 'rt ' , self.robot.theta

        # Compute 4 known points
        x2 = m1 * np.cos(ang0)
        y2 = m1 * np.sin(ang0)
        x3 = delta_x
        y3 = delta_y
        x4 = x3 + m2 * np.cos(ang0 + delta_theta)
        y4 = y3 + m2 * np.sin(ang0 + delta_theta)

        if(y2 != y4 and x2 != x4): # Prevent division by 0
            # Calculate perpendicular to wall through current sensor position
            # m = (y2 - y4) / (x2 - x4)
            # x_perp = (y4 + x3 * np.tan(delta_theta) - y3 - x4*m) / (np.tan(delta_theta) - m)
            # y_perp = m * (x_perp - x4) + y4
            #
            # # Calculate distances between calculated points
            # a_2 = (x4 - x3)*(x4 - x3) + (y4 - y3)*(y4 - y3)
            # b_2 = (x4 - x_perp)*(x4 - x_perp) + (y4 - y_perp)*(y4 - y_perp)
            # c_2 = (x_perp - x3)*(x_perp - x3) + (y_perp - y3)*(y_perp - y3)
            #
            # d_wall = np.sqrt(c_2)
            # if(a_2 > 1e-9 and b_2 > 1e-9):
            #     theta_wall = np.arccos((a_2 + b_2 - c_2) / (2 * np.sqrt(a_2) * np.sqrt(b_2))) + self.robot.sensor_theta - np.pi / 2
            # else:
            #     theta_wall = 0 # No calculation available, best ignore
            #
            # if theta_wall < 0:
            #     pdb.set_trace()
            a1 = (y2 - y4) / float(x2 - x4)
            a2 = np.tan(self.robot.theta - np.pi/2)
            xi = (y3 - y4 + a1*x4 - x3*a2)/float(a1 - a2)
            yi = y3 + a2 * (xi - x3)

            a3 = -1 / a2
            xp = -(y3 - y4 - a2*x3 + a3*x4)/(a2 - a3)
            yp = y3 + a2 * (xp - x3)

            a_2 = (x4 - xi)*(x4 - xi) + (y4 - yi)*(y4 - yi)
            b_2 = (x4 - xp)*(x4 - xp) + (y4 - yp)*(y4 - yp)
            c_2 = (xi - xp)*(xi - xp) + (yi - yp)*(yi - yp)

            if(a_2 > 1e-9 and b_2 > 1e-9 and c_2 > 1e-9):
                theta_wall = np.arccos((a_2 + b_2 - c_2)/(2 * np.sqrt(a_2) * np.sqrt(b_2)))
                #print a_2, b_2, c_2
                if xp > xi:
                    theta_wall = -theta_wall
            else:
                theta_wall = 0 # No calculation available, best ignore

            d_wall = np.sqrt((x3 - xi)*(x3 - xi) + (y3 - yi)*(y3 - yi))

        else:
            d_wall = 0
            theta_wall = 0

        return d_wall, theta_wall

    def plot(self, interactive = True):
        if interactive:
            self.robot.plot(10)
        else:
            self.robot.plot(1)
        self.wall.plot()

        # Plot sensor measurement
        if interactive:
            valid_tof, m = self.sampleTOFSensor()
            if valid_tof:
                plt.plot([self.robot.x, self.robot.x + m * np.cos(self.robot.theta + self.robot.sensor_theta  - np.pi/2)], [self.robot.y, self.robot.y + m * np.sin(self.robot.theta + self.robot.sensor_theta - np.pi/2)], 'k')
            else:
                plt.plot([self.robot.x, self.robot.x + m * np.cos(self.robot.theta + self.robot.sensor_theta - np.pi/2)], [self.robot.y, self.robot.y + m * np.sin(self.robot.theta + self.robot.sensor_theta - np.pi/2)], 'k')

        if interactive:
            plt.draw()
            plt.pause(0.001)
            raw_input("Press [enter] to continue.")

def simulateStep(scn, x0, T, pol, w):
    scn.initScenario(x0)
    scn.plot(False)
    x = x0
    for t in range(T):
        u = pol.sample(w, x)
        print 'Distance ', x[0]
        print 'Computed angle ', x[1] * 180 / np.pi
        print 'Wheel speed ', u
        x = scn.step(u)
        scn.plot(False)
        # print 'Real angle ', (scn.wall.theta - scn.robot.theta) * 180 / np.pi
    plt.show()

def simulateResults(scn, x0, T, pol, w, id = 0):
    dist_errors = []
    ang_errors = []
    max_overshoot = 0
    time_to_distance = -1
    collision = False

    scn.initScenario(x0)
    x = x0
    plt.figure((id-1) * 2 + 1)
    scn.plot(False)
    for t in range(T):
        u = pol.sample(w, x)
        x = scn.step(u)

        scn.plot(False)
        if x[0] <= pol.target[0]:
            dist_errors.append(pol.target[0] - x[0])
            ang_errors.append((pol.target[1] - x[1])*180 / np.pi)
            if time_to_distance == -1:
                time_to_distance = t+1
            if pol.target[0] - x[0] > max_overshoot:
                max_overshoot = pol.target[0] - x[0]
            if x[0] < 0.1:
                collision = True

    if id:
        plt.title('Weights ' + str(id) + ' trajectory')
    else:
        plt.title('Robot trajectory')
    plt.xlabel('x axis (mm)')
    plt.ylabel('y axis (mm)')
    print '\n\n------------------------------------------------------------'
    if id:
        print 'Simulation results (', str(id), '):'
    else:
        print 'Simulation results: '
    print '------------------------------------------------------------'
    print 'Collision: ', collision
    if time_to_distance == -1:
        print 'Target distance was not reached'
    else:
        print 'Maximum overshoot (mm): ', np.round(max_overshoot, 1)
        print 'Time to target distance (s): ', time_to_distance * scn.dt

        time = np.linspace(time_to_distance * scn.dt, T * scn.dt, len(dist_errors)).tolist()
        plt.figure((id-1) * 2 + 2)
        plt.subplot(211)
        if id:
            plt.title('Weights ' + str(id) + ' metrics')
        else:
            plt.title('Metrics')
        plt.plot(time, dist_errors)
        plt.ylabel('Distance error (mm)')
        plt.subplot(212)
        plt.plot(time, ang_errors)
        plt.ylabel('Angle error (deg)')
        plt.xlabel('Time (s)')
        if not id: plt.show()
    print '\n'

def compareWeights(scn, x0, T, pol, w1, w2):
    simulateResults(scn, x0, T, pol, w1, id = 1)
    simulateResults(scn, x0, T, pol, w2, id = 2)
    plt.show()

if __name__ == '__main__':
    # scn = Scenario(0.1)
    # x0 = np.array([240, np.pi/3])
    # scn.initScenario(x0)
    # simulate(scn, x0)

    scn = Scenario(0.1)
    x0 = np.array([240, np.pi/3])
    target = np.array([10, 0]).reshape(-1)
    offset = np.array([150, 150]).reshape(-1)
    pol = Proportional(-324, 324, target, offset)
    w = np.array([-5.5, 133, -5.4, -105]).reshape(-1)
    w2 = np.array([-2, 100, 2, -100]).reshape(-1)
    T = 300

    #simulateStep(scn, x0, T, pol, w)
    #simulateResults(scn, x0, T, pol, w)
    compareWeights(scn, x0, T, pol, w, w2)
