""" Includes varius functions to measure the performance of the lower-level
    policy within the wall following robot environment.
"""

import numpy as np
import matplotlib.pyplot as plt

from scenario import Cost
from scenario import LowerPolicy
from scenario import sampleContext

def robotModel(odoL, odoR, theta):
    '''
    Dynamic model for a differential drive robot.

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
    '''
    Robot class contains information of the robot state (x, y, theta).
    '''
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.sensor_theta = 18 * np.pi / 180 # 18 deg
        self.m = 0

    def resetPosition(self):
        # Global frame of reference is robot initial position, pointing to the
        # y-axis (forwards/upwards)
        self.x = 0
        self.y = 0
        self.theta = np.pi / 2

    def plot(self, size_dot = 1):
        # Compute position of body of robot (as a square)
        rt = self.theta - np.pi / 2
        robo_pos = np.array([[0, 0, 0], [0, -300, 0]]) # [[0, 0, 0], [0, -d, 0]]
        trans_mat = np.array([[np.cos(rt), -np.sin(rt)], [np.sin(rt), np.cos(rt)]])
        robo_trans = np.matmul(trans_mat, robo_pos)

        plt.plot(self.x, self.y, 'bo', markersize = size_dot) # Plot sensor location
        plt.plot(robo_trans[0, :] + self.x, robo_trans[1, :] + self.y, 'b') # Plot robot body

class Wall:
    '''
    Wall class contains information of the wall state (x, y, theta).
    '''
    def __init__(self):
        self.x = 0
        self.y = 0
        self.theta = 0

    def setPosition(self, m, theta, theta_sensor):
        '''
        Inputs
            m       distance from robot to wall as measured by TOF sensor
            theta   angle of the robot with respect to wall
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

class Environment:
    '''
    Environment simulating a differential drive robot approaching a straight wall.
    '''
    def __init__(self, dt = 0.02, noise = True):
        self.robot = Robot()
        self.wall = Wall()
        self.dt = dt
        self.noise = noise

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

        # New robot position
        delta_x, delta_y, delta_theta = self.stepMotors(odoL, odoR)

        # New ToF sample
        valid_sample, r_m2, ob_m2 = self.sampleTOFSensor()

        # Robot observed and real state
        observed_dist, observed_theta = self.computeDistanceAngle(self.robot.m, ob_m2, delta_x, delta_y, delta_theta)
        real_dist, real_theta = self.computeDistanceAngle(self.robot.m, ob_m2, delta_x, delta_y, delta_theta)
        self.robot.m = ob_m2

        # Return both observed and real state
        observed_x = np.array([observed_dist, observed_theta]).reshape(-1)
        real_x     = np.array([real_dist, real_theta]).reshape(-1)
        return observed_x, real_x

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
        return delta_x, delta_y, delta_theta

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
        # Interception of wall and sensor line of action
        x_i = -(self.robot.y - self.wall.y - self.robot.x*np.tan(self.robot.theta - np.pi / 2 + self.robot.sensor_theta) + self.wall.x*np.tan(self.wall.theta))/(np.tan(self.robot.theta - np.pi / 2 + self.robot.sensor_theta) - np.tan(self.wall.theta))
        y_i = np.tan(self.robot.sensor_theta + self.robot.theta - np.pi / 2) * (x_i - self.robot.x) + self.robot.y

        # Cehck if measurement is valid
        o_abs = np.remainder(abs(self.robot.sensor_theta + self.robot.theta - np.pi / 2), 2*np.pi)
        if (o_abs >= 0) and (o_abs <= np.pi / 2):
            valid = ((x_i - self.robot.x) >= 0) and ((y_i - self.robot.y) >= 0)
        elif (o_abs >= np.pi/2) and (o_abs <= np.pi):
            valid = ((x_i - self.robot.x) <= 0) and ((y_i - self.robot.y) >= 0)
        elif (o_abs >= np.pi) and (o_abs <= 3 * np.pi / 2):
            valid = ((x_i - self.robot.x) <= 0) and ((y_i - self.robot.y) <= 0)
        elif (o_abs >= 3*np.pi/2) and (o_abs <= 2 * np.pi):
            valid = ((x_i - self.robot.x) >= 0) and ((y_i - self.robot.y) <= 0)

        if valid:
            # Distance to wall
            m = np.sqrt((self.robot.x - x_i)**2 + (self.robot.y - y_i)**2)

            # Add noise
            if self.noise:
                observed_m = m + (np.random.rand() - 0.5) * 4
            else:
                observed_m = np.copy(m)

            # Sensor out of range
            if m > 255:
                valid = False
                m = 255
        else:
            m = 255
            observed_m = 255

        return valid, m, observed_m

    def computeDistanceAngle(self, m1, m2, delta_x, delta_y, delta_theta):
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
        theta_s = self.robot.sensor_theta
        ang0 = theta_s + self.robot.theta - np.pi/2

        # Compute 4 known points
        x2 = m1 * np.cos(ang0)
        y2 = m1 * np.sin(ang0)
        x3 = delta_x
        y3 = delta_y
        x4 = x3 + m2 * np.cos(ang0 + delta_theta)
        y4 = y3 + m2 * np.sin(ang0 + delta_theta)
        if(np.abs(y2 - y4) > 0.000001 and np.abs(x2 - x4) > 0.000001): # Prevent division by 0
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
            valid_tof, m, om = self.sampleTOFSensor()
            if valid_tof:
                plt.plot([self.robot.x, self.robot.x + m * np.cos(self.robot.theta + self.robot.sensor_theta  - np.pi/2)], [self.robot.y, self.robot.y + m * np.sin(self.robot.theta + self.robot.sensor_theta - np.pi/2)], 'k')
            else:
                plt.plot([self.robot.x, self.robot.x + m * np.cos(self.robot.theta + self.robot.sensor_theta - np.pi/2)], [self.robot.y, self.robot.y + m * np.sin(self.robot.theta + self.robot.sensor_theta - np.pi/2)], 'k')

        # Pause execution
        if interactive:
            plt.draw()
            plt.pause(0.001)
            raw_input("Press [enter] to continue.")

def simulateStepByStep(env, x0, T, pol, w):
    '''
    Simulate and plot environment step by step
    '''
    env.initScenario(x0)
    env.plot(True)

    pol.reset()

    x = x0
    for t in range(T):
        u = pol.sample(w, x)[0]
        x, rx = env.step(u)
        env.plot(True)
        print('Distance ', x[0])
        print('Computed angle ', x[1] * 180 / np.pi)
        print('Wheel speed ', u)

    plt.show()

def performanceMetric(env, x0, T, pol, w, plot = False):
    '''
    Simulate T rollouts and compute:
        - Whether the robot collisioned with the wall
        - Whether the target distance was achieved
        - The time taken to reach the target distance
        - The maximum overshoot from the target distance
        - Array with the distance error throughout the simulation
        - Array with the angle error throughout the simulation
    '''
    # Varibles
    dist_errors = []
    ang_errors = []
    max_overshoot = 0
    time_to_distance = 0
    reached_target = False
    collision = False

    env.initScenario(x0)
    if plot: env.plot(False)

    pol.reset()

    x = x0
    for t in range(T):
        u = pol.sample(w, x)[0]
        x, rx = env.step(u)

        if plot: env.plot(False)

        # Compute relevant parameters when already past target distance
        if rx[0] <= pol.target[0]:
            reached_target = True

            # Maximum observed overshoot
            if pol.target[0] - rx[0] > max_overshoot:
                max_overshoot = pol.target[0] - x[0]

            # Register collision (too close to wall within tolerance)
            if rx[0] < 0.1:
                collision = True

        # Reached target distance, log angle and distance error
        if reached_target:
            ang_errors.append((pol.target[1] - rx[1])*180 / np.pi)
            dist_errors.append(pol.target[0] - rx[0])
        else:
            time_to_distance = t+1

    return collision, reached_target, time_to_distance, max_overshoot, dist_errors, ang_errors

def validatePolicy(N, T, dt, pol, hpol, verbose = True):
    '''
    Validete the performance of the policy for N different contexts
    '''
    # Performance metrics
    a_collision = []
    a_time_to_distance = []
    a_max_overshoot = []
    n_collisions = 0;
    n_converge = 0;

    # Contexts
    x0s = sampleContext(N)

    # Environment
    env = Environment(dt, noise = False)

    for i in range(N):
        x0 = x0s[i, :]
        w = hpol.mean(x0.reshape(1,-1)).T
        collision, reached_target, time_to_distance, max_overshoot, dist_errors, ang_errors = performanceMetric(env, x0, T, pol, w, plot = False)

        a_collision.append(collision)

        if verbose:
            print('-----------------------------------------------------------------')
            print('Initial state: Distance ', x0[0], ' mm Angle: ', round(x0[1]*180/np.pi))

        if collision:
            if verbose: print('COLLISION OCURRED')
            n_collisions += 1
        else:
            if verbose: print('No collision ocurred')

            if time_to_distance != -1:
                n_converge += 1
                a_time_to_distance.append(time_to_distance)
                a_max_overshoot.append(max_overshoot)

                if verbose:
                    print('Maximum overshoot (mm): ', np.round(max_overshoot, 1))
                    print('Time to target distance (s): ', time_to_distance * env.dt)
            else:
                if verbose: print('DID NOT ACHIEVE THE TARGET DISTANCE')

    print('\n\nNumber of collisions: ', n_collisions, ' , failure rate ', np.round(n_collisions / float(N), 2))

    if N - n_collisions != 0:
        print('Number reached target distance: ', n_converge, ' converge rate, ', np.round(n_converge / float(N - n_collisions), 2))

    if len(a_time_to_distance) > 0:
        print('Mean time to target distance: ', np.round(np.mean(a_time_to_distance)*env.dt, 2), ' std: ', np.round(np.std(a_time_to_distance), 2))
        print('Mean overshoot: ', np.round(np.mean(a_max_overshoot), 2), ' std: ', np.round(np.std(a_max_overshoot), 4))

def simulateResults(env, x0, T, pol, w, id = 0):
    # Get performance metrics
    collision, time_to_distance, max_overshoot, dist_errors, ang_errors = performanceMetric(env, x0, T, pol, w, plot = True)

    if id: # If multiple will be ploted
        plt.title('Weights ' + str(id) + ' trajectory')
    else:
        plt.title('Robot trajectory')
    plt.xlabel('x axis (mm)')
    plt.ylabel('y axis (mm)')

    print('\n\n------------------------------------------------------------')
    if id:
        print('Simulation results (', str(id), '):')
    else:
        print('Simulation results: ')

    print('------------------------------------------------------------')
    print('Collision: ', collision)
    if time_to_distance == -1:
        print('Target distance was not reached')
    else:
        print('Maximum overshoot (mm): ', np.round(max_overshoot, 1))
        print('Time to target distance (s): ', time_to_distance * env.dt)

        plt.figure((id-1) * 2 + 2)
        plt.subplot(211)
        if id:
            plt.title('Weights ' + str(id) + ' metrics')
        else:
            plt.title('Metrics')

        time = np.linspace(time_to_distance * env.dt, T * env.dt, len(dist_errors)).tolist()
        plt.plot(time, dist_errors)
        plt.ylabel('Distance error (mm)')

        plt.subplot(212)
        plt.plot(time, ang_errors)
        plt.ylabel('Angle error (deg)')
        plt.xlabel('Time (s)')

        if not id: plt.show()
    print('\n')

def compareWeights(env, x0, T, pol, w1, w2):
    '''
    Compare the performance of two different controller weights for the same context
    '''
    simulateResults(env, x0, T, pol, w1, id = 1)
    simulateResults(env, x0, T, pol, w2, id = 2)
    plt.show()
