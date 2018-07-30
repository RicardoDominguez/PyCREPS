import numpy as np
import pdb
from numpy.random import multivariate_normal as mvnrnd
from policy import Proportional
from cost import CostExpQuad

class OptMod:
    def __init__(self, dt, pol, cost, noise = False):
        self.dt = dt
        self.pol = pol
        self.cost = cost
        self.noise = noise

    def simulateRobot(self, M, H, x0, w):
        '''
            x0 - (M x 2)
            w  - (4 x N)
        '''
        dt = self.dt
        pol = self.pol
        pol.reset()
        cost = self.cost

        TICKS_TO_MM = 12.0
        ROBOT_LENGTH = 280.0 # wheel to wheel in mm

        # Store numpy functions locally
        zeros = np.zeros
        ones = np.ones
        cos = np.cos
        sin = np.sin
        tan = np.tan
        empty = np.empty
        invert = np.invert
        np_or = np.logical_or
        np_and = np.logical_and
        remainder = np.remainder
        sqrt = np.sqrt
        power = np.power
        abs = np.abs
        arccos = np.arccos

        # Initialization variables
        pi = np.pi
        pi_2 = pi / 2
        pi2 = pi * 2
        large_pi = 1000*pi
        theta_sensor = 18 / (180.0) * pi

        # Initialize robot position
        robot_x = zeros((M,))
        robot_y = zeros((M,))
        robot_theta = ones((M,)) * pi_2
        robot_m = x0[:, 0].reshape(-1)

        # Initialize wall
        wall_x = robot_m * cos(theta_sensor)
        wall_y = robot_m * sin(theta_sensor)
        wall_theta = pi_2 + x0[:, 1].reshape(-1)

        delta_x = empty((M,))
        delta_y = empty((M,))
        delta_theta = empty((M,))

        x = x0
        arr1 = empty((M,))
        arr2 = empty((M,))
        arr3 = empty((M,))
        arr4 = empty((M,))
        arr5 = empty((M,))
        arr5 = empty((M,))
        indx1 = empty((M,), dtype = 'bool')
        indx2 = empty((M,), dtype = 'bool')
        indx3 = empty((M,), dtype = 'bool')
        indx4 = empty((M,), dtype = 'bool')
        indx5 = empty((M,), dtype = 'bool')
        indx6 = empty((M,), dtype = 'bool')
        R = np.zeros((M,1))
        for t in xrange(H):
            # Control action to expected odometry ----------------------------------
            # u (M x 2) - not used afterwards
            # delta_l is arr1 - (M x 1) - used until model step
            # delta_r is arr2 - (M x 1) - used until model step
            u = pol.sampleMat(w, x) * dt
            arr1 = u[:, 0].reshape(-1)
            arr2 = u[:, 1].reshape(-1)
            u = None # free memory for u

            # Robot model ----------------------------------------------------------
            # No rotation
            indx1 = arr1 == arr2
            if np.any(indx1):
                delta_l_eq = arr1[indx1]
                robot_theta_eq = robot_theta[indx1]
                delta_x[indx1] = delta_l_eq * cos(robot_theta_eq)
                delta_y[indx1] = delta_l_eq * sin(robot_theta_eq)
                delta_theta[indx1] = 0
                delta_l_eq = None # free memory
                robot_theta_eq = None
            # Rotation
            indx1 = invert(indx1)
            arrIndx11 = arr1[indx1]
            arrIndx12 = arr2[indx1]
            delta_theta[indx1] = (arrIndx12 - arrIndx11) / ROBOT_LENGTH # wd
            arrIndx13 = (arrIndx11 + arrIndx12) / (2 * delta_theta[indx1]) # arrIndx11, arrIndx12 rewritten
            arrIndx11 = robot_theta[indx1]
            arrIndx12 = delta_theta[indx1] + arrIndx11
            delta_x[indx1] = arrIndx13 * (sin(arrIndx12) - sin(arrIndx11))
            delta_y[indx1] = arrIndx13 * (cos(arrIndx11) - cos(arrIndx12))
            arrIndx13 = None # free memory
            arrIndx11 = None
            arrIndx12 = None

            # Detect collision -----------------------------------------------------
            arr1 = tan(wall_theta)
            arr2 = (delta_y*robot_x + delta_x * (wall_y - robot_y - wall_x*arr1))/(delta_y - delta_x*arr1) # xi
            arr3 = robot_y + (delta_y*(arr2 - robot_x))/delta_x # a1

            # Check if point of collision is within robot displacement
            arr4 = robot_x + delta_x # x_delta_x
            indx1 = delta_x >= 0 # pos_delta_x
            arrIndx11 = arr2[indx1]  # pos_xi
            indx2[indx1] = np_or(arrIndx11 < robot_x[indx1], arrIndx11 > arr4[indx1]) # arr8
            arrIndx11 = None

            indx1 = invert(indx1) # neg_delta_x
            arrIndx11 = arr2[indx1] # neg_xi
            indx2[indx1] = np_or(arrIndx11 < arr4[indx1], arrIndx11 > robot_x[indx1])
            arrIndx11 = None

            robot_x[indx2] += delta_x[indx2]
            robot_y[indx2] += delta_y[indx2]
            indx2 = invert(indx2) # Not arr8
            robot_x[indx2] = arr2[indx2] - 0.01
            robot_y[indx2] = arr3[indx2] - 0.01
            robot_theta += delta_theta

            # Sample TOF -----------------------------------------------------------
            arr4 = robot_theta + (theta_sensor - pi_2) # robot angle
            arr5 = tan(arr4) # tan robot angle
            arr2 = (robot_y - wall_y - robot_x*arr5 + wall_x*arr1)/(arr1 - arr5)
            arr3 = arr5 * (arr2 - robot_x) + robot_y
            arr4 = remainder(arr4 + large_pi, pi2) # o abs
            arr2 = arr2 - robot_x # xi minux x
            arr3 = arr3 - robot_y # a1 minus y

            indx1 = arr2 >= 0 # pos xi minus x
            indx2 = arr3 >= 0 # pos a1 minus y
            indx3 = invert(indx1) # neg xi minus x
            indx4 = arr4 <= pi_2 # under pi/2
            indx5[indx4] = np_and(indx1[indx4], indx2[indx4]) # valid_
            indx4 = invert(indx4) # over pi /2
            indx6 = arr4 <= pi # under pi
            arr4 = arr4 <= (1.5 * pi) # under 3pi/2
            indx4 = np_and(indx4, indx6) # arr7
            indx6 = invert(indx6) # over pi
            indx5[indx4] = np_and(indx3[indx4], indx2[indx4]) # valid_
            indx2 = invert(indx2) # arr5
            indx6 = np_and(indx6, arr4) #c3
            arr4 = invert(arr4) # over 3pi/2
            indx5[indx6] = np_and(indx3[indx6], indx2[indx6])
            indx5[arr4] = np_and(indx1[arr4], indx2[arr4])

            arr1[indx5] = sqrt(power(arr2[indx5], 2) + power(arr3[indx5],2)) # m2
            if self.noise: arr1[indx5] += (np.random.rand(np.sum(indx5)) - 0.5) * 4
            #arr1[indx5] = np.round(arr1[indx5])
            indx5 = invert(indx5) # not valid
            arr1[np_or(indx5, arr1 > 255)] = 255

            # Calculate angles -----------------------------------------------------
            arr2 = (theta_sensor + robot_theta) - pi_2 # ang 0
            x2 = robot_m * cos(arr2)
            y2 = robot_m * sin(arr2)
            arr2 = arr2 + delta_theta # ang 1
            x4 = delta_x + arr1 * np.cos(arr2)
            y4 = delta_y + arr1 * np.sin(arr2)
            robot_m = np.copy(arr1)

            arr1 = y2 - y4 # diff y2 y4
            arr2 = x2 - x4 # diff x2 x4
            indx1 = np_and(abs(arr1) > 0.000001, abs(arr2 > 0.000001)) # no division 0

            dx_no0 = delta_x[indx1]
            dy_no0 = delta_y[indx1]
            y4_no0 = y4[indx1]
            x4_no0 = x4[indx1]

            a1 = arr1[indx1] / arr2[indx1]
            a2 = tan(robot_theta[indx1] - pi_2)

            xi = (dy_no0 - y4_no0 + a1*x4_no0 - dx_no0*a2)/(a1 - a2)
            a1 = dy_no0 + a2 * (xi - dx_no0) # y i
            arr1[indx1] = sqrt(power(dx_no0 - xi, 2) + power(dy_no0 - a1, 2)) # d wall all

            a3 = empty(a2.shape)
            sz = abs(a2) < 1e-6
            a3[sz] = -1 / 0.0000001
            sz = invert(sz)
            a3[sz] = -1 / a2[sz]
            sz = None
            xp = (dy_no0 - y4_no0 - a2*dx_no0 + a3*x4_no0)/(a3 - a2)
            a2 = dy_no0 + a2 * (xp - dx_no0) # y p
            dx_no0 = power(x4_no0 - xi, 2) + power(y4_no0 - a1, 2) # a 2
            dy_no0 = power(x4_no0 - xp, 2) + power(y4_no0 - a2, 2) # b 2
            y4_no0 = power(xi - xp, 2) + power(a1 - a2, 2) # c2

            a2 = None

            indxSp1 = np_and(np_and(dx_no0 > 1e-9, dy_no0 > 1e-9) , y4_no0 > 1e-9) # ok triang
            a_2_ok = dx_no0[indxSp1]
            b_2_ok = dy_no0[indxSp1]

            a_2_ok = arccos((a_2_ok + b_2_ok - y4_no0[indxSp1])/(2 * sqrt(a_2_ok) * sqrt(b_2_ok))) # theta wall

            a_2_ok[xp[indxSp1] > xi[indxSp1]] *= -1
            a1[indxSp1] = a_2_ok
            a1[invert(indxSp1)] = 0

            arr2[indx1] = a1 # theta all
            indx1 = invert(indx1) # div0
            arr1[indx1] = 0
            arr2[indx1] = 0

            a1 = None
            b_2_ok = None
            y4_no0 = None
            x4_no0 = None
            d_wall = None
            dx_no0 = None
            dy_no0 = None
            a_2_ok = None
            indxSp1 = None

            x = np.concatenate([arr1.reshape(-1,1), arr2.reshape(-1,1)], 1)
            #pdb.set_trace()
            R += cost.sampleMat(x)
            #print x[0, :]
            #print R[0]
        return R

if __name__ == '__main__':
    M = 100
    H = 1000
    dt = 0.1

    x_mu = np.array([180, np.pi/4]).reshape(-1)
    x_sigma = np.eye(x_mu.shape[0]) * [50, np.pi/10]
    x0 = mvnrnd(x_mu, x_sigma, M)

    hpol_mu =  np.array([-2, 100, 2, -100]).reshape(-1)
    hpol_sigma = np.eye(hpol_mu.shape[0]) * [20, 200, 200, 20]
    w = mvnrnd(hpol_mu, hpol_sigma, M).T

    target = np.array([10, 0]).reshape(-1)
    offset = np.array([150, 150]).reshape(-1)
    pol = Proportional(-324, 324, target, offset)

    Kcost = np.array([0.005, 100]).reshape(1, -1)
    target = np.array([10, 0]).reshape(1, -1)
    cost = CostExpQuad(Kcost, target)

    x0[0, :] = np.array([130, np.pi/3])
    w[:, 0]  = np.array([-6.36, 95.62, 18.9, -104.17]).reshape(-1)
    x0[1, :] = np.array([240, np.pi/3])
    w[:, 1]  = np.array([-12.44, 147.56, -4.77, -106.8]).reshape(-1)

    print 'Initializing....'
    mod = OptMod(dt, pol, cost)
    print 'Done...'
    mod.simulateRobot(M, H, x0, w)
