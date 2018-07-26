from theano import *
import theano.tensor as T

from numpy.random import multivariate_normal as mvnrnd

from policy import Proportional
from cost import CostExpQuad

import numpy as np
import pdb

def compile_fwd_move():
    delta_l = T.dvector('delta_l')
    robot_theta = T.dvector('robot_theta')
    delta_x = T.dvector('delta_x')
    delta_y = T.dvector('delta_y')

    delta_x = delta_l * T.cos(robot_theta)
    delta_y = delta_l * T.sin(robot_theta)
    return function([delta_l, robot_theta], [delta_x, delta_y])

def compile_rot_move():
    delta_l = T.dvector('delta_l')
    delta_r = T.dvector('delta_r')
    robot_theta = T.dvector('robot_theta')
    delta_x = T.dvector('delta_x')
    delta_y = T.dvector('delta_y')
    delta_theta = T.dvector('delta_theta')
    ROBOT_LENGTH = T.constant(280.)

    wd = (delta_r - delta_l) / ROBOT_LENGTH
    R = (delta_l + delta_r) / (2 * wd)
    sum_wd_theta = wd + robot_theta
    delta_x = R * (T.sin(sum_wd_theta) - T.sin(robot_theta))
    delta_y = R * (T.cos(robot_theta) - T.cos(sum_wd_theta))
    delta_theta = wd
    return function([delta_l, delta_r, robot_theta], [delta_x, delta_y, delta_theta])

def compile_wall_position():
    robot_x = T.dvector('robot_x')
    robot_y = T.dvector('robot_y')
    delta_x = T.dvector('delta_x')
    delta_y = T.dvector('delta_y')
    wall_x = T.dvector('wall_x')
    wall_y = T.dvector('wall_y')
    wall_theta = T.dvector('wall_theta')
    tan_wall_theta = T.tan(wall_theta)
    x_i = (delta_y*robot_x + delta_x * (wall_y - robot_y - wall_x*tan_wall_theta))/(delta_y - delta_x*tan_wall_theta)
    y_i = robot_y + (delta_y*(x_i - robot_x))/delta_x
    x_delta_x = robot_x + delta_x
    return function([robot_x, robot_y, delta_x, delta_y, wall_x, wall_y, wall_theta], [x_i, y_i, x_delta_x])

def compile_sample_tof():
    robot_theta = T.dvector('robot_theta')
    robot_x = T.dvector('robot_x')
    wall_y = T.dvector('wall_y')
    robot_y = T.dvector('robot_y')
    wall_x = T.dvector('wall_x')
    wall_theta = T.dvector('wall_theta')
    tan_wall_theta = T.tan(wall_theta)
    sensor_theta = T.constant(18 / (180.0) * np.pi)
    robot_angle = robot_theta + (sensor_theta - np.pi/2)
    tan_robot_theta = T.tan(robot_angle)
    tan_wall_theta = T.tan(wall_theta)
    xi = (robot_y - wall_y - robot_x*tan_robot_theta + wall_x*tan_wall_theta)/(tan_wall_theta - tan_robot_theta)
    xi_minus_x = xi - robot_x
    yi_minus_y = tan_robot_theta * (xi - robot_x)
    ms = T.sqrt(xi_minus_x ** 2 + yi_minus_y ** 2)
    return function([robot_x, robot_y, robot_theta, wall_x, wall_y, wall_theta], [xi_minus_x, yi_minus_y, ms])

def compile_points():
    robot_theta = T.dvector('robot_theta')
    delta_x = T.dvector('delta_x')
    delta_y = T.dvector('delta_y')
    delta_theta = T.dvector('delta_theta')
    m1 = T.dvector('m1')
    m2 = T.dvector('m2')

    sensor_theta = T.constant(18 / (180.0) * np.pi)
    ang0 = (sensor_theta + robot_theta) - np.pi/2
    ang1 = ang0 + delta_theta

    x2 = m1 * T.cos(ang0)
    y2 = m1 * T.sin(ang0)
    x4 = delta_x + m2 * T.cos(ang1)
    y4 = delta_y + m2 * T.sin(ang1)

    diff_x2_x4 = x2 - x4
    diff_y2_y4 = y2 - y4

    return function([robot_theta, delta_x, delta_y, delta_theta, m1, m2], [x4, y4, diff_x2_x4, diff_y2_y4])

def compile_triangle_lengths():
    x4 = T.dvector('x4')
    y4 = T.dvector('y4')
    delta_x = T.dvector('delta_x')
    delta_y = T.dvector('delta_y')
    diff_y2_y4 = T.dvector('diff_y2_y4')
    diff_x2_x4 = T.dvector('diff_x2_x4')
    robot_theta = T.dvector('robot_theta')
    a1 = diff_y2_y4 / diff_x2_x4
    a2 = T.tan(robot_theta - np.pi/2)
    xi = (delta_y - y4 + a1*x4 - delta_x*a2)/(a1 - a2)
    yi = delta_y + a2 * (xi - delta_x)

    a3 = -1.0 / a2
    xp = (delta_y - y4 - a2*delta_x + a3*x4)/(a3 - a2)
    yp = delta_y + a2 * (xp - delta_x)

    a_2 = (x4 - xi) ** 2 + (y4 - yi) ** 2
    b_2 = (x4 - xp) ** 2 + (y4 - yp) ** 2
    c_2 = (xi - xp) ** 2 + (yi - yp) ** 2

    d_wall = T.sqrt((delta_x - xi) ** 2 + (delta_y - yi) ** 2)

    return function([x4, y4, delta_x, delta_y, diff_x2_x4, diff_y2_y4, robot_theta],[xi, xp, a_2, b_2, c_2, d_wall])

def compile_arrcos():
    a_2 = T.dvector('a_2')
    b_2 = T.dvector('b_2')
    c_2 = T.dvector('c_2')
    theta_wall = T.arccos((a_2 + b_2 - c_2)/(2 * T.sqrt(a_2) * T.sqrt(b_2)))
    return function([a_2, b_2, c_2], theta_wall)


class OptMod:
    def __init__(self, dt, pol, cost):
        # Precompile theano functions
        self.fcn_fwd_move = compile_fwd_move()
        self.fcn_rot_move = compile_rot_move()
        self.fcn_wall_position = compile_wall_position()
        self.fcn_sample_tof = compile_sample_tof()
        self.fcn_points = compile_points()
        self.fcn_triangle_lengths = compile_triangle_lengths()
        self.fcn_arrcos = compile_arrcos()

        self.dt = dt
        self.pol = pol
        self.cost = cost

    def simulateRobot(self, M, H, x0, w):
        fcn_fwd_move = self.fcn_fwd_move
        fcn_rot_move = self.fcn_rot_move
        fcn_wall_position = self.fcn_wall_position
        fcn_sample_tof = self.fcn_sample_tof
        fcn_points = self.fcn_points
        fcn_triangle_lengths = self.fcn_triangle_lengths
        fcn_arrcos = self.fcn_arrcos

        dt = self.dt
        pol = self.pol
        cost = self.cost

        # Frequently used numpy operations
        empty = np.empty
        invert = np.invert
        np_or = np.logical_or
        np_and = np.logical_and
        remainder = np.remainder
        abs = np.abs
        any = np.any
        copy = np.copy

        # Initialize robot position
        robot_x = np.zeros(M)
        robot_y = np.zeros(M)
        robot_theta = np.ones(M) * np.pi/2
        robot_m = x0[:, 0].reshape(-1)
        m2 = empty(M)

        # Initialize wall
        theta_sensor = 18 / (180.0) * np.pi
        wall_x = robot_m * np.cos(theta_sensor)
        wall_y = robot_m * np.sin(theta_sensor)
        wall_theta = np.pi/2 + x0[:, 1].reshape(-1)

        delta_x = empty(M)
        delta_y = empty(M)
        delta_theta = empty(M)

        x = x0
        arr1 = empty(M)
        arr2 = empty(M)
        arr3 = empty(M)
        arr4 = empty(M)

        indx1 = empty((M,), dtype = 'bool')
        indx2 = empty((M,), dtype = 'bool')
        valid = empty((M,), dtype = 'bool')
        pos_xi_minus_x = empty((M,), dtype = 'bool')
        neg_xi_minus_x = empty((M,), dtype = 'bool')
        pos_yi_minus_y = empty((M,), dtype = 'bool')

        # Rewards
        R = np.zeros((M,1))
        for t in xrange(H):
            # ******************************************************************************
            # COMPUTE SIMULATOR STEP
            # ******************************************************************************
            # ------------------------------------------------------------------------------
            # Get control action...
            x = pol.sampleMat(w, x) * dt
            arr1 = x[:, 0].reshape(-1)
            arr2 = x[:, 1].reshape(-1)

            # ------------------------------------------------------------------------------
            # Robot step when robot theta = 90...
            indx1 = arr1 == arr2
            if(any(indx1)):
                delta_x[indx1], delta_y[indx1] = fcn_fwd_move(arr1[indx1], robot_theta[indx1])
                delta_theta[indx1] = 0

            # ------------------------------------------------------------------------------
            # Robot step when robot theta != 90...
            indx1 = np.invert(indx1) # Reuse...
            delta_x[indx1], delta_y[indx1], delta_theta[indx1] = fcn_rot_move(arr1[indx1], arr2[indx1], robot_theta[indx1])

            # ------------------------------------------------------------------------------
            # Detect collision.
            # [xi, yi, x_delta_x]
            [arr1, arr2, arr3] = fcn_wall_position(robot_x, robot_y, delta_x, delta_y, wall_x, wall_y, wall_theta)

            indx1 = delta_x >= 0
            pos_xi = arr1[indx1]
            valid[indx1] = np_or(pos_xi < robot_x[indx1], pos_xi > arr3[indx1])
            pos_xi = None

            indx1 = invert(indx1)
            pos_xi = arr1[indx1]
            valid[indx1] = np_or(pos_xi < arr3[indx1], pos_xi > robot_x[indx1])
            pos_xi = None

            # ------------------------------------------------------------------------------
            # Update robot position.
            robot_x[valid] += delta_x[valid]
            robot_y[valid] += delta_y[valid]
            valid = invert(valid)
            robot_x[valid] = arr1[valid] - 0.01
            robot_y[valid] = arr2[valid] - 0.01
            robot_theta += delta_theta

            # ------------------------------------------------------------------------------
            # Sample TOF sensor
            # [xi_minus_x, yi_minus_y, ms])
            [arr1, arr2, arr3] = fcn_sample_tof(robot_x, robot_y, robot_theta, wall_x, wall_y, wall_theta)

            pos_xi_minus_x = arr1 >= 0
            pos_yi_minus_y = arr2 >= 0
            neg_xi_minus_x = invert(pos_xi_minus_x)
            arr1 = remainder(robot_theta + (1000*np.pi + theta_sensor - np.pi/2), 2*np.pi) #theta_abs

            indx1 = arr1 <= np.pi / 2
            valid[indx1] = np_and(pos_xi_minus_x[indx1], pos_yi_minus_y[indx1]) # under pi /2

            indx2 = arr1 <= np.pi # over pi/2
            indx1 = np_and(invert(indx1), indx2) # c2
            valid[indx1] = np_and(neg_xi_minus_x[indx1], pos_yi_minus_y[indx1]) # c2

            indx2 = invert(indx2)
            indx1 = arr1 <= (1.5 * np.pi)
            indx2 = np_and(indx2, indx1)
            pos_yi_minus_y = invert(pos_yi_minus_y)
            valid[indx2] = np_and(neg_xi_minus_x[indx2], pos_yi_minus_y[indx2])

            indx1 = invert(indx1)
            valid[indx1] = np_and(pos_xi_minus_x[indx1], pos_yi_minus_y[indx1])

            m2[valid] = arr3[valid]
            valid = invert(valid)
            m2[np_or(valid, m2 > 255)] = 255

            # ------------------------------------------------------------------------------
            # Compute three points to triangulate robot position
            # [x4, y4, diff_x2_x4, diff_y2_y4]
            arr1, arr2, arr3, arr4 = fcn_points(robot_theta, delta_x, delta_y, delta_theta, robot_m, m2)

            # ------------------------------------------------------------------------------
            # Compute triangle lengths
            indx1 = np_and(abs(arr3) > 0.000001, abs(arr4) > 0.000001)
            xi, xp, a_2, b_2, c_2, d_wall = fcn_triangle_lengths(arr1[indx1], arr2[indx1], delta_x[indx1], delta_y[indx1], arr3[indx1], arr4[indx1], robot_theta[indx1])

            # ------------------------------------------------------------------------------
            # Compute angle
            sIndx1 = np_and(np_and(a_2 > 1e-9, b_2 > 1e-9), c_2 > 1e-9)
            theta_out = fcn_arrcos(a_2[sIndx1], b_2[sIndx1], c_2[sIndx1])

            # ------------------------------------------------------------------------------
            # Update state
            theta_out[xp[sIndx1] > xi[sIndx1]] *= -1
            a_2[sIndx1] = theta_out
            a_2[invert(sIndx1)] = 0

            arr2[indx1] = d_wall
            arr3[indx1] = a_2
            indx1 = invert(indx1)
            arr2[indx1] = 0
            arr3[indx1] = 0

            robot_m = copy(m2)

            # ------------------------------------------------------------------------------
            # Free memory
            statesIndx1 = None
            xi = None
            xp = None
            a_2 = None
            b_2 = None
            c_2 = None
            d_wall = None
            theta_wall = None
            theta_out = None

            x = np.concatenate([arr2.reshape(-1,1), arr3.reshape(-1,1)], 1)
            R += cost.sampleMat(x)
        return R

if __name__ == '__main__':
    M = 10000
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

    x0[0, :] = np.array([240, np.pi/3])
    w[:, 0]  = np.array([-12.44, 147.56, -4.77, -106.8]).reshape(-1)
    x0[1, :] = np.array([240, np.pi/3])
    w[:, 1]  = np.array([-12.44, 147.56, -4.77, -106.8]).reshape(-1)

    print 'Initializing....'
    mod = OptMod(dt, pol, cost)
    print 'Done...'
    mod.simulateRobot(M, H, x0, w)
