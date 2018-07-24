def simulateRobot(M, H, x0, dt, w, cost, pol):
    TICKS_TO_MM = 12.0
    ROBOT_LENGTH = 280.0 # wheel to wheel in mm

    # Store numpy functions locally
    zeros = np.zeros
    ones = np.ones
    cos = np.cos
    sin = np.sin
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
    robot_x = zeros((M, 1))
    robot_y = zeros((M, 1))
    robot_theta = ones((M, 1)) * pi_2
    robot_m = x0[:, 0]

    # Initialize wall
    wall_x = robot_m * cos(theta_sensor)
    wall_y = robot_m * sin(theta_sensor)
    wall_theta = pi_2 + x0[:, 1]

    for t in xrange(H):
        # Control action to expected odometry
        u = pol.sample(w, x) * dt
        delta_l = u[:, 0]
        delta_r = u[:, 1]

        # Variables used
        delta_x = empty((M, 1))
        delta_y = empty((M, 1))
        delta_theta = empty((M, 1))
        equal_disp = delta_l == delta_r
        not_eq_disp = invert(equal_disp)

        # Robot moved forward (no rotation)
        delta_l_eq = delta_l[equal_disp]
        robot_theta_eq = robot_theta[equal_disp]
        delta_x[equal_disp] = delta_l_eq * cos(robot_theta)
        delta_y[equal_disp] = delta_l_eq * sin(robot_theta)
        delta_theta[equal_disp] = 0

        delta_l_neq = delta_l[not_eq_disp]
        delta_r_neq = delta_r[not_eq_disp]
        robot_theta_neq = robot_theta[not_eq_disp]
        wd = delta_r_neq - delta_l_neq / ROBOT_LENGTH
        R = (delta_l_neq + delta_r_neq) / (2 * wd)
        sum_wd_theta = wd + robot_theta_neq
        delta_x[not_eq_disp] = R * (sin(sum_wd_theta) - sin(robot_theta_neq))
        delta_y[not_eq_disp] = R * (cos(robot_theta_neq) - cos(sum_wd_theta))
        delta_theta[not_eq_disp] = wd

        # Detect collision
        tan_wall_theta = tan(wall_theta)
        x_i = (delta_y*robot_x + delta_x * (wall_y - robot_y - wall_x*tan_wall_theta))/(delta_y - delta_x*tan_wall_theta)
        y_i = robot_y + (delta_y*(x_i - robot_x))/delta_x

        # Check if point of collision is within robot displacement
        x_delta_x = robot_x + delta_x
        pos_delta_x = delta_x >= 0
        pos_xi = delta_x[pos_delta_x]
        valid[pos_delta_x] = np_or(pos_xi < robot_x[pos_delta_x], pos_xi > x_delta_x[pos_delta_x])

        neg_delta_x = invert(pos_delta_x)
        neg_xi = delta_x[neg_delta_x]
        valid[neg_delta_x] = np_or(neg_xi < x_delta_x[neg_delta_x], neg_xi > robot_x[neg_delta_x])

        not_valid = invert(valid)

        robot_x[valid] += delta_x[valid]
        robot_y[valid] += delta_y[valid]
        robot_x[not_valid] = x_i[not_valid] - 0.01
        robot_y[not_valid] = y_i[not_valid] - 0.01
        robot_theta += delta_theta

        # Sample TOF
        robot_angle = robot_theta + (sensor_theta - pi_2)
        tan_robot_theta = tan(robot_angle)
        tan_wall_theta = tan(wall_theta)
        x_i = (robot_y - wall_y - robot_x*tan_robot_theta + wall_x*tan_wall_theta)/(tan_wall_theta - tan_robot_theta)
        y_i = tan_robot_theta * (x_i - robot_x) + robot_y

        o_abs = remainder(robot_angle + large_pi, pi2)
        xi_minus_x = x_i - robot_x
        yi_minus_y = y_i - robot_y
        pos_xi_minus_x = xi_minus_x >= 0
        pos_yi_minus_y = yi_minus_y >= 0
        neg_xi_minus_x = invert(pos_xi_minus_x)
        neg_yi_minus_y = invert(pos_yi_minus_y)

        over_pi_2 = o_abs > pi_2
        under_pi_2 = invert(over_pi_2)
        over_pi = o_abs > pi
        under_pi = invert(over_pi)
        over_3pi_2 = o_abs > (1.5 * pi)
        under_3pi_2 = invert(over_3pi_2)

        c2 = np_and(over_pi_2, under_pi)
        c3 = np_and(over_pi, under_3pi_2)

        valid[under_pi_2] = np_and(pos_xi_minus_x[under_pi_2], pos_yi_minus_y[under_pi_2])
        valid[c2] = np_and(neg_xi_minus_x[c2], pos_yi_minus_y[c2])
        valid[c3] = np_and(neg_xi_minus_x[c3], neg_yi_minus_y[c3])
        valid[over_3pi_2] = np_and(pos_xi_minus_x[over_3pi_2], neg_yi_minus_y[over_3pi_2])
        not_valid = invert(valid)

        m[valid] = sqrt(power(xi_minus_x, 2) + power(yi_minus_y,2))
        m[np_and(not_valid, m > 255)] = 255

        # # WARNING:  I SHOULD REUSE
        ang0 = (sensor_theta + robot_theta) - pi_2
        ang1 = ang0 + delta_theta

        x2 = m1 * cos(ang0)
        y2 = m1 * sin(ang0)
        x4 = delta_x + m2 * np.cos(ang1)
        y4 = delta_y + m2 * np.sin(ang1)

        diff_y2_y4 = y2 - y4
        diff_x2_x4 = x2 - x4
        no_div_0 = np_and(abs(diff_y2_y4) > 0.000001, abs(diff_x2_x4 > 0.000001))

        dx_no0 = delta_x[no_div_0]
        dy_no0 = delta_y[no_div_0]
        y4_no0 = y4[no_div_0]
        x4_no0 = x4[no_div_0]

        a1 = diff_y2_y4[no_div_0] / diff_x2_x4[no_div_0]
        a2 = tan(robot_theta[no_div_0] - pi_2)
        xi = (dy_no0 - y4_no0 + a1*x4_no0 - dx_no0*a2)/(a1 - a2)
        yi = dy_no0 + a2 * (xi - dx_no0)

        a3 = -1 / a2
        xp = (dy_no0 - y4_no0 - a2*dx_no0 + a3*x4_no0)/(a3 - a2)
        yp = dy_no0 + a2 * (xp - dx_no0)

        a_2 = power(x4_no0 - xi, 2) + power(y4_no0 - yi, 2)
        b_2 = power(x4_no0 - xp, 2) + power(y4_no0 - yp, 2)
        c_2 = power(xi - xp, 2) + power(yi - yp, 2)

        ok_triang = np_and(a_2 > 1e-9, b_2 > 1e-9, c_2 > 1e-9)
        a_2_ok = a_2[ok_triang]
        b_2_ok = b_2[ok_triang]
        theta_wall = arccos((a_2_ok + b_2_ok - c_2[ok_triang])/(2 * sqrt(a_2_ok) * sqrt(b_2_ok)))
        indx_inv_theta = xp[ok_triang] > xi[ok_triang]
        theta_wall[xp[ok_triang] > xi[ok_triang]] *= -1
        theta_wall[inverse(ok_triang)] = 0
        d_wall = sqrt(power(dx_no0 - xi, 2) + power(dy_no0 - yi, 2))

        d_wall_all[no_div_0] = d_wall
        theta_wall_all[no_div_0] = theta_wall
        div_0 = inverse(no_div_0)
        d_wall_all[div_0] = 0
        theta_wall_all[div_0] = 0

        robot_m = m2

    x = np.concatenate([d_wall_all, theta_wall_all], 1)
