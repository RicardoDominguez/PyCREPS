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
    arr1 = x0[:, 0]

    # Initialize wall
    wall_x = robot_m * cos(theta_sensor)
    wall_y = robot_m * sin(theta_sensor)
    wall_theta = pi_2 + x0[:, 1]

    delta_x = empty((M, 1))
    delta_y = empty((M, 1))
    delta_theta = empty((M, 1))
    u = empty((M, 2))

    #
    arr1 = empty((M, 1))
    arr2 = empty((M, 1))
    arr3 = empty((M, 1))
    arr4 = empty((M, 1))
    arr5 = empty((M, 1))
    arr5 = empty((M, 1))
    arr6 = empty((M, 1))
    arr7 = empty((M, 1))
    arr8 = empty((M, 1))
    arr9 = empty((M, 1))
    for t in xrange(H):
        # Control action to expected odometry ----------------------------------
        # u (M x 2) - not used afterwards
        # delta_l is arr1 - (M x 1) - used until model step
        # delta_r is arr2 - (M x 1) - used until model step
        u = pol.sample(w, x) * dt
        arr1 = u[:, 0]
        arr2 = u[:, 1]
        u = None # free memory for u

        # Robot model ----------------------------------------------------------
        # No rotation
        arr5 = arr1 == arr2
        if np.any(arr5):
            delta_l_e = arr1[arr5]
            robot_theta_eq = robot_theta[arr5]
            delta_x[arr5] = delta_l_eq * cos(robot_theta_eq)
            delta_y[arr5] = delta_l_eq * sin(robot_theta_eq)
            delta_theta[arr5] = 0
            delta_l_eq = None # free memory
            robot_theta_eq = None
        # Rotation
        arr5 = inverse(arr5)
        arrIndx11 = arr1[arr5]
        arrIndx12 = arr2[arr5]
        delta_theta[arr5] = arrIndx12 - arrIndx11 / ROBOT_LENGTH
        arrIndx13 = (arrIndx11 + arrIndx12) / (2 * wd) # arrIndx11, arrIndx12 rewritten
        arrIndx11 = robot_theta[arr5]
        arrIndx12 = delta_theta[arr5] + arrIndx11
        delta_x[arr5] = arrIndx13 * (sin(arrIndx12) - sin(arrIndx11))
        delta_y[arr5] = arrIndx13 * (cos(arrIndx11) - cos(arrIndx12))
        arrIndx13 = None # free memory
        arrIndx11 = None
        arrIndx12 = None

        # Detect collision -----------------------------------------------------
        arr1 = tan(wall_theta)
        arr2 = (delta_y*robot_x + delta_x * (wall_y - robot_y - wall_x*arr1))/(delta_y - delta_x*arr1) # xi
        arr3 = robot_y + (delta_y*(arr2 - robot_x))/delta_x # a1

        # Check if point of collision is within robot displacement
        arr4 = robot_x + delta_x # x_delta_x
        arr5 = delta_x >= 0 # pos_delta_x
        arrIndx11 = delta_x[arr5]  # pos_xi
        arr6[arr5] = np_or(arrIndx11 < robot_x[arr5], arrIndx11 > arr4[arr5]) # arr8
        arrIndx11 = None

        arr5 = invert(arr5) # neg_delta_x
        arrIndx11 = delta_x[arr5] # neg_xi
        arr6[arr5] = np_or(arrIndx11 < arr4[arr5], arrIndx11 > robot_x[arr5])
        arrIndx11 = None

        robot_x[arr6] += delta_x[arr6]
        robot_y[arr6] += delta_y[arr6]
        arr6 = invert(arr6) # Not arr8
        robot_x[arr6] = arr2[arr6] - 0.01
        robot_y[arr6] = arr3[arr6] - 0.01
        robot_theta += delta_theta

        # Sample TOF -----------------------------------------------------------
        arr4 = robot_theta + (sensor_theta - pi_2) # robot angle
        arr5 = tan(arr4) # tan robot angle
        arr1 = tan(wall_theta)
        arr2 = (robot_y - wall_y - robot_x*arr5 + wall_x*arr1)/(arr1 - arr5)
        arr3 = arr5 * (arr2 - robot_x) + robot_y

        arr4 = remainder(arr4 + large_pi, pi2) # o abs
        arr2 = arr2 - robot_x # xi minux x
        arr3 = arr3 - robot_y # a1 minus y

        arr1 = arr2 >= 0 # pos xi minus x
        arr5 = arr3 >= 0 # pos a1 minus y
        arr6 = invert(arr1) # neg xi minus x
        arr7 = arr4 <= pi_2 # under pi/2
        arr8[arr7] = np_and(arr1[arr7], arr5[arr7]) # valid_
        arr7 = invert(arr7) # over pi /2
        arr9 = arr4 <= pi # under pi
        arr4 = arr4 <= (1.5 * pi) # under 3pi/2
        arr7 = np_and(arr7, arr9) # arr7
        arr9 = inverse(arr9) # over pi
        arr8[arr7] = np_and(arr6[arr7], arr5[arr7]) # valid_
        arr5 = invert(arr5) # arr5
        arr9 = np_and(arr9, arr4) #c3
        arr4 = inverse(arr4) # over 3pi/2
        arr8[arr9] = np_and(arr6[arr9], arr5[arr9])
        arr8[arr4] = np_and(arr1[arr4], arr5[arr4])

        arr1[arr8] = sqrt(power(arr2, 2) + power(arr3,2)) # m2
        arr8 = invert(arr8) # not valid
        arr1[np_and(arr8, arr1 > 255)] = 255

        arr9 = None
        arr7 = None
        arr6 = None
        arr4 = None
        arr5 = None
        arr8 = None

        # Calculate angles -----------------------------------------------------
        arr2 = (sensor_theta + robot_theta) - pi_2 # ang 0
        x2 = robot_m * cos(arr2)
        y2 = robot_m * sin(arr2)
        arr2 = arr2 + delta_theta # ang 1
        x4 = delta_x + arr1 * np.cos(arr2)
        y4 = delta_y + arr1 * np.sin(arr2)
        robot_m = arr1

        arr1 = y2 - y4 # diff y2 y4
        arr2 = x2 - x4 # diff x2 x4
        arr3 = np_and(abs(arr1) > 0.000001, abs(arr2 > 0.000001)) # no division 0

        dx_no0 = delta_x[arr3]
        dy_no0 = delta_y[arr3]
        y4_no0 = y4[arr3]
        x4_no0 = x4[arr3]

        a1 = arr1[arr3] / arr2[arr3]
        a2 = tan(robot_theta[arr3] - pi_2)

        xi = (dy_no0 - y4_no0 + a1*x4_no0 - dx_no0*a2)/(a1 - a2)
        a1 = dy_no0 + a2 * (xi - dx_no0) # y i

        a3 = -1 / a2
        xp = (dy_no0 - y4_no0 - a2*dx_no0 + a3*x4_no0)/(a3 - a2)
        a2 = dy_no0 + a2 * (xp - dx_no0) # y p
        dx_no0 = power(x4_no0 - xi, 2) + power(y4_no0 - a1, 2) # a 2
        dy_no0 = power(x4_no0 - xp, 2) + power(y4_no0 - a2, 2) # b 2
        y4_no0 = power(xi - xp, 2) + power(a1 - a2, 2) # c2

        a2 = None
        xi = None
        a1 = None

        x4_no0 = np_and(dx_no0 > 1e-9, dy_no0 > 1e-9, y4_no0 > 1e-9) # ok triang
        a_2_ok = dx_no0[x4_no0]
        b_2_ok = dy_no0[x4_no0]
        a_2_ok = arccos((a_2_ok + b_2_ok - y4_no0[x4_no0])/(2 * sqrt(a_2_ok) * sqrt(b_2_ok))) # theta wall
        a_2_ok[xp[x4_no0] > xi[x4_no0]] *= -1
        a_2_ok[inverse(x4_no0)] = 0

        arr1[arr3] = sqrt(power(dx_no0 - xi, 2) + power(dy_no0 - a1, 2)) # d wall all
        arr2[arr3] = a_2_ok # theta all
        arr3 = inverse(arr3) # div0
        arr1[arr3] = 0
        arr2[arr3] = 0

        b_2_ok = None
        y4_no0 = None
        x4_no0 = None
        d_wall = None
        dx_no0 = None
        dy_no0 = None
        a_2_ok = None

    x = np.concatenate([arr1, arr2], 1)
