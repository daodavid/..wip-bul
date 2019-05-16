import numpy as np


def solveODE(u, v, u0, v0, t0=0, h=0.1, n=100):
    """
    solve system ode
    :param u: u(x,y,t) e1
    :param v: v(x,y,t) e2
    :param u0: initial value
    :param v0: initial value
    :param t0: initial value
    :param h:  increment step
    :param n:  interval multiple by h give the interval for integration
    :return:
    """

    y_args = [v0]  # y_args array for y arguments
    x_args = [u0]  # x_args array for arguments
    t_args = [t0]  # time interval
    x = u0
    y = v0
    t = t0
    for i in range(n):
        k1 = h * u(t, x, y)
        l1 = h * v(t, x, y)
        k2 = h * u(t + h / 2, x + k1 / 2, y + l1 / 2)
        l2 = h * v(t + h / 2, x + k1 / 2, y + l1 / 2)
        k3 = h * u(t + h / 2, x + k2 / 2, y + l2 / 2)
        l3 = h * v(t + h / 2, x + k2 / 2, y + l2 / 2)
        k4 = h * u(t + h, x + k3, y + l3)
        l4 = h * v(t + h, x + k3, y + l3)
        k = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        l = (1 / 6) * (l1 + 2 * l2 + 2 * l3 + l4)
        x = x + k
        y = y + l
        t = t + h
        x_args.append(x)
        y_args.append(y)
        t_args.append(t)

        # ax.plot(x_args, y_args, label="solved", color="black")

    return np.array([x_args, y_args, t_args]).T


def intergrate(x0, dxdt, t):
    x_args = [x0]
    t_args = [0]

    f = x0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        f = f + dxdt[i] * dt

        x_args.append(f)
        t_args.append(t)
    return np.array([x_args, t]).T


def derivate(x_args, t_args):
    res = []
    for i in range(len(x_args) - 1):
        dfdt = (x_args[i + 1] - x_args[i]) / (t_args[i + 1] - t_args[i])
        res.append(dfdt)
    return np.array(res)


def solve2Order(u_vx, v_vy, x0, y0, u0, v0, t0=0, h=0.1, n=10000):
    y_args = [y0]  # y_args array for y arguments
    x_args = [x0]  # x_args array for arguments
    t_args = [t0]  # time interval
    x_speed = [u0 ]
    y_speed = [v0]
    vx_next = u0
    vy_next = v0
    vx= u0
    vy=v0
    t = t0
    x = x0
    y = y0
    z =h
    h=h/2
    for i in range(n):
        k1 = h * u_vx(t, x, y)
        l1 = h * v_vy(t, x, y)
        k2 = h * u_vx(t + h / 2, x + k1 / 2, y + l1 / 2)
        l2 = h * v_vy(t + h / 2, x + k1 / 2, y + l1 / 2)
        k3 = h * u_vx(t + h / 2, x + k2 / 2, y + l2 / 2)
        l3 = h * v_vy(t + h / 2, x + k2 / 2, y + l2 / 2)
        k4 = h* u_vx(t + h, x + k3, y + l3)
        l4 = h * v_vy(t + h, x + k3, y + l3)
        k = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        l = (1 / 6) * (l1 + 2 * l2 + 2 * l3 + l4)
        vx_next = vx_next + k
        vy_next = vy_next + l
        x_speed.append(vx)
        y_speed.append(vy)
        t = t + h
        x = x + vx * z
        y = y + vy * z
        x_args.append(x)
        y_args.append(y)
        t_args.append(t)
        vx = vx_next
        vy=vy_next

    return np.array([x_args, y_args, t_args]).T
