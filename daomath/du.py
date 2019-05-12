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
        k1 = h * u(x, y, t)
        l1 = h * v(x, y, t)
        k2 = h * u(t + h / 2, x + k1 / 2, y + l1 / 2)
        l2 = h * v(t + h / 2, x + k1 / 2, y + l1 / 2)
        k3 = h * u(t + h / 2, x + k2 / 2, y + l2 / 2)
        l3 = h * v(t + h / 2, x + k2 / 2, y + l2 / 2)
        k4 = h * u(t + h , x + k3 , y + l3 )
        l4 = h * v(t + h , x + k3, y + l3 )
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
    f=x0
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        f = f+dxdt[i] * dt

        x_args.append(f)
    return np.array(x_args)



