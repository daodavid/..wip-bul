import numpy as np


def solveODE(u, v, u0, v0, t0=0, h=0.1, n=100):
    """
    Runge Kutta method
    :return:
   """

    y_args = [v0]
    x_args = [u0]
    t_args = [t0]
    x = u0
    y = v0
    t = t0
    for i in range(n):
        k1 = -h * u(x, y)
        l1 = -h * v(x, y)
        k2 = u(x + k1 / 2, y + l1 / 2)
        l2 = v(x + k1 / 2, y + l1 / 2)
        k3 = u(x + k2 / 2, y + l2 / 2)
        l3 = v(x + k2 / 2, y + l2 / 2)
        k4 = u(x + k3 / 2, y + l3 / 2)
        l4 = v(x + k3 / 2, y + l3 / 2)
        k = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        l = (1 / 6) * (l1 + 2 * l2 + 2 * l3 + l4)
        x = x + k
        y = y + l
        t = t + h
        x_args.append(x)
        y_args.append(y)
        t_args.append(t)


        # ax.plot(x_args, y_args, label="solved", color="black")

    return np.array([x_args,y_args,t_args]).T


def intergrate(x0,dxdt,t):
    x_args=[]
    for i in range(len(t)-1):
        dt = t[i+1]- t[i]
        f = dxdt*dt+x0
        x0=f
        x_args.append(f)
    return np.array(x_args)
