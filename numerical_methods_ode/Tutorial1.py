import numpy as np
import matplotlib.pyplot as plt

from daomath.fields import VectorField


def draw_function(my_funct, from_x, to_x, label='y'):
    ax = plt.gca()
    x_args = np.linspace(from_x, to_x, 10000)
    y_args = [my_funct(x) for x in x_args]
    # plt.plot(x_args, y_args,label=r'$\sin (x)$')
    # plt.scatter(x_args,y_args,color='black')
    ax.plot(x_args, y_args, label=label, color="black")
    print(x_args)
    print(y_args)
    ax.legend()
    # plt.show()


def solve(function, x0, y0, h, n, actual_func):
    x_args = [x0]
    y_args = [y0]
    x = x0
    y = y0
    for i in range(n):
        y = y + function(x, y) * h
        y_args.append(y)
        x = x + h
        x_args.append(x)

    ax = plt.gca()
    ax.scatter(x_args, y_args, label="solved", color="blue", s=10)
    print(x_args)
    print(y_args)
    draw_function(actual_func, 1, 10)


dydx = lambda x, y: 2 * y


# solve(dydx,0,1,0.01,1000,lambda x : np.exp(2*x))
# plt.show()


def sysODE(functions, x0, y0, t0, h=0.01, n=1500):
    u = functions[0]
    v = functions[1]
    field = VectorField(u, v)

    x_args = [x0]
    y_args = [y0]
    t_atgs = [t0]
    """
    u' = u+v 
    v' = v - u
    u_n+1 = u_m -v_n + du/dt'2&h"2/2
   """
    # for i in range(n):
    #     x = x + u(x,y,t)*(-h)
    #     y = y+  v(x,y,t)*(-h)
    #     t = t+ h
    #     x_args.append(x)
    #     y_args.append(y)
    #     t_atgs.append(t)

    x = x0
    y = y0
    t = t0
    for i in range(n):
        if len(x_args) > 1:
            dx = x - x_args[len(x)]
        x = x + u(x, y, t) * h
        y = y + v(x, y, t) * h
        t = t + h
        x_args.append(x)
        y_args.append(y)
        t_atgs.append(t)
    field.plot_field(append=True)
    ax = plt.gca()
    plt.xlim(-12, 12)
    plt.ylim(-10, 10)
    ax.plot(x_args, y_args, label="solved", color="black")


u = lambda x, y, t=0: 4 * np.sin(np.pi * x / 180)
v = lambda x, y, t=0: 4 * np.cos(np.pi * x / 180)
##3sysODE([u,v],-1,3,0)
plt.show()


def solveODE(u, v, u0, v0, t0, h=0.1, n=100):
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
        k = (1 / 6)*(k1 + 2 * k2 + 2 * k3 + k4)
        l = (1 / 6)*(l1 + 2 * l2 + 2 * l3 + l4)
        x = x + k
        y = y + l
        t = t + h
        x_args.append(x)
        y_args.append(y)
        t_args.append(t)
        field = VectorField(u, v)

        field.plot_field(append=True)
        ax = plt.gca()
        plt.xlim(-12, 12)
        plt.ylim(-10, 10)
        V = np.array([x, y])
        #q = plt.quiver(v[:, 0], v[:, 1], v[:, 2], v[:, 3], angles='xy', scale_units='xy', scale=1, color='z',width=0.003)
        ax.plot(x_args, y_args, label="solved", color="black")

    return v


#
# u_x= lambda x,y:-y/np.sqrt(x**2+y**2)
# v_y = lambda x,y:-x/np.sqrt(x**2+y**2)


u_x= lambda x,y:-x/np.sqrt(x**2+y**2)
v_y = lambda x,y:-y/np.sqrt(x**2+y**2)
solveODE(u_x,v_y,-10*np.sqrt(2)/2,-10*np.sqrt(2)/2,0)
plt.show()