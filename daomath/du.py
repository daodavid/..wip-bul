import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani


def solveODE(u, v, u0, v0, t0=0, h=0.001, n=10000):
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


def solve2Order(u_vx, v_vy, x0, y0, u0, v0, t0=0, h=0.001, n=100000):
    y_args = [y0]  # y_args array for y arguments
    x_args = [x0]  # x_args array for arguments
    t_args = [t0]  # time interval
    x_speed = [u0]
    y_speed = [v0]
    vx_next = u0
    vy_next = v0
    vx = u0
    vy = v0
    t = t0
    x = x0
    y = y0
    p = h
    h = h / 2
    for i in range(n):
        x = x + vx * h
        y = y + vy * h
        x_args.append(x)
        y_args.append(y)
        k1 = h * u_vx(t, x, y)
        l1 = h * v_vy(t, x, y)
        k2 = h * u_vx(t + h / 2, x + k1 / 2, y + l1 / 2)
        l2 = h * v_vy(t + h / 2, x + k1 / 2, y + l1 / 2)
        k3 = h * u_vx(t + h / 2, x + k2 / 2, y + l2 / 2)
        l3 = h * v_vy(t + h / 2, x + k2 / 2, y + l2 / 2)
        k4 = h * u_vx(t + h, x + k3, y + l3)
        l4 = h * v_vy(t + h, x + k3, y + l3)
        k = (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        l = (1 / 6) * (l1 + 2 * l2 + 2 * l3 + l4)

        vx = vx + k
        vy = vy + l
        t = t + h
        x_speed.append(vx)
        y_speed.append(vy)
        t_args.append(t)

    return np.array([x_args, y_args, x_speed, y_speed, t_args]).T


#
# def leap_frog(v,u_0, x0, y0, u0, v0, t0=0, h=0.1, n=10000):
#     x_n=x0
#     y_n=y0
#     Vx_n=u0
#     Uy_n=v0
#
#     for i in range(n):
#         x = v()

#
# fx = lambda t,x,y : -10*x/(np.sqrt(x**2+y**2))
# fy = lambda t,x,y :-10*y/(np.sqrt(x**2+y**2))
#
# r = solve2Order(fx,fy,3,8,4,-10)
# #plt.plot(r[:,4],r[:,3])
# plt.plot(r[:,0],r[:,1])
# plt.show()    tor figure

def leapFrog(u, v, x0, y0, vx_0, vy_0, n=1000, h=0.001):
    #https://people.ucsc.edu/~mmrosent/talks_notes/num_int_pres.pdf
    x = x0
    y = y0
    vx = vx_0
    vy = vy_0
    t = 0
    y_args = [y]  # y_args array for y arguments
    x_args = [x]  # x_args array for arguments
    t_args = [t]  # time interval
    x_speed = [vx]
    y_speed = [vy]

    for i in range(n):
        x = x +  vx* (h / 2)
        y = y + vy* (h/2)

        vx = vx+ h* u(t,x,y)
        vy = vy + h * v(t, x,y)


        x = x + vx*(h/2)
        y = y + vy*(h/2)

        t = t + h
        x_speed.append(vx)
        y_speed.append(vy)
        t_args.append(t)
        x_args.append(x)
        y_args.append(y)
    return np.array([x_args, y_args, x_speed, y_speed, t_args]).T



def solveSystemOdeRungeKutta(u, v, x0, y0, vx_0, vy_0, n=1000, h=0.000001):
    # http://www.astro.utu.fi/~cflynn/galdyn/lecture6.html?fbclid=IwAR31E-ezjjqEmMxxui7CkpEjrc32ACuMErAHQgK1sSsaifNRv2ss3FcPGaY
    x = x0
    y = y0
    vx = vx_0
    vy = vy_0
    t = 0
    y_args = [y]  # y_args array for y arguments
    x_args = [x]  # x_args array for arguments
    t_args = [t]  # time interval
    x_speed = [vx]
    y_speed = [vy]

    for i in range(n):
        A_xn = h * u(t, x, y)/2
        A_yn = h * v(t, x, y)/2

        b_xn = h * (vx + A_xn/2) / 2
        b_yn = h * (vy + A_yn/2) / 2

        B_xn = (x + b_xn) / 2
        B_yn = (y + b_yn) / 2

        C_xn = h * (x + b_xn)
        C_yn = h * (y + b_yn)

        beta_xn = h * (vx + C_xn)
        beta_yn = h * (vy + C_yn)

        D_xn = h * (x + beta_xn) / 2
        D_yn = h * (y + beta_yn) / 2

        K_xn = (A_xn + B_xn + C_xn) / 3
        K_yn = (A_yn + B_yn + C_yn) / 3

        J_xn = (A_xn + 2 * B_xn + 2* C_xn + D_xn) / 3
        J_yn = (A_yn + 2 * B_yn + 2 *C_yn + D_yn) / 3

        x = x + h * (vx + K_xn)
        y = y + h * (vy + K_yn)

        vx = vx + J_xn
        vy = vy + J_yn

        t = t + h
        x_speed.append(vx)
        y_speed.append(vy)
        t_args.append(t)
        x_args.append(x)
        y_args.append(y)

    return np.array([x_args, y_args, x_speed, y_speed, t_args]).T

k = 300




def animate(i):
    # fx = lambda t, x, y: -k * x / (np.sqrt(x ** 2 + y ** 2) * (x ** 2 + y ** 2))
    # fy = lambda t, x, y: -k * y / (np.sqrt(x ** 2 + y ** 2) * (x ** 2 + y ** 2))
    # r = solve2Order(fx, fy, 6, 8, -20, -15)
    k = 420
    fx = lambda t, x, y: -k * x / ((x ** 2 + y ** 2)**(3/2))
    fy = lambda t, x, y: -k * y / ((x ** 2 + y ** 2)**(3/2))
    r = leapFrog(fx, fy, 10,10, -4*np.sin(np.pi/2), 4*np.cos(np.pi/2),h=0.01)

    plt.scatter(r[i, 0], r[i, 1])

print(4**(1/2))

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
anim = ani.FuncAnimation(fig, animate, interval=1)
plt.show()







