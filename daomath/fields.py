import numpy as np
import matplotlib.pyplot as plt


class RadiusVector:
    def __init__(self, x_function, y_function, z_function=None, range=[-10, 10], split=60):
        self.x_t = x_function
        self.y_t = y_function
        time = np.linspace(range[0], range[1], split)
        self.radius_vector = [[0, 0, self.x_t(t), self.y_t(t), t] for t in time]
        self.radius_vector = np.array(self.radius_vector)
        self.x_args = self.radius_vector[:, 2]
        self.y_args = self.radius_vector[:, 3]
        self.t_args = self.radius_vector[:, 4]

    def draw_radios_vector(self, color='black', append=False):
        self.ax = plt.gca()
        plt.title('Arrows scale with plot width, not view')
        plt.text(1, 1, r'$2 \frac{m}{s}$', fontsize=20, verticalalignment='center', transform=self.ax.transAxes)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        v = self.radius_vector
        self.ax.spines['top'].set_color('none')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['right'].set_color('none')
        q = plt.quiver(v[:, 0], v[:, 1], self.x_args, self.y_args, color=color,
                       width=0.003, angles='xy', scale_units='xy', scale=1)

        if append:
            """
            """
        else:
            plt.show()

    def derivative(self):
        v_x = []
        v_x0 = []  # start position
        v_y = []
        v_y0 = []
        time = []
        print(self.x_args)
        for i in range(len(self.x_args) - 2):
            dt = self.t_args[i + 1] - self.t_args[i]
            dx = self.x_args[i + 1] - self.x_args[i]
            dy = self.y_args[i + 1] - self.y_args[i]
            time.append(self.t_args[i])
            v_x0.append(self.x_args[i])
            v_y0.append(self.y_args[i])
            v_x.append(dx / dt)
            v_y.append(dy / dt)
        self.speed_space = np.array([v_x0, v_y0, v_x,v_y ,time]).T

        return self.speed_space

    def acceleration(self):
        a_x = []
        a_x0 = []  # start position
        a_y = []
        a_y0 = []
        time = []
        vx_sp = self.speed_space[:,2]
        vy_sp =self.speed_space[:,3]

        for i in range(len(vx_sp) - 2):
            dt = self.t_args[i + 1] - self.t_args[i]
            dvx = vx_sp[i + 1] - vx_sp[i]
            dvy = vy_sp[i + 1] - vy_sp[i]
            time.append(self.t_args[i])
            a_x0.append(self.x_args[i])
            a_y0.append(self.y_args[i])
            a_x.append(dvx / dt)
            a_y.append(dvy / dt)

        self.acceleration_space = np.array([a_x0, a_y0, a_x,a_y, time]).T
        return self.acceleration_space

    def draw_curve(self, color='blue'):
        self.ax.plot(self.x_args, self.y_args, label="solved", color=color)

    def draw_speed(self):
        self.ax = plt.gca()
        plt.title('Arrows scale with plot width, not view')
        plt.text(1, 1, r'$2 \frac{m}{s}$', fontsize=20, verticalalignment='center', transform=self.ax.transAxes)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        v = self.speed_space
        self.ax.spines['top'].set_color('none')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['right'].set_color('none')
        q = plt.quiver(v[0:100:4, 0], v[0:100:4, 1], v[0:100:4, 2], v[0:100:4, 3], color='red',
                       width=0.003, angles='xy', scale_units='xy', scale=1)

        plt.show()
    def draw_accelaration(self):
        self.ax = plt.gca()
        plt.title('Arrows scale with plot width, not view')
        plt.text(1, 1, r'$2 \frac{m}{s}$', fontsize=20, verticalalignment='center', transform=self.ax.transAxes)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        v = self.acceleration_space
        self.ax.spines['top'].set_color('none')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['right'].set_color('none')
        print(v[:,2])
        print(v[1,2])
        q = plt.quiver(v[0:100:4, 0], v[0:100:4, 1], v[0:100:4, 2], v[0:100:4, 3], color='blue',
                       width=0.003, angles='xy', scale_units='xy', scale=1)
    def draw(self):
        self.draw_radios_vector(append=True)
        self.draw_curve()
        plt.show()


# v = RadiusVector(lambda t: 10 * np.sin(np.pi * (t / 360)), lambda t: 4 * np.cos(np.pi * (t / 360)),range[0,10])
v = RadiusVector(lambda t: t * 3, lambda t: t * t, range=[0, 10], split=100)
v.derivative()
v.acceleration()
v.draw_accelaration()
v.draw_speed()

class VectorField:

    def __init__(self, x_function, y_function, z_function=None, range=[-10, 10]):
        """

        :param x_function: e1 = U(x,y)
        :param y_function: e2 = V(x,y)
        :param z_function: e3 = P(x,y)
        :param range:
        """
        self.U = x_function
        self.V = y_function
        self.z = z_function
        self.range = range

    def evaluate_cord(self):
        x = np.linspace(self.range[0], self.range[1], 10)
        y = np.linspace(self.range[0], self.range[1], 10)
        matrix = [[x0, y0, self.U(x0, y0), self.V(x0, y0)] for x0 in x for y0 in
                  y]  ### matrix v1([x0.0,y0.0,x0.1,y.0.1])v2([x2.0,y2.0,x2.1,y2,2])
        self.quiver_cords = np.array(matrix)

    def plot_field(self, append=False, color='b'):
        self.evaluate_cord()
        self.ax = plt.gca()
        plt.title('Arrows scale with plot width, not view')
        v = self.quiver_cords
        self.ax.spines['top'].set_color('none')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['right'].set_color('none')
        q = plt.quiver(v[:, 0], v[:, 1], v[:, 2], v[:, 3], angles='xy', scale_units='xy', scale=1, color=color,
                       width=0.003)
        # plt.quiver([0], [0], [2 , angles='xy', scale_units='xy', scale=1)
        plt.quiverkey(q, 10, 10, 100, r'$2 \frac{m}{s}$', labelpos='E', coordinates='figure')
        plt.text(1, 1, r'$2 \frac{m}{s}$', fontsize=20, verticalalignment='center', transform=self.ax.transAxes)
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.axis('equal')
        if append:
            """
            """
        else:
            plt.show()

    def append_vector(self):
        pass


"""
references 
book : Numerical Methods for Solving SYstems of Nonlinear Euqtions
by Courtney Remani

"""


class SystemODE(VectorField):
    def __init__(self, x_function, y_function, z_function=None, range=[-10, 10]):
        VectorField.__init__(x_function, y_function, z_function, range)

    def solve(self):
        pass
