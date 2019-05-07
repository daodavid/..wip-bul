import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.tools as tls
from plotly import figure_factory as ff
import plotly.plotly as py


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
        matrix = [[x0, y0, self.U(x0, y0), self.V(x0, y0)] for x0 in x for y0 in y]  ### matrix v1([x0.0,y0.0,x0.1,y.0.1])v2([x2.0,y2.0,x2.1,y2,2])
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