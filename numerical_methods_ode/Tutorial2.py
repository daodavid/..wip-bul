import numpy as np
import matplotlib.pyplot as plt
from daomath.fields import VectorField





class Motion:
    def __init__(self,x_t,y_t,range=[-10,10]):
        time = np.linspace(1,10,20)
        self.radius_vector = [[ 0, 0, x_t(t),y_t(t),t ] for t in time ]
        self.radius_vector = np.array(self.radius_vector)
        print(self.radius_vector)
        self.x_args = self.radius_vector[:,2]
        self.y_args = self.radius_vector[:,3]



    def draw_radios_vector(self,color='black',append=False):
        self.ax = plt.gca()
        plt.title('Arrows scale with plot width, not view')
        plt.text(1, 1, r'$2 \frac{m}{s}$', fontsize=20, verticalalignment='center', transform=self.ax.transAxes)
        plt.xlim(0, 10)
        plt.ylim(0, 100)
        v = self.radius_vector
        self.ax.spines['top'].set_color('none')
        self.ax.spines['bottom'].set_position('zero')
        self.ax.spines['left'].set_position('zero')
        self.ax.spines['right'].set_color('none')
        q = plt.quiver(v[:, 0], v[:, 1], self.x_args, self.y_args,  color=color,
                       width=0.003,angles='xy', scale_units='xy', scale=1)

        if append:
            """
            """
        else:
            plt.show()


    def draw_curve(self,color='blue'):
        self.ax.plot(self.x_args, self.y_args, label="solved", color=color)


    def draw(self):
       self.draw_radios_vector(append=True)
       self.draw_curve()
       plt.show()

body = Motion(lambda t:t,lambda t :100 - t*t )
body.draw()
