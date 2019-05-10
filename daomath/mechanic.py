import numpy as np
import matplotlib.pyplot as plt
from daomath.fields import VectorField
from daomath.du import *
class Force(VectorField):

    def __init__(self,u,v,p=0,r=[0,10]):
        super().__init__(u,v,p)

class MaterialPoint():


    def __init__(self,x0=0,y0=0,mass=1):
        self.x0 =x0
        self.y0=y0
        self.mass = mass
        pass


    def __get_cordinate__(self,time):
        pass

    def __get_cordinate__(self):
        pass

    def get_speed(self):
        pass

    def get_acceleration(self):
        pass

    def get_mass(self):
        pass

    def get_radius_vecotor(self):
        pass

    def add_force_field(self,f):
        self.force=f

    def calculate_speed(self,vx0,vy0,time_range=[0,100],step=60):
        t = np.linspace(time_range[0],time_range[1],step)
        self.speed_space = solveODE(self.force.U,self.force.V,vx0,vy0)
        return self.speed_space

    def calculate_radius_vector(self,vx0,vy0):
        self.speed_space = solveODE(self.force.U,self.force.V,vx0,vy0)
        self.x_args = intergrate(self.x0,self.speed_space[:,1],self.speed_space[:,0])
        self.y_args = intergrate(self.x0, self.speed_space[:,2], self.speed_space[:,0])
        return np.array([self.x_args,self.y_args]).T

    def plot_graph_motion(self):
        x0 = self.x_args*0
        y0 = self.y_args*0
        q = plt.quiver(x0, x0,self.x_args, self.y_args, angles='xy', scale_units='xy', scale=0.001, color='red', width=3)

f_x = lambda x,y : -x


force = Force(lambda x,y :-x,lambda x,y: -y)
force.plot_field()
force.plot_p()
plt.show()

# point = MaterialPoint(x0=5,y0=5)
# point.add_force_field(force)
# v = point.calculate_speed(1,4)
# point.calculate_radius_vector(-1,-1)
# point.plot_graph_motion()
# plt.show()
