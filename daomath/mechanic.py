import numpy as np
import matplotlib.pyplot as plt
from daomath.fields import VectorField
from daomath.du import *
from daomath.utility import *
class Force(VectorField):

    def __init__(self,u,v,p=0,r=[-10,10]):
        super().__init__(u,v,p,range=r)

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
        x = reduce_array(self.x_args,100)
        y = reduce_array(self.y_args,100)
        x0 = x*0
        y0 = y*0
        q = plt.quiver(x0, x0,x, y, angles='xy', scale_units='xy', scale=50, color='r', width=3)

f_x = lambda x,y : -x


force = Force(lambda x,y,t=0 :1,lambda x,y,t=0: 1)
force.plot_force_field()
force.plot_p()
#plt.show()

# point = MaterialPoint(x0=6,y0=6)
# point.add_force_field(force)
# v = point.calculate_speed(1,-1)
# print(v)
# point.calculate_radius_vector(-1,-1)
# point.plot_graph_motion()
# plt.show()

u = lambda t,x,y:1
v = lambda t,x,y:2*t

z = solveODE(u,v,0,0)
#z = intergrate(1,u,100)
# print(z)
def runge_kutta(f,x0,y0,n=100,h=0.1):
   x_args = np.linspace(0,10,10)
   y = y0
   x = x0
   x_args=[x]
   y_args =[y]
   for i in range(n):
       k1 = h*f(x,y)
       k2 = h*f(x+h/2,y+k1/2)
       k3 = h*f(x+h/2,y+k2/2)
       k4 = h*f(x+h,y+k3)
       y = y + (k1+2*k2+2*k3+k4)/6
       x = x+h
       x_args.append(x)
       y_args.append(y)

   return  np.array([[x],[y]]).T
# u = lambda x,y:1
# v = lambda x,y,:2*x
# v = runge_kutta(v,0,0)
print(v)