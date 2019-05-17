import numpy as np
import matplotlib.pyplot as plt
from daomath.fields import VectorField
from daomath.du import *
from daomath.utility import *
class Force(VectorField):

    def __init__(self,u,v,p=0,r=[-1,100]):
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
        self.vx0= vx0
        self.vy0=vy0
        self.speed_space = solveODE(self.force.U,self.force.V,vx0,vy0)

    def draw_speed(self):
        self.force.plot_field(self.speed_space,color='green')

    def calculate_radius_vector(self):
        #self.speed_space = solveODE(self.force.U,self.force.V,vx0,vy0)
        self.x_args = intergrate(self.x0,self.speed_space[:,0],self.speed_space[:,2])
        self.y_args = intergrate(self.y0, self.speed_space[:,1], self.speed_space[:,2])
        return np.array([self.x_args,self.y_args]).T
    def plot_radios_vector(self):
        n=10
        v_space = solve2Order(self.force.U,self.force.V,self.x0,self.y0,self.vx0,self.vy0,n=9000)
        x = reduce_array(v_space[:, 0], n)
        y = reduce_array(v_space[:, 1], n)
        x0 = x * 0
        y0 = y * 0
        q = plt.quiver(x0, x0, x, y, angles='xy', scale_units='xy', scale=1, color='r', width=0.003)
        #plt.plot(x,y)

    def plot_graph_motion(self,scale = 6000):
        n = scale
        x = reduce_array(self.x_args[:,0],n)
        y = reduce_array(self.y_args[:,0],n)
        speed_x = reduce_array(self.speed_space[:,0],n)
        speed_y = reduce_array(self.speed_space[:, 1],n)
        sp1x = derivate(self.x_args[:,0],self.x_args[:,1])
        sp1y=derivate(self.y_args[:,0],self.y_args[:,1])
        sp1x = reduce_array(sp1x,n)
        sp1y = reduce_array(sp1y,n)
        # x=self.x_args
        # y=self.y_args
        x0 = x*0
        y0 = y*0
        spe0 = plt.quiver(self.x0,self.y0,self.vx0,self.vy0,scale=40,color='violet',width=0.009)#\vec a
        line, = plt.plot([4, 2, 1], label=r'$\vec v_0$', linewidth=1, color='violet')
        sp = plt.quiver(x, y, sp1x, sp1y, scale=60, color='green', width=0.003)
        line2, = plt.plot([4, 2, 1], label=r'$\vec v$', linewidth=1,color='green')
        q = plt.quiver(x0, x0,x, y, angles='xy', scale_units='xy', scale=1, color='r', width=0.003)
        line3, = plt.plot([4, 2, 1], label=r'$\vec r$', linewidth=1, color='r')



        line4, = plt.plot([4, 3, 2], label='low of motion', linewidth=1, color='black')
        plt.legend(handles=[line,line2,line3,line4], loc=7)
        plt.plot(x,y,color='black')

f_x = lambda x,y : -x


force = Force(lambda t,x,y:-10*x/(np.sqrt(x**2+y**2))**3,lambda t,x,y:-10*y/(np.sqrt(x**2+y**2)**3))
#force = Force(lambda t,x,y:0,lambda t,x,y:-2)
force.plot_force_field()

force.plot_p()


point = MaterialPoint(x0=4,y0=5)
point.add_force_field(force)
point.calculate_speed(-2,-4)
#point.draw_speed()
#plt.show()
# print(v)
#point.calculate_radius_vector()
#point.plot_graph_motion(scale=200)
point.plot_radios_vector()
plt.axis([-10, 10, -10, 20])
plt.show()
# link http://physics.bu.edu/py502/lectures3/cmotion.pdf
#leapfrog algoritam
#https://www.uio.no/studier/emner/matnat/astro/AST2000/h18/undervisningsmateriell_h2018/Numerical%20Compendium/integration_of_differential_equations.pdf