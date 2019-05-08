import numpy as np
import matplotlib.pyplot as plt
from daomath.fields import VectorField

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

    def add_force_field(self,force):
        self.force+=force

    def calculate_speed(self,vx0,vy0,time_range=[0,100],step=60):
        t = np.linspace(time_range[0],time_range[1],step)



force = Force(lambda x,y : -x ,lambda y,x: -y)
force.plot_field()