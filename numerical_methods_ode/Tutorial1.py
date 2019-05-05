import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.tools as tls
from plotly import figure_factory as ff
import plotly.plotly as py


def draw_function(my_funct, from_x, to_x,label='y'):
    ax = plt.gca()
    x_args = np.linspace(from_x, to_x, 10000)
    y_args = [my_funct(x) for x in x_args]
    #plt.plot(x_args, y_args,label=r'$\sin (x)$')
    #plt.scatter(x_args,y_args,color='black')
    ax.plot(x_args,y_args, label=label,color="black")
    ax.legend()
    #plt.show()



def solve(function,x0,y0,h,n,actual_func):

    x_args = [x0]
    y_args = [y0]
    x=x0
    y=y0
    for i in range(n):
        y = y + function(x, y) * h
        y_args.append(y)
        x =x+h
        x_args.append(x)

    ax = plt.gca()
    ax.scatter(x_args, y_args, label="solved", color="blue",s=1)
    draw_function(actual_func, 0, 10)


dydx = lambda x,y:2*x

solve(dydx,0,1,0.00009,100000,lambda x : x**2 +1)
plt.show()