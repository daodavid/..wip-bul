import numpy as np
import matplotlib.pyplot as plt

ax = plt.gca()
plt.xlim(0, 10)
plt.ylim(0, 100)

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')


def plt_properties():
    ax = plt.gca()
    plt.xlim(-10, 10)
    plt.ylim(-10, 100)

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')


def calculate_derivative_at_point(function, point, precision=1e-7):
    """
    Calculates a numerical approximation to the derivative of the specified function
    at the given point
    """
    der = (function(point + precision) - function(point)) / precision
    return der


def plot_derivative(function, derivative=None, min_x=1, max_x=10):
    """
    Plots the function and its derivative.
    The `derivative` parameter is optional and can be provided as a separate function.
    If it's not provided, the derivative will be calculated automatically
    """
    # We're using vectorized functions to make our code simpler: this only hides the for-loop,
    # it doesn't provide any performance gain
    vectorized_function = np.vectorize(function)
    print(vectorized_function)
    x = np.linspace(min_x, max_x, 10)
    y = function(x)
    print(y)

    dy = []
    if derivative is None:
        dy = np.vectorize(calculate_derivative_at_point)(function, x)
    else:
        dy = np.vectorize(derivative)(x)

    plt.plot(x, y)
    plt.plot(x, dy, color='b')
    plt.show()

    # TODO: Plot the function and its derivative.
    # Don't forget to add axis labels.
    # Feel free to make the plot as pretty as you wish - you may add titles,
    # tick marks, legends, etc.


def plot_derivative_at_point(function, point, derivative=None, min_x=-10, max_x=10):
    """
    Plots the function in the range [x_min; x_max]. Computes the tangent line to the function
    at the given point and also plots it
    """
    plt_properties()
    vectorized_function = np.vectorize(function)

    x = np.linspace(min_x, max_x, 1000)
    y = vectorized_function(x)

    slope = 0  # Slope of the tangent line
    if derivative is None:
        slope = calculate_derivative_at_point(function, point)
    else:
        slope = derivative(point)

    intercept = function(point) - slope * point
    tangent_line_x = np.linspace(point - 2, point + 2, 4)
    tangent_line_y = slope * tangent_line_x + intercept
    print(tangent_line_x)
    print(tangent_line_y)
    plt.plot(x, y)
    plt.plot(tangent_line_x, tangent_line_y)
    plt.show()

# plot_derivative(lambda x: x ** 2, lambda x: 2 * x)  # The derivative is calculated by hand

def calculate_integral(function, x_min, x_max, num_points = 5000,init_value=0):
    """
    Calculates a numerical approximation of the definite integral of the provided function
    between the points x_min and x_max.
    The parameter n specifies the number of points at which the integral will be calculated
    """

    x = np.linspace(x_min,x_max,num_points)
    f = np.vectorize(function,x)
    y = [0]
    step = 0
    for i in range(len(x)-1):
        d = x[i+1]-x[i]
        z = x[i] + d/2
        step += function(z)*d
        y.append(step)
    return np.array(y)

f = lambda x : 2*x
F = lambda x :x**2


x = np.linspace(0,10,10)
v = np.vectorize(f)
f1 = v(x)
plt.plot(x,f1)

v= np.vectorize(F)
F1 = v(x)
plt.plot(x,F1)


F2 = calculate_integral(f,0,10,10)

plt.plot(x,F2+1)

plt.show()


