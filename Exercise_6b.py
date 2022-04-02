from scipy.optimize import newton
import matplotlib.pyplot as plt
import numpy as np


# Objective function formalization
def obj_fun():
    return lambda x: 2 * (x ** 4) - 4 * (x ** 2) + x - 0.5


# Plot objective function
def plot_obj_fun(x):
    return 2 * x ** 4 - 4 * (x ** 2) + x - 0.5


# First derivative
def der_obj_fun(x):
    return (8 * (x ** 3)) - (8 * x) + 1


# Second derivative
def sec_der_obj_func(x):
    return (24 * (x ** 2)) - (8 * x)


# DESCENT GRADIENT
print("SOLVING WITH GRADIENT DESCENT METHOD")
# Initial guess
x0s = [-2, -0.5, 0.5, 2]
# stop criterion
accuracy = pow(10, -4)
for x0 in x0s:
    label_x0 = "initial point [" + str(x0) + ", " + str(plot_obj_fun(x0)) + "]"
    plt.scatter(x0, plot_obj_fun(x0), color='yellow', edgecolor='black', label=label_x0)
    # maximum number of iterations
    max_iters = 1000
    # iteration count
    iters = 0
    previous_step_size = 1
    # Learning rate
    rate = 0.01
    current = x0
    while previous_step_size > accuracy and iters < max_iters:
        # Current x will be previous in next step
        previous = current
        # Gradient descent
        current = current - rate * der_obj_fun(previous)
        # Distance moved
        previous_step_size = abs(current - previous)
        # iteration count
        iters += 1
        # plot
        plt.scatter(current, plot_obj_fun(current), color='red', edgecolor='black')
    print("Initial point: ", x0)
    print("Local minimum: ", current)
    print("Number of iterations: ", iters)
    # Plot
    # define range for input
    x0 = np.linspace(-2, 2)
    plt.plot(x0, plot_obj_fun(x0))
    # plot minimal point
    label_min = "minimal point [" + str(current) + ", " + str(plot_obj_fun(current)) + "]"
    plt.scatter(current, plot_obj_fun(current), color='green', edgecolor='black', label=label_min)
    plt.legend(prop={'size': 6}, loc='best')
    plt.show()


# NEWTON'S METHOD
print("SOLVING WITH NEWTON'S METHOD")
x0s = [-2, -0.5, 0.5, 2]
for x0 in x0s:
    g = newton(func=der_obj_fun, x0=x0, fprime=sec_der_obj_func, tol=0.0001, full_output=True)
    # print(g)
    r = g[1]
    print("initial point:", x0)
    print("Value:", r.root)
    print("Number of iterations: ", r.iterations)