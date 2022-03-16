from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import math
import numdifftools as ndt


# Jacobian
def jacobian(x):
    dx = math.e**(x[0]) * (4*x[0]**2 + 4*x[0]*(x[1]+2) + 2*x[1]**2 + 6*x[1] + 1)
    dy = math.e**(x[0]) * (4*x[1] + 4*x[0] + 2)
    return np.array((dx, dy))


# Objective function formalization
def obj_fun():
    return lambda x: math.e**(x[0]) * (4 * x[0] ** 2 + 2 * x[1] ** 2 + 4 * x[0] * x[1] + 2 * x[1] + 1)


# Objective function for plot
def plot_obj_fun(x):
    return math.e**(x[0]) * (4 * x[0] ** 2 + 2 * x[1] ** 2 + 4 * x[0] * x[1] + 2 * x[1] + 1)


# Constraint functions (have to be >= 0
cons = ({'type': 'ineq', 'fun': lambda x: -x[0] * x[1] + x[0] + x[1] - 1.5},
        {'type': 'ineq', 'fun': lambda x: x[0] * x[1] + 10})

# Bounds, if any, e.g. x1 and x2 have to be positive
bounds = ((None, None),) * 2

# Initial guess: [0,0], [10,20], [-10,1], [-30,-30]
x0s = [np.asarray((0, 0)), np.asarray((10, 20)), np.asarray((-10, 1)), np.asarray((-30, -30))]

# Save minimal points and variables to plot
minimals = []    # minimals[i][0] = x, minimals[i][1] = y, minimals[i][2] = z
for x0 in x0s:
    # Method SLSQP uses Sequential Least SQuires Programming to minimize a function
    # of several variables with any combination of bounds, equality and inequality constraints.
    obj = obj_fun()
    # res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons, jac=jacobian)
    minimals.append((res.x[0], res.x[1], res.fun))
    print("optimal value p*", res.fun)
    print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])

# define range for input
r_min, r_max = -10.0, 3.0
# sample input range uniformly at 0.7 increments
xaxis = np.arange(r_min, r_max, 0.7)
yaxis = np.arange(r_min, r_max, 0.7)
# create a mesh from the axis
x0, x1 = np.meshgrid(xaxis, yaxis)
# compute targets
results = np.array(plot_obj_fun((x0, x1)))
# create a surface plot with the jet color scheme
figure = plt.figure()
axis = figure.gca(projection='3d')
for p in minimals:
    axis.scatter(p[0], p[1], p[2], color='red', edgecolor='black')
axis.plot_surface(x0, x1, results, cmap='jet')
# show the plot
plt.show()
