from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


# Jacobian
def jacobian(x):
    dx = 2 * x[0]
    dy = 2 * x[1]
    return np.array((dx, dy))


# Objective function formalization
def obj_fun():
    return lambda x: x[0] ** 2 + x[1] ** 2


# Objective function for plot
def plot_obj_fun(x):
    return x[0] ** 2 + x[1] ** 2


# Minimize objective function
def min_func(x):
    return minimize(obj_fun(), x, method='SLSQP', bounds=bounds, constraints=cons)


# Minimize objective function with gradient
def min_func_jacobian(x):
    return minimize(obj_fun(), x, method='SLSQP', bounds=bounds, constraints=cons, jac=jacobian)


# Constraint functions (inequalities have to be >= 0)
cons = ({'type': 'ineq', 'fun': lambda x: -0.5 + x[0]},
        {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 1},
        {'type': 'ineq', 'fun': lambda x: x[0]**2 + x[1]**2 - 1},
        {'type': 'ineq', 'fun': lambda x: 9*x[0]**2 + x[1]**2 - 9},
        {'type': 'ineq', 'fun': lambda x: x[0]**2 - x[1]},
        {'type': 'ineq', 'fun': lambda x: x[1]**2 - x[0]})

# Bounds, if any, e.g. x1 and x2 have to be positive
bounds = ((None, None),) * 2

# Initial guess
x0s = [np.asarray((3, 0)), np.asarray((1, 1))]

# Save minimal points and variables to plot
minimals = []    # minimals[i][0] = x, minimals[i][1] = y, minimals[i][2] = z
for x0 in x0s:
    # Method SLSQP uses Sequential Least SQuires Programming to minimize a function
    # of several variables with any combination of bounds, equality and inequality constraints.
    res = min_func(x0)
    print(res)
    minimals.append((res.x[0], res.x[1], res.fun))
    print("optimal value p*", res.fun)
    print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])

# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.7 increments
xaxis = np.arange(r_min, r_max, 0.2)
yaxis = np.arange(r_min, r_max, 0.2)
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
