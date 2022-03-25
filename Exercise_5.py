from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt

# Create two scalar optimization variables.
x = Variable(2, name='x')

# Constraints
c1 = cvxpy.square((x[0] - 1)) + cvxpy.square((x[1] - 1))
c2 = cvxpy.square((x[0] - 1)) + cvxpy.square((x[1] + 1))
constraints = [c1 <= 1., c2 <= 1.]

# Form objective.
f0 = cvxpy.square(x[0]) + cvxpy.square(x[1])
obj = Minimize(f0)

# Form and solve problem.
prob = Problem(obj, constraints)
print("solve", prob.solve())  # Returns the optimal value.
print("is DCP?:", prob.is_dcp())
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x[0].value, x[1].value)


# Objective function for plot
def plot_obj_fun(x_0, x_1):
    return pow(x_0, 2) + pow(x_1, 2)


# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.7 increments
xaxis = np.arange(r_min, r_max, 0.7)
yaxis = np.arange(r_min, r_max, 0.7)
# create a mesh from the axis
x0, x1 = np.meshgrid(xaxis, yaxis)
# compute targets
results = np.array(plot_obj_fun(x0, x1))
# create a surface plot with the jet color scheme
figure = plt.figure()
axis = figure.gca(projection='3d')
axis.scatter(x[0].value, x[1].value, prob.value, color='red', edgecolor='black')
axis.plot_surface(x0, x1, results, cmap='jet')
# show the plot
plt.show()
