from cvxpy import *
import numpy as np
import matplotlib.pyplot as plt

# Create two scalar optimization variables.
x = Variable(1, name='x')

# Constraints
c1 = cvxpy.square(x) - 6. * x + 8.
constraints = [c1 <= 0.]

# Form objective.
f0 = cvxpy.square(x) + 1.
obj = Minimize(f0)

# Form and solve problem.
prob = Problem(obj, constraints)
print("solve", prob.solve())  # Returns the optimal value.
print("is DCP?:", prob.is_dcp())
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x.value)
print("optimal dual variables lambda1 = ", constraints[0].dual_value)


# Objective function for plot
def plot_obj_fun(x):
    return pow(x, 2) + 1


# define range for input
x0 = np.linspace(-10, 10)
plt.plot(x0, plot_obj_fun(x0))
# create a surface plot with the jet color scheme
plt.scatter(x.value, prob.value, color='red', edgecolor='black')
label = "local minimum [2, 5]"
plt.scatter(x.value, prob.value, color='red', edgecolor='black', label=label)
plt.legend(prop={'size': 6}, loc='upper right')
# show the plot
plt.show()
