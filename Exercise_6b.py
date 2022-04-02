from scipy.optimize import newton
import matplotlib.pyplot as plt


# Objective function formalization
def obj_fun():
    return lambda x: 2 * (x ** 4) - 4 * (x ** 2) + x - 0.5


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
        iters += 1  # iteration count
    print("Initial point: ", x0)
    print("Local minimum: ", current)
    print("Number of iterations: ", iters)


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