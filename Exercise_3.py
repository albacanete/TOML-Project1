# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 10:59:49 2018

@author: joseb
"""

#  min     x1^2 + x2^2
#  s.t.    x1^2 +x1*x2 + x2^2 - 3   <= 0
#  var     x1, x2

# %%
from scipy.optimize import minimize
import numpy as np
from cvxpy import *

print('\nSOLVING USING SCIPY\n')


# Jacobian
def jacobian(x):
    dx = 2 * x[0]
    dy = 2 * x[1]
    return np.array((dx, dy))


# Jacobian
def hessian():
    d11 = 2.
    d12 = 0.
    d22 = 2.
    return np.array(((d11, d12), (d12, d22)))


# Objective function
def obj_fun():
    return lambda x: x[0] ** 2 + x[1] ** 2


# Minimize objective function
def min_func(x):
    return minimize(obj_fun(), x, method='SLSQP', bounds=bounds, constraints=cons)


# Minimize objective function with gradient
def min_func_jacobian(x):
    return minimize(obj_fun(), x, method='SLSQP', bounds=bounds, constraints=cons, jac=jacobian)


# Minimize objective function with hessian
def min_func_hessian(x):
    return minimize(obj_fun(), x, bounds=bounds, constraints=cons, jac=jacobian, hess=hessian)


# constraints functions
cons = ({'type': 'ineq', 'fun': lambda x: -x[0] ** 2 - x[0] * x[1] - x[1] ** 2 + 3},
        {'type': 'ineq', 'fun': lambda x: 3 * x[0] + 2 * x[1] - 3})

# bounds, if any, e.g. x1 and x2 have to be positive
bounds = ((None, None), (None, None))

# initial guess
x0 = np.asarray((10, 10))

# Method SLSQP uses Sequential Least SQuares Programming to minimize a function 
# of several variables with any combination of bounds, equality and inequality constraints. 
res = min_func(x0)
print("optimal value p*", res.fun)
print("optimal var: x1 = ", res.x[0], " x2 = ", res.x[1])

res2 = min_func_jacobian(x0)
print('\n', res2)
print("JAC: optimal value p*", res2.fun)
print("JAC: optimal var: x1 = ", res2.x[0], " x2 = ", res2.x[1])

# print 'C1',res2.x[0]**2+res2.x[1]**2+res2.x[0]*res2.x[1],'C2',3*res2.x[0]+2*res2.x[1]

res3 = min_func_hessian(x0)
print('\n', res3)
print("JAC+HESS: optimal value p*", res3.fun)
print("JAC*HESS: optimal var: x1 = ", res3.x[0], " x2 = ", res3.x[1])


print('\n SOLVING USING CVXPY\n')

# Create two scalar optimization variables.
x = Variable(2, name='x')

# Constraints
P1 = np.array(np.mat('1. 0.5; 0.5 1.'))
f1 = quad_form(x, P1)
f2 = 3. * x[0] + 2. * x[1]
constraints = [f1 <= 3., f2 >= 3.]

# Form objective.
P0 = np.array(np.mat('1. 0.; 0. 1.'))
f0 = quad_form(x, P0)
obj = Minimize(f0)

# Form and solve problem.
prob = Problem(obj, constraints)
print("solve", prob.solve())  # Returns the optimal value.
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x[0].value, " x2 = ", x[1].value)
print("optimal dual variables lambda1 = ", constraints[0].dual_value)
print("optimal dual variables lambda2 = ", constraints[1].dual_value)
