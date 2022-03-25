from cvxpy import *


# Create two scalar optimization variables.
x = Variable(1, name='x')

# Constraints
c1 = pow(x, 2.) - 6.*x + 8.
print(c1)
constraints = [c1 <= 0.]

# Form objective.
f0 = pow(x, 2.) + 1.
obj = Minimize(f0)

# Form and solve problem.
prob = Problem(obj, constraints)
print("solve", prob.solve())  # Returns the optimal value.
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x.value)
print("optimal dual variables lambda1 = ", constraints[0].dual_value)
