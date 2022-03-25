from cvxpy import *


# Create two scalar optimization variables.
x = Variable(2, name='x')

# Constraints
c1 = cvxpy.square((x[0]-1)) + cvxpy.square((x[1]-1))
c2 = cvxpy.square((x[0]-1)) + cvxpy.square((x[1]+1))
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
