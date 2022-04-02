from cvxpy import *

# Create optimization variables.
# x[0] = x1, x[1] = x2, x[2] = x3
# r[0] = R12, r[1] = R23, r[2] = R32
x = Variable(3, name='x')
r = Variable(3, name='r')

# Constraints
f1 = x[0] + x[1]
f2 = x[0]
f3 = x[2]
f4 = r[0] + r[1] + r[2]
constraints = [f1 <= r[0], f2 <= r[1], f3 <= r[2], f4 <= 1]

# Form objective.
f0 = cvxpy.sum(cvxpy.log(x))
obj = Maximize(f0)

# Form and solve problem.
prob = Problem(obj, constraints)
print("solve", prob.solve())  # Returns the optimal value.
print("status:", prob.status)
print("optimal value p* = ", prob.value)
print("optimal var: x1 = ", x[0].value)
print("optimal var: x2 = ", x[1].value)
print("optimal var: x3 = ", x[2].value)
print("optimal var: R12 = ", r[0].value)
print("optimal var: R23 = ", r[1].value)
print("optimal var: R32 = ", r[2].value)
print("optimal dual variables lambda1 = ", constraints[0].dual_value)
print("optimal dual variables lambda2 = ", constraints[1].dual_value)
print("optimal dual variables lambda3 = ", constraints[2].dual_value)
print("optimal dual variables mu1 = ", constraints[3].dual_value)
