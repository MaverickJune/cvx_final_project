import cvxpy as cp
import numpy as np

# Number of requests
n = 5

# Given data
p = np.array([2, 3, 1, 4, 2])      # Processing times
T = np.array([5, 7, 3, 8, 6])      # SLOs
alpha = 1.0                        # Weighting parameter

# Variables
s = cp.Variable(n, nonneg=True)     # Start times
f = s + p                           # Finish times
y = cp.Variable((n, n), boolean=True)

# Penalty functions (example: quadratic penalty for lateness)
penalties = cp.sum(cp.pos(f - T)**2)

# Objective
objective = cp.Minimize(penalties + alpha * cp.sum(f))

# Constraints
constraints = []

M = 1e5  # A large constant for big-M method

for i in range(n):
    for j in range(n):
        if i != j:
            constraints += [
                s[i] + p[i] <= s[j] + M * (1 - y[i, j]),
                s[j] + p[j] <= s[i] + M * y[i, j],
                y[i, j] + y[j, i] == 1,
            ]

# Solve the problem using MOSEK
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.MOSEK)

# results
print("Optimal start times:", s.value)
print("Optimal finish times:", f.value)