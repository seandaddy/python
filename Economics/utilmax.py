from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

def utility(x):
    x1, x2 = x
    return -(x1**(1/3) + x2**(1/3))

def constraint(x):
    x1, x2 = x
    return 100 - x1 - x2

x0 = [0, 0]
b = (0, 100)
bnds = (b, b)
con = {'type': 'eq', 'fun': constraint}

sol = minimize(utility, x0, method='SLSQP', bounds=bnds, constraints=con)

print(sol.x)
