import numpy as np
import matplotlib.pyplot as plt

def utility(x):
    x1, x2 = x
    return (x1**(1/3) + x2**(1/3))

def constraint(x):
    x1, x2 = x
    return 100 - x1 - x2

budgets = np.linspace(0, 100, 6)
    
x1 = np.linspace(0, 100, 100)
x2 = np.linspace(0, 100, 100)
X1, X2 = np.meshgrid(x1, x2)

U = np.zeros(X1.shape)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        U[i,j] = utility([X1[i,j], X2[i,j]])

C = X1 + X2

# min_U = np.min(U)
# max_U = np.max(U)
# contour_levels = np.linspace(min_U, max_U, 25)
#levels=contour_levels
plt.contour(X1, X2, U, levels=[7.368], cmap='RdYlBu')
plt.contour(X1, X2, C, levels=[100], cmap='gray')

plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()
