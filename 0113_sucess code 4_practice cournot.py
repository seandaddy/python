from sympy import Symbol, symbols, diff
from sympy.solvers import solve
import sympy
sympy.init_printing()

q1, q2 = symbols('q1 q2')
b = symbols('b')
c1, c2 = symbols('c1 c2')
x1, x2 = symbols('x1 x2')

def p(q1,q2):
    return 5 - b* q1 - b* q2

def cost(q,c):
    return c*q

def invest(q,x):
    return x*q-x*x

def profit(q1,q2,c,x):
    return p(q1,q2)*q1 - cost(q1,c) + invest(q1,x) 

foc1 = diff(profit(q1,q2,c1,x1),q1)
foc2 = diff(profit(q2,q1,c2,x2),q2)

ans=sympy.solve([foc1,foc2], [q1, q2])
print(ans[q1])
print(ans[q2])

foc3 = diff(profit(ans[q1],ans[q2],c1,x1),x1)
foc4 = diff(profit(ans[q2],ans[q1],c2,x2),x2)

z=sympy.solve([foc3, foc4], [x1, x2])
print(z[x1])
print(z[x2])

z1 = z[x1].subs([(c1,1), (c2,1), (b,1)])
z2 = z[x2].subs([(c1,1), (c2,1), (b,1)])
print(z1)
print(z2)

ans1=ans[q1].subs([(x1,z[x1]), (x2,z[x2])])
ans2=ans[q2].subs([(x1,z[x1]), (x2,z[x2])])
eq1 = ans1.subs([(c1,1), (c2,1), (b,1)])
eq2 = ans2.subs([(c1,1), (c2,1), (b,1)])
print(eq1)
print(eq2)

pro1=profit(ans1,ans2,c1,z[x1]).subs([(c1,1), (c2,1), (b,1)])
pro2=profit(ans1,ans2,c2,z[x2]).subs([(c1,1), (c2,1), (b,1)])
print(pro1)
print(pro2)