from scipy import optimize,arange
from numpy import array

def demand(q1,q2):
     return 1-b1*q1-b2*q2
     
def cost(q,c):
    return c*q
     
def profit(q1,q2,c1):
    return demand(q1,q2)*q1-cost(q1,c1)
  
b1 = 1
b2 = 1  

def reaction(q2,c1):
      q1=optimize.fminbound(lambda q: -profit(q,q2,c1),0,1,full_output=1)
      return q1[0]
      
def fixed_point(q,param):  
    return [q[0]-reaction(q[1],param[0]), q[1]-reaction(q[0],param[1])]
      
param = [0.1,0.1,1,1]
intitial_guess = [0,0]

ans = optimize.fsolve(lambda q: fixed_point(q,[param[0],param[1]]), intitial_guess)
print(ans)

