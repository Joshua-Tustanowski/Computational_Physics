import numpy as np
from numpy import *
from numpy import pi
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import optimize

def pend(y,t):
    theta, omega = y
    dydt = [omega, -sin(theta) - q*omega + F * sin(Q*t)]
    return dydt

# ---Initial conditions---
y0 = [pi/2, 0] 
q = 0.5
F = 1.465
Q = 2/3

# ---Generate equally spaced points--- 
t = np.linspace(0, 20, 1000)
y = odeint(pend, y0, t)
angle = y[:,0]
Omega = y[:,1]

# ---producing plots for different values of q---

z = []
for i in range(len(y)):
    for q in range(1,10):
        z.append(y[i,0])
#print(z)

# ---finding the value when the function is zero---
for i in range(1, len(y)):
    if abs(Omega[i] + Omega[i-1]) == abs(Omega[i]) + abs(Omega[i-1]):
            continue
    else:
        print(t[i+1])

#start = zeros((5,len(y)))
#print(start)

def func(x):
    return 0.01*np.cos(x)

Energy = []
for i in range(len(y)):
    #print("{:8.4g} {:8.4g} {:8.4g} {:8.4g}".format(t[i],y[i,0],y[i,1],50*(y[i,1]**2)+50*(y[i,0]**2)))
    Energy.append(50*(y[i,1]**2)+50*(y[i,0]**2))
    
y3 = y[:,0] 
# ---plotting the solutions out---
#for q in range(1,12,5):
plt.plot(t, angle, 'b', label='theta(t)')
#plt.show()    
plt.plot(t, Omega, 'g', label='omega(t)')
#plt.plot(t,func(t),'r',label='Theoretical angle')
#plt.plot(Periods,theta0)
plt.legend(loc='best')
plt.title('q = 0.5, F = 1.465')
plt.xlabel('Period/4 (s)')
plt.ylabel("Initial angle and angular speed (arbituary angle)")
plt.grid()
plt.show()
## For theta = pi/2, the period is 1.86181*4
theta0 = [0.01,0.1,0.50,1.00,1.50,pi/2,2.00,2.50,3.00]
Periods = [1.5736,1.5756,1.5996,1.6777,1.8279,1.8619,2.0941,2.5867, 4.0440]

#Damped oscillator
F = [0.5,1.2,1.44,1.465] 
Period = [3.5235,2.3824,2.0220,1.9820]

plt.plot(F,Period)
plt.grid()
plt.xlabel('Forcing constant (arbituary units)')
plt.ylabel('Period/4 (s)')
plt.title('The influence of a forcing term on a Pendulums period ')
plt.show
