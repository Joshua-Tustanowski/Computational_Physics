import numpy as np
from numpy import *
from scipy import integrate
import matplotlib.pyplot as plt
#%matplotlib inline

def f(x):
    res1 = np.sin(pi/2*x**2)
    return res1

def g(x):
    res2 = np.cos(pi/2*x**2)
    return res2

X = np.arange(-5,5,0.01)

#plot(X,f(X))

def F(x):
    res1 = np.zeros_like(x)
    for i,val in enumerate(x):
        y,err = integrate.quad(f,0,val)
        res1[i]=y
    return res1


def G(x):
    res2 = np.zeros_like(x)
    for i,val in enumerate(x):
        y,err = integrate.quad(g,0,val)
        res2[i]=y
    return res2

#plt.plot(X,F(X))
#plt.plot(X,G(X))
plt.plot(F(X),G(X))
