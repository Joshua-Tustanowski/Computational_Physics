import numpy
from scipy import random
from numpy import pi
import matplotlib.pyplot as plt

b=pi/8
V = numpy.power(b,8)
N = 10

#Generates a set of 8 random real numbers on the range set
def func1():
    return random.rand(8)*b

#Defines the function to be integrated
def func():
    return numpy.sin(numpy.sum(func1()))

#Performs the integration estimate by performing a summation
def func2():
    acc =0
    for i in range(0,N):
        acc += func()
        value = V/N * acc
    return value
print(func2())
# Print out a sample for the integration estimates
def func3():
    x = numpy.zeros(25)
    for j in range(len(x)):
        x[j] = numpy.power(10,6)*func2()
    return x
