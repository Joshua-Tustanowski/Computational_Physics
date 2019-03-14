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
# Print the estimate of the integrals as a check
#print(func3())

#calculates the squared average of the function in the range
def func5():
    count = 0
    for i in range(0,N):
        count += numpy.power(func(),2)
        result = 1/N * count
    return result   
    
#calculating the mean and standard deviation
def mean(x):
    sum1 = 0
    for j in range(len(x)):
        sum1 += x[j]
    return sum1/len(x)
#print the mean of the set of integral estimates
#print(mean(func3()))

#calculate the mean2 for the integral estimates
def mean2(x):
    sum2 = 0
    for j in range(0,len(x)):
        sum2 += numpy.power(x[j],2)
    return sum2/len(x)
#Check the function is working properly
#print(mean2(func3()))

def sdv(x):
    meansq = numpy.power(mean(x),2)
    diff = (mean2(x) - meansq)/len(x)
    return numpy.power(diff,0.5)


y=[]
z=[]
error = []
values = []
#for N in range(1,100):
#    error.append(mean(func3())-exact)
#    values.append(mean(func3()))
#    z.append(N)
#print(values)
#print(error)
for N in range(1,1000):
    #print(mean(func3()),mean2(func3()),sdv(func3()),N)
    y.append(sdv(func3()))
    z.append(N)
#plot the data
plt.plot(z,y)
plt.show()   

print('done')
