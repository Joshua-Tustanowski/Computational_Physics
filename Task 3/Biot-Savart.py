import numpy as np
from numpy import *
import matplotlib.pyplot as plt

class Vector():
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

    def getx(self):
        return self.x
    def gety(self):
        return self.y
    def getz(self):
        return self.z

    def setx(self, x):
        self.x = x
    def sety(self, y):
        self.y = y
    def setz(self, z):
        self.z = z

    def cross(self, a):
        return Vector(self.gety()*a.getz()-self.getz()*a.gety(),
                     -self.getx()*a.getz() + self.getz()*a.getx(),-self.gety()*a.getx()+self.getx()*a.gety())

    def scalar(self,mu):
        self.setx(mu * self.getx())
        self.sety(mu * self.gety())
        self.setz(mu * self.getz())
        return self

    def __sub__(self, vector):
        return Vector(self.getx()-vector.getx(), self.gety()-vector.gety(), self.getz()-vector.getz())

    def __add__(self, vector):
        return Vector(self.getx()+vector.getx(), self.gety()+vector.gety(), self.getz()+vector.getz())

    def magnitude(self):
        return np.sqrt(self.getx()**2 + self.gety()**2 +self.getz()**2)

    def print(self):
        print("(" + str(self.getx()) + "," + str(self.gety()) + "," + str(self.getz())+ ")")


class LineSegment():
    def __init__(self, current, startpoint: Vector, endpoint: Vector):
        self.current = current
        self.startpoint = startpoint
        self.endpoint = endpoint


    def getlengthvector(self):
        return (self.endpoint - self.startpoint)

    def getI(self):
        return self.current

    def setI(self, current):
        self.current = current

    def getstartpoint(self):
        return self.startpoint

    def setstartpoint(self, startpoint):
        self.startpoint = startpoint

    def getendpoint(self):
        return self.getendpoint

    def getendpoint(self, endpoint):
        self.endpoint = endpoint

    def printstart(self):
        self.startpoint.print()

    def print_x(self):
        print(self.startpoint.getx())

    def print_y(self):
        print(self.startpoint.gety())

    def print_z(self):
        print(self.startpoint.getz())


class Wire():
    def __init__(self, linearray):
        self.linearray = linearray

    def getlinearray(self):
        return self.linearray

    def getlinesegmentn(self, n):
        if (n>=len(self.linearray)):
            print("Array Out Of Bounds")
        else:
            return self.linearray[n]

class Wires():
    def __init__(self, wire: Wire):
        self.wire = wire

    def getWire(self):
        return self.wire

    def getWireSegment(self, n):
        if (n>=len(self.wire)):
            print("Array Out Of Bounds")
        else:
            return self.wire[n]


#Start setting up the single coil case
current = 1

# Initialise the splitting of the loop of wire

    # Basic parameters
r=1
t=np.linspace(0,1,1000)
a = 0
b = r * cos(2*pi*t)
c = r * sin(2*pi*t)

#-------------Initilised vectors-----------------------
Vector(a,b,c)
startpoint = []
endpoint = []
circle = Vector(a,b,c)

#----------Initialising the array of line segments---------

startpoint.append(Vector(a,b[len(t)-1],c[len(t)-1]))
endpoint.append(Vector(a,b[0],c[0]))

for i in range(len(t)-1):
    startpoint.append(Vector(a,b[i],c[i]))
    endpoint.append(Vector(a,b[i+1],c[i+1]))
    #print(startpoint[i].print())

#---------Assign all the linesegment vectors to a array------
linesegarray = []
for i in range(len(startpoint)):
    linesegarray.append(LineSegment(current, startpoint[i], endpoint[i]))

#----Assign the linesegments to a wire so we can print out the vector values more effectively-----
wire1 = Wire(linesegarray)

#-----Print out ALL values for the split up loop of wire--------
#for i in range(len(linesegarray)):
#    wire1.getlinesegmentn(i).printstart()


# ----print out the circle and check -----
y_values = []
z_values = []
for i in range(len(linesegarray)):
    y_values.append(wire1.getlinesegmentn(i).startpoint.gety())
    z_values.append(wire1.getlinesegmentn(i).startpoint.getz())

#Plot the circle out
##plt.plot(y_values, z_values)####

#------Calculate the magnetic field components-----------------
def dB(linesegment,vector):
    direction = linesegment.getlengthvector().cross(vector)
    Field = direction.scalar(linesegment.getI()*(1/(4*pi))*(1/(vector.magnitude()**3)))
    return Field

Bx = []
#--------Summing all the field components-------------------
x_value = []
noofpoints=9
for j in range(noofpoints):
    accumulator = Vector(0,0,0)
    for i in range(len(wire1.getlinearray())):
        accumulator += dB(wire1.getlinesegmentn(i),Vector(j*0.5,0,0) - wire1.getlinesegmentn(i).startpoint)
    x_value.append(accumulator.x)
            # Extracting just the B-values along the x-axis

#-----------Plotting the theoretical magnetic field for a single coil-------------

def Btheory(x):
    field = 1/(2*((r ** 2 + x ** 2 ) ** (3/2)))
    return field
xspread = linspace(0,4,100)
Bpred = []
for i in range(len(xspread)):
    Bpred.append(Btheory(xspread[i]))
# Check the printed values
#print(Bpred)

#----------Compare the residuals for the two plots----------------
residual = []
#for i in range(len(t)):
#    residual.append(Bx[i] - Bpred[i])
#print(residual)

array = linspace(0,4,noofpoints)
#print(array)
#plt.show()
#plt.scatter(array,x_value)
#plt.show()
#plt.plot(xspread,Btheory(xspread))
#plt.show()

# --- plotting the vector field ---

# grid time bby
n = 15
m = 15
zeta=[[Vector(0,0,0) for j in range(m)]for i in range(n)]

#for i in range(len(zeta)):
#    for j in range(len(zeta[i])):
#        zeta[i][j].print()

startpointx = -2
startpointy = -2
endpointx = 2
endpointy = 2
StepSizex = (endpointx-startpointx)/n
StepSizey = (endpointy-startpointy)/m

for i in range(len(zeta)):
    for j in range(len(zeta[i])):
        zeta[i][j] = Vector(startpointx+StepSizex*i,startpointy+StepSizey*j,0)
        #zeta[i][j].print()

MagneticGrid = [[Vector(0,0,0) for j in range(m)]for i in range(n)]
#for i in range(len(MagneticGrid)):
#    for j in range(len(MagneticGrid[i])):
#        MagneticGrid[i][j].print()

for i in range(len(zeta)):
    for j in range(len(zeta[i])):
        count = Vector(0,0,0)
        for k in range(len(wire1.getlinearray())):
            if((zeta[i][j] - wire1.getlinesegmentn(k).startpoint).magnitude()>0.25):
                count += dB(wire1.getlinesegmentn(k),zeta[i][j] - wire1.getlinesegmentn(k).startpoint)
            else:
                continue
        MagneticGrid[i][j] = count
        #MagneticGrid[i][j].print()

xcomponents = zeros((m,n))
ycomponents = zeros((m,n))
xcoord = zeros((m,n))
ycoord = zeros((m,n))
for i in range(len(zeta)):
    for j in range(len(zeta[i])):
        xcomponents[i][j] = MagneticGrid[i][j].getx()
        ycomponents[i][j] = MagneticGrid[i][j].gety()
        xcoord[i][j] = zeta[i][j].getx()
        ycoord[i][j] = zeta[i][j].gety()
        #print(ycomponents[i][j])

plt.quiver(xcoord,ycoord,xcomponents,ycomponents,pivot='mid', scale=5)
## Calculating the number of points needed to give a sensible result

def circlepoint(noPoints, centre):
    t = linspace(0,1,noPoints)
    r=1
    a = centre
    b = r * cos(2*pi*t)
    c = r * sin(2*pi*t)
    startpoint = []
    endpoint = []
    startpoint.append(Vector(a,b[len(t)-1],c[len(t)-1]))
    endpoint.append(Vector(a,b[0],c[0]))
    for i in range(len(t)-1):
        startpoint.append(Vector(a,b[i],c[i]))
        endpoint.append(Vector(a,b[i+1],c[i+1]))
    linesegarray = []
    for i in range(len(startpoint)):
        linesegarray.append(LineSegment(current, startpoint[i], endpoint[i]))
    return linesegarray
## Checking the circle function works
#circle = circlepoint(100,0)
#wireA = Wire(circle)

#y_values = []
#z_values = []
#for i in range(len(circle)):
#    y_values.append(wireA.getlinesegmentn(i).startpoint.gety())
#    z_values.append(wireA.getlinesegmentn(i).startpoint.getz())

#plt.plot(y_values,z_values)
#plt.show()
value = []
for j in range(1,50):
    wire = Wire(circlepoint(j,0))
    counter = Vector(0,0,0)
    for i in range(len(wire.getlinearray())):
        counter += dB(wire.getlinesegmentn(i),Vector(0,0,0) - wire.getlinesegmentn(i).startpoint)
    value.append(counter.x)

# comparing the value of the approximation against the actual value
Number_of_points = []
for i in range(len(value)):
    value[i] = value[i] - Btheory(0)
    Number_of_points.append(i)
#print(value)
fig, ax = plt.subplots()
ax.plot(Number_of_points,value, color='red')
ax.set(xlabel='Number of points', ylabel='Magnetic field residual error at the origin',
       title='How the number of points affects the magnetic field calculation')
ax.grid()
plt.show()

# ---------- Helmholtz coils --------------

# Initialise the coils of wire

wire3 = Wire(circlepoint(30, 0.5))
wire4 = Wire(circlepoint(30, -0.5))

# Calculate the magnetic field for them being in the vicinity of each other

field_value = []
displace = []
for j in range(-10,11):
    displace.append(j*0.1)
    accumulator_new = Vector(0,0,0)
    for i in range(len(wire4.getlinearray())):
        accumulator_new += dB(wire4.getlinesegmentn(i),Vector(j*0.1,0,0) - wire4.getlinesegmentn(i).startpoint) + dB(wire3.getlinesegmentn(i),Vector(j*0.1,0,0) - wire3.getlinesegmentn(i).startpoint)
    field_value.append(accumulator_new.x)

Bcenter = (0.8) ** (3/2)

# --- Test the looped values ----
print(field_value)
print(displace)

# --- Plot the uniformity of the field ---
fig, ax = plt.subplots()
ax.plot(displace,field_value)

# --- Plotting the residual for the uniformity of the magnetic field ---
for i in range(len(field_value)):
    if(field_value[i] < 0.70):
        field_value[i] = 0
    else:
        field_value[i] = field_value[i] - Bcenter

#print(field_value)
ax.plot(displace, field_value)
ax.legend(['field','residuals'])
plt.show()

# --- Plotting the fields of the Helmhotz coils ---

# grid time bby
q = 15
p = 15
zeta1=[[Vector(0,0,0) for j in range(p)]for i in range(q)]

#for i in range(len(zeta)):
#    for j in range(len(zeta[i])):
#        zeta[i][j].print()

startpointx1 = -1.5
startpointy1 = -1.5
endpointx1 = 1.5
endpointy1 = 1.5
StepSizex1 = (endpointx1-startpointx1)/q
StepSizey1 = (endpointy1-startpointy1)/p

for i in range(len(zeta1)):
    for j in range(len(zeta1[i])):
        zeta1[i][j] = Vector(startpointx1+StepSizex1*i,startpointy1+StepSizey1*j,0)
        #zeta[i][j].print()

MagneticGrid1 = [[Vector(0,0,0) for j in range(p)]for i in range(q)]
#for i in range(len(MagneticGrid)):
#    for j in range(len(MagneticGrid[i])):
#        MagneticGrid[i][j].print()

for i in range(len(zeta1)):
    for j in range(len(zeta1[i])):
        count1 = Vector(0,0,0)
        for k in range(len(wire4.getlinearray())):
            if((zeta1[i][j] - wire3.getlinesegmentn(k).startpoint).magnitude()>0.25 and (zeta1[i][j] - wire4.getlinesegmentn(k).startpoint).magnitude()>0.25):
                count1 += dB(wire4.getlinesegmentn(k),zeta1[i][j] - wire4.getlinesegmentn(k).startpoint) + dB(wire3.getlinesegmentn(k),zeta1[i][j] - wire3.getlinesegmentn(k).startpoint)
            else:
                continue
        MagneticGrid1[i][j] = count1
        #MagneticGrid[i][j].print()

xcomponents1 = zeros((q,p))
ycomponents1 = zeros((q,p))
xcoord1 = zeros((q,p))
ycoord1 = zeros((q,p))
for i in range(len(zeta1)):
    for j in range(len(zeta1[i])):
        xcomponents1[i][j] = MagneticGrid1[i][j].getx()
        ycomponents1[i][j] = MagneticGrid1[i][j].gety()
        xcoord1[i][j] = zeta1[i][j].getx()
        ycoord1[i][j] = zeta1[i][j].gety()
        #print(ycomponents[i][j])

plt.quiver(xcoord1,ycoord1,xcomponents1,ycomponents1,pivot='mid', scale=8)
# Add a cylinder of the appropriate dimensions here

## ---N coils in a row--- ##
#N = 10 # Number of coils
R = 1 # Radius of coil
circles = []
for j in range(-5,6*R):
    circles.append(circlepoint(30,10*j))
    #print(j)

# Generating all the wires
wires = []
print(len(circles))
for j in range(len(circles)):
    wires.append(Wire(circles[j]))

# Picking out the singular wires and calculating the magnetic field at zero
x_values2 = []
for k in range(1,11):
    accumulator = Vector(0,0,0)
    for i in range(len(wires[0].getlinearray())):
        for j in range(len(wires)):
            accumulator += dB(wires[j].getlinesegmentn(i),Vector(k*0.1,0,0) - wires[j].getlinesegmentn(i).startpoint)
    #accumulator.print()

#print(x_values2)
# grid time bby
u = 20
v = 20
zeta2=[[Vector(0,0,0) for j in range(v)]for i in range(u)]

#for i in range(len(zeta)):
#    for j in range(len(zeta[i])):
#        zeta[i][j].print()

startpointx2 = -3
startpointy2 = -3
endpointx2 = 3
endpointy2 = 3
StepSizex2 = (endpointx2-startpointx2)/u
StepSizey2 = (endpointy2-startpointy2)/v

for i in range(len(zeta2)):
    for j in range(len(zeta2[i])):
        zeta2[i][j] = Vector(startpointx2+StepSizex2*i,startpointy2+StepSizey2*j,0)
        #zeta[i][j].print()

MagneticGrid2 = [[Vector(0,0,0) for j in range(v)]for i in range(u)]
#for i in range(len(MagneticGrid)):
#    for j in range(len(MagneticGrid[i])):
#        MagneticGrid[i][j].print()

for i in range(len(zeta2)):
    for j in range(len(zeta2[i])):
        count2 = Vector(0,0,0)
        for m in range(len(wires[0].getlinearray())):
            for n in range(len(wires)):
                count2 += dB(wires[n].getlinesegmentn(m),zeta2[i][j] - wires[n].getlinesegmentn(m).startpoint)
        MagneticGrid2[i][j] = count2
        #MagneticGrid[i][j].print()

xcomponents2 = zeros((u,v))
ycomponents2 = zeros((u,v))
xcoord2 = zeros((u,v))
ycoord2 = zeros((u,v))
for i in range(len(zeta2)):
    for j in range(len(zeta2[i])):
        xcomponents2[i][j] = MagneticGrid2[i][j].getx()
        ycomponents2[i][j] = MagneticGrid2[i][j].gety()
        xcoord2[i][j] = zeta2[i][j].getx()
        ycoord2[i][j] = zeta2[i][j].gety()
        #print(ycomponents[i][j])

plt.quiver(xcoord2,ycoord2,xcomponents2,ycomponents2,pivot='mid', scale=8)
