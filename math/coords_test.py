from coords import RectCoord, PolarCoord, Polygon, RectFunc, PolarFunc
import numpy as np 
import matplotlib.pyplot as plt 

def f(x):
    return 5-5*np.cos(x)

F = PolarFunc(f,0,2*np.pi)

F.graph()

F.scale(1.6).graph()

plt.show()





