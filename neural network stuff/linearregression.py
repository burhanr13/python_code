from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt 

x = np.arange(20)
y = 2.*x + 1
y += np.random.uniform(low=-5,high=5,size=20)

w = np.random.uniform(low=-5,high=5,size=2)

x = np.reshape(x,(20,1))

x = np.hstack((np.ones((20,1)),x))

plt.subplot(121).plot(x[:,1],y,"ro")
l, = plt.subplot(121).plot(x[:,1],x@w,"b-")

def cost(x,y,w):
    return 1/40*np.sum((x@w-y)**2)

history = [cost(x,y,w)]

l2, = plt.subplot(122).plot(history,"g-")

for i in range(0,70):
    w -= 0.0005*1/20*x.T@(x@w-y)
    
    l.set_ydata(x@w)
    history.append(cost(x,y,w))
    l2.set_xdata(np.arange(i+2))
    l2.set_ydata(history)
    plt.subplot(122).relim()
    plt.subplot(122).autoscale_view()
    plt.pause(0.1)
   

print(cost(x,y,w))
print(w)
plt.show()
