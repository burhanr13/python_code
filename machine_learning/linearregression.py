from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt 

n = 20
iterations = 100
alpha = 0.01

x = np.random.uniform(-10,10,n)
y = 2.*x + 1
y += np.random.uniform(low=-5,high=5,size=n)

w = np.zeros(2) #np.random.uniform(low=-5,high=5,size=2)

x = np.reshape(x,(n,1))

x = np.hstack((np.ones((n,1)),x))

plt.subplot(121).plot(x[:,1],y,"ro")
l, = plt.subplot(121).plot(x[:,1],x@w,"b-")

def cost(x,y,w):
    return 1/(2*n)*np.sum((x@w-y)**2)

history = [cost(x,y,w)]

l2, = plt.subplot(122).plot(history,"g-")

for i in range(iterations):
    w -= alpha*1/n*x.T@(x@w-y)
    
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
