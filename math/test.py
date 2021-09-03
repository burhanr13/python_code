import calculus as cs
import numpy as np
from coords import RectFunc
import matplotlib.pyplot as plt

def graph(f):
    a = np.linspace(-10,10,10000)
    plt.plot(a,[f(x) for x in a])

# graph(lambda x : np.sin(np.cos(np.tan(x))))
# graph(lambda x : cs.derivative(lambda a: np.sin(np.cos(np.tan(a))), x))


# plt.show()

print(cs.integral(lambda x : x**-2,1,1000))