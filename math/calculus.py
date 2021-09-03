import numpy as np


def limit(f, x):
    e = 1e-3
    d = np.inf
    while np.abs(d) > 1e-10:
        d = f(x+e) - f(x+e/2)
        e /= 2
        if e <= 1e-15:
            return np.inf
    return f(x+e)
    

def derivative(f, x):
    try:
        return limit(lambda h : (f(x+h)-f(x))/h, 0)
    except Exception:
        return np.nan

def integral(f, a, b):
    try:
        n = 1e6
        h = (b-a)/n
        return sum([f(a+i*h)*h for i in range(int(n))])
    except Exception:
        return np.nan