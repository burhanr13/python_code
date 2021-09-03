import numpy as np
import matplotlib.pyplot as plt


class RectCoord:
    def __init__(self, x, y):
        makeNdarray(x, y)
        self.x = x
        self.y = y

    def values(self):
        return self.x, self.y

    def translate(self, dx, dy):
        return RectCoord(self.x+dx, self.y+dy)

    def reflect(self, axis):
        if axis == "x":
            return RectCoord(self.x, -self.y)
        elif axis == "y":
            return RectCoord(-self.x, self.y)
        elif axis == "xy":
            return RectCoord(-self.x, -self.y)
        else:
            raise ValueError("axis must be either 'x' or 'y' or 'xy'")

    def scale(self, scalar, axis="xy"):
        if axis == "x":
            return RectCoord(self.x, self.y*scalar)
        elif axis == "y":
            return RectCoord(self.x*scalar, self.y)
        elif axis == "xy":
            return RectCoord(self.x*scalar, self.y*scalar)
        else:
            raise ValueError("axis must be either 'x' or 'y' or 'xy'")

    def toPolar(self):
        r = np.sqrt(self.x**2 + self.y**2)
        theta = np.arctan2(self.y, self.x)
        return PolarCoord(r, theta)

    def rotate(self, dtheta):
        pol = self.toPolar()
        pol = pol.rotate(dtheta)
        return pol.toRect()

    def graph(self):
        plt.plot(self.x, self.y)


class Polygon(RectCoord):
    def __init__(self, x, y):
        RectCoord.__init__(self, x, y)
        self.x = np.append(self.x, self.x[0])
        self.y = np.append(self.y, self.y[0])

    def center(self):
        cx = np.mean(self.x[:-1])
        cy = np.mean(self.y[:-1])
        return RectCoord(cx, cy)

    def scaleFromCenter(self, scalar):
        cx = self.center().x
        cy = self.center().y
        scaled = self.translate(-cx, -cy)
        scaled = scaled.scale(scalar)
        scaled = scaled.translate(cx, cy)
        return scaled

    def rotateFromCenter(self, dtheta):
        cx = self.center().x
        cy = self.center().y
        rot = self.translate(-cx, -cy)
        rot = rot.rotate(dtheta)
        rot = rot.translate(cx, cy)
        return rot


class RectFunc(RectCoord):
    def __init__(self, func, start=-10, end=10):
        x = np.arange(start, end, 0.01)
        y = np.array([func(a) for a in x])
        RectCoord.__init__(self, x, y)
        self.func = func


class PolarCoord:
    def __init__(self, r, theta):
        makeNdarray(r, theta)
        for i in range(0, np.size(r)):
            if r[i] < 0:
                r[i] = -r[i]
                theta[i] += np.pi
            theta[i] %= 2*np.pi
            if theta[i] > np.pi:
                theta[i] -= 2*np.pi
        self.r = r
        self.theta = theta

    def values(self):
        return self.r, self.theta

    def rotate(self, dtheta):
        return PolarCoord(self.r, self.theta+dtheta)

    def scale(self, scalar):
        return PolarCoord(self.r*scalar, self.theta)

    def toRect(self):
        x = self.r*np.cos(self.theta)
        y = self.r*np.sin(self.theta)
        return RectCoord(x, y)

    def graph(self):
        rect = self.toRect()
        plt.plot(rect.x, rect.y)


class PolarFunc(PolarCoord):
    def __init__(self, func, start=0, end=2*np.pi):
        theta = np.arange(start, end, 0.01)
        r = np.array([func(a) for a in theta])
        PolarCoord.__init__(self, r, theta)
        self.func = func


def makeNdarray(in1, in2):
    if isinstance(in1, (int, float)) and isinstance(in2, (int, float)):
        x = np.array([in1])
        y = np.array([in2])
    elif isinstance(in1, list) and isinstance(in2, list) and np.size(in1) == np.size(in2):
        x = np.array(in1)
        y = np.array(in2)
    elif isinstance(in1, np.ndarray) and isinstance(in2, np.ndarray) and np.shape(in1) == np.shape(in2):
        pass
    else:
        raise ValueError("in1 and in2 must be numbers, lists, or ndarrays")
