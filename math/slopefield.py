import numpy as np
import matplotlib.pyplot as plt

import calculus as cs


def slopeField(dydx):
    for x in range(-10, 10):
        for y in range(-10, 10):
            try:
                slope = dydx(x, y)
            except ZeroDivisionError:
                plt.plot([x, x], [y+0.5, y-0.5], 'b-')
            else:
                mag = np.sqrt(slope ** 2 + 1)
                plt.plot([x - 1 / (2 * mag), x + 1 / (2 * mag)],
                         [y - slope / (2 * mag), y + slope / (2 * mag)], 'b-')
    plt.show()


slopeField(lambda x, y: cs.derivative(lambda t: t**3-5*t, x))
