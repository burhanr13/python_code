import numpy as np


def reduce(a: np.array):
    b = np.copy(a).astype(np.float64)

    i, j = 0, 0
    while i < min(np.size(b, 0), np.size(b, 1)) and j < np.size(b, 0):
        k = j
        while b[k, i] == 0:
            k += 1
            if k >= np.size(b, 0):
                i += 1
                k = j
                if i >= np.size(b, 1):
                    return b
        b[j], b[k] = b[k], b[j]

        b[j] /= b[j, i]
        for m in range(np.size(b, 0)):
            if m == j:
                continue
            b[m] -= b[m, i]/b[j, i] * b[j]
        i += 1
        j += 1

    return b


def solveSystem(a):
    b = reduce(a)
    pivots = []
    free = []
    i = 0
    for row in b:
        while i < len(row) and row[i] == 0:
            free.append(i)
            i += 1
        if i >= len(row):
            break
        if i == len(row) - 1:
            return None
        pivots.append(i)
        i += 1
    
    m = -b[:,free]
    m[:,-1] *= -1

    soln = np.zeros((np.size(b,1)-1,len(free)))
    for n, i in enumerate(pivots):
        soln[i] = m[n]
    
    for n, i in enumerate(free[:-1]):
        soln[i,n] = 1

    return soln

    


def randMat():
    return np.random.randint(0, 20, (np.random.randint(1, 10), np.random.randint(1, 10)))


A = np.random.randint(-10, 10, (3, 4))

A= np.array([[2,-2,1,-2,-2],[1,-1,-1,2,-1],[-1,1,2,-4,1],[3,-3,0,0,-3]])


print(A)
print(np.round(reduce(A), 1))

print(np.round(solveSystem(A), 1))
