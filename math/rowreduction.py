import numpy as np


def reduce(a: np.array):
    b = np.copy(a).astype(np.float64)

    i, j = 0, 0
    while i < np.size(b,1) and j < np.size(b, 0):
        k = j
        while b[k, i] == 0:
            k += 1
            if k >= np.size(b, 0):
                i += 1
                k = j
                if i >= np.size(b, 1):
                    return b
        b[[j,k]] = b[[k,j]]

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
            if i!=np.size(b,1)-1: free.append(i)
            i += 1
        if i >= len(row):
            break
        if i == len(row) - 1:
            return None
        pivots.append(i)
        i += 1

    free += range(pivots[-1]+1 if free == [] else max(pivots[-1],free[-1])+1, np.size(b,1))
    
    m = -b[:,free]
    m[:,-1] *= -1

    soln = np.zeros((np.size(b,1)-1,len(free)))
    for n, i in enumerate(pivots):
        soln[i] = m[n]
    
    for n, i in enumerate(free[:-1]):
        soln[i,n] = 1

    return soln

    


def randMat():
    return np.random.randint(-20, 20, (np.random.randint(2, 10), np.random.randint(2, 10)))


#A = np.random.randint(-10, 10, (3, 4))

#A = np.array([[2,-2,1,-2,-2],[1,-1,-1,2,-1],[-1,1,2,-4,1],[3,-3,0,0,-3]])
#A = randMat()

#A = np.array([[1,2,1,0,1],[0,1,4,3,2],[0,0,2,2,4]])

#A = np.array([[1,0,3,7],[0,1,-4,1],[0,0,0,0]])

A = np.array([[2,1,3,4,0,-1],
              [-2,-1,-3,-4,5,6],
              [4,2,7,6,1,-1]])

print(A)
print(np.round(reduce(A), 1))

sol = solveSystem(A)

print(sol if sol is None else np.round(sol,1))
