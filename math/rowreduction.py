import numpy as np


def old_reduce(a: np.array):
    b = np.copy(a)
    b = b.astype(np.float64)

    for i in range(min(np.size(b, 1), np.size(b, 0))):
        if b[i, i] == 0:
            continue
        for j in range(i+1, np.size(b, 0)):
            b[j] -= b[j, i]/b[i, i] * b[i]

    for i in reversed(range(min(np.size(b, 1), np.size(b, 0)))):
        if b[i, i] == 0:
            continue
        for j in reversed(range(i)):
            b[j] -= b[j, i]/b[i, i] * b[i]

    for i in range(min(np.size(b, 0), np.size(b, 1))):
        if b[i, i] == 0:
            continue
        b[i] /= b[i, i]

    return b


def reduce(a: np.array):
    b = np.copy(a)
    b = b.astype(np.float64)

    i = 0
    j = 0
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


def randMat():
    return np.random.randint(0, 20, (np.random.randint(1, 10), np.random.randint(1, 10)))


A = np.random.randint(-10, 10, (3, 4))

A = [[0,1,3,4],[1,0,3,3],[1,2,9,11]]


print(A)
print(np.round(reduce(A), 1))
