def preprocess(Z):
    A = []
    r = len(Z)
    c = len(Z[0])
    for i in range(r):
        row = []
        if i > 0:
            sum = A[i-1][0]
        else:
            sum = 0
        for j in range(c):
            if i > 0 and j > 0:
                sum += A[i-1][j] - A[i-1][j-1]
            sum += Z[i][j]
            row.append(sum)
        A.append(row)    
    return A

def subsum(Z, A, r1, r2, c1, c2):
    sum = 0
    if r1 > 0:
        sum -= A[r1-1][c2]
    if c1 > 0:
        sum -= A[r2][c1-1]
    if r1 > 0 and c1 > 0:
        sum += A[r1-1][c1-1]
    sum += A[r2][c2]
    return sum

def printmat(M):
    for i in M:
        print(i)

mat = [[0,0,1,0],
       [1,0,1,1],
       [0,0,0,1],
       [1,0,0,1]]

matsums = preprocess(mat)

printmat(matsums)
print(subsum(mat, matsums, 0, 2, 1, 3))
    