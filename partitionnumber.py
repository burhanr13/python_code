def findArrayNum(n) -> int:
    if n % 2 == 0:
        return (n/2)*(n/2+1)/2 + (n/2)**2
    return ((n-1)/2)*((n-1)/2+1)/2 + ((n+1)/2)**2


def partitionNum(n):
    a = [0 for i in range(n+1)]
    for i in range(n+1):
        if i == 0:
            a[0] = 1
        else:
            sign = 0
            for ind in [int(findArrayNum(j)) for j in range(1,i+1)]:
                if ind > i:
                    break
                a[i] += (-1, 1)[sign % 4 < 2] * a[i-ind]
                sign+=1
    return a[n]


print(partitionNum(666))
