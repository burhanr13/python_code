def fact(n):
    if n == 0:
        return 1
    
    res = 1
    for i in range(n):
        res *= i+1
    return res

print(fact(10))