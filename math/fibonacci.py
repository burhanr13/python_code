def fib(n):
    a = 1
    b = 1
    for i in range(n-2):
        c = a + b
        a = b
        b = c
    return c

print(fib(1000))