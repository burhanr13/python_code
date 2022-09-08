import math
import sys

def lcm(n):
    res = 1
    for i in range(2,n+1):
        if isPrime(i):
            x = i
            while x < n:
                x *= i
            if x > n:
                x /= i
            res *= x
    return res

def isPrime(n):
    for i in range(2,int(math.sqrt(n))+1):
        if not n%i:
            return False
    return True

print(lcm(int(sys.argv[1])))
