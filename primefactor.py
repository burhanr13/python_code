import math
import sys
import time

def largestPrimeFactor(n):
    divisor = 2
    while(divisor <= math.sqrt(n)):
        if n%divisor:
            divisor += 1
        else:
            n /= divisor
    return n

start = time.time()

print(largestPrimeFactor(int(sys.argv[1])))

print(time.time()-start)
