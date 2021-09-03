def fact(x):
	if x == 0:
		result = 1
	else:
		result = x
		x -= 1
		while x >= 1:
			result *= x
			x -= 1
	return result

def c(n,r):
	return (fact(n))/(fact(r)*fact(n-r))

def pascal(a):
	ans = []
	for i in range(0,a+1):
		ans.append(c(a,i))
	return ans


