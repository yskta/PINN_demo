#ref：https://sites.google.com/site/dueyama/home-jp/python_simulation/6
import math
import matplotlib.pyplot as plt

T = 0.2
M = 1000
N = 50 
D = 1.0

h = 1.0/float(N)
tau = T/float(M)
alpha = D*tau/(h*h)
INTV = 200
def u0(x):
	return 2*x*(1 - x)

x, u, new_u = [0.0]*(N + 2), [0.0]*(N + 2), [0.0]*(N + 2)
ug = [0.0]*(N + 2)

for j in range(1, N + 1):
	x[j] = (j - 0.5)*h
	u[j] = u0(x[j])
x[0] = 0
x[N+1] = 1.0
t = 0
for j in range(1, N + 1):
	ug[j] = u[j]
plt.plot(x, ug, label="t="+str(t))

for k in range(1, M + 1):
	t = k*tau
	u[0] = -u[1]
	u[N+1] = -u[N]
	for j in range(1, N + 1):
		new_u[j] = alpha*u[j-1] + (1 - 2*alpha)*u[j] + alpha*u[j+1]
	for j in range(1, N + 1):
		u[j] = new_u[j] 

	if k%INTV == 0:
		for j in range(1, N + 1):
			ug[j] = u[j]
		plt.plot(x, ug, label="t="+str(('%.2f'%t)))

plt.title("6-1")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.show()