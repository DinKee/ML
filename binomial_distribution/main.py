import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.interpolate import make_interp_spline, BSpline

#Q1
p=[0.25,0.05]
e=[0.01,0.05]
n=[100,3000,50000]

pro=[[0]*2 for i in range(3)]

for i in range(3):
    for j in range(2):
        for k in range(math.ceil(p[j]*n[i]*(1-e[j])),math.floor(p[j]*n[i]*(1+e[j]))+1):
            pro[i][j] = pro[i][j] + binom.pmf(k,n[i],p[j])

print("-----------------------------------------")
print("| n\p   | p =0.7,e=0.01 | p =0.5,e=0.01 |")
print("|n=100  | %.4f          | %.4f          |" % (pro[0][0],pro[0][1]))
print("-----------------------------------------")
print("|n=3000 | %.4f          | %.4f          |" % (pro[1][0],pro[1][1]))
print("-----------------------------------------")
print("|n=50000| %.4f          | %.4f          |" % (pro[2][0],pro[2][1]))
print("-----------------------------------------")

################################
#Q2

n_2=[5,50]
p=0.1
pro_2 = [[0]*(n_2[0]+1),[0]*(n_2[1]+1)]
com_2 = [[0]*(n_2[0]+1),[0]*(n_2[1]+1)]
bino_2= [[0]*(n_2[0]+1),[0]*(n_2[1]+1)]
for j in range(2):
    for i in range(n_2[j]+1):
        pro_2[j][i] = math.log(pow(p,i)*pow(1-p,n_2[j]-i))
        com_2[j][i] = math.log(math.factorial(n_2[j])/math.factorial(n_2[j]-i)/math.factorial(i))
        bino_2[j][i] = pro_2[j][i] + com_2[j][i]


x_1 = [i/5 for i in range(6)]
x_2 = [i/50 for i in range(51)]
x = np.linspace(0, 1, 300)

pro_2[0]=make_interp_spline(x_1,pro_2[0])(x)
com_2[0]=make_interp_spline(x_1,com_2[0])(x)
bino_2[0]=make_interp_spline(x_1,bino_2[0])(x)
pro_2[1]=make_interp_spline(x_2,pro_2[1])(x)
com_2[1]=make_interp_spline(x_2,com_2[1])(x)
bino_2[1]=make_interp_spline(x_2,bino_2[1])(x)

plt.plot(x,pro_2[0],label='probability_n=5_p=0.1')
# plt.plot(x_1,gaussian_filter1d(pro_2[0],2),label='probability_n=5_p=0.1 filtered')
plt.plot(x,com_2[0],label='combination_n=5_p=0.1')
# plt.plot(x_1,gaussian_filter1d(com_2[0],2),label='combination_n=5_p=0.1 filtered')
plt.plot(x,bino_2[0],label='binomial_n=5_p=0.1')
# plt.plot(x_1,gaussian_filter1d(bino_2[0],2),label='binomial_n=5_p=0.1 filtered')
plt.plot(x,pro_2[1],label='probability_n=50_p=0.1')
plt.plot(x,com_2[1],label='combination_n=50_p=0.1')
plt.plot(x,bino_2[1],label='binomial_n=50_p=0.1')
plt.xlabel("X")
plt.ylabel("(log10)Y")
plt.legend(loc='best')
plt.show()
