import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('cs.csv')
#print(df)

E = np.array(df['E_nu'].to_list())
mc_cs = np.array(df['mc_cs'].to_list())
cs = np.array(df['cs'].to_list())
npoints = np.array(df['used_points'].to_list())

a = 3.95
b = -0.121


#plt.plot(npoints)
#plt.show()

#plt.plot(E, cs, label='cs')
#plt.plot(E, mc_cs, label='mc cs')
x = np.linspace(5e3, 5e8, int(1e5))
y = [a + b*np.log(i) for i in x]

est = [a + b*np.log(i) for i in E]
cor = mc_cs/est
plt.plot(E, cs, label='cs')
plt.plot(E, mc_cs, label='mc cs')
plt.plot(E, cor, label='correction')
#plt.xscale('log')
plt.legend()
plt.show()


plt.plot(E, mc_cs/cs, label='frac')
plt.plot(E, cor/cs, label='cor frac')
plt.plot(x, y, label='estimate')
#plt.xscale('log')
plt.legend()
plt.show()