import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('cs.csv')
#print(df)

E = np.array(df['E_nu'].to_list())
mc_cs = np.array(df['mc_cs'].to_list())
cs = np.array(df['cs'].to_list())
npoints = np.array(df['used_points'].to_list())

a = 2.4 + .42
b = -0.121

plt.plot(npoints)
plt.show()

#plt.plot(E, cs, label='cs')
#plt.plot(E, mc_cs, label='mc cs')
plt.plot(E, mc_cs/cs, label='frac')
x = np.linspace(5e3, 5e8, int(1e5))
y = [a + b*np.log10(i) for i in x]
plt.plot(x, y, label='estimate')
#plt.xscale('log')
plt.legend()
plt.show()