import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('cs_3.csv')
#print(df)

E = np.array(df['E_nu'].to_list())
s = 2*0.938*np.array(df['E_nu'].to_list())
mc_cs = np.array(df['mc_cs'].to_list())
cs = np.array(df['cs'].to_list())
npoints = np.array(df['used_points'].to_list())

plt.plot(E, cs, label='cs ref')
plt.plot(E, mc_cs, label='cs calc')
#plt.plot(E, cor, label='correction')
plt.xlabel('E_nu')
plt.ylabel('cross section in pb')
plt.xscale('log')
plt.legend()
plt.show()


plt.plot(s, mc_cs/cs, label='frac')
#plt.xscale('log')
plt.legend()
plt.show()