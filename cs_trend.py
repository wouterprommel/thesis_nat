import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('cs_3.csv')
#print(df)
df_ref = pd.read_csv('ref_cs.csv')
E_nu = np.array(df_ref['E_nu'])
ref_cs = np.array(df_ref['cs'])

df_w = pd.read_csv('ref_wrong.csv')
E_nu2 = np.array(df_w['E_nu'])
ref_w = np.array(df_w['cs'])

E = np.array(df['E_nu'].to_list())
s = 2*0.938*np.array(df['E_nu'].to_list())
pdf21 = np.array(df['PDF21'].to_list())
pdf31 = np.array(df['PDF31'].to_list())
pdf40 = np.array(df['PDF40'].to_list())
struc = np.array(df['struc'].to_list())
convert = np.array(df['convert'].to_list())
cs = np.array(df['cs'].to_list())
npoints = np.array(df['used_points'].to_list())

plt.plot(E, cs, label='cs ref')
plt.plot(E_nu, ref_cs, label='ref')
#plt.plot(E_nu2, ref_w, label='w')
#plt.plot(E, pdf21, label='pdf21')
plt.plot(E, pdf31, label='pdf31')
plt.plot(E, convert, label='convert')
#plt.plot(E, pdf40, label='pdf40')
#plt.plot(E, struc, label='struc')
#plt.plot(E, cor, label='correction')
plt.xlabel('E_nu')
plt.ylabel('cross section in pb')
plt.xscale('log')
#plt.xlim(5e3, 5e9)
#plt.ylim(0, 1.7e4)
plt.legend()
plt.show()

if True:
    #plt.plot(s, pdf31/cs, label='frac31')
    #plt.plot(s, struc/cs, label='frac_struc')
    plt.plot(s, convert/cs)
    plt.xscale('log')
    plt.legend()
    plt.show()