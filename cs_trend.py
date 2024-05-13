import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-colorblind')
#plt.style.use('Solarize_Light2')
plt.rcParams['text.usetex'] = True

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
pdf21 = np.array(df['PDF21'].to_list())/4
pdf31 = np.array(df['PDF31'].to_list())
pdf40 = np.array(df['PDF40'].to_list())/4
struc = np.array(df['struc'].to_list())
logstruc = np.array(df['logstruc'].to_list())

log21 = np.array(df['log21'].to_list())
log31 = np.array(df['log31'].to_list())
log40 = np.array(df['log40'].to_list())

log21_err = np.array(df['log21_err'].to_list())
log31_err = np.array(df['log31_err'].to_list())
log40_err = np.array(df['log40_err'].to_list())

cs = np.array(df['cs'].to_list())
npoints = np.array(df['used_points'].to_list())

plt.scatter(E, cs, label='cs ref')
#plt.plot(E_nu, ref_cs, label='ref')
#plt.plot(E_nu2, ref_w, label='w')
#plt.plot(E, pdf21, label='pdf21')
#plt.plot(E, pdf31, label='pdf31')
plt.scatter(E, log21, label='log21')
plt.scatter(E, log31, label='log31')
plt.scatter(E, log40, label='log40')
plt.scatter(E, logstruc, label='logstruc')
#plt.plot(E, pdf40, label='pdf40')
#plt.plot(E, struc, label='struc')
#plt.plot(E, cor, label='correction')
plt.xlabel(r'$E_{\nu}$')
plt.ylabel('cross section in pb')
plt.xscale('log')
plt.yscale('log')
#plt.xlim(5e3, 5e9)
#plt.ylim(0, 1.7e4)
plt.legend()
plt.show()

if True:
    #21_err = np.stack(log)
    #plt.plot(s, pdf31/cs, label='frac31')
    #plt.plot(s, struc/cs, label='frac_struc')
    ratio = log21/cs
    plt.errorbar(s[:-6], ratio[:-6], yerr=21_err, label='PDF2.1 LO')
    plt.errorbar(s, log31/cs, yerr=31_err, label='PDF3.1 LO')
    plt.errorbar(s, log40/cs, yerr=40_err, label='PDF4.0 LO')
    plt.scatter(s, logstruc/cs, label='PDF3.1 NLO')
    plt.xscale('log')
    plt.legend()
    plt.xlabel('Neutrino Energy [GeV]')
    plt.ylabel(r'$\sigma_{\nu} /\ \sigma_{\nu}^{ref}$')
    #plt.ylabel(r'$\sigma$')

    plt.savefig(f"Figs/pdfs_ratio.pdf", format="pdf", bbox_inches="tight")
    plt.show()