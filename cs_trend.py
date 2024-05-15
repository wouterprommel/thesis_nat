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

err21 = np.stack([log21 + log21_err, log21 - log21_err])
err31 = np.stack([log31 + log31_err, log31 - log31_err])
err40 = np.stack([log40 + log40_err, log40 - log40_err])

cs = np.array(df['cs'].to_list())
anti_cs = np.array(df['anti_cs'].to_list())
npoints = np.array(df['used_points'].to_list())


proton_pdf31 = np.array(df['proton_pdf31'].to_list())
proton_pdf31_err = np.array(df['proton_pdf31_err'].to_list())

neutron_pdf31 = np.array(df['neutron_pdf31'].to_list())
neutron_pdf31_err = np.array(df['neutron_pdf31_err'].to_list())

anti_proton_pdf31 = np.array(df['anti_proton_pdf31'].to_list())
anti_proton_pdf31_err = np.array(df['anti_proton_pdf31_err'].to_list())

anti_neutron_pdf31 = np.array(df['anti_neutron_pdf31'].to_list())
anti_neutron_pdf31_err = np.array(df['anti_neutron_pdf31_err'].to_list())

anti_pdf31 = np.array(df['anti_pdf31'].to_list())

mean = (proton_pdf31 + neutron_pdf31) / 2
print(neutron_pdf31.shape)
print(mean.shape)

fig, axis = plt.subplots()
axis.errorbar(E, proton_pdf31, yerr=proton_pdf31_err, fmt='.', label='proton pdf31 LO')
axis.errorbar(E, neutron_pdf31, yerr=neutron_pdf31_err, fmt='.', label='neutron pdf31 LO')
axis.errorbar(E, mean, fmt='.', label='mean')
axis.errorbar(E, log31, yerr=log31_err, fmt=".", label='isoscalar pdf31 LO')
axis.set_xscale('log')
axis.set_yscale('log')
axis.set_xlabel(r'$E_{\nu}$')
axis.set_ylabel(r'$\sigma_{\nu}$ [Pb]')
plt.legend()
plt.savefig(f"Figs/proton31.pdf", format="pdf", bbox_inches="tight")
plt.show()

fig, axis = plt.subplots()
axis.errorbar(E, proton_pdf31/cs, yerr=proton_pdf31_err/cs, fmt='.', label='proton pdf31 LO')
axis.errorbar(E, neutron_pdf31/cs, yerr=neutron_pdf31_err/cs, fmt='.', label='neutron pdf31 LO')
axis.errorbar(E, mean/cs, fmt='.', label='mean')
axis.errorbar(E, log31/cs, yerr=log31_err/cs, fmt=".", label='isoscalar pdf31 LO')
axis.set_xscale('log')
axis.set_yscale('log')
axis.set_xlabel(r'$E_{\nu}$')
axis.set_ylabel(r'$\sigma_{\nu} /\ \sigma_{\nu}^{ref}$')
plt.legend()
plt.savefig(f"Figs/proton_ratio.pdf", format="pdf", bbox_inches="tight")
plt.show()


fig, axis = plt.subplots()
axis.errorbar(E, anti_proton_pdf31/anti_cs, fmt='.', label='proton pdf31 LO')
axis.errorbar(E, anti_neutron_pdf31/anti_cs, fmt='.', label='neutron pdf31 LO')
axis.errorbar(E, anti_pdf31/cs, fmt=".", label='isoscalar pdf31 LO')
axis.set_xscale('log')
axis.set_yscale('log')
axis.set_xlabel(r'$E_{\bar{\nu}}$')
axis.set_ylabel(r'$\sigma_{\bar{\nu}} /\ \sigma_{\bar{\nu}}^{ref}$')
plt.legend()
plt.savefig(f"Figs/anti_proton_ratio.pdf", format="pdf", bbox_inches="tight")
plt.show()

if False:
    fig, axis = plt.subplots()

    #plt.plot(E_nu, ref_cs, label='ref')
    #plt.plot(E_nu2, ref_w, label='w')
    #plt.plot(E, pdf21, label='pdf21')
    #plt.plot(E, pdf31, label='pdf31')
    axis.errorbar(E, logstruc, fmt='.', label='logstruc')
    axis.errorbar(E, log21, yerr=log21_err, fmt=".", label='log21')
    axis.errorbar(E, log31, yerr=log31_err, fmt=".", label='log31')
    axis.errorbar(E, log40, yerr=log40_err, fmt=".", label='log40')
    axis.errorbar(E, cs, fmt='.', label='cs ref')
    #pltot(E, pdf40, label='pdf40')
    #pltot(E, struc, label='struc')
    #pltot(E, cor, label='correction')
    axis.set_xlabel(r'$E_{\nu}$')
    axis.set_ylabel(r'$\sigma_{\nu}$ [Pb]')
    axis.set_xscale('log')
    axis.set_yscale('log')
    #pltim(5e3, 5e9)
    #pltim(0, 1.7e4)
    axis.legend()
    plt.savefig(f"Figs/sigma_E.pdf", format="pdf", bbox_inches="tight")
    plt.show()

    #21_err = np.stack(log)
    #plt.plot(s, pdf31/cs, label='frac31')
    #plt.plot(s, struc/cs, label='frac_struc')
    fig, axis = plt.subplots()
    ratio = log21/cs
    err_r = log21_err/cs
    axis.errorbar(s, logstruc/cs, fmt="o", label='PDF3.1 NLO')
    axis.errorbar(s[:-6], ratio[:-6], yerr=err_r[:-6], fmt="o", label='PDF2.1 LO')
    axis.errorbar(s, log31/cs, yerr=log31_err/cs, fmt="o", label='PDF3.1 LO')
    axis.errorbar(s, log40/cs, yerr=log40_err/cs, fmt="o", label='PDF4.0 LO')
    axis.set_xscale('log')
    axis.legend()
    axis.set_xlabel('Neutrino Energy [GeV]')
    axis.set_ylabel(r'$\sigma_{\nu} /\ \sigma_{\nu}^{ref}$')

    plt.savefig(f"Figs/pdfs_ratio.pdf", format="pdf", bbox_inches="tight")
    plt.show()