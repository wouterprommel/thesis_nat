import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import path

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
pdf21 = np.array(df['PDF21'].to_list())
pdf31 = np.array(df['PDF31'].to_list())
pdf40 = np.array(df['PDF40'].to_list())
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

pdf31_x7 = np.array(df['pdf31_x7'].to_list())
pdf31_x7_err = np.array(df['pdf31_x7_err'].to_list())
pdf40_x7 = np.array(df['pdf40_x7'].to_list())
pdf40_x7_err = np.array(df['pdf40_x7_err'].to_list())


pdf31_strange40 = np.array(df['pdf31_strange40'].to_list()) 
pdf31_strange40_err = np.array(df['pdf31_strange40_err'].to_list())

pdf31_DIY = np.array(df['pdf31_DIY'].to_list()) / 2 
pdf31_DIY_err = np.array(df['pdf31_DIY_err'].to_list())

pdf31_LO_x2 = np.array(df['pdf31_LO_x2'].to_list()) 
pdf31_LO_x2_err = np.array(df['pdf31_LO_x2_err'].to_list())

pdf31_NLO= np.array(df['pdf31_NLO_acc_1'].to_list()) 
pdf31_NLO_err = np.array(df['pdf31_NLO_acc_1_err'].to_list())
pdf31n_NLO_x2 = np.array(df['pdf31n_NLO_acc_1_x2'].to_list()) #* 2 
pdf31n_NLO_x2_err = np.array(df['pdf31n_NLO_acc_1_x2_err'].to_list())
pdf31n_NLO_x2_v2 = np.array(df['pdf31n_NLO_acc_1_x2_v2'].to_list()) * 2 
pdf31n_NLO_x2_v2_err = np.array(df['pdf31n_NLO_acc_1_x2_v2_err'].to_list())

pdf31n_NLO_acc_1_x2_v3 = np.array(df['pdf31n_NLO_acc_1_x2_v3'].to_list())
pdf31n_NLO_acc_1_x2_v3_err = np.array(df['pdf31n_NLO_acc_1_x2_v3_err'].to_list())

pdf31n_LO_x9 = np.array(df['pdf31n_LO_acc_1_x9'].to_list()) 
pdf31n_LO_x9_err = np.array(df['pdf31n_LO_acc_1_x9_err'].to_list())
pdf31_LO_x9 = np.array(df['pdf31_LO_acc_1_x9'].to_list()) 
pdf31_LO_x9_err = np.array(df['pdf31_LO_acc_1_x9_err'].to_list())

mean = (proton_pdf31 + neutron_pdf31) / 2
print(neutron_pdf31.shape)
print(mean.shape)

if False:
    fig, axis = plt.subplots(2, sharex='col', figsize=(6,6), height_ratios=[2, 1])
    axis[0].errorbar(E, proton_pdf31, yerr=proton_pdf31_err, linestyle='-', marker='.', label='proton PDF31 LO')
    axis[0].errorbar(E, neutron_pdf31, yerr=neutron_pdf31_err, linestyle='-', marker='.', label='neutron PDF31 LO')
    axis[0].errorbar(E, log31, yerr=log31_err, linestyle='-', marker='.', label='isoscalar PDF31 LO')
    axis[0].set_xscale('log')
    axis[0].set_yscale('log')
    #axis[0].set_xlabel(r'$E_{\nu}$ [GeV]')
    axis[0].set_ylabel(r'$\sigma_{\nu}$ [Pb]')
    axis[1].errorbar(E, proton_pdf31/cs, yerr=proton_pdf31_err/cs, linestyle='-', marker='.', label='proton PDF31 LO')
    axis[1].errorbar(E, neutron_pdf31/cs, yerr=neutron_pdf31_err/cs, linestyle='-', marker='.', label='neutron PDF31 LO')
    axis[1].errorbar(E, log31/cs, yerr=log31_err/cs, linestyle='-', marker='.', label='isoscalar PDF31 LO')
    axis[1].set_xscale('log')
    axis[1].set_xlim(1e3, 1e10)
    #axis[1].set_yscale('log')
    axis[1].set_xlabel(r'$E_{\nu}$ [GeV]')
    #axis[1].set_xlabel(r'$E_{\nu}$')
    axis[1].set_ylabel(r'$\sigma_{\nu} /\ \sigma_{\nu}^{ref}$')
    axis[0].legend()
    plt.savefig(f"Figs/proton31.pdf", format="pdf", bbox_inches="tight")
    plt.show()


if False:
    fig, axis = plt.subplots(2, sharex='col', figsize=(6,6), height_ratios=[2, 1])
    axis[0].errorbar(E, anti_proton_pdf31, linestyle='-', marker='.', label='anti-proton PDF31 LO')
    axis[0].errorbar(E, anti_neutron_pdf31, linestyle='-', marker='.', label='anti-neutron PDF31 LO')
    axis[0].errorbar(E, anti_pdf31, linestyle='-', marker='.', label='isoscalar PDF31 LO')
    axis[0].errorbar(E, anti_cs, linestyle='-', marker='.', label='ref anti-neutrino')
    axis[0].set_xscale('log')
    axis[0].set_yscale('log')
    axis[1].set_ylabel(r'$\sigma_{\bar{\nu}}$ [Pb]')
    axis[1].errorbar(E, anti_proton_pdf31/anti_cs, linestyle='-', marker='.', label='anti-proton PDF31 LO')
    axis[1].errorbar(E, anti_neutron_pdf31/anti_cs, linestyle='-', marker='.', label='anti-neutron PDF31 LO')
    axis[1].errorbar(E, anti_pdf31/anti_cs, linestyle='-', marker='.', label='isoscalar PDF31 LO')
    axis[1].set_xscale('log')
    axis[1].set_xlim(1e3, 1e10)
    axis[1].set_xlabel(r'$E_{\bar{\nu}}$ [GeV]')
    axis[1].set_ylabel(r'$\sigma_{\bar{\nu}} /\ \sigma_{\bar{\nu}}^{ref}$')
    axis[0].legend()
    plt.savefig(f"Figs/anti_proton.pdf", format="pdf", bbox_inches="tight")
    plt.show()

if True:
    x31 = log31
    x31_err = log31_err
    label1 = 'PDF3.1 LO xmin=1e-9'
    x40 = pdf40_x7
    x40_err = pdf40_x7_err
    x21 = pdf31_LO_x2
    x21_err = pdf31_LO_x2_err
    label2 = 'PDF3.1 LO xmin=1e-2'
    exp = pdf31n_NLO_acc_1_x2_v3
    exp_err = pdf31n_NLO_acc_1_x2_v3_err
    label3 = 'PDF3.1 NLO v3 with xmin = 1e-2'
    exp2 = pdf31n_NLO_x2 * 2
    exp2_err = pdf31n_NLO_x2_err
    label4 = 'PDF3.1 NLO with xmin = 1e-2'

    fig, axis = plt.subplots(2, sharex='col', figsize=(6,6), height_ratios=[2, 1])
    #plt.plot(E_nu, ref_cs, label='ref')
    #plt.plot(E_nu2, ref_w, label='w')
    #plt.plot(E, pdf21, label='pdf21')
    #plt.plot(E, pdf31, label='pdf31')
    #axis[0].errorbar(E, logstruc, linestyle='-', marker='.', label='PDF3.1 NLO')
    axis[0].errorbar(E, x31, yerr=x31_err, linestyle='-', marker='.', label=label1)
    axis[0].errorbar(E, x21, yerr=x21_err, linestyle='-', marker='.', label=label2)
    axis[0].errorbar(E, exp, yerr=exp_err, linestyle='-', marker='.', label=label3)
    axis[0].errorbar(E, exp2, yerr=exp2_err, linestyle='-', marker='.', label=label4)
    #axis[0].errorbar(E, x40, yerr=x40_err, linestyle='-', marker='.', label='PDF4.0 LO')
    axis[0].errorbar(E, cs, linestyle='-', marker='.', label='reference')
    #pltot(E, pdf40, label='pdf40')
    #pltot(E, struc, label='struc')
    #pltot(E, cor, label='correction')
    axis[1].set_xlabel(r'$E_{\nu}$ [GeV]')
    axis[0].set_ylabel(r'$\sigma_{\nu}$ [Pb]')
    axis[0].set_xscale('log')
    axis[0].set_yscale('log')
    #pltim(5e3, 5e9)
    #pltim(0, 1.7e4)
    axis[0].legend()
    #plt.savefig(f"Figs/sigma_E.pdf", format="pdf", bbox_inches="tight")
    #plt.show()

    #21_err = np.stack(log)
    #plt.plot(s, pdf31/cs, label='frac31')
    #plt.plot(s, struc/cs, label='frac_struc')
    #fig, axis = plt.subplots()
    ratio = x21/cs
    err_r = x21_err/cs
    #axis[1].errorbar(E, logstruc/cs, linestyle='-', marker='.', label='PDF3.1 NLO')
    #axis[1].errorbar(E[:-6], ratio[:-6], yerr=err_r[:-6], linestyle='-', marker='.', label='PDF2.1 LO')
    axis[1].errorbar(E, x31/cs, yerr=x31_err/cs, linestyle='-', marker='.', label=label1)
    axis[1].errorbar(E, x21/cs, yerr=x21_err/cs, linestyle='-', marker='.', label=label2)
    axis[1].errorbar(E, exp/cs, yerr=exp_err/cs, linestyle='-', marker='.', label=label3)
    axis[1].errorbar(E, exp2/cs, yerr=exp2_err/cs, linestyle='-', marker='.', label=label4)
    #axis[1].errorbar(E, x40/cs, yerr=x40_err/cs, linestyle='-', marker='.', label='PDF4.0 LO')
    axis[1].set_xscale('log')
    #axis[1].legend()
    #axis[1].set_xlabel('Neutrino Energy [GeV]')
    axis[1].set_xlim(1e3, 1e10)
    axis[1].set_ylabel(r'$\sigma_{\nu} /\ \sigma_{\nu}^{ref}$')

    #plt.savefig(path.fig_path() + "pdf31_NLO_x2_v3.pdf", format="pdf", bbox_inches="tight")
    plt.show()

if False:
    SE23 = np.square(log21 - pdf31_x7)
    SE24 = np.square(log21 - pdf40_x7)
    SE34 = np.square(pdf31_x7 - pdf40_x7)
    plt.plot(SE23, label='SE23')
    plt.plot(SE34, label='SE34')
    plt.plot(SE24, label='SE24')
    plt.show()