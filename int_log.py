
import lhapdf
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd


class cs_neutrino_nucleon:

    def __init__(self, E_nu, pdf):
        self.Mn = 0.938
        self.pdf = pdf

        GF = 1.663787e-5
        self.GF2 = GF*GF
        Mw = 80.385
        self.Mw2 = Mw*Mw
        self.s = 2*E_nu*self.Mn
        self.st = self.s + self.Mn*self.Mn
        self.conv1 = 389379290.4730569
        self.conv = 0.3894e9
        print(self.conv/self.conv1)

        self.calc_count = 0
        assert self.s < self.pdf.q2Max, 'E_nu too high, s > q2max'

        self.lnQ2min = np.log(pdf.q2Min)
        self.lnQ2max = np.log(self.s)
        self.xmin = pdf.xMin
        print(f'{self.lnQ2min=}, {self.lnQ2max}')



    def calc(self):
        cs, err = integrate.quad(self.diff_lnQ2, self.lnQ2min, self.lnQ2max)
        return cs

    def diff_lnQ2(self, lnQ2):
        lnxmin = lnQ2 - np.log(self.s)
        lnxmax = -1e-9 # or just 0 ?
        Q2 = np.exp(lnQ2)
        #print(lnxmin, lnxmax)
        ddif, err = integrate.quad(self.ddiff_lnx_lnQ2, lnxmin, lnxmax, args=(Q2,))
        return Q2 * ddif

    def ddiff_lnx_lnQ2(self, lnx, Q2):
        x = np.max([np.exp(lnx), self.xmin])
        y = Q2 / x / self.s
        omy2 = np.power((1-y), 2)
        Yp = 1 + omy2
        Ym = 1 - omy2
        fact = self.conv * self.GF2 / 4 / np.pi * np.power(self.Mw2 / (self.Mw2 + Q2 ), 2) / 2
        return fact * (Yp * self.pdf.xfxQ2(2001, x, Q2) - y*y * self.pdf.xfxQ2(2002, x, Q2) + Ym * self.pdf.xfxQ2(2003, x, Q2))

pdf = lhapdf.mkPDF("NNPDF31sx_nnlonllx_as_0118_LHCb_nf_6_SF")
#cs = cs_neutrino_nucleon(1e6, pdf)

df = pd.read_csv('cs_3.csv')
df['convert'] = 19*[0.0]
for i in range(0, 19):
    E_nu = df.at[i, 'E_nu']
    cs = cs_neutrino_nucleon(E_nu, pdf)
    sigma = cs.calc()
#print(cs.calc_count, cs.error_count)
    print(f'cs: {sigma}, E_nu: {E_nu}')

    if True:
        df.at[i, 'convert'] = sigma
        df.to_csv('cs_3.csv', index=False)

import cs_trend