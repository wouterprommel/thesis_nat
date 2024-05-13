
import lhapdf
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
import multiprocessing
import datetime


class cs_neutrino_nucleon:

    def __init__(self, E_nu, pdf):
        self.Mn = 0.938
        self.pdf = pdf

        GF = 1.1663787e-5
        self.GF2 = GF*GF
        Mw = 80.385
        self.Mw2 = Mw*Mw
        self.s = 2*E_nu*self.Mn
        self.st = self.s + self.Mn*self.Mn
        self.conv_p = 389379290.4730569
        self.conv = 0.3894e9

        self.calc_count = 0
        if self.s < self.pdf.q2Max:
           self.physical = True 
        else:
            self.physical = False

        self.lnQ2min = np.log(pdf.q2Min)
        self.lnQ2max = np.log(self.s)

        #self.lnQ2min = pdf.q2Min
        #self.lnQ2max = self.s

        self.xmin = pdf.xMin
        #print(f'{self.lnQ2min=}, {self.lnQ2max}')

        self.calc_count = 0



    def calc(self):
        if self.physical:
            cs, err = integrate.quad(self.diff_lnQ2, self.lnQ2min, self.lnQ2max)
            return cs
        else:
            print('E_nu to high; s > q2max')
            return None

    def diff_lnQ2(self, lnQ2):
        lnxmin = lnQ2 - np.log(self.s)
        lnxmax = -1e-9 # or just 0 ?
        #lnxmin = np.exp(lnQ2 - np.log(self.s))
        #lnxmin = np.log(lnQ2/self.s)
        #lnxmax = 1
        Q2 = np.exp(lnQ2)
        #Q2 = lnQ2
        #print(lnxmin, lnxmax)
        #pool = multiprocessing.Pool(processes=4)
        ddif, err = integrate.quad(self.ddiff_lnx_lnQ2, lnxmin, lnxmax, args=(Q2,))
        return Q2 * ddif
        return ddif

    def ddiff_lnx_lnQ2(self, lnx, Q2):
        self.calc_count += 1
        x = np.max([np.exp(lnx), self.xmin])
        #x = np.exp(lnx)
        #x = lnx
        #x = 0.1
        #Q2 = 100

        y = Q2 / x / self.s
        omy2 = np.power((1-y), 2)
        Yp = 1 + omy2
        Ym = 1 - omy2
        fact = self.conv * self.GF2 / 4 / np.pi * np.power(self.Mw2 / (self.Mw2 + Q2 ), 2)
        #print('fact', fact, 'conf', self.conv, 'pow', np.power(self.Mw2 / (self.Mw2 + Q2 ), 2), 'GF2', self.GF2)
        #print('Yp', Yp)
        #quit()
        # was 2*() for f2 and f3
        F2 = (self.pdf.xfxQ2(1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) + self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(-2, x, Q2) + 2*self.pdf.xfxQ2(3, x, Q2) + 2*self.pdf.xfxQ2(-4, x, Q2))
        xF3 = ((self.pdf.xfxQ2(2, x, Q2) - self.pdf.xfxQ2(-2, x, Q2)) + (self.pdf.xfxQ2(1, x, Q2) - self.pdf.xfxQ2(-1, x, Q2)) + 2*self.pdf.xfxQ2(3, x, Q2) - 2*self.pdf.xfxQ2(-4, x, Q2))
        #return fact * (Yp * self.pdf.xfxQ2(2001, x, Q2) - y*y * self.pdf.xfxQ2(2002, x, Q2) + Ym * self.pdf.xfxQ2(2003, x, Q2))
        return fact * (Yp * F2 + Ym * xF3)

#pdf_struc = lhapdf.mkPDF("NNPDF31sx_nnlonllx_as_0118_LHCb_nf_6_SF")
pdf_31 = lhapdf.mkPDF("NNPDF31_lo_as_0118")
pdf_40 = lhapdf.mkPDF("NNPDF40_lo_as_01180")
#pdf = lhapdf.mkPDF("NNPDF21_lo_as_0119_100")
#cs = cs_neutrino_nucleon(1e6, pdf)

df = pd.read_csv('cs_3.csv')

for name, pdf in [('log31', pdf_31), ('log40', pdf_40)]:
#df['log40'] = 19*[0.0]
# 0, 19 all 
# 7, 8 for 1e6
    for i in range(19, 28): # 19 to end
        E_nu = df.at[i, 'E_nu']
        dt_start = datetime.datetime.now()
        cs = cs_neutrino_nucleon(E_nu, pdf)
        print('physical', cs.physical)
        if cs.physical:
            sigma = cs.calc()
        else:
            sigma = 0.0
        dt_end = datetime.datetime.now()
        print(cs.calc_count)
        print()
        print(f'cs: {sigma}, E_nu: {E_nu}, cs/cs-ref: {sigma/df.at[i, "cs"]}')
        print(f'Time of calc: {(dt_end - dt_start)}')
        print()

        if True:
            df.at[i, name] = sigma
            df.to_csv('cs_3.csv', index=False)

import cs_trend