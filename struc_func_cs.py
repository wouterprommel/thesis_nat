import lhapdf
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd

def GeV_to_pb(cs):
    return round((cs)/(2.56819e-9), 3)

class cs_neutrino_nucleon:

    def __init__(self, E_nu, pdf):
        self.Mn = 0.938
        self.pdf = pdf

        self.GF = 1.663787e-5
        self.Mw = 80.385
        self.s = 2*E_nu*self.Mn
        self.st = self.s + self.Mn*self.Mn

        self.calc_count = 0

        assert self.s < self.pdf.q2Max, 'E_nu too high, s > q2max'
    
    def calc(self):
        xmin = lambda q2: np.max([pdf.xMin, q2/self.s])
        #print('xmin', xmin(1.2), xmin(2), xmin(1000))
        qmax = np.min([self.s, (500*self.Mw)**2])
        sigma, err = integrate.dblquad(self._ddiff_neutrino_nucleon, pdf.q2Min, qmax, xmin, pdf.xMax)
        return sigma, err
    
    def _ddiff_neutrino_nucleon(self, x, q2):
        self.calc_count += 1
        assert 0 < q2/(self.s * x) < 1, 'y must be between 0 and 1'
        A = (self.GF*self.GF)/(4*np.pi) * ((self.Mw*self.Mw)/(self.Mw*self.Mw + q2))**2
        y = q2/x/self.s
        Yp = 1 + (1 - y)**2
        Ym = 1 - (1 - y)**2
        F2 = self.pdf.xfxQ2(2001, x, q2)
        FL = self.pdf.xfxQ2(2002, x, q2)
        xF3 = self.pdf.xfxQ2(2003, x, q2)
        #return A*(Yp*F2 - y*y*FL + Ym*xF3)
        return A*(Yp*F2 + Ym*xF3)
    

df = pd.read_csv('cs_3.csv')

pdf = lhapdf.mkPDF("NNPDF31sx_nnlonllx_as_0118_LHCb_nf_6_SF")
#pdf = lhapdf.mkPDF("NNPDF40_lo_as_01180")
#pdf = lhapdf.mkPDF("NNPDF31_lo_as_0118")

#df['PDF21'] = 19*[0.0]
#df['PDF31'] = 19*[0.0]

for i in range(0, 19):
    E_nu = df.at[i, 'E_nu']
    cs = cs_neutrino_nucleon(E_nu, pdf)
    sigma, err = cs.calc()
#print(cs.calc_count, cs.error_count)
    print(f'cs: {GeV_to_pb(sigma)}, E_nu: {E_nu}')

    if True:
        df.at[i, 'struc'] = GeV_to_pb(sigma)
        df.at[i, 'err'] = err 
        df.at[i, 'used_points'] = cs.calc_count
        df.to_csv('cs_3.csv', index=False)

import cs_trend