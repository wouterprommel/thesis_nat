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
        xmax = pdf.xMax
        xmin = lambda q2: np.max([pdf.xMin, q2/self.s])
        #xmin = 1e-7#pdf.xMin
        qmax = np.min([self.s, (500*self.Mw)**2])
        qmin = pdf.q2Min
        sigma, err = integrate.dblquad(self._ddiff2, qmin, qmax, xmin, xmax)
        #sigma, err = integrate.dblquad(self._ddiff_neutrino_nucleon, pdf.q2Min, self.s, xmin, pdf.xMax)
        return sigma, err

    def calc_split(self):
        xmax = pdf.xMax
        xmin = pdf.xMin #lambda q2: np.max([pdf.xMin, q2/self.s])
        qmax = self.s #np.min([self.s, (500*self.Mw)**2])
        qmin = pdf.q2Min

        q_range = np.linspace(np.log(qmin), np.log(qmax), 1000)
        print(q_range.shape)
        sigma = 0.0
        for i in range(1, q_range.shape[0]):
            ql = np.exp(q_range[i - 1])
            qh = np.exp(q_range[i])
            dq = qh - ql
            ds, err = integrate.dblquad(self._ddiff2, ql, qh, xmin, xmax)
            sigma += ds
            if i % 100 == 0:
                print(f'i: {i}, cs: {GeV_to_pb(sigma)} pb, ds: {GeV_to_pb(ds)}, qmin:{ql}, qmax:{qh}')
        return sigma, err
    
    def _ddiff_neutrino_nucleon(self, x, q2):
        self.calc_count += 1
        assert 0 < q2/(self.s * x) < 1, 'y must be between 0 and 1'
        A = (self.GF*self.GF*self.s)/(self.st*4*np.pi*(1 + q2/(self.Mw*self.Mw))**2)
        Yp = 1 + (1 - q2/(x*self.s))**2
        Ym = 1 - (1 - q2/(x*self.s))**2
        F2 = 2*(self.pdf.xfxQ2(1, x, q2) + self.pdf.xfxQ2(2, x, q2) + self.pdf.xfxQ2(-1, x, q2) + self.pdf.xfxQ2(-2, x, q2) + 2*self.pdf.xfxQ2(3, x, q2) + 2*self.pdf.xfxQ2(-4, x, q2))
        xF3 = 2*((self.pdf.xfxQ2(2, x, q2) - self.pdf.xfxQ2(-2, x, q2)) + (self.pdf.xfxQ2(1, x, q2) - self.pdf.xfxQ2(-1, x, q2)) + 2*self.pdf.xfxQ2(3, x, q2) - 2*self.pdf.xfxQ2(-4, x, q2))
        return A*(Yp*F2 + Ym*xF3)/x

    def _ddiff2(self, x, Q2):
        assert 0 < Q2/(self.s * x) < 1, 'y must be between 0 and 1'
        self.calc_count += 1
        A = (self.GF*self.GF)/np.pi/4
        a = 1/(1 + Q2/(self.Mw*self.Mw))**2
        b = (self.pdf.xfxQ2(1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) + 2*self.pdf.xfxQ2(3, x, Q2))
        c = (1 - Q2/(x*self.s))**2
        d = (self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(-2, x, Q2) + 2*self.pdf.xfxQ2(-4, x, Q2))
        return A*a*((b) + c*(d)) / x
    
df = pd.read_csv('cs_3.csv')

#pdf = lhapdf.mkPDF("NNPDF21_lo_as_0119_100")
#pdf = lhapdf.mkPDF("NNPDF40_lo_as_01180")
pdf = lhapdf.mkPDF("NNPDF31_lo_as_0118")

#df['PDF21'] = 19*[0.0]
#df['PDF31'] = 19*[0.0]
#df['PDF40'] = 19*[0.0]

for i in range(0,19):
    E_nu = df.at[i, 'E_nu']
    cs = cs_neutrino_nucleon(E_nu, pdf)
    sigma, err = cs.calc()
#print(cs.calc_count, cs.error_count)
    print(GeV_to_pb(sigma), E_nu)

    if True:
        df.at[i, 'PDF31'] = GeV_to_pb(sigma)
        df.at[i, 'err'] = err 
        df.at[i, 'used_points'] = cs.calc_count
        df.to_csv('cs_3.csv', index=False)
import cs_trend