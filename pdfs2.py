import lhapdf
import numpy as np
import matplotlib.pyplot as plt


class mc_cs:

    def __init__(self, pdf):
        self.pdf = pdf
        self.X, self.Q2 = self.get_range(self) 
        self.Gf = 1.663787e-5
        self.Mn = 0.9
        self.Mw = 80
    
    def mc_Q2(self):
        s = 0.0
        n_points = 0
        for q2 in self.Q2:
            xmin = np.max([self.pdf.xMin, q2/(2*self.E_nu*self.Mn)])
        for x in self.X:
            n_points += 1
            if ... :
                A = self.Gf*self.Gf/(4*np.pi(1 + q2/(self.Mw*self.Mw))**2)
                Yp = 1 + (1 - q2/(x*self.s))**2
                Ym = 1 - (1 - q2/(x*self.s))**2
                F2 = 2*(self.pdf.xfxQ2(1, x, q2) + self.pdf.xfxQ2(2, x, q2) + self.pdf.xfxQ2(-1, x, q2) + self.pdf.xfxQ2(-2, x, q2) + 2*self.pdf.xfxQ2(3, x, q2) + 2*self.pdf.xfxQ2(-4, x, q2))
                xF3 = 2*((self.pdf.xfxQ2(2, x, q2) - self.pdf.xfxQ2(-2, x, q2)) + (self.pdf.xfxQ2(1, x, q2) - self.pdf.xfxQ2(-1, x, q2)) + 2*self.pdf.xfxQ2(3, x, q2) - 2*self.pdf.xfxQ2(-4, x, q2))
                s += A*(Yp*F2 + Ym*xF3)/x
        avg_value = s/n_points
        area = (pdf.xMax - pdf.xMin)

    def get_range(self):
        steps = np.arange(1, 10)
        macht = np.arange(np.log10(self.pdf.xMin), 0)
        print(steps)
        print(macht)
        X = []
        for m in macht:
            for s in steps:
                X.append(s*10**m)
        print(X)
        print(np.log10(self.pdf.xMin))

        steps = np.linspace(self.pdf.q2Min, 10, 10)
        print(steps)
        macht = np.arange(0, 10)
        Q2 = []
        for m in macht:
            for s in steps:
                Q2.append(s*10**m)
        #print(Q2)
        #plt.plot(X)
        return X, Q2


pdf = lhapdf.mkPDF("NNPDF40_lo_as_01180")
mc_cs(pdf)