import lhapdf
import numpy as np
import matplotlib.pyplot as plt


pdf = lhapdf.mkPDF("NNPDF21_lo_as_0119_100")
print(pdf.xMin, pdf.xMax)
print(pdf.q2Min, pdf.q2Max)
xrange = np.linspace(1.0000e-4, 1, 10)
print(min(xrange), max(xrange))

#f_u = [pdf.xfxQ2(2, i, 10) for i in xrange]
#f_d = [pdf.xfxQ2(1, i, 10) for i in xrange]
#print(f_u)
#print(f_d)
if False:
    plt.plot(xrange, f_d)
    plt.plot(xrange, f_u)
    plt.xscale('log')
    plt.show()


class mc_cross_section():

    def __init__(self, E_nu, pdf):
        Mn = 0.938

        self.GF = 1.663787e-5
        self.Mw = 80.385
        self.s = 2*E_nu*Mn

        self.num_samples=1000000

        self.x_samples = np.random.uniform(pdf.xMin, 1, self.num_samples)
        self.Q2_samples = np.random.uniform(pdf.q2Min, self.s, self.num_samples)
        
        #Q2_max(x) = s*x
        bool_array = self.x_samples * self.Q2_samples < self.s
        self.x_region = self.x_samples[bool_array]
        self.Q2_region = self.Q2_samples[bool_array]
        self.area = sum(bool_array)/self.num_samples * ((1 - pdf.xMin) * (self.s - pdf.q2Min))
        print('INT fraction', sum(bool_array)/self.num_samples)
         

    def plot_mc_samples(self):
            print('INT area', self.area)
            plt.title('MC samples')
            plt.scatter(self.x_samples, self.Q2_samples)
            plt.scatter(self.x_region, self.Q2_region)
            plt.show()


    def calc(self):
        integral = self._mc()
        return (self.GF*self.GF * self.Mw^4)/np.pi * integral


    def _mc(self, func):
        
        integral = 0.0
        for i in range(self.num_samples):
            integral += self._dsig_dxdQ2(self.x_samples[i], self.Q2_samples[i])
        
        # int =  func-average * area
        integral *= self.area / self.num_samples
        
        return integral

    def _dsig_dxdQ2(self, x, Q2):
        pass

a = np.array([1,2,3,4])
print(a[a < 3])

pdf = lhapdf.mkPDF("NNPDF21_lo_as_0119_100")
mc_cs = mc_cross_section(1e11, pdf)
cs = mc_cs.plot_mc_samples()
print("Cross-Section Neutrino-Proton:", cs/(2.56819e-9), 'pb')
