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

    def __init__(self, E_nu, pdf, num_samples):
        Mn = 0.938
        self.pdf = pdf

        self.GF = 1.663787e-5
        self.Mw = 80.385
        self.s = 2*E_nu*Mn
        print(self.s, self.pdf.q2Max, self.pdf.q2Max - self.s)
        assert self.s < self.pdf.q2Max, 'E_nu too high, s > q2max'

        #self.num_samples=100000000
        self.num_samples=num_samples

        self.x_samples = np.random.uniform(pdf.xMin, 1, self.num_samples)
        self.Q2_samples = np.random.uniform(pdf.q2Min, self.s, self.num_samples)
        #self.area = ((1 - pdf.xMin) * (self.s - pdf.q2Min))

        bool_array = self.x_samples * self.s > self.Q2_samples

        self.x_region = self.x_samples[bool_array]
        self.Q2_region = self.Q2_samples[bool_array]
        self.area = sum(bool_array)/self.num_samples * ((1 - pdf.xMin) * (self.s - pdf.q2Min))
        print('sum bool array', sum(bool_array))
        print('INT fraction', sum(bool_array)/self.num_samples)

    def plot_mc_samples(self):
            print('Integration area', self.area)
            plt.title('MC samples')
            plt.scatter(self.x_samples, self.Q2_samples)
            plt.scatter(self.x_region, self.Q2_region)
            plt.show()


    def calc(self):
        integral = self._mc()
        return (self.GF*self.GF * self.Mw**4)/(2*np.pi) * integral # factor 2 from (N + P)/2


    def _mc(self):
        integral = 0.0
        n_samples = len(self.x_region)
        print('#samples:', n_samples)
        for i in range(n_samples):
            if i % 1000000 == 0:
                 print(i)
            integral += self._differential_cs_neutrino_nuclei(self.x_region[i], self.Q2_region[i])
        
        # int =  func-average * area
        integral *= self.area / n_samples 
        
        return integral

    def _differential_cs_neutrino_nuclei(self, x, Q2):
        assert Q2 < self.pdf.q2Max, f'Q2 out of range {Q2 - self.pdf.q2Max}'
        a = 1/(self.Mw*self.Mw + Q2*Q2)**2
        b = (self.pdf.xfxQ2(1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) + 2*self.pdf.xfxQ2(3, x, Q2))/(x)
        c = (1 - Q2/(x*self.s))**2
        d = (self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(-2, x, Q2) + 2*self.pdf.xfxQ2(-4, x, Q2))/(x)
        return a*(b + c*d)


assert True, 'test is true'
a = np.array([1,2,3,4])
a = a * 2
print(a < np.array([4,5,5,5]))

pdf = lhapdf.mkPDF("NNPDF21_lo_as_0119_100")
E_nu = 5e6
n_samples = 100000000
n_samples = int(1e7)
for i in range(4):
    mc_cs = mc_cross_section(E_nu, pdf, n_samples)
    #mc_cs.plot_mc_samples()
    #print(len(str(E_nu)) - 3)
    cs = mc_cs.calc()
    print("Cross-Section Neutrino-Proton:", round(cs/(2.56819e-9), 3), 'pb, at E_nu: 5e', len(str(E_nu)) - 3, 'GeV\n\n')
