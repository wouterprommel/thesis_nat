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

def GeV_to_pb(cs):
    return round((cs)/(2.56819e-9), 3)


class mc_cross_section():

    def __init__(self, E_nu, pdf, regions, n_samples_list):
        Mn = 0.938
        self.pdf = pdf

        self.GF = 1.663787e-5
        self.Mw = 80.385
        self.s = 2*E_nu*Mn

        self.cs = 0.0
        #print(f's: {self.s}, Q2_max: {self.pdf.q2Max}, diff s q2max: {self.pdf.q2Max - self.s}')
        print(f'pdf ranges: x min: {pdf.xMin}, x max:{pdf.xMax}, Q2 min:{pdf.q2Min}, Q2 max{pdf.q2Max}')
        assert self.s < self.pdf.q2Max, 'E_nu too high, s > q2max'
        assert len(regions) - 1 == len(n_samples_list), 'regions must be 1 longer than n_samples'
        n_samples_list = [0] + n_samples_list

        # regions example [0, 1e-5, 1e-4, 1e-3, 0.1, 1]
        for idx in range(1, len(regions)):

            num_samples = int(n_samples_list[idx])
            xmin = regions[idx - 1]
            xmax = regions[idx]


            if xmin < pdf.xMin:
                x_set_min = pdf.xMin
            else:
                x_set_min = xmin
            x_set_max = xmax

            q2_set_min = pdf.q2Min
            q2_set_max = 1e4 #self.s

            print(f'set ranges: x min: {x_set_min}, x max:{x_set_max}, Q2 min:{q2_set_min}, Q2 max{q2_set_max}')

            # initial sample points
            x_samples = np.random.uniform(x_set_min, x_set_max, num_samples)
            Q2_samples = np.random.uniform(q2_set_min, q2_set_max, num_samples)
            #self.area = ((1 - pdf.xMin) * (self.s - pdf.q2Min))

            # validate points
            bool_array = x_samples * self.s > Q2_samples

            # usable region
            self.x_region = x_samples[bool_array]
            self.Q2_region = Q2_samples[bool_array]
            
            # number of new samples needed based on fraction of usable points
            n_new_samples = round((num_samples - sum(bool_array))/(sum(bool_array)/num_samples))

            # new samples
            x_samples = np.random.uniform(x_set_min, x_set_max, n_new_samples)
            Q2_samples = np.random.uniform(q2_set_min, q2_set_max, n_new_samples)

            # validate new samples
            bool_array = x_samples * self.s > Q2_samples

            # add usable samples to the region
            self.x_region = np.concatenate((self.x_region, x_samples[bool_array]), axis=0)
            self.Q2_region = np.concatenate((self.Q2_region, Q2_samples[bool_array]), axis=0)

            # total used samples
            num_samples += n_new_samples

            print('Used points:', len(self.x_region))
            print('INT fraction after addition', len(self.x_region)/num_samples)

            area = len(self.x_region)/num_samples * ((x_set_max - x_set_min) * (q2_set_max - q2_set_min))

            cs_region = self.calc(self.x_region, self.Q2_region, area)
            self.cs += cs_region
            print(f'cs: {GeV_to_pb(cs_region)}, for x region: {xmin} - {xmax}')
            print(f'cs: {GeV_to_pb(self.cs)}, from 0 to {xmax}')

    def plot_mc_samples(self):
        print('Integration area', self.area)
        plt.title('MC samples')
        #plt.scatter(self.x_samples, self.Q2_samples)
        plt.scatter(self.x_region, self.Q2_region)
        plt.show()

    def plot_mc_samples_eval(self, eval_list):
        plt.title('evaluated MC samples')
        scat = plt.scatter(self.x_region, self.Q2_region, c=eval_list, cmap='hot')
        plt.colorbar(scat, label='Color')
        plt.yscale('log')
        plt.show()


    def calc(self, X, Q2, A):
        integral = self._mc(X, Q2, A)
        return (self.GF*self.GF * self.Mw**4)/(4*np.pi) * integral # factor 2 from (N + P)/2 extra 2 factor from anti-neutrino

    def calc_vis(self):
        integral, int_list = self._mc_list()
        cs = (self.GF*self.GF * self.Mw**4)/(4*np.pi) * integral # factor 2 from (N + P)/2
        eval_list = (self.GF*self.GF * self.Mw**4)/(4*np.pi) * np.array(int_list)
        print("All points: Cross-Section Neutrino-Proton:", cs/(2.56819e-9), 'pb')
        cs_top10 = 0.0
        for i in range(10):
            index = np.argmax(eval_list)
            cs_top10 += eval_list[index]
            eval_list[index] = 0.0

        cs_top10 *= self.area / len(self.x_region) 
        print("top10 points: Cross-Section Neutrino-Proton:", cs_top10/(2.56819e-9), 'pb')
        self.plot_mc_samples_eval(eval_list)
        return cs


    def _mc(self, X, Q2, A):
        integral = 0.0
        n_samples = len(X)
        print('#samples:', n_samples)
        for i in range(n_samples):
            if i % 1000000 == 0:
                 print(i)
            integral += self._differential_cs_neutrino_nuclei(X[i], Q2[i])
        
        # int =  func-average * area
        integral *= A / n_samples 
        
        return integral

    def _mc_list(self):
        n_samples = len(self.x_region)
        int_list = []
        print('#samples:', n_samples)
        for i in range(n_samples):
            if i % 1000000 == 0:
                 print(i)
            int_list.append(self._differential_cs_neutrino_nuclei(self.x_region[i], self.Q2_region[i]))
        
        # int =  func-average * area
        integral = sum(int_list)
        integral *= self.area / n_samples 
        
        return integral, int_list

    def _differential_cs_neutrino_nuclei(self, x, Q2):
        assert Q2 < self.pdf.q2Max, f'Q2 out of range {Q2 - self.pdf.q2Max}'
        a = 1/(self.Mw*self.Mw + Q2*Q2)**2
        b = (self.pdf.xfxQ2(1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) + 2*self.pdf.xfxQ2(3, x, Q2))/(x)
        anti_b = (self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(-2, x, Q2) + 2*self.pdf.xfxQ2(-3, x, Q2))/(x)
        c = (1 - Q2/(x*self.s))**2
        d = (self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(-2, x, Q2) + 2*self.pdf.xfxQ2(-4, x, Q2))/(x)
        anti_d = (self.pdf.xfxQ2(1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) + 2*self.pdf.xfxQ2(4, x, Q2))/(x)
        return a*((b + anti_b) + c*(d + anti_d))


pdf = lhapdf.mkPDF("NNPDF21_lo_as_0119_100")
E_nu = 1e6
n_samples = 100000000
regions = [0, 1e-4, 1e-3, 1e-2, 1]
n_samples_list = [1e3, 1e3, 1e3, 1e3]
for i in range(1):
    mc_cs = mc_cross_section(E_nu, pdf, regions, n_samples_list)

    print("Cross-Section Neutrino-Proton:", GeV_to_pb(mc_cs.cs) , 'pb, at E_nu: 1e', len(str(E_nu)) - 3, 'GeV\n\n')

