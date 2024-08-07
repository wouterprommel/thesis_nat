import lhapdf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def GeV_to_pb(cs):
    return round((cs)/(2.56819e-9), 3)

def list_GeV_to_pb(cs):
    return [round((i)/(2.56819e-9), 3) for i in cs]


class mc_cross_section():

    def __init__(self, E_nu, pdf, regions, n_samples_list):
        self.Mn = 0.938
        self.pdf = pdf

        self.GF = 1.663787e-5
        self.Mw = 80.385
        self.s = 2*E_nu*self.Mn

        self.cs = 0.0
        #print(f's: {self.s}, Q2_max: {self.pdf.q2Max}, diff s q2max: {self.pdf.q2Max - self.s}')
        #print(f'pdf ranges: x min: {pdf.xMin}, x max:{pdf.xMax}, Q2 min:{pdf.q2Min}, Q2 max{pdf.q2Max}')
        assert self.s < self.pdf.q2Max, 'E_nu too high, s > q2max'
        assert len(regions) - 1 == len(n_samples_list), 'regions must be 1 longer than n_samples'
        n_samples_list = [0] + n_samples_list

        # regions example [0, 1e-5, 1e-4, 1e-3, 0.1, 1]
        self.tot_used_points = 0
        for idx in range(1, len(regions)):

            self.num_samples = int(n_samples_list[idx])
            xmin = regions[idx - 1]
            xmax = regions[idx]


            if xmin < pdf.xMin:
                x_set_min = pdf.xMin
            else:
                x_set_min = xmin
            x_set_max = xmax

            q2_set_min = pdf.q2Min
            q2_set_max = self.s

            #print(f'set ranges: x min: {x_set_min}, x max:{x_set_max}, Q2 min:{q2_set_min}, Q2 max{q2_set_max}')

            # initial sample points
            x_samples = np.random.uniform(x_set_min, x_set_max, self.num_samples)
            Q2_samples = np.random.uniform(q2_set_min, q2_set_max, self.num_samples)
            #self.area = ((1 - pdf.xMin) * (self.s - pdf.q2Min))

            # validate points
            bool_array = x_samples * self.s > Q2_samples

            bool_array2 = self.Mn*self.Mn + Q2_samples*(1 - x_samples)/x_samples > 4 # should be True for all
            #print('number of sample below resonance prod threshold', self.num_samples - sum(bool_array2))

            bool_array = bool_array & bool_array2
            # usable region
            x_region = x_samples[bool_array]
            Q2_region = Q2_samples[bool_array]
            f = len(x_region)/self.num_samples
            print(f'fraction of samples in region:{f}, number of samples in region:{len(x_region)}, number of random samples:{self.num_samples}')
            # number of new samples needed based on fraction of usable points
            if f == 0:
                n_new_samples = self.num_samples
            else:
                n_new_samples = round((self.num_samples - sum(bool_array))/(sum(bool_array)/self.num_samples))

            if n_new_samples >= self.num_samples*5:
                n_new_samples = self.num_samples*5

            # new samples
            x_samples = np.random.uniform(x_set_min, x_set_max, n_new_samples)
            Q2_samples = np.random.uniform(q2_set_min, q2_set_max, n_new_samples)

            # validate new samples
            bool_array = x_samples * self.s > Q2_samples

            #bool_array2 = self.Mn*self.Mn + Q2_samples/x_samples - Q2_samples > 4 
            bool_array2 = self.Mn*self.Mn + Q2_samples*(1 - x_samples)/x_samples > 4 
            #print('number of sample below resonance prod threshold', n_new_samples - sum(bool_array2))


            bool_array = bool_array & bool_array2
            # add usable samples to the region
            x_region = np.concatenate((x_region, x_samples[bool_array]), axis=0)
            Q2_region = np.concatenate((Q2_region, Q2_samples[bool_array]), axis=0)

            bool_array3 = x_region < Q2_region/self.s
            print(f'x0 is lower than needed?', sum(bool_array3))

            # total used samples
            self.num_samples += n_new_samples
            self.tot_used_points += self.num_samples

            #area = len(x_region)/self.num_samples * ((x_set_max - x_set_min) * (q2_set_max - q2_set_min))
            area = ((x_set_max - x_set_min) * (q2_set_max - q2_set_min))

            #cs_region = self.calc(x_region, Q2_region, area)
            cs_region = self.calc_vis(x_region, Q2_region, area)
            self.cs += cs_region
            print(f'cs: {GeV_to_pb(cs_region)}, for x region: {xmin} - {xmax}')
        print(f'cs: {GeV_to_pb(self.cs)}, from 0 to {xmax}')

    def plot_mc_samples(self):
        print('Integration area', self.area)
        plt.title('MC samples')
        #plt.scatter(self.x_samples, self.Q2_samples)
        plt.scatter(self.x_region, self.Q2_region)
        plt.show()

    def plot_mc_samples_eval(self, X, Q2, eval_list):
        print(f'Number of points in region:{len(X)}')
        plt.title('evaluated MC samples')
        scat = plt.scatter(X, Q2, c=list_GeV_to_pb(eval_list), cmap='viridis')
        plt.colorbar(scat, label='Color')
        plt.yscale('log')
        plt.show()


    def calc(self, X, Q2, A):
        integral = self._mc(X, Q2, A)
        #return (self.GF*self.GF * self.Mw**4 * integral)/(2*np.pi) # factor 2 from (N + P)/2   But just 4 for the structure functions bc those are multiplied by 2...
        return (self.GF*self.GF * integral)/(np.pi) 

    def calc_vis(self, X, Q2, A):
        integral, int_list = self._mc_list(X, Q2, A)

        cs = (self.GF*self.GF * integral)/(np.pi)

        eval_list = (self.GF*self.GF * np.array(int_list))/(np.pi) 

        print("All points CS:", cs/(2.56819e-9), 'pb')
        cs_top10 = 0.0
        for i in range(10):
            index = np.argmax(eval_list)
            cs_top10 += eval_list[index]
            eval_list[index] = 0.0

        cs_top10 *= A / self.num_samples 
        print("top10 points Cs:", cs_top10/(2.56819e-9), 'pb')

        self.plot_mc_samples_eval(X, Q2, eval_list)

        return cs


    def _mc(self, X, Q2, A):
        integral = 0.0
        integral2 = 0.0
        integral3 = 0.0
        n_samples = len(X)
        for i in range(n_samples):
            if False and i % 1000000 == 0:
                print(i)
            integral += self._diff_cs_neutrino_nuclei(X[i], Q2[i])

        
        integral *= A / self.num_samples
        
        return integral3

    def _mc_list(self, X, Q2, A):
        n_samples = len(X)
        int_list = []
        for i in range(n_samples):
            if False and i % 1000000 == 0:
                 print(i)
            diff_cs = self._diff_cs_neutrino_nuclei(X[i], Q2[i])
            if diff_cs == 0:
                print(f'x:{X[i]}, Q2:{Q2[i]}' )
            int_list.append(diff_cs)
        
        # int =  func-average * area
        integral = sum(int_list)
        integral *= A / self.num_samples 
        
        return integral, int_list

    def _diff_cs_neutrino_nuclei(self, x, Q2):
        assert Q2 < self.pdf.q2Max, f'Q2 out of range {Q2 - self.pdf.q2Max}'
        assert x > Q2/self.s
        assert self.Mn*self.Mn + Q2*(1-x)/x > 4, 'W2 is below resonance production threshold'
        a = 1/(1 + Q2/(self.Mw*self.Mw))**2
        b = (self.pdf.xfxQ2(1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) + 2*self.pdf.xfxQ2(3, x, Q2))/(x)
        c = (1 - Q2/(x*self.s))**2
        d = (self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(-2, x, Q2) + 2*self.pdf.xfxQ2(-4, x, Q2))/(x)
        return a*((b) + c*(d))


df = pd.read_csv('cs.csv')

#pdf = lhapdf.mkPDF("NNPDF21_lo_as_0119_100")
pdf = lhapdf.mkPDF("NNPDF40_lo_as_01180")
regions_small = [0, 1e-3, 1e-2, 1e-1, 0.2, 1]
n_samples_small = [ 1e5, 1e5, 1e5, 1e4, 1e4]

regions = [0, 1e-3, 1e-2, 5e-2, 1e-1, 0.2, 1]
n_samples = [2e7, 2e6, 2e6, 2e6, 2e6, 2e6]
n_samples = [2e5, 2e5, 2e5, 2e5, 2e5, 2e5]
if False:
    E_nu = 1e6
    regions = [0, 1e-3, 1e-2, 5e-2, 1e-1, 0.2, 1]
    n_samples_list = [2e6, 2e6, 2e6, 2e6, 2e6, 2e6]
    goal = 628
if False:
    E_nu = 1e8
    regions = [0, 1e-5, 1e-4, 5e-4, 1e-3, 1e-2, 0.1, 1]
    n_samples_list = [1e6, 2e5, 2e5, 2e5, 2e5, 2e5, 2e5]
    goal = 4250
if False:
    E_nu = 1e4
    regions = [0, 1e-3, 1e-2, 1e-1, 0.2, 1]
    n_samples_list = [ 1e5, 1e5, 1e5, 1e4, 1e4]
    goal = 44.6
for i in range(7,8):
    E_nu = df.at[i, 'E_nu']
    if E_nu < 1e5:
        reg = regions_small
        n_samp = n_samples_small
    else:
        reg = regions
        n_samp = n_samples

    mc_cs = mc_cross_section(E_nu, pdf, reg, n_samp)

    print("Cross-Section Neutrino-Nucleon:", GeV_to_pb(mc_cs.cs), GeV_to_pb(mc_cs.cs)/(df.at[i, 'cs']), 'pb, at E_nu: ', df.at[i, 'E_nu'], 'GeV\n\n')
    if False:
        df.at[i, 'mc_cs'] = GeV_to_pb(mc_cs.cs)
        df.at[i, 'used_points'] = mc_cs.tot_used_points
        df.to_csv('cs.csv', index=False)
