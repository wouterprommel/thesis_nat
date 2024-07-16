from NLO_functions2 import Structure_Functions
import lhapdf
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
#import multiprocessing
import datetime
from tqdm import tqdm
import warnings
import concurrent.futures as cf
import pickle

def save_pickle(obj, name):
    print(f'Saving file: {name}')
    file = open(name + '.pickle', 'wb')
    pickle.dump(obj, file)
    file.close()

class cs_neutrino_nucleon:

    def __init__(self, E_nu, pdf, anti=False, target='isoscalar', NLO=False, multithread=True, precision=1):
        self.Mn = 0.938
        self.pdf = pdf

        GF = 1.1663787e-5
        self.GF2 = GF*GF
        Mw = 80.385
        self.Mw2 = Mw*Mw
        self.s = 2*E_nu*self.Mn
        self.st = self.s + self.Mn*self.Mn
        self.convert_p = 389379290.4730569
        self.convert = 0.3894e9

        self.target = target 
        self.anti = anti
        self.NLO = NLO
        self.strucf = Structure_Functions(pdf, precision)

        self.calc_count = 0
        if self.s < self.pdf.q2Max:
           self.physical = True 
        else:
            self.physical = False

        self.lnQ2min = np.log(pdf.q2Min)
        self.lnQ2max = np.log(self.s)

        #self.lnQ2min = pdf.q2Min
        #self.lnQ2max = self.s

        #self.xmin = pdf.xMin
        #self.xmin = 1.5e-5 #1e-7
        self.xmin = 1e-9
        #print(f'{self.lnQ2min=}, {self.lnQ2max}')

        self.calc_count = 0
        self.count_xmin = 0
        self.maxworkers = 8
        self.multithread = multithread
        self.precision = precision

        self.compare_lo = []
        self.compare_nlo = []
        self.compare_x = []
        self.compare_q = []

    def calc(self):
        if self.physical:
            self.pbar = tqdm(total=3528)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if self.multithread:
                    list_Q2 = np.linspace(self.lnQ2min, self.lnQ2max, self.maxworkers + 1)
                    errs = []
                    splits_Q2 = []
                    for i in range(self.maxworkers):
                        splits_Q2.append((list_Q2[i], list_Q2[i+1]))

                    print(splits_Q2)
                    cs = 0
                    with cf.ThreadPoolExecutor(max_workers=self.maxworkers) as executor:
                        future_split = {executor.submit(integrate.quad, self.diff_lnQ2, split_Q2[0], split_Q2[1], epsabs=self.precision, epsrel=self.precision): split_Q2 for split_Q2 in splits_Q2}
                        for future in cf.as_completed(future_split):
                            result = future.result()
                            cs += result[0]
                            errs.append(result[1])
                            err = np.sqrt(np.sum([i*i for i in errs]))
                            print(f'cumulative cs: {cs}, combined err: {err}')

                    err = np.sqrt(np.sum([i*i for i in errs]))

                else:
                    cs, err = integrate.quad(self.diff_lnQ2, self.lnQ2min, self.lnQ2max, epsabs=self.precision)
            self.pbar.close()
            return cs, err
        else:
            print('E_nu to high; s > q2max')
            return None
    
    def calc_deviation(self):
        if self.physical:
            self.cs_list = []
            self.err_list = []
            for i in range(10):
                cs, err = integrate.quad(self.diff_lnQ2, self.lnQ2min, self.lnQ2max)
                self.cs_list.append(cs)
                self.err_list.append(err)
            cs, err = self.deviation()
            return cs, err 

        else:
            print('E_nu to high; s > q2max')
            return None

    def deviation(self):
        cs = np.mean(self.cs_list)
        err = np.std(self.cs_list)
        return cs, err

    def diff_lnQ2(self, lnQ2):
        lnxmin = lnQ2 - np.log(self.s)
        #print(f'-----> lnxmin: {np.exp(lnxmin)}')
        #lnxmin = np.max([lnxmin, np.log(self.xmin)])
        lnxmax = -1e-9 # or just 0 ?
        #lnxmin = np.exp(lnQ2 - np.log(self.s))
        #lnxmin = np.log(lnQ2/self.s)
        #lnxmax = 1
        Q2 = np.exp(lnQ2)
        #Q2 = lnQ2
        #print(lnxmin, lnxmax)
        ddif, err = integrate.quad(self.ddiff_lnx_lnQ2, lnxmin, lnxmax, args=(Q2,), epsabs=self.precision, epsrel=self.precision)
        return Q2 * ddif
        return ddif
    
    def struc(self, x, Q2):
        if not self.anti:
            if self.target == 'isoscalar':
                #F2 = (self.pdf.xfxQ2(1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) + self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(-2, x, Q2) + 2*self.pdf.xfxQ2(3, x, Q2) + 2*self.pdf.xfxQ2(-4, x, Q2))
                #xF3 = ((self.pdf.xfxQ2(2, x, Q2) - self.pdf.xfxQ2(-2, x, Q2)) + (self.pdf.xfxQ2(1, x, Q2) - self.pdf.xfxQ2(-1, x, Q2)) + 2*self.pdf.xfxQ2(3, x, Q2) - 2*self.pdf.xfxQ2(-4, x, Q2))
                # LO method 2
                F2 = (self.pdf.xfxQ2(1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) + self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(-2, x, Q2) + self.pdf.xfxQ2(3, x, Q2) + self.pdf.xfxQ2(-3, x, Q2) + self.pdf.xfxQ2(4, x, Q2) + self.pdf.xfxQ2(-4, x, Q2))
                xF3 = ((self.pdf.xfxQ2(2, x, Q2) - self.pdf.xfxQ2(-2, x, Q2)) + (self.pdf.xfxQ2(1, x, Q2) - self.pdf.xfxQ2(-1, x, Q2)) + self.pdf.xfxQ2(3, x, Q2) - self.pdf.xfxQ2(-3, x, Q2) + self.pdf.xfxQ2(4, x, Q2) - self.pdf.xfxQ2(-4, x, Q2))

            elif self.target == 'proton':
                F2 = 2*(self.pdf.xfxQ2(1, x, Q2) + self.pdf.xfxQ2(-2, x, Q2) + self.pdf.xfxQ2(3, x, Q2) + self.pdf.xfxQ2(-4, x, Q2))
                xF3 = 2*(self.pdf.xfxQ2(1, x, Q2) - self.pdf.xfxQ2(-2, x, Q2) + self.pdf.xfxQ2(3, x, Q2) - self.pdf.xfxQ2(-4, x, Q2))
        
            elif self.target == 'neutron':
                F2 = 2*(self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) + self.pdf.xfxQ2(3, x, Q2) + self.pdf.xfxQ2(-4, x, Q2))
                xF3 = 2*( -self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) + self.pdf.xfxQ2(3, x, Q2) - self.pdf.xfxQ2(-4, x, Q2))

            # NLO
            #return fact * (Yp * self.pdf.xfxQ2(2001, x, Q2) - y*y * self.pdf.xfxQ2(2002, x, Q2) + Ym * self.pdf.xfxQ2(2003, x, Q2))
        elif self.anti:
            if self.target == 'isoscalar':
                F2 = (self.pdf.xfxQ2(1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) + self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(-2, x, Q2) + 2*self.pdf.xfxQ2(-3, x, Q2) + 2*self.pdf.xfxQ2(4, x, Q2))
                xF3 = ((self.pdf.xfxQ2(2, x, Q2) - self.pdf.xfxQ2(-2, x, Q2)) + (self.pdf.xfxQ2(1, x, Q2) - self.pdf.xfxQ2(-1, x, Q2)) - 2*self.pdf.xfxQ2(-3, x, Q2) + 2*self.pdf.xfxQ2(4, x, Q2))

            elif self.target == 'proton':
                F2 = 2*(self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) + self.pdf.xfxQ2(3, x, Q2) + self.pdf.xfxQ2(-4, x, Q2))
                xF3 = 2*(-self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) - self.pdf.xfxQ2(-3, x, Q2) + self.pdf.xfxQ2(4, x, Q2))
        
            elif self.target == 'neutron':
                F2 = 2*(self.pdf.xfxQ2(1, x, Q2) + self.pdf.xfxQ2(-2, x, Q2) + self.pdf.xfxQ2(-3, x, Q2) + self.pdf.xfxQ2(4, x, Q2))
                xF3 = 2*(+self.pdf.xfxQ2(1, x, Q2) - self.pdf.xfxQ2(-2, x, Q2) - self.pdf.xfxQ2(-3, x, Q2) + self.pdf.xfxQ2(4, x, Q2))
        return F2, xF3

    def ddiff_lnx_lnQ2(self, lnx, Q2):
        self.calc_count += 1
        x = np.max([np.exp(lnx), self.xmin])
        if x == self.xmin:
            self.count_xmin += 1
        #x = np.exp(lnx)
        #x = lnx
        #x = 0.1
        #Q2 = 100
        self.pbar.update(1)

        y = Q2 / x / self.s
        omy2 = np.power((1-y), 2)
        Yp = 1 + omy2
        Ym = 1 - omy2
        fact = self.convert * self.GF2 / 4 / np.pi * np.power(self.Mw2 / (self.Mw2 + Q2 ), 2)

        if False:
            if self.calc_count >= 200:
                save_pickle([self.compare_lo, self.compare_nlo, self.compare_x, self.compare_q], 'compare_lo_nlo')
                quit()

            F2, xF3 = self.struc(x, Q2)
            lo = fact * (Yp * F2 + Ym * xF3)

            fact2 = self.convert * self.GF2 / np.pi * np.power(self.Mw2 / (self.Mw2 + Q2 ), 2) # the paper has extra factor 2

            xF1, F2, xF3 = NLO_functions.struc_NLO_m(x, Q2) #self.struc_LO(x, Q2)
            nlo = fact2 * (F2*(1-y) + xF1*y*y + xF3*y*(1 - y/2))

            #xF1, F2, xF3 = NLO_functions.struc_LO(x, Q2) #self.struc_LO(x, Q2)
            #lo = fact2 * (F2*(1-y) + xF1*y*y + xF3*y*(1 - y/2))
            abs_diff = np.abs(nlo - lo)
            if abs_diff > 0.02:
                print(f'{x=}, {Q2=}, {lnx=}, {abs_diff=} ')

            self.compare_lo.append(lo)
            self.compare_nlo.append(nlo)
            self.compare_x.append(x)
            self.compare_q.append(Q2)
            return nlo

        if self.NLO:
            fact = self.convert * self.GF2 / np.pi * np.power(self.Mw2 / (self.Mw2 + Q2 ), 2) # the paper has extra factor 2
            #xF1, F2, xF3 = self.struc_NLO(x, Q2)

            #xF1, F2, xF3 = self.strucf.struc_NLO_m(x, Q2) #self.struc_LO(x, Q2)
            xF1, F2, xF3 = self.strucf.strucf(x, Q2) #self.struc_LO(x, Q2)
            #print(f'diff xF1 NLO: {xF1}, xF1 lo: {xF1_lo}, diff: {np.abs(xF1 - xF1_lo)}')
            if self.anti:
                return fact * (F2*(1-y) + xF1*y*y - xF3*y*(1 - y/2))
            else:
                return fact * (F2*(1-y) + xF1*y*y + xF3*y*(1 - y/2))
            # NLO from paper
            #return fact * (Yp * self.pdf.xfxQ2(2001, x, Q2) - y*y * self.pdf.xfxQ2(2002, x, Q2) + Ym * self.pdf.xfxQ2(2003, x, Q2))

        elif self.anti:
            F2, xF3 = self.struc(x, Q2)
            return fact * (Yp * F2 - Ym * xF3)
        else:
            F2, xF3 = self.struc(x, Q2)
            return fact * (Yp * F2 + Ym * xF3)

#pdf_struc = lhapdf.mkPDF("NNPDF31sx_nnlonllx_as_0118_LHCb_nf_6_SF")
pdf_31 = lhapdf.mkPDF("NNPDF31_lo_as_0118")
nlo_pdf_31 = lhapdf.mkPDF("NNPDF31_nlo_as_0118_mc")
nlo_pdf_40 = lhapdf.mkPDF("NNPDF40_nlo_as_01180_nf_6")
pdf_40 = lhapdf.mkPDF("NNPDF40_lo_as_01180")
pdf_21 = lhapdf.mkPDF("NNPDF21_lo_as_0119_100")
nlo_pdf_23 = lhapdf.mkPDF("NNPDF23_nlo_as_0118")
#cs = cs_neutrino_nucleon(1e6, pdf)

df = pd.read_csv('cs_3.csv')
name = 'pdf31_LO_acc_0.01_x9_anti'
df[name] = 19*[0.0]
df[name + '_err'] = 19*[0.0]

for name, pdf in [(name, pdf_31), ('pdf40_LO_acc_0.01_x9_anti', pdf_40), ('pdf21_LO_acc_0.01_x9_anti', pdf_21)]: #, ('pdf31n_NLO_acc_0.01_x9_v5', nlo_pdf_31)]:#,  ('log21', pdf_21)]:
    print(f'PDF values: {pdf.q2Min=}')
    for i in range(0, 19): # 19 to end
        E_nu = df.at[i, 'E_nu']
        dt_start = datetime.datetime.now()
        cs = cs_neutrino_nucleon(E_nu, pdf, anti=True, target='isoscalar', NLO=False, multithread=True, precision=0.01)
        print('physical', cs.physical)
        print(f'Calculating for E_nu: {E_nu}')
        if cs.physical:
            sigma, err = cs.calc()
        else:
            sigma = 0.0
            err = 0.0
        dt_end = datetime.datetime.now()
        print(f'Calc count: {cs.calc_count}, xmin count: {cs.count_xmin}')
        print(cs.calc_count)
        print()
        print(f'cs: {sigma}, err: {err}, E_nu: {E_nu}, cs/cs-ref: {sigma/df.at[i, "cs"]}')
        print(f'Time of calc: {(dt_end - dt_start)}')
        print()

        if True:
            df.at[i, name] = sigma
            df.at[i, name + "_err"] = err
            df.to_csv('cs_3.csv', index=False)

#import cs_trend