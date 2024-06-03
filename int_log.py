import NLO_functions
import lhapdf
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import pandas as pd
import multiprocessing
import datetime
from tqdm import tqdm
import warnings


class cs_neutrino_nucleon:

    def __init__(self, E_nu, pdf, anti=False, target='isoscalar', NLO=False):
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
        self.xmin = 1e-2
        #print(f'{self.lnQ2min=}, {self.lnQ2max}')

        self.calc_count = 0

    def calc(self):
        if self.physical:
            self.pbar = tqdm(total=2e6)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cs, err = integrate.quad(self.diff_lnQ2, self.lnQ2min, self.lnQ2max)
            self.pbar.close()
            print(f'Took {self.calc_count} cals to ddif_sigma')
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
        lnxmax = -1e-9 # or just 0 ?
        #lnxmin = np.exp(lnQ2 - np.log(self.s))
        #lnxmin = np.log(lnQ2/self.s)
        #lnxmax = 1
        Q2 = np.exp(lnQ2)
        #Q2 = lnQ2
        #print(lnxmin, lnxmax)
        ddif, err = integrate.quad(self.ddiff_lnx_lnQ2, lnxmin, lnxmax, args=(Q2,))
        return Q2 * ddif
        return ddif
    
    def struc(self, x, Q2):
        if not self.anti:
            if self.target == 'isoscalar':
                F2 = (self.pdf.xfxQ2(1, x, Q2) + self.pdf.xfxQ2(2, x, Q2) + self.pdf.xfxQ2(-1, x, Q2) + self.pdf.xfxQ2(-2, x, Q2) + 2*self.pdf.xfxQ2(3, x, Q2) + 2*self.pdf.xfxQ2(-4, x, Q2))
                xF3 = ((self.pdf.xfxQ2(2, x, Q2) - self.pdf.xfxQ2(-2, x, Q2)) + (self.pdf.xfxQ2(1, x, Q2) - self.pdf.xfxQ2(-1, x, Q2)) + 2*self.pdf.xfxQ2(3, x, Q2) - 2*self.pdf.xfxQ2(-4, x, Q2))

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


    def delta(self, x):
        epsilon = 0.001
        if x <= epsilon and x >= -epsilon:
            return 1.0
        else:
            return 0.0

    def Cqiz(self, z):
        '''Coefficient function for the quarks'''
        a = (1 + z*z) * (np.log(1 - z)/(1 - z))
        b = - np.divide((1 + z*z), (1 - z)) * np.log(z)
        c = -3/2 / (1 - z) + 3 + 2*z
        d = -(9/2 + np.pi*np.pi/3) * self.delta(1 - z)

        return a + b + c + d -2*z
        #elif i == 2:
        #    return a + b + c + d 
        #elif i == 3:
        #    return a + b + c + d - (1 + z) 
    def Cg(self, z):
        a = np.power((1 - z), 2) + z*z
        b = np.log((1 - z) / z)
        c = 6*z*(1 - z)
        return a*b + c + 4*z*(1 - z)
    
    def Cq_lo(self, z):
        return self.delta(1 - z)
    
    def convolution(self, C, q, x, Q2):
        conv_int = lambda  z: C(x/z) * self.pdf.xfxQ2(q, z, Q2) / z
        conv, err = integrate.quad(conv_int, x, 1)
        return conv

    def struc_NLO(self, x, Q2):
        quarks = ['up', 'down', 'strange', 'charm']
        flavours = {'up':2, 'down':1, 'strange':3, 'charm':4,}
        anti_flavours = {'up':-2, 'down':-1, 'strange':-3, 'charm':-4}
        K1 = {'up':1/4, 'down':1/4, 'strange':1/4, 'charm':1/4,}
        K2 = {'up':1/2, 'down':1/2, 'strange':1/2, 'charm':1/2,}

        print(f'alpha Q2:{self.pdf.alphasQ2(Q2)}, \nconvolution quark down: {self.convolution(self.Cqiz, flavours["down"], x, Q2)}, \nconvolution gluon: {self.convolution(self.Cg, 21, x, Q2)}')
        xF1 = np.array([K1[q]*(self.pdf.xfxQ2(flavours[q], x, Q2) + self.pdf.xfxQ2(anti_flavours[q], x, Q2)
                               + self.pdf.alphasQ2(Q2)*(self.convolution(self.Cqiz, flavours[q], x, Q2) + 
                                                       self.convolution(self.Cqiz, anti_flavours[q], x, Q2) + 
                                                       self.convolution(self.Cg, 21, x, Q2))
                               ) for q in quarks]).sum()
        F2 = np.array([K2[q]*(self.pdf.xfxQ2(flavours[q], x, Q2) + self.pdf.xfxQ2(anti_flavours[q], x, Q2)) for q in quarks]).sum()
        xF3 = np.array([K2[q]*(self.pdf.xfxQ2(flavours[q], x, Q2) - self.pdf.xfxQ2(anti_flavours[q], x, Q2)) for q in quarks]).sum()
        return xF1, F2, xF3

    def struc_LO(self, x, Q2):
        quarks = ['up', 'down', 'strange', 'charm']
        anti_quarks = ['anti-up', 'anti-down', 'anti-strange', 'anti-charm']
        flavours = {'up':2, 'down':1, 'strange':3, 'charm':4,}
        anti_flavours = {'up':-2, 'down':-1, 'strange':-3, 'charm':-4}
        #anti_flavours = {'anti-up':-2, 'anti-down':-1, 'anti-strange':-3, 'anti-charm':-4}
        #f1 = {'up':0.0, 'down':0.0, 'strange':0.0, 'charm':0.0,}
        #f1a = {'anti-up':0.0, 'anti-down':0.0, 'anti-strange':0.0, 'anti-charm':0.0}
        #flavours = {'up':2, 'anti-up':-2, 'down':1, 'anti-down':-1, 'strange':3, 'anti-strange':-3, 'charm':4, 'anti-charm':-4}
        #f1 = {'up':0.0, 'anti-up':0.0, 'down':0.0, 'anti-down':0.0, 'strange':0.0, 'anti-strange':0.0, 'charm':0.0, 'anti-charm':0.0}
        #f1 = self.f_LO(1, x, Q2)
        #f1_g = 0
        #g = self.pdf(21, x, Q2)
        K1 = {'up':1/4, 'down':1/4, 'strange':1/4, 'charm':1/4,}
        K2 = {'up':1/2, 'down':1/2, 'strange':1/2, 'charm':1/2,}

        #F1 = [self.convolution(flavours[qi], f1[qi]) for qi in quarks] + [self.convolution(flavours[aqi], f1[aqi]) for aqi in anti_quarks] + self.convolution(g, f1_g)
        # quick 
        xF1 = np.array([K1[q]*(self.pdf.xfxQ2(flavours[q], x, Q2) + self.pdf.xfxQ2(anti_flavours[q], x, Q2)) for q in quarks]).sum()
        F2 = np.array([K2[q]*(self.pdf.xfxQ2(flavours[q], x, Q2) + self.pdf.xfxQ2(anti_flavours[q], x, Q2)) for q in quarks]).sum()
        xF3 = np.array([K1[q]*(self.pdf.xfxQ2(flavours[q], x, Q2) - self.pdf.xfxQ2(anti_flavours[q], x, Q2)) for q in quarks]).sum()
        # slow with conv.
        xF1 = np.array([K1[q]*(self.convolution(self.Cq_lo, flavours[q], x, Q2) + self.convolution(self.Cq_lo, flavours[q], x, Q2)) for q in quarks]).sum()
        F2 = np.array([K2[q]*(self.convolution(self.Cq_lo, flavours[q], x, Q2) + self.convolution(self.Cq_lo, flavours[q], x, Q2)) for q in quarks]).sum()
        xF3 = np.array([K2[q]*(self.convolution(self.Cq_lo, flavours[q], x, Q2) - self.convolution(self.Cq_lo, flavours[q], x, Q2)) for q in quarks]).sum()
        return xF1, F2, xF3

    def ddiff_lnx_lnQ2(self, lnx, Q2):
        self.calc_count += 1
        x = np.max([np.exp(lnx), self.xmin])
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

        if self.NLO:
            fact = self.convert * self.GF2 / np.pi * np.power(self.Mw2 / (self.Mw2 + Q2 ), 2) # the paper has extra factor 2
            #xF1, F2, xF3 = self.struc_NLO(x, Q2)

            xF1, F2, xF3 = NLO_functions.struc_NLO_m(x, Q2) #self.struc_LO(x, Q2)
            #print(f'diff xF1 NLO: {xF1}, xF1 lo: {xF1_lo}, diff: {np.abs(xF1 - xF1_lo)}')
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
#pdf_40 = lhapdf.mkPDF("NNPDF40_lo_as_01180")
#pdf_21 = lhapdf.mkPDF("NNPDF21_lo_as_0119_100")
#cs = cs_neutrino_nucleon(1e6, pdf)

df = pd.read_csv('cs_3.csv')
name = 'pdf31_NLO_x2'
df[name] = 19*[0.0]
df[name + '_err'] = 19*[0.0]

for name, pdf in [(name, pdf_31)]:#, ('log40', pdf_40), ('log21', pdf_21)]:
#for name, pdf in [('log40', pdf_40)]:
# 0, 19 all 
# 7, 8 for 1e6
    for i in range(0, 19): # 19 to end
        E_nu = df.at[i, 'E_nu']
        dt_start = datetime.datetime.now()
        cs = cs_neutrino_nucleon(E_nu, pdf, anti=False, target='isoscalar', NLO=True)
        print('physical', cs.physical)
        if cs.physical:
            sigma, err = cs.calc()
        else:
            sigma = 0.0
            err = 0.0
        dt_end = datetime.datetime.now()
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