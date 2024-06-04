import warnings
import numpy as np
import matplotlib.pyplot as plt
import lhapdf
import path
from tqdm import tqdm
from scipy import integrate
import pickle
import multiprocessing
import functools
from datetime import datetime

pdf = lhapdf.mkPDF("NNPDF31_lo_as_0118")

Flavours = [1, 2, 3, 4, -1, -2, -3, -4]

def test(solver, xmin, flavour):
    X = np.logspace(-xmin, 1, 1000, endpoint=False)
    X = X[X < 1.0]
    Q2 = 1e6
    R = []
    G = []
    for x in tqdm(X):
        assert x < 1.0
        G.append(C_sum(x, Q2, 2) * x)
        #R.append(np.sum([solver(x, Q2, flavour, 2) * x for flavour in Flavours]))
    #plt.plot(X, R, label=f'F2 NLO')
    plt.plot(X, G, label='F2 NLO sum')
    plt.xscale('log')
    plt.xlabel('x')
    plt.legend()
    name = path.fig_path() + f"F2_NLO_x{xmin}_Q2.1e{np.log10(Q2)}"
    if False:
        file = open(name + '.pickle', 'ab')
        pickle.dump([X, R, G], file)
        file.close()

    plt.savefig(name + ".pdf", format="pdf", bbox_inches="tight")
    plt.show()

def MC(func, a, b):
    N = 1000
    x = np.random.uniform(a, b, N)
    R = func(x)
    return (b-a)*R.mean()

def MC_log(func, a, b):
    N = 1000
    x = np.exp(np.random.uniform(np.log(a), np.log(b), N))
    R = func(x)
    return (b-a)*R.mean()

def MC_div(f, a, b):
    N = 10000
    #x = np.random.uniform(a, b, N)
    x = np.exp(np.random.uniform(np.log(a), np.log(b), N))
    R = (f(x) - f(np.array([b])))/(1-x)
    return (b-a)*R.mean()

def g(x, Q2):
    g = pdf.xfxQ2(21, x, Q2)/x
    return g

def Cg(x, Q2, i):
    if i == 1:
        f = lambda z: (10*(x/z/z * (1 - x)) + (1/z - 2*x/z/z + x*x/z/z/z) *np.log((z- x)/x) )* g(z, Q2)
    else:
        f = lambda z: (6*(x/z/z * (1 - x)) + (1/z - 2*x/z/z + x*x/z/z/z) *np.log((z- x)/x) )* g(z, Q2)

    int, err = integrate.quad(f, x, 1)
    #f2 = lambda z: (1/z - 2*x/z/z + x*x/z/z/z) *np.log((z- x)/x)* g(z, Q2)
    #int2, err = integrate.quad(f2, x, 1)
    #print(f'{int=} \n{int2=}')
    return int #+ int2

def q_2(u, x, Q2, flavour):
    return np.array([pdf.xfxQ2(flavour, x/i, Q2)*i/x for i in u])

def q_2_sum(u, x, Q2):
    return np.array([np.sum([pdf.xfxQ2(flavour, x/i, Q2) for flavour in Flavours])*i/x for i in u])

def q(z, Q2, flavour):
    return np.array([pdf.xfxQ2(flavour, i, Q2)/i for i in z])

def q_sum(z, Q2):
    return np.array([np.sum([pdf.xfxQ2(flavour, i, Q2) for flavour in Flavours])/i for i in z])

def q_s(z, Q2, flavour):
    return pdf.xfxQ2(flavour, z, Q2)/z

def q_s_sum(z, Q2):
    return np.sum([pdf.xfxQ2(flavour, z, Q2) for flavour in Flavours])/z

def Cc(x, Q2, flavour):
    #x = 0.129
    f = lambda u: -3/2/u * q_2(u, x, Q2, flavour)
    r = MC_div(f, x, 1)
    return r

def Cc_sum(x, Q2):
    #x = 0.129
    f = lambda u: -3/2/u * q_2_sum(u, x, Q2)
    r = MC_div(f, x, 1)
    return r

def Ce(x, Q2, flavour):
    return - pdf.xfxQ2(flavour, x, Q2)/x * (9/2 + np.pi**2/3)

def Ce_sum(x, Q2):
    return - q_s_sum(x, Q2) * (9/2 + np.pi**2/3)

def Cd(x, Q2, flavour, i):
    if i == 1:
        f = lambda z: (3/z) * q_s(z, Q2, flavour) 
    else:
        f = lambda z: (3/z + 2*x/z/z) * q_s(z, Q2, flavour) 
    r, err = integrate.quad(f, x, 1)
    #r = MC(f, x, 1)
    return r

def Cd_sum(x, Q2, i):
    if i == 1:
        f = lambda z: (3/z) * q_s_sum(z, Q2) 
    else:
        f = lambda z: (3/z + 2*x/z/z) * q_s_sum(z, Q2) 
    r, err = integrate.quad(f, x, 1)
    #r = MC(f, x, 1)
    return r

def Cb(x, Q2, flavour):
    f = lambda z: (1 + x*x/z/z)/(z - x) * np.log(x/z) * q_s(z, Q2, flavour)
    r, err = integrate.quad(f, 1, x)
    #r = MC_log(f, x, 1)
    return r

def Cb_sum(x, Q2):
    f = lambda z: (1 + x*x/z/z)/(z - x) * np.log(x/z) * q_s_sum(z, Q2)
    r, err = integrate.quad(f, 1, x)
    #r = MC_log(f, x, 1)
    return r

def Ca(x, Q2, flavour):
    #f = lambda u: (1/u + u)*q_2(u, x, Q2)*np.log(1 - u) / (1 - u)
    f = lambda z: 1/z * (1 + x*x/z/z) * np.log(1 - x/z) * q(z, Q2, flavour)
    r = MC_div(f, x, 1)
    return r

def Ca_sum(x, Q2):
    #f = lambda u: (1/u + u)*q_2(u, x, Q2)*np.log(1 - u) / (1 - u)
    f = lambda z: 1/z * (1 + x*x/z/z) * np.log(1 - x/z) * q_sum(z, Q2)
    r = MC_div(f, x, 1)
    return r

def C3(x, Q2, flavour):
    f = lambda z: -(1/z + x/z/z) * q_s(z, Q2, flavour)
    r, err = integrate.quad(f, x, 1)
    return r

def C3_sum(x, Q2):
    f = lambda z: -(1/z + x/z/z) * q_s_sum(z, Q2)
    r, err = integrate.quad(f, x, 1)
    return r

def C(x, Q2, flavour, i):
    #Cbd = lambda z: ((1 + x*x/z/z)/(z - x) * np.log(x/z) + (3/z + 2*x/z/z)) * q_s(z, Q2) 
    #r, err = integrate.quad(Cbd, 1, x)
    r = Cb(x, Q2, flavour) + Cd(x, Q2, flavour, i)
    # Ce
    #r += pdf.xfxQ2(flavour, x, Q2)/x * (9/2 + np.pi**2/3)
    r += Ce(x, Q2, flavour)
    r += Cc(x, Q2, flavour) + Ca(x, Q2, flavour)
    if i == 3:
        r += C3(x, Q2, flavour)
    return r

def C_sum(x, Q2, i):
    #Cbd = lambda z: ((1 + x*x/z/z)/(z - x) * np.log(x/z) + (3/z + 2*x/z/z)) * q_s(z, Q2) 
    #r, err = integrate.quad(Cbd, 1, x)
    r = Cb_sum(x, Q2) + Cd_sum(x, Q2, i)
    # Ce
    #r += pdf.xfxQ2(flavour, x, Q2)/x * (9/2 + np.pi**2/3)
    r += Ce_sum(x, Q2)
    r += Cc_sum(x, Q2) + Ca_sum(x, Q2)
    if i == 3:
        r += C3_sum(x, Q2)
    return r

def F1_nlo(x, Q2):
    #x, Q2 = args
    K1 = 1/4
    quarks = ['up', 'down', 'strange', 'charm']
    flavours = {'up':2, 'down':1, 'strange':3, 'charm':4,}
    anti_flavours = {'up':-2, 'down':-1, 'strange':-3, 'charm':-4}
    F1 = np.sum(K1*np.array([q_s(x, Q2, flavours[q]) +  q_s(x, Q2, anti_flavours[q]) 
                          + pdf.alphasQ2(Q2)*(C(x, Q2, flavours[q], 1) + C(x, Q2, anti_flavours[q], 1) + 2*Cg(x, Q2, 1)) for q in quarks]))
    return F1

def F2_nlo(x, Q2):
    #x, Q2 = args
    K2 = 1/2
    quarks = ['up', 'down', 'strange', 'charm']
    flavours = {'up':2, 'down':1, 'strange':3, 'charm':4,}
    anti_flavours = {'up':-2, 'down':-1, 'strange':-3, 'charm':-4}
    F2 = np.sum(K2*x*np.array([q_s(x, Q2, flavours[q]) +  q_s(x, Q2, anti_flavours[q]) 
                          + pdf.alphasQ2(Q2)*(C(x, Q2, flavours[q], 2) + C(x, Q2, anti_flavours[q], 2) + 2*Cg(x, Q2, 2)) for q in quarks]))
    return F2

def Fi_nlo_sum(x, Q2, i):
    #x, Q2 = args

    quarks = ['up', 'down', 'strange', 'charm']
    flavours = {'up':2, 'down':1, 'strange':3, 'charm':4,}
    anti_flavours = {'up':-2, 'down':-1, 'strange':-3, 'charm':-4}
    gluon = 2*Cg(x, Q2, i)
    if i == 1:
        K = 1/4
        Fi_lo = np.sum(np.array([q_s(x, Q2, flavours[q]) + q_s(x, Q2, anti_flavours[q]) for q in quarks]))
    elif i == 2:
        K = 1/2
        Fi_lo = np.sum(np.array([q_s(x, Q2, flavours[q]) + q_s(x, Q2, anti_flavours[q]) for q in quarks]))
    elif i == 3:
        K = 1/2
        Fi_lo = np.sum(np.array([q_s(x, Q2, flavours[q]) - q_s(x, Q2, anti_flavours[q]) for q in quarks]))

    Fi_nlo = pdf.alphasQ2(Q2)*(C_sum(x, Q2, i) + gluon)
    return K*x*(Fi_lo + Fi_nlo)

def F3_nlo(x, Q2):
    #x, Q2 = args
    K2 = 1/2
    quarks = ['up', 'down', 'strange', 'charm']
    flavours = {'up':2, 'down':1, 'strange':3, 'charm':4,}
    anti_flavours = {'up':-2, 'down':-1, 'strange':-3, 'charm':-4}
    F3 = np.sum(K2*np.array([q_s(x, Q2, flavours[q]) -  q_s(x, Q2, anti_flavours[q]) 
                          + pdf.alphasQ2(Q2)*(C(x, Q2, flavours[q], 3) - C(x, Q2, anti_flavours[q], 3)) for q in quarks]))
    return F3

def F_lo(x, Q2):
    K1 = 1/2
    quarks = ['up', 'down', 'strange', 'charm']
    flavours = {'up':2, 'down':1, 'strange':3, 'charm':4,}
    anti_flavours = {'up':-2, 'down':-1, 'strange':-3, 'charm':-4}
    F1 = np.sum(K1*np.array([pdf.xfxQ2(flavours[q], x, Q2) - pdf.xfxQ2(anti_flavours[q], x, Q2) for q in quarks]))
    return F1

def smap(f, *args):
    return f(*args)

def struc_NLO(x, Q2):
    return F1_nlo(x, Q2), F2_nlo(x, Q2), F3_nlo(x, Q2)

def struc_NLO_sum(x, Q2):
    return [Fi_nlo_sum(x, Q2, i) for i in [1, 2, 3]] 

def struc_NLO_m(x, Q2):
    pool = multiprocessing.Pool(processes=3)
#    f1 = Process(target=F1_nlo, args=(x, Q2))
    #F1_nlo(x, Q2), F2_nlo(x, Q2), F3_nlo(x, Q2)
    #res = pool.map(smap, ([F1_nlo, F2_nlo, F3_nlo], [x, x, x], [Q2, Q2, Q2]))
    f1 =functools.partial(F1_nlo, x, Q2)
    f2 =functools.partial(F2_nlo, x, Q2)
    f3 =functools.partial(F3_nlo, x, Q2)
    res = pool.map(smap, [f1, f2, f3])
    return res

def struc_NLO_m_sum(x, Q2):
    pool = multiprocessing.Pool(processes=3)
#    f1 = Process(target=F1_nlo, args=(x, Q2))
    #F1_nlo(x, Q2), F2_nlo(x, Q2), F3_nlo(x, Q2)
    #res = pool.map(smap, ([F1_nlo, F2_nlo, F3_nlo], [x, x, x], [Q2, Q2, Q2]))
    f1 =functools.partial(Fi_nlo_sum, x, Q2, 1)
    f2 =functools.partial(Fi_nlo_sum, x, Q2, 2)
    f3 =functools.partial(Fi_nlo_sum, x, Q2, 3)
    res = pool.map(smap, [f1, f2, f3])
    return res

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = datetime.now()
        a = np.sum([struc_NLO_m_sum(x, 1e6) for x in tqdm(np.linspace(0.001, 1, 40))])
        #test(C, 2, 1)
        te = datetime.now()
        print(f'{a} took {te - ts} sum in int')
        ts = datetime.now()
        b = np.sum([struc_NLO_m(x, 1e6) for x in tqdm(np.linspace(0.001, 1, 40))])
        te = datetime.now()
        print(f'{a} took {te - ts} single thread')
        print(f'{a=} and {b=}, difference is {a-b=}')
    #r = MC(lambda x: np.sin(x), 0, 1)
    #r = MC_div(f, x, 1)
    #print(r)
    #test(F3_nlo, 3, 1)
    #print(pdf.xfxQ2(1, 1, 100))