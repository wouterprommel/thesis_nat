import numpy as np
import matplotlib.pyplot as plt
import lhapdf
import path
from tqdm import tqdm
from scipy import integrate
import pickle

pdf = lhapdf.mkPDF("NNPDF31_lo_as_0118")

def test(solver, xmin, flavour):
    X = np.logspace(-xmin, 1, 1000, endpoint=False)
    X = X[X < 1.0]
    Q2 = 1e6
    R = []
    G = []
    for x in tqdm(X):
        assert x < 1.0
        G.append(F1_lo(x, Q2) * x)
        R.append(solver(x, Q2) * x)
    plt.plot(X, R, label=f'F3 NLO')
    plt.plot(X, G, label='F3 LO')
    plt.xscale('log')
    plt.xlabel('x')
    plt.legend()
    name = path.fig_path() + f"F3_NLO_x{xmin}_Q2.1e{np.log10(Q2)}"
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

def q(z, Q2, flavour):
    return np.array([pdf.xfxQ2(flavour, i, Q2)/i for i in z])

def q_s(z, Q2, flavour):
    return pdf.xfxQ2(flavour, z, Q2)/z

def Cc(x, Q2, flavour):
    #x = 0.129
    f = lambda u: -3/2/u * q_2(u, x, Q2, flavour)
    r = MC_div(f, x, 1)
    return r

def Ce(x, Q2, flavour):
    return - pdf.xfxQ2(flavour, x, Q2)/x * (9/2 + np.pi**2/3)

def Cd(x, Q2, flavour, i):
    if i == 1:
        f = lambda z: (3/z) * q_s(z, Q2, flavour) 
    else:
        f = lambda z: (3/z + 2*x/z/z) * q_s(z, Q2, flavour) 
    r, err = integrate.quad(f, x, 1)
    #r = MC(f, x, 1)
    return r

def Cb(x, Q2, flavour):
    f = lambda z: (1 + x*x/z/z)/(z - x) * np.log(x/z) * q_s(z, Q2, flavour)
    r, err = integrate.quad(f, 1, x)
    #r = MC_log(f, x, 1)
    return r

def Ca(x, Q2, flavour):
    #f = lambda u: (1/u + u)*q_2(u, x, Q2)*np.log(1 - u) / (1 - u)
    f = lambda z: 1/z * (1 + x*x/z/z) * np.log(1 - x/z) * q(z, Q2, flavour)
    r = MC_div(f, x, 1)
    return r

def C3(x, Q2, flavour):
    f = lambda z: -(1/z + x/z/z) * q(z, Q2, flavour)
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

def F1_nlo(x, Q2):
    K1 = 1/4
    quarks = ['up', 'down', 'strange', 'charm']
    flavours = {'up':2, 'down':1, 'strange':3, 'charm':4,}
    anti_flavours = {'up':-2, 'down':-1, 'strange':-3, 'charm':-4}
    F1 = np.sum(K1*np.array([q_s(x, Q2, flavours[q]) +  q_s(x, Q2, anti_flavours[q]) 
                          + pdf.alphasQ2(Q2)*(C(x, Q2, flavours[q], 1) + C(x, Q2, anti_flavours[q], 1) + 2*Cg(x, Q2, 1)) for q in quarks]))
    return F1

def F2_nlo(x, Q2):
    K2 = 1/2
    quarks = ['up', 'down', 'strange', 'charm']
    flavours = {'up':2, 'down':1, 'strange':3, 'charm':4,}
    anti_flavours = {'up':-2, 'down':-1, 'strange':-3, 'charm':-4}
    F2 = np.sum(K2*x*np.array([q_s(x, Q2, flavours[q]) +  q_s(x, Q2, anti_flavours[q]) 
                          + pdf.alphasQ2(Q2)*(C(x, Q2, flavours[q], 2) + C(x, Q2, anti_flavours[q], 2) + 2*Cg(x, Q2, 2)) for q in quarks]))
    return F2

def F3_nlo(x, Q2):
    K2 = 1/2
    quarks = ['up', 'down', 'strange', 'charm']
    flavours = {'up':2, 'down':1, 'strange':3, 'charm':4,}
    anti_flavours = {'up':-2, 'down':-1, 'strange':-3, 'charm':-4}
    F3 = np.sum(K2*x*np.array([q_s(x, Q2, flavours[q]) -  q_s(x, Q2, anti_flavours[q]) 
                          + pdf.alphasQ2(Q2)*(C(x, Q2, flavours[q], 2) - C(x, Q2, anti_flavours[q], 2)) for q in quarks]))
    return F3

def F1_lo(x, Q2):
    K1 = 1/2
    quarks = ['up', 'down', 'strange', 'charm']
    flavours = {'up':2, 'down':1, 'strange':3, 'charm':4,}
    anti_flavours = {'up':-2, 'down':-1, 'strange':-3, 'charm':-4}
    F1 = np.sum(K1*np.array([pdf.xfxQ2(flavours[q], x, Q2) - pdf.xfxQ2(anti_flavours[q], x, Q2) for q in quarks]))
    return F1


#r = MC(lambda x: np.sin(x), 0, 1)
#r = MC_div(f, x, 1)
#print(r)
test(F3_nlo, 3, 1)
#print(pdf.xfxQ2(1, 1, 100))