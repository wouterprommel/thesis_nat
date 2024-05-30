import lhapdf
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import signal
from datetime import datetime as dt
from tqdm import tqdm
import path

def delta_old(x):
    epsilon = 0.001
    if x <= epsilon and x >= -epsilon:
        return 1.0
    else:
        return 0.0
    
def delta(x):
    epsilon = 0.001
    bm1 = x <= epsilon
    bm2 = x >= -epsilon
    return np.array((bm1 & bm2)*1.0)


def C1q(z):
    '''Coefficient function for the quarks'''
    a = (1 + z*z) * (np.log(1 - z)/(1 - z))
    b = - np.divide((1 + z*z), (1 - z)) * np.log(z)
    c = -3/2 / (1 - z) + 3 + 2*z
    d = -(9/2 + np.pi*np.pi/3) * delta(1 - z)

    return a + b + c + d -2*z
    #elif i == 2:
    #    return a + b + c + d 
    #elif i == 3:
    #    return a + b + c + d - (1 + z) 
def Cg(z):
    a = np.power((1 - z), 2) + z*z
    b = np.log((1 - z) / z)
    c = 6*z*(1 - z)
    return a*b + c + 4*z*(1 - z)

def Cq_lo(z):
    print(z)
    return delta(1 - z)

def convolution(C, q, x, Q2):
    conv_int = lambda  z: C(x/z) * pdf.xfxQ2(q, z, Q2) / z
    conv, err = integrate.quad(conv_int, x, 1)
    return conv

def conv2(C, q, x, Q2):
    Z = np.linspace(x, 1, 1000, endpoint=False)
    p = np.array([pdf.xfxQ2(q, z, Q2) for z in Z])
    C = lambda y: delta(1 - y) * Z
    #print(C(x/Z))
    conv = p * C(x/Z) / Z
    conv = p * C1q(x/Z) / Z
    #print(p.shape, C1q(Z).shape)
    #conv = np.convolve(p, C1q(Z))
    #print(conv.max())
    return conv.sum()

def gs(xs, Q2):
    g = np.array([pdf.xfxQ2(21, x, Q2)/x for x in xs])
    return g

def g(x, Q2):
    g = pdf.xfxQ2(21, x, Q2)/x
    return g

def Cg_conv_g(x, Q2):
    f = lambda z: (6*(x/z/z * (1 - x)) + (1/z - 2*x/z/z + x*x/z/z/z) *np.log((z- x)/x) )* g(z, Q2)
    int, err = integrate.quad(f, x, 1)
    #f2 = lambda z: (1/z - 2*x/z/z + x*x/z/z/z) *np.log((z- x)/x)* g(z, Q2)
    #int2, err = integrate.quad(f2, x, 1)
    #print(f'{int=} \n{int2=}')
    return int #+ int2

pdf = lhapdf.mkPDF("NNPDF31_lo_as_0118")

def gluon():
    #conv = convolution(C1q, 1, 0.1, 1000)
    X = np.logspace(-7, 1, 1000, endpoint=False)
    X = X[X < 1.0]
    Q2 = 1e6
    R = []
    G = []
    for x in tqdm(X):
        R.append(Cg_conv_g(x, Q2) * x)
        G.append(pdf.xfxQ2(21, x, Q2))
    plt.plot(X, R, label='x * C @ g')
    plt.plot(X, G, label='x * gluon')
    plt.xscale('log')
    plt.xlabel('x')
    plt.legend()
    plt.savefig(path.fig_path() + "C_g_x7_Q6.pdf", format="pdf", bbox_inches="tight")
    plt.show()


    x = 1e-3
    #test()

    #conv2 = conv2(Cq_lo, 1, x, 1000)
    s = dt.now()
    int = Cg_conv_g(x, Q2)
    e = dt.now()
    print('result int:', int)
    print('gluon value: ', pdf.xfxQ2(21, x, Q2)/x)
    print('took: ', e - s)
    #print(conv2)

def Cq_conv_q(x, Q2):
    Caa = lambda z: (1 + x*x/z/z) * np.log(1 - x/z)/(z - x) #* (pdf.xfxQ2(1, x, Q2)/x)
    Cb = lambda z: -(1 + x*x/z/z)/(z - x) * np.log(x/z) #* (pdf.xfxQ2(1, x, Q2)/x) # added Cd
    Cd = lambda z: 3 + 2*z
    Ce = - pdf.xfxQ2(1, x, Q2)/x * (9/2 + np.pi**2 /3)
    Cca = lambda z: - 3/2 * (1/(z-x))
    C = lambda z: (Caa(z) + Cb(z) + Cca(z) + Cd(z)) * (pdf.xfxQ2(1, x, Q2)/x)
    int_y, _ = integrate.quad(lambda y: np.log(1 - y)/(1 - y), 0, 1)
    int_y2, _ = integrate.quad(lambda y: 1/(1 - y), 0, 1)
    print(int_y)
    print(Ce)
    Cab = - pdf.xfxQ2(1, x, Q2)/x * 2 * int_y
    Ccb = pdf.xfxQ2(1, x, Q2)/x * 3 / 2 * int_y2
    int, _ = integrate.quad(C, x, 1)
    #int2, _ = integrate.quad(Caa, x, 1)
    return int +Cab + Ccb + Ce


#x = 4e-7
x = 0.04#4e-3
Q2 = 1e6
s = dt.now()
int = Cq_conv_q(x, Q2)
e = dt.now()
print('result int:', int)
print('quark value: ', pdf.xfxQ2(1, x, Q2)/x)
print('took: ', e - s)