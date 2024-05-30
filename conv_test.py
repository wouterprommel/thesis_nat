import lhapdf
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import signal

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

def Cg_conv_g(x, Q2):
    f = lambda z: 6*pdf.xfxQ2(21, z, Q2)/z # g
    f2 = lambda z: 6*pdf.xfxQ2(21, z, Q2) # z*g
    int, err = integrate.quad(f, x, 1)
    int2, err = integrate.quad(f2, x, 1)
    return int - int2


pdf = lhapdf.mkPDF("NNPDF31_lo_as_0118")

#conv = convolution(C1q, 1, 0.1, 1000)
x = 0.01
Q2 = 1000
#conv2 = conv2(Cq_lo, 1, x, 1000)
int = Cg_conv_g(x, Q2)
print(int)
print(pdf.xfxQ2(21, x, Q2))
#print(conv2)