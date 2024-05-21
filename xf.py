import lhapdf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

pdf21 = lhapdf.mkPDF("NNPDF21_lo_as_0119_100")
pdf31 = lhapdf.mkPDF("NNPDF31_lo_as_0118")
pdf40 = lhapdf.mkPDF("NNPDF40_lo_as_01180")
pdf31sx = lhapdf.mkPDF("NNPDF31sx_nnlonllx_as_0118_LHCb_nf_6_SF")

def plot(pdf):
    x = np.linspace(pdf.xMin, 1, 10000)
    Q2 = 10

    plt.plot(x, [pdf.xfxQ2(1, i, Q2) for i in x], label='d')
    plt.plot(x, [pdf.xfxQ2(2, i, Q2) for i in x], label='u')
    plt.plot(x, [pdf.xfxQ2(3, i, Q2) for i in x], label='s')
    plt.plot(x, [pdf.xfxQ2(-4, i, Q2) for i in x], label='-c')
    plt.legend()
    plt.ylim((-2, 2))
    plt.show()


def plot_up(pdfs, name, Q2):
    for name, pdf in zip(name, pdfs):
        x = np.linspace(pdf.xMin, 1, 10000)
        plt.plot(x, [pdf.xfxQ2(2, i, Q2) for i in x], label=name)
    #plt.plot(x, [pdf31sx.xfxQ2(2001, i, Q2) for i in x], label='F2')
    plt.legend()
    plt.ylim((-2, 2))
    plt.show()

def plot_down(pdfs, name, Q2):
    for name, pdf in zip(name, pdfs):
        x = np.linspace(pdf.xMin, 1, 10000)
        plt.plot(x, [pdf.xfxQ2(1, i, Q2) for i in x], label=name)
    #plt.plot(x, [pdf31sx.xfxQ2(2001, i, Q2) for i in x], label='F2')
    plt.legend()
    plt.ylim((-2, 2))
    plt.show()

def plot_strange(pdfs, name, Q2):
    for name, pdf in zip(name, pdfs):
        x = np.linspace(pdf.xMin, 1, 10000)
        plt.plot(x, [pdf.xfxQ2(3, i, Q2) for i in x], label=name)
    #plt.plot(x, [pdf31sx.xfxQ2(2001, i, Q2) for i in x], label='F2')
    plt.legend()
    plt.ylim((-2, 2))
    plt.show()

def plot_charm(pdfs, name, Q2):
    for name, pdf in zip(name, pdfs):
        x = np.linspace(pdf.xMin, 1, 10000)
        plt.plot(x, [pdf.xfxQ2(-4, i, Q2) for i in x], label=name)
    #plt.plot(x, [pdf31sx.xfxQ2(2001, i, Q2) for i in x], label='F2')
    plt.legend()
    plt.ylim((-2, 2))
    plt.show()


def plot_struc():
    x = np.linspace(pdf31sx.xMin, 1, 10000)
    Q2 = 10
    plt.plot(x, [pdf31sx.xfxQ2(2001, i, Q2) for i in x], label='F2')
    plt.plot(x, [pdf31sx.xfxQ2(2003, i, Q2) for i in x], label='xF3')
    #plt.plot(x, [pdf31sx.xfxQ2(2002, i, Q2) for i in x], label='FL')

    F2 = lambda x, q2: 2*(pdf31.xfxQ2(1, x, q2) + pdf31.xfxQ2(2, x, q2) + pdf31.xfxQ2(-1, x, q2) + pdf31.xfxQ2(-2, x, q2) + 2*pdf31.xfxQ2(3, x, q2) + 2*pdf31.xfxQ2(-4, x, q2))
    xF3 =lambda x, q2: 2*((pdf31.xfxQ2(2, x, q2) - pdf31.xfxQ2(-2, x, q2)) + (pdf31.xfxQ2(1, x, q2) - pdf31.xfxQ2(-1, x, q2)) + 2*pdf31.xfxQ2(3, x, q2) - 2*pdf31.xfxQ2(-4, x, q2))

    plt.plot(x, [F2(i, Q2) for i in x], label='F2 - partons')
    plt.plot(x, [xF3(i, Q2) for i in x], label='xF3 - partons')


    plt.legend()
    plt.ylim((-2, 2))
    plt.show()

def diff_234():
    pdfs = [pdf21, pdf31, pdf40]
    Q2 = 1000
    q2max = 100000000.0
    #q2max = 1e10

    xmin = 1e-7
    x = np.linspace(xmin, 1, 10000)
    q2_space = [np.exp(i) for i in np.linspace(0, np.log(q2max), 100)]

    MSE23 = []
    MSE34 = []
    MSE24 = []

    for q2 in tqdm(q2_space):
        xfs = []
        for pdf in pdfs:
            assert q2 <= pdf.q2Max + 1, f'Q2: {q2} but Q2 max: {pdf.q2Max}'
            xfs.append(np.array([pdf.xfxQ2(-2, i, q2) for i in x]))
        #    plt.plot(x, xfs[-1])
        #plt.show()

        MSE23.append(np.mean(np.square(xfs[0] - xfs[1])))
        MSE34.append(np.mean(np.square(xfs[1] - xfs[2])))
        MSE24.append(np.mean(np.square(xfs[0] - xfs[2])))

    plt.plot(q2_space, MSE23, linestyle='-', marker='.', label='23')
    plt.plot(q2_space, MSE34, linestyle='-', marker='.', label='34')
    plt.plot(q2_space, MSE24, linestyle='-', marker='.', label='24')
    plt.xscale('log')
    plt.legend()
    plt.show()

def diff_34():
    pdfs = [pdf31, pdf40]
    q2max = 1e10

    xmin = 1e-9
    x = np.linspace(xmin, 1, 10000)
    q2_space = [np.exp(i) for i in np.linspace(0, np.log(q2max), 100)]

    MSE34 = []

    for q2 in tqdm(q2_space):
        xfs = []
        for pdf in pdfs:
            assert q2 <= pdf.q2Max + 1, f'Q2: {q2} but Q2 max: {pdf.q2Max}'
            xfs.append(np.array([pdf.xfxQ2(-2, i, q2) for i in x]))
        #    plt.plot(x, xfs[-1])
        #plt.show()

        MSE34.append(np.mean(np.square(xfs[0] - xfs[1])))

    plt.plot(q2_space, MSE34, linestyle='-', marker='.', label='34')
    plt.xscale('log')
    plt.legend()
    plt.show()
    
#plot_up([pdf21, pdf31, pdf40], ['21', '31','40'])
#plot_struc()
#print(pdf31sx.xfxQ(2001, 0.1, 10))
#print(pdf31sx.xfxQ2(2001, 0.1, 100))
diff_234()
diff_34()

if False:
    Q2 = 1000
    plot_up([pdf40, pdf31, pdf21], ['PDF4.0', 'PDF3.1', 'PDF2.1'], Q2)
    plot_down([pdf40, pdf31, pdf21], ['PDF4.0', 'PDF3.1', 'PDF2.1'], Q2)
    plot_strange([pdf40, pdf31, pdf21], ['PDF4.0', 'PDF3.1', 'PDF2.1'], Q2)
    plot_charm([pdf40, pdf31, pdf21], ['PDF4.0', 'PDF3.1', 'PDF2.1'], Q2)