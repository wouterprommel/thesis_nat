import lhapdf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import path

pdf21 = lhapdf.mkPDF("NNPDF21_lo_as_0119_100")
pdf31 = lhapdf.mkPDF("NNPDF31_lo_as_0118")
pdf40 = lhapdf.mkPDF("NNPDF40_lo_as_01180")
#pdf31sx = lhapdf.mkPDF("NNPDF31sx_nnlonllx_as_0118_LHCb_nf_6_SF")

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


def plot_quarks(pdfs, name, Q2):
    fig, axis = plt.subplots(2, 3, figsize=(12, 6))
    for name, pdf in zip(name, pdfs):
        x = np.linspace(pdf.xMin, 1, 10000)
        axis[0, 0].plot(x, [pdf.xfxQ2(2, i, Q2) for i in x], label='up ' + name)
        axis[1, 0].plot(x, [pdf.xfxQ2(-2, i, Q2) for i in x], label='anti-up ' + name)
        axis[0, 1].plot(x, [pdf.xfxQ2(1, i, Q2) for i in x], label='down ' + name)
        axis[1, 1].plot(x, [pdf.xfxQ2(-1, i, Q2) for i in x], label='anti-down ' + name)
        axis[0, 2].plot(x, [pdf.xfxQ2(3, i, Q2) for i in x], label='strange' + name)
        axis[1, 2].plot(x, [pdf.xfxQ2(-4, i, Q2) for i in x], label='anti-charm' + name)
    #plt.plot(x, [pdf31sx.xfxQ2(2001, i, Q2) for i in x], label='F2')
    axis[0, 0].legend()
    axis[1, 0].legend()
    axis[1, 1].legend()
    axis[0, 1].legend()
    axis[1, 2].legend()
    axis[0, 2].legend()
    axis[0, 0].set_ylim((-.5, 1))
    axis[1, 0].set_ylim((-.5, 1))
    axis[1, 1].set_ylim((-.5, 1))
    axis[0, 1].set_ylim((-.5, 1))
    axis[1, 2].set_ylim((-.5, 1))
    axis[0, 2].set_ylim((-.5, 1))
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

    #df['up'] = N*[0.0]

def plot_quarks_MSE(q2_space, results):
    fig, axis = plt.subplots(4, 2, figsize=(14, 14))
    all = False
    for (quark, dic), ax in zip(results.items(), axis.reshape(-1)):
        if all:
            for name, res in dic.items():
                ax.plot(q2_space, res, label=name)
        else:
            ax.plot(q2_space, dic['MSE34'], label='MSE34')

        ax.set_title(quark)
        ax.set_xscale('log')
            #axis[1, 0].plot(q2_space, 
            #axis[0, 1].plot(q2_space, 
            #axis[1, 1].plot(q2_space, 
            #axis[0, 2].plot(q2_space, 
            #axis[1, 2].plot(q2_space, 

    if all:
        plt.savefig(path.fig_path() + "diff234.pdf", format="pdf", bbox_inches="tight")
    else:
        plt.savefig(path.fig_path() + "diff34_x9.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def diff_234():
    pdfs = [pdf21, pdf31, pdf40]
    Q2 = 1000
    q2max = 8
    q2max = 10

    xmin = 1e-7
    x = np.linspace(xmin, 1, 10000)
    #q2_space = [np.exp(i) for i in np.linspace(0, np.log(q2max), 100)]
    q2_space = np.logspace(0, q2max, 1000, endpoint=False)


    quarks = {'up':2, 'anti-up':-2, 'down':1, 'anti-down':-1, 'strange':3, 'anti-strange':-3, 'charm':4, 'anti-charm':-4}
    quark_results = {}
    for quark, id in quarks.items():
        MSE23 = []
        MSE34 = []
        MSE24 = []
        print(quark, id)
        for q2 in tqdm(q2_space):
            xfs = []
            for pdf in pdfs:
                #assert q2 <= pdf.q2Max + 1, f'Q2: {q2} but Q2 max: {pdf.q2Max}'
                xfs.append(np.array([pdf.xfxQ2(id, i, q2) for i in x]))
        #    plt.plot(x, xfs[-1])
        #plt.show()

            MSE23.append(np.mean(np.square(xfs[0] - xfs[1])))
            MSE34.append(np.mean(np.square(xfs[1] - xfs[2])))
            MSE24.append(np.mean(np.square(xfs[0] - xfs[2])))
        quark_results[quark] = {'MSE23': MSE23, 'MSE34':MSE34, 'MSE24': MSE24}

    plot_quarks_MSE(q2_space, quark_results)
    if False:
        plt.plot(q2_space, MSE23, linestyle='-', marker='.', label='23')
        plt.plot(q2_space, MSE34, linestyle='-', marker='.', label='34')
        plt.plot(q2_space, MSE24, linestyle='-', marker='.', label='24')
        plt.xscale('log')
        plt.legend()
        plt.savefig(f"Figs/diff234.pdf", format="pdf", bbox_inches="tight")
        plt.show()

def diff_34():
    pdfs = [pdf31, pdf40]
    q2max = 1e10

    xmin = 1e-7
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
    plt.savefig(f"Figs/diff34.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
#plot_up([pdf21, pdf31, pdf40], ['21', '31','40'])
#plot_struc()
#print(pdf31sx.xfxQ(2001, 0.1, 10))
#print(pdf31sx.xfxQ2(2001, 0.1, 100))
#diff_234()
#diff_34()

diff_234()

Q2 = 1.6e4 # so hugh dips at x>>0 stops at (higher)>1.5e4 Q2 energies !
#plot_quarks([pdf21, pdf31, pdf40], ['PDF2.1', 'PDF3.1', 'PDF4.0'], Q2)

if False:
    plot_up([pdf40, pdf31, pdf21], ['PDF4.0', 'PDF3.1', 'PDF2.1'], Q2)
    plot_down([pdf40, pdf31, pdf21], ['PDF4.0', 'PDF3.1', 'PDF2.1'], Q2)
    plot_strange([pdf40, pdf31, pdf21], ['PDF4.0', 'PDF3.1', 'PDF2.1'], Q2)
    plot_charm([pdf40, pdf31, pdf21], ['PDF4.0', 'PDF3.1', 'PDF2.1'], Q2)
