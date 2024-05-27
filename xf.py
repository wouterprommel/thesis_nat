import lhapdf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import path
import pickle

plt.style.use('seaborn-v0_8-colorblind')
#plt.style.use('Solarize_Light2')
plt.rcParams['text.usetex'] = True

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
    fig, axis = plt.subplots(2, 4, figsize=(12, 6))
    for name, pdf in zip(name, pdfs):
        #x = np.linspace(1e-3, 1, 10000)
        x = np.logspace(-3, 1, 10000, endpoint=False)
        ba = x < 1.0
        x = x[ba]
        #print(x)
        axis[0, 0].plot(x, [pdf.xfxQ2(2, i, Q2) for i in x], label='up ' + name)
        axis[1, 0].plot(x, [pdf.xfxQ2(-2, i, Q2) for i in x], label='anti-up ' + name)
        axis[0, 1].plot(x, [pdf.xfxQ2(1, i, Q2) for i in x], label='down ' + name)
        axis[1, 1].plot(x, [pdf.xfxQ2(-1, i, Q2) for i in x], label='anti-down ' + name)
        axis[0, 2].plot(x, [pdf.xfxQ2(3, i, Q2) for i in x], label='strange ' + name)
        axis[1, 2].plot(x, [pdf.xfxQ2(-4, i, Q2) for i in x], label='anti-charm ' + name)
        axis[0, 3].plot(x, [pdf.xfxQ2(21, i, Q2)/10 for i in x], label='gluon/10 ' + name)
    #plt.plot(x, [pdf31sx.xfxQ2(2001, i, Q2) for i in x], label='F2')
    axis[0, 0].legend()
    axis[1, 0].legend()
    axis[1, 1].legend()
    axis[0, 1].legend()
    axis[1, 2].legend()
    axis[0, 2].legend()
    axis[0, 3].legend()
    axis[0, 0].set_ylim((-.5, 1))
    axis[1, 0].set_ylim((-.5, 1))
    axis[1, 1].set_ylim((-.5, 1))
    axis[0, 1].set_ylim((-.5, 1))
    axis[1, 2].set_ylim((-.5, 1))
    axis[0, 2].set_ylim((-.5, 1))
    axis[0, 3].set_ylim((-.5, 1))

    axis[0, 0].set_xscale('log')
    axis[1, 0].set_xscale('log')
    axis[1, 1].set_xscale('log')
    axis[0, 1].set_xscale('log')
    axis[1, 2].set_xscale('log')
    axis[0, 2].set_xscale('log')
    axis[0, 3].set_xscale('log')
    plt.savefig(path.fig_path() + "quarks_Q2.1e8.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def plot_up(pdfs, name, Q2):
    for name, pdf in zip(name, pdfs):
        x = np.linspace(pdf.xMin, 1, 100)
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
    all = True
    use = np.array([q < pdf21.q2Max + 1 for q in q2_space])
    for (quark, dic), ax in zip(results.items(), axis.reshape(-1)):
        if all:
            for name, res in dic.items():
                if name == 'MSE23' or name == 'MSE24':
                    ax.plot(q2_space[use], np.array(res)[use], label=name)
                else:
                    ax.plot(q2_space, res, label=name)
        else:
            ax.plot(q2_space, dic['MSE34'], label='MSE34')

        ax.set_title(quark)
        ax.legend()
        ax.set_xscale('log')
            #axis[1, 0].plot(q2_space, 
            #axis[0, 1].plot(q2_space, 
            #axis[1, 1].plot(q2_space, 
            #axis[0, 2].plot(q2_space, 
            #axis[1, 2].plot(q2_space, 

    if all:
        #plt.savefig(path.fig_path() + "diff234.pdf", format="pdf", bbox_inches="tight")
        pass
    else:
        plt.savefig(path.fig_path() + "diff34_x9.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def diff_234():
    pdfs = [pdf21, pdf31, pdf40]
    Q2 = 1000
    q2max = 8
    q2max = 10

    xmin = 1e-7
    #x = np.linspace(xmin, 1, 10000) # N=10000
    x = np.logspace(-7, 1, 10000, endpoint=False)
    ba = x < 1.0
    x = x[ba]
    #q2_space = [np.exp(i) for i in np.linspace(0, np.log(q2max), 100)] 
    q2_space = np.logspace(4.5, q2max, 1000, endpoint=False) #N= 1000


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

    #return quark_results
    file = open('quark_results.pickle', 'ab')
    #pickle.dump(quark_results, file)
    file.close()
    plot_quarks_MSE(q2_space, quark_results)

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
    
def diff_struc():
    pdfs = [pdf21, pdf31, pdf40]
    Q2 = 1000
    q2max = 8
    q2max = 10

    xmin = 1e-7
    x = np.linspace(xmin, 1, 10000) # N=10000
    #q2_space = [np.exp(i) for i in np.linspace(0, np.log(q2max), 100)] 
    q2_space = np.logspace(4.5, q2max, 100, endpoint=False) #N= 1000
    MSE23 = []
    MSE34 = []
    MSE24 = []
    use = []
    for Q2 in tqdm(q2_space):
        F2_pdf_list = []
        xF3_pdf_list = []
        use.append(Q2 <= pdf21.q2Max + 1)
        for pdf in pdfs:
            F2_pdf_list.append(np.array([pdf.xfxQ2(1, i, Q2) + pdf.xfxQ2(2, i, Q2) + pdf.xfxQ2(-1, i, Q2) + pdf.xfxQ2(-2, i, Q2) + 2*pdf.xfxQ2(-3, i, Q2) + 2*pdf.xfxQ2(4, i, Q2) for i in x]))
            xF3_pdf_list.append(np.array([(pdf.xfxQ2(2, i, Q2) - pdf.xfxQ2(-2, i, Q2)) + (pdf.xfxQ2(1, i, Q2) - pdf.xfxQ2(-1, i, Q2)) - 2*pdf.xfxQ2(-3, i, Q2) + 2*pdf.xfxQ2(4, i, Q2) for i in x]))

        #MSE23.append(np.mean(np.square(F2_pdf_list[0] - F2_pdf_list[1])))
        #MSE34.append(np.mean(np.square(F2_pdf_list[1] - F2_pdf_list[2])))
        #MSE24.append(np.mean(np.square(F2_pdf_list[0] - F2_pdf_list[2])))
        MSE23.append(np.mean(np.square(xF3_pdf_list[0] - xF3_pdf_list[1])))
        MSE34.append(np.mean(np.square(xF3_pdf_list[1] - xF3_pdf_list[2])))
        MSE24.append(np.mean(np.square(xF3_pdf_list[0] - xF3_pdf_list[2])))
    MSE23 = np.array(MSE23)
    MSE34 = np.array(MSE34)
    MSE24 = np.array(MSE24)
    use = np.array(use)
    if True:
        plt.plot(q2_space[use], MSE23[use], linestyle='-', marker='.', label='23')
        plt.plot(q2_space, MSE34, linestyle='-', marker='.', label='34')
        plt.plot(q2_space[use], MSE24[use], linestyle='-', marker='.', label='24')
        plt.xscale('log')
        plt.legend()
        plt.savefig(path.fig_path() + "struc_MSE.pdf", format="pdf", bbox_inches="tight")
        plt.show()


def alphas(pdfs):
    for q in range(10, 1000):
        for pdf in pdfs:
            print(pdf.alphasQ(q))
#plot_up([pdf21, pdf31, pdf40], ['21', '31','40'])
#plot_struc()
#print(pdf31sx.xfxQ(2001, 0.1, 10))
#print(pdf31sx.xfxQ2(2001, 0.1, 100))
#diff_234()
#diff_34()

#quark_results = diff_234()
diff_234()
#diff_struc()
if False:
    ratio_up = np.array(quark_results['up']['MSE23']) / np.array(quark_results['anti-up']['MSE23'])
    ratio_down = np.array(quark_results['down']['MSE23']) / np.array(quark_results['anti-down']['MSE23'])
    ratio_strange = np.array(quark_results['strange']['MSE23']) / np.array(quark_results['anti-strange']['MSE23'])
    ratio_charm = np.array(quark_results['charm']['MSE23']) / np.array(quark_results['anti-charm']['MSE23'])
    plt.plot(ratio_up, label='up')
    plt.plot(ratio_down, label='down')
    plt.plot(ratio_strange, label='strange')
    plt.plot(ratio_charm, label='charm')
    plt.legend()
    plt.show()

Q2 = 1e8 #1.6e4 # so hugh dips at x>>0 stops at (higher)>1.5e4 Q2 energies !
#plot_quarks([pdf21, pdf31, pdf40], ['PDF2.1', 'PDF3.1', 'PDF4.0'], Q2)


#alphas([pdf21, pdf31, pdf40])

if False:
    plot_up([pdf40, pdf31, pdf21], ['PDF4.0', 'PDF3.1', 'PDF2.1'], Q2)
    plot_down([pdf40, pdf31, pdf21], ['PDF4.0', 'PDF3.1', 'PDF2.1'], Q2)
    plot_strange([pdf40, pdf31, pdf21], ['PDF4.0', 'PDF3.1', 'PDF2.1'], Q2)
    plot_charm([pdf40, pdf31, pdf21], ['PDF4.0', 'PDF3.1', 'PDF2.1'], Q2)
