import numpy as np
import matplotlib.pyplot as plt
import pickle
import path

def open_pickle(name):
    print(f'Opening file: {name}')
    file = open(name + '.pickle', 'rb')
    obj = pickle.load(file)
    file.close()
    return obj

plt.style.use('seaborn-v0_8-colorblind')
#plt.style.use('Solarize_Light2')
plt.rcParams['text.usetex'] = True

lo, nlo, X, Q2 = open_pickle('compare_lo_nlo')
#print(lo)
E = 200
lo = np.array(lo[:E])
nlo = np.array(nlo[:E])
X = np.array(X[:E])
Q2 = np.array(Q2[:E])
abs_diff = np.abs(lo - nlo)
#plt.plot(X, abs_diff)
print(f'mean abs diff: {np.mean(abs_diff)}')

#x = np.arange(0, len(lo))
plt.scatter(X, lo)
plt.scatter(X, nlo)
plt.xscale('log')
plt.ylabel(r'$\frac{d\sigma(x, Q^2)}{dxdQ^2}$') #evaluation of 
plt.xlabel(r'$x$')
#plt.savefig(path.fig_path() + "pdf31_point_sample_comparison_x9.pdf", format="pdf", bbox_inches="tight")
plt.show()

#for i, j in zip(lo, nlo):
    