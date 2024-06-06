import numpy as np
import matplotlib.pyplot as plt
import pickle

def open_pickle(name):
    print(f'Opening file: {name}')
    file = open(name + '.pickle', 'rb')
    obj = pickle.load(file)
    file.close()
    return obj

lo, nlo = open_pickle('compare_lo_nlo')
#print(lo)
lo = np.array(lo)
nlo = np.array(nlo)
abs_diff = np.abs(lo - nlo)

print(f'mean abs diff: {np.mean(abs_diff)}')

x = np.arange(0, len(lo))
plt.scatter(x, lo)
plt.scatter(x, nlo)
plt.show()

#for i, j in zip(lo, nlo):
    