import numpy as np
import matplotlib.pyplot as plt
import lhapdf


pdf = lhapdf.mkPDF("NNPDF31_lo_as_0118")

q2min = pdf.q2Min
q2max = pdf.q2Max

#r = np.arange(q2min, q2max, 1000)
#lr = np.arange(np.log(q2min), np.log(q2max), 1000)
r = np.linspace(q2min, q2max, 1000)
lr = np.linspace(np.log(q2min), np.log(q2max), 1000)

print(np.log(q2min), np.log(q2max))
plt.plot(r)
plt.plot(np.exp(lr))
plt.yscale('log')
plt.show()