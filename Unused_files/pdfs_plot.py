import matplotlib.pyplot as plt
import lhapdf
import numpy as np

pdf21 = lhapdf.mkPDF("NNPDF21_lo_as_0119_100")
pdf40 = lhapdf.mkPDF("NNPDF40_lo_as_01180")
pdf31 = lhapdf.mkPDF("NNPDF31_lo_as_0118")

x = np.linspace(1e-9, 1, 1000)

plt.set_cmap('cividis')
plt.plot(x, [pdf21.xfxQ2(1, i, 10) for i in x], label='pdf 21 lo')
plt.plot(x, [pdf31.xfxQ2(1, i, 10) for i in x], label='pdf 31 lo')
plt.plot(x, [pdf40.xfxQ2(1, i, 10) for i in x], label='pdf 40 lo')
plt.ylim((-0.2, 1))
plt.legend()
plt.show()