import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 0.1, 0.001)
s1 = np.sin(2*np.pi*30*t) + np.sin(2*np.pi*60*t)
idxs = np.arange(0, len(t), len(t)/50)
idxs = [0] + idxs + [t[-1]]
idxs = [int(x) for x in idxs]

fix, axs = plt.subplots(2,1)
axs[0].plot(t, s1)
axs[0].set_xlabel('Señal con continuo dominio')
axs[1].set_xlabel('Muestreo de la señal')
axs[1].scatter(t[idxs], s1[idxs], s=7)
axs[0].xaxis.set_major_locator(plt.NullLocator())
axs[0].xaxis.set_major_locator(plt.NullLocator())
axs[1].xaxis.set_major_locator(plt.NullLocator())
axs[1].xaxis.set_major_locator(plt.NullLocator())
axs[0].yaxis.set_major_locator(plt.NullLocator())
axs[0].yaxis.set_major_locator(plt.NullLocator())
axs[1].yaxis.set_major_locator(plt.NullLocator())
axs[1].yaxis.set_major_locator(plt.NullLocator())
plt.show()
