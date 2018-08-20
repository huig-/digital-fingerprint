import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0, 2*np.pi, 0.001)
s1 = np.sin(t) 
#idxs = np.arange(0, len(t), len(t)/50)
#idxs = [0] + idxs + [t[-1]]
#idxs = [int(x) for x in idxs]
idxs = np.arange(0, 2*np.pi, np.pi/6)
idxs = [x + np.pi/12 for x in idxs]

fig, axs = plt.subplots(2,1)
axs[0].plot(t, s1)
for x,y in zip(idxs, np.sin(idxs)):
    if y > 0:
        axs[0].arrow(x,0,0,y-0.1, color='k', head_width=0.1, head_length=0.1)
    else:
        axs[0].arrow(x,0,0,y+0.1, color='k', head_width=0.1, head_length=0.1)

axs[0].axhline(y = 0, color='k')
axs[0].axvline(x = 0, color='k')
axs[1].axhline(y = 0, color='k')
axs[1].axvline(x = 0, color='k')


s2 = np.sin(20*t)
axs[1].plot(t,s2)
axs[1].plot(t, np.sin(5*t), linestyle='--')

axs[0].set_xlabel('Muestro de una señal de baja frecuencia')
axs[1].set_xlabel('Muestreo de una señal de alta frecuencia')
axs[0].axis('off')
axs[1].axis('off')
fig.tight_layout()
plt.show()
