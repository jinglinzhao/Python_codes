
import numpy as np
import matplotlib.pyplot as plt

x       = np.linspace(-30,30,201,endpoint=True)

plt.figure()
y_se    = 1 - np.exp(-((x-2)/5)**2)
y_se2   = 1 - np.exp(-((x)/5)**2)
plt.plot(x-(y_se/max(y_se)*2-1), y_se, 'r--', linewidth=3.0)
plt.plot(-x+(y_se/max(y_se)*2-1), y_se, 'r--', linewidth=3.0)
plt.plot(x, y_se2, 'k', linewidth=3.0)
plt.text(0, -0.1, 'A', fontsize=16, horizontalalignment='center')
plt.text(-3, -0.1, 'b', fontsize=16, horizontalalignment='center', color='red')
plt.text(3, -0.1, 'd', fontsize=16, horizontalalignment='center', color='red')
plt.box(on=None)
plt.xlim(-15,15)
plt.axis('off')
plt.savefig('other_indicators_suck.png')
plt.show()
