
import numpy as np
import matplotlib.pyplot as plt

x       = np.linspace(-30,30,201,endpoint=True)

plt.figure()
y_se    = np.exp(-(x/5)**2)
y_se2   = np.exp(-((x+2)/5)**2)
plt.plot(x, y_se, 'k--', linewidth=3.0)
plt.plot(x, y_se2, 'k', linewidth=3.0)
plt.box(on=None)
plt.axis('off')
plt.xlim(-15,15)
plt.savefig('Line_shift.png')
plt.show()

plt.figure()
y_se    = np.exp(-((x)/5)**2)
y_se2   = np.exp(-((x+2)/5)**2)
plt.plot(x+(y_se/max(y_se)*2-1), y_se, 'k--', linewidth=3.0)
plt.plot(x, y_se2, 'k', linewidth=3.0)
plt.box(on=None)
plt.xlim(-15,15)
plt.axis('off')
plt.savefig('Line_deform.png')
plt.show()
