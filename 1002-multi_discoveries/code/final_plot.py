import numpy as np
import matplotlib.pyplot as plt

left  = 0.1  # the left side of the subplots of the figure
right = 0.95    # the right side of the subplots of the figure
bottom = 0.05   # the bottom of the subplots of the figure
top = 0.95      # the top of the subplots of the figure
wspace = 0.4   # the amount of width reserved for blank space between subplots
hspace = 0.3   # the amount of height reserved for white space between subplots

plt.rcParams.update({'font.size': 16})
fig, axes = plt.subplots(figsize=(14, 14))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

# fig.add_subplot(111)    # The big subplot
# plt.axis('off')     # hide frame
# plt.xticks([])                        # don't want to see any ticks on this axis
plt.yticks([])
# plt.xlabel("JD - 2,400,000")
# plt.ylabel("$RV_{HARPS} - RV_{FT,L}$ [m/s]")

# ax1 = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)
color = "#ff7f0e"

t 	= np.loadtxt('../data/HD128621/plot/2009/plot_t.txt')
y 	= np.loadtxt('../data/HD128621/plot/2009/plot_y.txt')
yerr = np.loadtxt('../data/HD128621/plot/2009/plot_yerr.txt')
x 	= np.loadtxt('../data/HD128621/plot/2009/plot_x.txt')
mu 	= np.loadtxt('../data/HD128621/plot/2009/plot_mu.txt')
std = np.loadtxt('../data/HD128621/plot/2009/plot_std.txt')

plt.subplot(311)
plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.1, capsize=0)
plt.plot(x, mu, color=color)
plt.xlim(54872, 54962)
plt.ylim(-3.5, 3.1)
plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.title('Epoch 1: 2009-02-15..2009-05-06')
plt.ylabel(r"$\Delta RV_L$ [m/s]")

t 	= np.loadtxt('../data/HD128621/plot/2010/plot_t.txt')
y 	= np.loadtxt('../data/HD128621/plot/2010/plot_y.txt')
yerr = np.loadtxt('../data/HD128621/plot/2010/plot_yerr.txt')
x 	= np.loadtxt('../data/HD128621/plot/2010/plot_x.txt')
mu 	= np.loadtxt('../data/HD128621/plot/2010/plot_mu.txt')
std = np.loadtxt('../data/HD128621/plot/2010/plot_std.txt')

plt.subplot(312)
plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.1, capsize=0)
plt.plot(x, mu, color=color)
plt.xlim(55273, 55363)
plt.ylim(-3.5, 3.1)
plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.title('Epoch 2: 2010-03-23..2010-06-12')
plt.ylabel(r"$\Delta RV_L$ [m/s]")

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# ###########################
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(x, y1)
# ax1.set_ylabel('y1')

# ax2 = ax1.twinx()
# ax2.plot(x, y2, 'r-')
# ax2.set_ylabel('y2', color='r')
# for tl in ax2.get_yticklabels():
#     tl.set_color('r')
# ###########################

t 	= np.loadtxt('../data/HD128621/plot/2011/plot_t.txt')
y 	= np.loadtxt('../data/HD128621/plot/2011/plot_y.txt')
yerr = np.loadtxt('../data/HD128621/plot/2011/plot_yerr.txt')
x 	= np.loadtxt('../data/HD128621/plot/2011/plot_x.txt')
mu 	= np.loadtxt('../data/HD128621/plot/2011/plot_mu.txt')
std = np.loadtxt('../data/HD128621/plot/2011/plot_std.txt')

plt.subplot(313)
plt.errorbar(t, y, yerr=yerr, fmt=".k", alpha=0.1, capsize=0)
plt.plot(x, mu, color=color)
plt.xlim(55608, 55698)
plt.ylim(-3.5, 3.1)
plt.fill_between(x, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
plt.title('Epoch 3: 2011-02-18..2011-05-15')
plt.xlabel("JD - 2,400,000")
plt.ylabel(r"$\Delta RV_L$ [m/s]")

plt.savefig('final_plot.png')
plt.show()
