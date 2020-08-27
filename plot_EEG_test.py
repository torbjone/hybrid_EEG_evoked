import os
from os.path import join
import sys
import numpy as np
import matplotlib.pyplot as plt

sim_folder = "evoked_cdm"


populations = [f for f in os.listdir(join(sim_folder, "cdm"))
               if os.path.isdir(join(sim_folder, "cdm", f))]
sub_pop_groups_dict = {"L5E": ["p5(L56)", "p5(L23)"],
                       "L4E": ["p4", "ss4(L4)", "ss4(L23)"],
                       "L6E": ["p6(L4)", "p6(L56)"],
                       "L23E": ["p23"],
                       "L5I": ["b5", "nb5"],
                       "L4I": ["b4", "nb4"],
                       "L6I": ["b6", "nb6"],
                       "L23I": ["b23", "nb23"],
                       }
print(populations)
summed_eeg = np.load(join(sim_folder, "summed_EEG.npy"))#[500:]
simple_eeg = np.load(join(sim_folder, "simple_EEG.npy"))#[500:]

summed_eeg -= np.average(summed_eeg)
simple_eeg -= np.average(simple_eeg)

fig = plt.figure()
ax1 = fig.add_subplot(211, xlim=[875, 950], ylim=[-0.7, 0.7])
ax2 = fig.add_subplot(212, xlim=[875, 950])

pop_sum = []

for pop in sub_pop_groups_dict.keys():
    pop_eeg = np.load(join(sim_folder, "EEG_{}.npy".format(pop)))#[500:]
    pop_eeg -= np.average(pop_eeg)
    pop_sum.append(pop_eeg)
    ax1.plot(pop_eeg, c='gray')

pop_sum = np.sum(pop_sum, axis=0)
ax1.plot(summed_eeg, 'k', lw=2)
ax1.plot(simple_eeg, 'r', ls=':', lw=2)

rel_diff = (summed_eeg - simple_eeg) / np.max(np.abs(summed_eeg[850:950]))


ax2.plot(rel_diff)
plt.savefig("EEG_plot_test.png")
plt.show()