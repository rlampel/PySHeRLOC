import numpy as np
import matplotlib.pyplot as plt
import os


dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "benchmark.log")

meas = np.loadtxt(filename, delimiter=",")

# iterate over array until entry is [0, 0]
num_entries = meas.shape[0]
start_ind = 0
line_styles = ["-", ":", "--", "-."]

for curr_ind in range(num_entries):
    if meas[curr_ind, 0] == 0 and meas[curr_ind, 1] == 0 or curr_ind == num_entries - 1:
        plt.scatter(meas[start_ind:curr_ind, 1],
                    meas[start_ind:curr_ind, 0])
        plt.plot(meas[start_ind:curr_ind, 1],
                 meas[start_ind:curr_ind, 0],
                 linestyle=line_styles[start_ind % 4])
        plt.xlabel("h")
        plt.ylabel("f")
        start_ind = curr_ind + 1

plt.show()
