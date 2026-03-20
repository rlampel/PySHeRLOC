import matplotlib.pyplot as plt
import numpy as np


avg_auto_lift = [
    12.4,
    30.6,
    17.0,
    17.0,
    15.8,
    4.0,
    6.0,
    4.0,
    10.0,
    36.2,
    19.2,
    19.4,
    6.0,
    24.4,
    43.8,
    40.6,
    9.6,
    19.2,
    19.0,
    5.0,
    9.6,
    52.2,
    33.4,
    34.0,
    18.6,
    22.4,
    15.2,
    21.4,
    6.2,
    11.2,
    25.0,
    6.0,
    34.75,  # [50, 25, 42, 22, inf],
    61.8,
    6.4,
    14.0,
    21.0,
    19.0,
    19.2,
    35.8,
    37.0,  # OEDs
    63.4,
    22.8,
    46.6
]
avg_default_lift = [
    12.8,
    30.4,
    17.4,
    17.8,
    15.8,
    4.0,
    6.0,
    4.0,
    10.2,
    41.6,
    20.8,
    19.0,
    6.0,
    24.4,
    163.2,
    93.2,
    9.6,
    19.6,
    18.2,
    5.0,
    9.8,
    50.2,
    39,  # [inf, 57, 35, 28, 36],
    37.4,
    18.8,
    22.4,
    15.6,
    16.2,
    6.2,
    10.6,
    24.0,
    6.0,
    32.75,  # [39, 25, 35, inf, 32],
    67.8,
    6.4,
    14.0,
    19.8,
    19.0,
    20.2,
    35.8,
    130.4,  # [120, 200, 83, 118, 131], # OEDs
    73.6,
    25.8,
    71.2
]
avg_single_lift = [
    24.8,
    24.8,
    44.8,
    51.0,
    14.6,
    2.0,
    2.0,
    2.0,
    4.0,
    28.0,
    22.6,
    19.8,
    9.0,
    np.inf,
    33.2,
    34.0,
    np.inf,
    np.inf,
    32.25,  # [inf, 46, 27, 29, 27],
    41.2,
    15.0,
    np.inf,
    91.4,
    99.0,
    42.0,
    40.6,
    51.0,
    48.8,
    23.0,
    23.6,
    22.25,  # [22, inf, 21, 28, 18],
    np.inf,
    np.inf,
    11.6,
    16.4,
    14.8,
    76.6,
    np.inf,
    np.inf,
    57.0,
    35.4,  # OEDs
    34.0,
    31.0,
    28.0
]

avg_fsinit_heur = [
    12.6,
    30.2,
    16.4,
    19.0,
    14.8,
    4.0,
    6.0,
    4.0,
    10.2,
    39.0,
    20.0,
    18.0,
    6.0,
    25.0,
    100.2,
    84.8,
    10.0,
    13.0,
    18.4,
    5.0,
    8.4,
    46.6,
    40.0,
    33.8,
    17.6,
    22.0,
    15.4,
    16.8,
    5.6,
    11.2,
    24.4,
    6.0,
    34.4,
    65.0,
    6.0,
    14.0,
    20.6,
    22.4,
    18.0,
    28.0,
    109.6,
    73.2,
    20.0,
    70.4
]


# Exact Hessian
default_lift_ex = [
    6.6,
    17.6,
    6.0,
    10.0,
    8.6,
    4.0,
    7.0,
    4.0,
    8.0,
    8.2,
    14.8,
    10.4,
    4.0,
    11.4,
    140.4,  # [200, 200, 36, 200, 66],
    38.0,
    5.0,
    19.0,  # [12, 23, 24, inf, 17],
    10.0,
    4.0,
    4.0,
    22.6,
    20.2,
    21.4,
    10.0,
    11.4,
    6.0,
    7.4,
    4.0,
    9.2,
    16.8,
    4.0,
    8.2,
    41.6,
    5.0,
    10.8,
    8.0,
    16.8,
    16.4,
    33.0,
    27.4,  # OEDs
    64.2,
    9.0,
    32.8,
]

single_lift_ex = [
    15.0,
    15.2,
    7.0,
    7.0,
    8.2,
    2.0,
    2.0,
    2.0,
    4.0,
    7.4,
    15.4,
    8.5,  # [9, 8, 8, 200, 9],
    6.0,
    np.inf,
    32.0,
    31.6,
    6.0,
    np.inf,
    16.0,
    4.0,
    4.0,
    np.inf,
    26.4,
    28.4,
    10.0,
    7.8,
    8.0,
    7.0,
    4.0,
    4.0,
    14.4,
    4.0,
    5.0,
    7.4,
    8.0,
    10.6,
    8.4,
    11.0,
    11.2,
    40.4,
    17.0,  # OEDs
    15.0,
    11.0,
    12.0
]

auto_lift_ex = [
    6.6,
    17.6,
    6.0,
    10.0,
    8.6,
    4.0,
    7.0,
    4.0,
    8.0,
    9.2,
    14.8,
    10.8,
    4.0,
    11.4,
    16.4,
    16.8,
    5.0,
    18.25,  # [12, 24, 25, inf, 12],
    10.0,
    6.4,
    4.0,
    17.0,
    20.2,
    22.2,
    10.0,
    11.4,
    6.0,
    7.8,
    4.0,
    9.2,
    17.6,
    4.0,
    8.2,
    33.0,
    5.0,
    11.4,
    8.0,
    16.6,
    17.4,
    32.2,   # Quadrotor -> has to be replaced
    24.8,  # OEDs
    55.8,
    11.0,
    26.6
]

fsinit_heur_ex = [
    7.0,
    17.6,
    6.0,
    9.8,
    9.6,
    4.0,
    7.0,
    4.0,
    8.0,
    8.2,
    15.0,
    11.2,
    4.0,
    10.6,
    104.4,
    36.6,
    5.0,
    7.4,
    12.4,
    4.0,
    4.0,
    17.0,  # [22, 16, inf, 14, 16],
    18.8,
    19.8,
    9.6,
    11.8,
    6.0,
    7.4,
    4.0,
    8.8,
    16.4,
    4.0,
    8.2,
    40.4,
    5.0,
    10.8,
    8.0,
    15.8,
    15.8,
    32.2,
    25.0,  # OEDs
    51.0,
    9.0,
    44.0,
]

# rework front
comb_lift = [
    9, 30, 17, 21, 21, 4, 6, 4, 10, 25, 13, 17, 6, 24, 32, 29, 10, 26, 19, 5, 9, 42,
    28, 30, 19, 22, 15, 19, 7, 11, 25, 6, 17, 32, 6, 15, 33, 23, 23, 18,
    # 37,
    # 26.8 [25, 23, 31, 23, 32],
    # 35.2 [24, 34, 34, 43, 41],
    # 37.4 [44, 44, 34, 25, 40],
    # 46.6 [48, 45, 51, 43, 46],
    # 51.0 [65, 70, 33, inf, 36],
    # 39.4 [40, 28, 53, 42, 34],
    # 73.2 [64, 79, 63, 76, 84],
    36.0,  # OEDs
    78.0,
    27.0,
    45.2,
]


comb_lift_ex = [
    7, 16, 6, 10, 24, 4, 7, 4, 8, 6, 14, 11, 4, 13, 17, 18, 5, 30, 10, 7,
    7, 15, 15, 14, 10, 12, 6, 7, 4, 8, 16, 4, 8, 20, 5, 11, 24, 8, 18, 19, 35,
    # 8.6 [8, 8, 8, 10, 9],
    # 15.6 [17, 14, 16, 15, 16],
    # 17.0 [17, 19, 17, 16, 16],
    # inf [22, 16, inf, 14, 16],
    # 19.4 [18, 24, 18, 17, 20],
    # 22.0 [21, 20, 25, 21, 23],
    # 34.4 [35, 40, 39, 32, 26],
    21.0,  # OEDs
    56.0,
    11.0,
    38.0,
]

# Settings
bar_height = 0.8
colors = {4: 'red', 3: 'orange', 2: 'yellow', 1: 'yellowgreen', 0: 'green', -1: 'black'}
labels = ["Single", "Multiple", "Adaptive"]

data = []

problems = [
    "Van der Pol",
    "Van der Pol Mayer",
    "Lotka Volterra",
    "Lotka Volterra Mayer",
    "Batch Reactor",
    "Bioreactor",
    "Bioreactor Mayer",
    "Bryson Denham",
    "Bryson Denham Mayer",
    "Catalyst Mixing",
    "Cushioned Oscillation Mayer",
    "Dielectrophoretic Particle",
    "Double Oscillator",
    "Ducted Fan",
    "Egerstedt",
    "Egerstedt Mayer",
    "Electric Car",
    "F8 Aircraft",
    "F8 Aircraft Mayer",
    "Fuller",
    "Fuller Mayer",
    "Hang Glider",
    "Hanging Chain Lagrange",
    "Hanging Chain",
    "Lotka Competitive",
    "Lotka Competitive Mayer",
    "Lotka Shared",
    "Lotka Shared Mayer",
    "LQR",
    "LQR Mayer",
    "Mountain Car",
    "Ocean",
    "Ocean Mayer",
    "Particle Steering Mayer",
    "Rao Mease",
    "Rao Mease Mayer",
    "Three Tank",
    "Tubular Reactor",
    "Tubular Reactor Mayer",
    "Quadrotor",
    "Lotka OED",
    "Dielectr Particle",
    "Jackson OED",
    "Van der Pol OED"
]

for j in range(len(problems)):
    # data += [avg_single_lift[j], avg_default_lift[j], avg_auto_lift[j]]
    data += [single_lift_ex[j], default_lift_ex[j], auto_lift_ex[j]]

data = np.reshape(data, (-1, 3))

# print all problems
for m in range(len(problems)):
    out = problems[m]
    ref = data[m][1]  # scale to default multiple shooting
    for el in data[m]:
        if np.isinf(el):
            out += ",inf"
        else:
            out += "," + str(int(el * 100 / ref) / 100)
    print(out)

# comp_label = "Custom Init"
# comp_label = "Auto Lifting"
comp_label = "FSInit"

N_means = {"Multiple": [x[0] for x in data],
           "Single": [x[1] for x in data],
           "Alg.": [x[2] for x in data],
           }
colors = ['red', 'orange', 'yellowgreen', 'green', 'black']

x = np.arange(len(problems))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
ax.bar(x - width, N_means["Multiple"], width, label=labels[0], color=colors[0])
ax.bar(x, N_means["Single"], width, label=labels[1], color=colors[1])
ax.bar(x + width, N_means["Alg."], width, label=labels[2], color=colors[2])

# ax.set_ylabel("Iterations (N)", fontsize='large')
ax.set_ylabel("Seconds (s)", fontsize='large')
ax.set_xticks(x)
ax.set_xticklabels(problems, rotation=45, ha="right", fontsize=11.0)
ax.legend(fontsize='large', loc='upper left')
plt.tight_layout()
plt.show()

table_data = []
for k in range(len(problems)):
    table_data += [avg_single_lift[k], avg_default_lift[k]]
    table_data += [avg_fsinit_heur[k], avg_auto_lift[k]]
    table_data += [single_lift_ex[k], default_lift_ex[k]]
    table_data += [fsinit_heur_ex[k], auto_lift_ex[k]]

table_data = np.reshape(table_data, (-1, 8))

# print for table output
for m in range(len(problems)):
    out = problems[m]
    col_prefix = "& \\multicolumn{1}{l|}{"
    col_suffix = "} "
    for n in range(table_data.shape[1]):
        el = table_data[m][n]
        out += col_prefix

        if np.isinf(el):
            out += "inf"
        else:
            out += str(el)

        out += col_suffix
    out += "\\\\ \\hline"
    print(out)
