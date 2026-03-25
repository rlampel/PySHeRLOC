import os
import numpy as np


def load_data(path):
    data = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip().rstrip(",")
            parts = [p.strip() for p in line.split(",") if p.strip()]

            name = parts[0]
            values = [float(x) for x in parts[1:]]
            data[name] = values

    return data


def print_array(arr):
    out = ""
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            x = 2**col if col > 0 else 0
            out += f"({x}, {str(arr[row, col])}) "
        out += "\n"
    print(out)


if __name__ == "__main__":
    # read list of problems
    dirname = os.path.dirname(__file__)
    output_file = os.path.join(dirname, "tex_output.txt")

    excluded = [
        "Catalyst Mixing",
        "Three Tank OED"
    ]

    ex_files = [
        "exact_iters.log",
        "fsinit_exact_iters.log",
        "condense_exact_iters.log",
        "fsinit_condense_exact_iters.log"
    ]

    qn_files = [
        "quasi_newton_iters.log",
        "fsinit_quasi_newton_iters.log",
        "condense_quasi_newton_iters.log",
        "fsinit_condense_quasi_newton_iters.log"
    ]

    x_labels = ["exact", "Quasi-Newton"]
    style_names = ["base", "fs", "cond", "comb"]

    num_datasets = len(ex_files)

    ex_datasets = [load_data(os.path.join(dirname, ex_files[j])) for j in range(num_datasets)]
    qn_datasets = [load_data(os.path.join(dirname, qn_files[j])) for j in range(num_datasets)]
    problem_names = list(ex_datasets[0].keys())
    # plot_rows(data)

    counter = 0
    default_iters_ex = 0

    ex_iters = np.array([0] * 4)
    qn_iters = np.array([0] * 4)
    for i in range(len(problem_names)):
        curr_name = problem_names[i]

        if curr_name not in excluded:

            for m in range(len(ex_datasets)):
                ex_data = ex_datasets[m]
                qn_data = qn_datasets[m]
                curr_ex_data = ex_data[curr_name]
                curr_qn_data = qn_data[curr_name]

                ex_iters[m] += curr_ex_data[0]
                qn_iters[m] += curr_qn_data[0]

            counter += 1

    print(ex_iters / counter, qn_iters / counter)

