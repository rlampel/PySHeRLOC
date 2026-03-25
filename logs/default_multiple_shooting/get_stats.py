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
    output_file = os.path.join(dirname, "tex_time_output.txt")

    excluded = [
        "Catalyst Mixing"
    ]

    files = [
        "IPOPT_exact_times.log",
        "IPOPT_quasi_newton_times.log",
        "blockSQP2_exact_times.log",
        "blockSQP2_quasi_newton_times.log"
    ]

    x_labels = [0, 2, 4, 8, 16, 32, 64]
    style_names = ["ipopt_ex", "ipopt_qu", "blocksqp_ex", "blocksqp_qu"]
    num_conv = np.zeros((len(style_names), len(x_labels)))
    avg_time = num_conv.copy()

    datasets = [load_data(os.path.join(dirname, files[j])) for j in range(4)]
    problem_names = list(datasets[0].keys())

    output = ""
    counter = 0
    # iterate over all problems
    for i in range(len(problem_names)):
        curr_name = problem_names[i]

        if curr_name not in excluded:

            # iterate over all datasets
            for m in range(len(datasets)):
                data = datasets[m]
                # get results for current problem and dataset
                curr_data = data[curr_name]
                conv_all = False

                # make sure that it converged for all discretizations
                if sum(curr_data) != np.inf:
                    conv_all = True
                    counter += 1

                for k in range(len(curr_data)):
                    if curr_data[k] != np.inf:
                        num_conv[m, k] += 1
                    if conv_all:
                        avg_time[m, k] += curr_data[k]

    print_array(num_conv)
    print_array(avg_time / counter)
