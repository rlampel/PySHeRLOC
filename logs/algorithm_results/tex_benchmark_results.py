import matplotlib.pyplot as plt
import os


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


def plot_rows(data):
    x_labels = [0, 2, 4, 8, 16, 32, 64]

    for name, values in data.items():
        plt.figure()
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(values)), x_labels)
        plt.xlabel("Scale")
        plt.ylabel("Value")
        plt.title(name)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # read list of problems
    dirname = os.path.dirname(__file__)
    output_file = os.path.join(dirname, "tex_output.txt")

    excluded = [
        "LQR Mayer",
        "Ocean Mayer",
        "F8 Aircraft Mayer",
        "Particle Steering Mayer",
        "Rao Mease Mayer",
        "Tubular Reactor Mayer"
    ]

    ex_files = [
        "baseline_exact.dat",
        "fsinit_exact_iters.dat",
        "auto_condense_exact_iters.dat",
        "comb_exact_iters.dat"
    ]

    qn_files = [
        "baseline_quasi.dat",
        "fsinit_quasi_newton_iters.dat",
        "auto_condense_quasi_newton_iters.dat",
        "comb_quasi_newton_iters.dat"
    ]

    x_labels = ["exact", "Quasi-Newton"]
    style_names = ["base", "fs", "cond", "comb"]

    num_datasets = len(ex_files)

    ex_datasets = [load_data(os.path.join(dirname, ex_files[j])) for j in range(num_datasets)]
    qn_datasets = [load_data(os.path.join(dirname, qn_files[j])) for j in range(num_datasets)]
    problem_names = list(ex_datasets[0].keys())
    # plot_rows(data)

    output = ""
    counter = 0
    for i in range(len(problem_names)):
        curr_name = problem_names[i]

        if curr_name not in excluded:
            output += "%" + "---" + str(counter + 1) + " " + curr_name + "---\n"
            output += "\\nextgroupplot[title={"
            output += curr_name + "}"

            # add y-labels
            if counter % 5 == 0:
                output += ", ylabel={Rel. Iterations}"

            if counter % 25 == 0:
                output += ",\nlegend columns=-1, % Horizontal legend \n"
                output += "legend entries={base, fs, cond, comb},\n"
                output += "legend style={draw=none, fill=none, font=\\footnotesize,\n"
                output += "column sep=0.5cm,\n"
                output += "/tikz/every odd column/.append style={column sep=0cm},},\n"
                output += "legend to name=CommonLegendAlg"
                output += str(counter // 25 + 1)

            output += "]\n"

            for m in range(len(ex_datasets)):
                ex_data = ex_datasets[m]
                qn_data = qn_datasets[m]
                curr_ex_data = ex_data[curr_name]
                curr_qn_data = qn_data[curr_name]

                output += "\t\\addplot[" + style_names[m] + "] coordinates {"

                # output only 3 decimal places of precision
                output += "(" + str(x_labels[0]) + ", " + f'{curr_ex_data[0]:.3f}' + ")"
                output += "(" + str(x_labels[1]) + ", " + f'{curr_qn_data[0]:.3f}' + ")"

                output += "};\n"
            output += "\n"
            counter += 1

    with open(output_file, 'a') as f:
        f.write(output)
