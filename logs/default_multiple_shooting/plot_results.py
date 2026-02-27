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

    files = [
        "ipopt_exact_iters.dat",
        "ipopt_quasi_newton_iters.dat",
        "blocksqp2_exact_iters.dat",
        "blocksqp2_quasi_newton_iters.dat"
    ]

    x_labels = [0, 2, 4, 8, 16, 32, 64]
    style_names = ["ipopt_ex", "ipopt_qu", "blocksqp_ex", "blocksqp_qu"]

    datasets = [load_data(os.path.join(dirname, files[j])) for j in range(4)]
    problem_names = list(datasets[0].keys())
    # plot_rows(data)

    output = ""
    for i in range(len(problem_names)):
        curr_name = problem_names[i]
        output += "%" + "---" + str(i + 1) + " " + curr_name + "---\n"
        output += "\\nextgroupplot[title={"
        output += curr_name + "}"

        # add y-labels
        if i % 4 == 0:
            output += ", ylabel={Rel. Iterations}"

        if i % 24 == 0:
            output += ",\nlegend columns=-1, % Horizontal legend \n"
            output += "legend entries={IPOPT exact, IPOPT Quasi-Newton, BlockSQP2 exact, BlockSQP2 Quasi-Newton},\n"
            output += "legend style={draw=none, fill=none, font=\\footnotesize,\n"
            output += "column sep=0.5cm,\n"
            output += "/tikz/every odd column/.append style={column sep=0cm},},\n"
            output += "legend to name=CommonLegendComp"
            output += str(i // 24 + 1)

        output += "]\n"

        for m in range(len(datasets)):
            data = datasets[m]
            curr_data = data[curr_name]

            # rescale the current data to relative iterations
            min_item = min(curr_data)
            curr_data = [el / min_item for el in curr_data]

            output += "\t\\addplot[" + style_names[m] + "] coordinates {"
            for k in range(len(curr_data)):
                # output only 3 decimal places of precision
                output += "(" + str(x_labels[k]) + ", " + f'{curr_data[k]:.3f}'
                output += ") "
            output += "};\n"
        output += "\n"
    with open(output_file, 'a') as f:
        f.write(output)

