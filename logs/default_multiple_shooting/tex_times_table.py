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

    datasets = [load_data(os.path.join(dirname, files[j])) for j in range(4)]
    problem_names = list(datasets[0].keys())
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
                output += ", ylabel={Time (s)}"

            if counter == 0:
                output += ",\nlegend columns=-1, % Horizontal legend \n"
                output += "legend entries={IPOPT exact, IPOPT Quasi-Newton, blockSQP2 exact, blockSQP2 Quasi-Newton},\n"
                output += "legend style={draw=none, fill=none, font=\\footnotesize,\n"
                output += "column sep=0.5cm,\n"
                output += "/tikz/every odd column/.append style={column sep=0cm},},\n"
                output += "legend to name=CommonLegendTimeComp"
                output += str(counter // 24 + 1)

            output += "]\n"

            for m in range(len(datasets)):
                data = datasets[m]
                curr_data = data[curr_name]

                # rescale the current data to relative iterations
                # min_item = min(curr_data)
                # curr_data = [el / min_item for el in curr_data]

                output += "\t\\addplot[" + style_names[m] + "] coordinates {"
                for k in range(len(curr_data)):
                    # output only 3 decimal places of precision
                    output += "(" + str(x_labels[k]) + ", " + f'{curr_data[k]:.3f}'
                    output += ") "
                output += "};\n"
            output += "\n"
            counter += 1

    with open(output_file, 'a') as f:
        f.write(output)

