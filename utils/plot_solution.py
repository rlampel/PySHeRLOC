import numpy as np
import matplotlib.pyplot as plt
from . import ode_solver


def plot_segmented(init, grid, ode, starting_ind=0):
    q_dim = init["q_dim"]
    s_dim = init["s_dim"]
    sol = init["sol"]

    control_points = grid["control"]
    time_points = grid["time"]
    lifting_points = grid["lift"]

    num_control_points = len(control_points)
    # convert to numpy array
    sol = np.array(sol).flatten()

    q_opt = sol[:q_dim * num_control_points]
    s_opt = sol[q_dim * num_control_points:]
    q_opt_plot = np.reshape(np.array(q_opt), (-1, q_dim))
    q_colors = ["yellow", "orange", "blue", "pink"]
    for k in range(q_dim):
        plt.step(control_points, q_opt_plot[:, k], where='post',
                 linestyle="-.", color=q_colors[k % len(q_colors)])

    plt.plot()

    colors = ["black", "red", "#777", "#955", "brown", "purple"]
    styles = ["-", "--", ":", "-."]

    curr_lift_point = 0
    curr_s = s_opt[curr_lift_point * s_dim:(curr_lift_point + 1) * s_dim]
    plot_list = [np.array(curr_s).flatten()]

    for j in range(len(time_points) - 1):
        curr_init = {}
        curr_init["controls"] = q_opt
        curr_init["s"] = curr_s
        curr_init["q_dim"] = q_dim

        curr_s, Jk = ode_solver.integrate_interval(curr_init, control_points, ode,
                                                   time_points[j], time_points[j + 1])
        plot_list += [np.array(curr_s).flatten()]

        if (lifting_points[j + 1] == 1):
            for d in range(s_dim):
                plt.plot(time_points[starting_ind:j + 2],
                         [el[d] for el in plot_list],
                         color=colors[d % len(colors)],
                         linestyle=styles[d % 4])

            starting_ind = j + 1
            curr_lift_point += 1
            curr_s = s_opt[curr_lift_point * s_dim:(curr_lift_point + 1) * s_dim]
            plot_list = [np.array(curr_s).flatten()]

    for d in range(s_dim):
        plt.plot(time_points[starting_ind:len(time_points)],
                 [el[d] for el in plot_list],
                 color=colors[d % len(colors)],
                 linestyle=styles[d % 4])

    legend_labels = []
    for lb in range(q_dim):
        label = r"$u_{" + str(lb) + "}$"
        legend_labels += [label]
    for lb in range(s_dim):
        label = r"$x_{" + str(lb) + "}$"
        legend_labels += [label]

    plt.legend(legend_labels)
    plt.xlabel(r"Time $t$")

