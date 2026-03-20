import numpy as np
import utils.ode_solver as ode_solver
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)


def plot_segmented(GUI, init, grid, ode,
                   state_labels=[],
                   control_labels=[],
                   state_indices=None,
                   state_scales=None,
                   time_scale=1, starting_ind=0):
    """Plot the current multiple shooting solution in the GUI.

    Keyword arguments:
        GUI -- list containing toolbar, canvas and figure
        init    -- values of current states and controls
        grid    -- dict containing discretization and lifting points
        ode     -- casadi function
        starting_ind    -- first state to plot
    """
    toolbar, canvas, fig = GUI
    q_dim = init["q_dim"]
    s_dim = init["s_dim"]
    sol = init["sol"]

    ax_state = fig.add_subplot(2, 1, 1)
    ax_control = fig.add_subplot(2, 1, 2)

    control_points = grid["control"]
    time_points = grid["time"]
    scaled_time_points = [time_scale * t for t in grid["time"]]
    scaled_control_points = [time_scale * t for t in grid["control"]]
    lifting_points = grid["lift"]

    # if no specific indices are supplied, plot everything
    if state_indices is None:
        state_indices = [i for i in range(s_dim)]

    num_control_points = len(control_points)
    # convert to numpy array
    sol = np.array(sol).flatten()

    q_opt = sol[:q_dim * num_control_points]
    s_opt = sol[q_dim * num_control_points:]
    # duplicate final entry for aesthetic reasons
    q_opt_plot = np.reshape(np.array(q_opt), (-1, q_dim))
    q_opt_plot = np.concatenate((q_opt_plot, [q_opt_plot[-1]]), axis=0)
    q_colors = ["orange", "blue", "magenta", "pink", "cyan"]
    for k in range(q_dim):
        ax_control.step(scaled_control_points + [scaled_time_points[-1]],
                        q_opt_plot[:, k],
                        where='post',
                        linestyle="-.",
                        color=q_colors[k % len(q_colors)])

    ax_control.plot()

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
                if d in state_indices:
                    if state_scales is not None:
                        curr_scale = state_scales[d]
                    else:
                        curr_scale = 1
                    ax_state.plot(scaled_time_points[starting_ind:j + 2],
                                  [el[d] * curr_scale for el in plot_list],
                                  color=colors[d % len(colors)],
                                  linestyle=styles[d % 4])

            starting_ind = j + 1
            curr_lift_point += 1
            curr_s = s_opt[curr_lift_point * s_dim:(curr_lift_point + 1) * s_dim]
            plot_list = [np.array(curr_s).flatten()]

    for d in range(s_dim):
        if d in state_indices:
            if state_scales is not None:
                curr_scale = state_scales[d]
            else:
                curr_scale = 1
            ax_state.plot(scaled_time_points[starting_ind:len(time_points)],
                          [el[d] * curr_scale for el in plot_list],
                          color=colors[d % len(colors)],
                          linestyle=styles[d % 4])

    for lb in range(len(control_labels), q_dim):
        label = r"$u_{" + str(lb) + "}$"
        control_labels += [label]

    for lb in range(len(state_labels), len(state_indices)):
        label = r"$x_{" + str(lb) + "}$"
        state_labels += [label]

    scaled_state_labels = state_labels.copy()
    # add the scaling to the matching labels
    if state_scales is not None:
        label_ind = 0
        for i in range(len(state_scales)):
            s = state_scales[i]
            if i in state_indices:
                if s != 1.:
                    scaled_state_labels[label_ind] += r"$\ \cdot \ " + str(s) + "$"
                label_ind += 1

    ax_control.legend(control_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18)
    ax_state.legend(scaled_state_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=18)
    ax_state.set_xlabel(r"Time $t$", fontsize=22)
    ax_control.set_xlabel(r"Time $t$", fontsize=22)
    ax_state.set_ylabel("State value", fontsize=22)
    ax_control.set_ylabel("Control value", fontsize=22)
    return ax_state


def plot_conv(GUI, conv_lists, legend_labels, viol_arr, obj_arr):
    """Plot the current multiple shooting solution in the GUI.

    Keyword arguments:
        GUI -- list containing toolbar, canvas and figure
        conv_lists -- list of lists, each entry creates one plot
    """
    toolbar, canvas, fig = GUI

    ax = fig.add_subplot(1, 2, 1)
    colors = ["black", "red", "#777", "#955", "brown", "purple"]
    styles = ["-", "--", ":", "-."]

    for k in range(len(conv_lists)):
        curr_list = conv_lists[k]
        ax.plot([i for i in range(len(curr_list))], curr_list,
                linestyle=styles[k % len(styles)],
                color=colors[k % len(colors)])

    ax.legend(legend_labels, fontsize=12)
    ax.set_yscale("log")
    ax.set_xlabel("iteration", fontsize=16)
    ax.set_ylabel("Component value", fontsize=16)

    ax2 = fig.add_subplot(1, 2, 2)

    ax2.plot(viol_arr, obj_arr, color="red")
    ax2.scatter(viol_arr, obj_arr, marker="x", color="black")

    ax2.set_xlabel("violation", fontsize=16)
    ax2.set_ylabel("objective value", fontsize=16)
    return


def plot_path(GUI, viol_arr, obj_arr):
    """Plot the current multiple shooting solution in the GUI.

    Keyword arguments:
        GUI -- list containing toolbar, canvas and figure
        conv_lists -- list of lists, each entry creates one plot
    """
    toolbar, canvas, fig = GUI

    ax = fig.add_subplot(1, 2, 2)

    ax.plot(viol_arr, obj_arr, color="red")
    ax.scatter(viol_arr, obj_arr, marker="x", color="black")

    ax.set_xlabel("violation", fontsize=16)
    ax.set_ylabel("objective value", fontsize=16)
    return ax

