import casadi as cs
import numpy as np
from .. import initialization
from .dyn_lifting import partial_eval, get_num_items, remove_nan, get_shortest_path
from scipy.sparse import csr_matrix


def best_init_lift(problem, starting_times, starting_vals, controls, grid, mu=1,
                   verbose=False, return_state_indx=False):
    """Compute the lifting with the optimal residual contraction for an OCP.

    Keyword arguments:
        problem         --  instance of optimal control problem class
        starting_times  --  times at which the candidates for the intermediate variables are
                            introduced (ordered list with ascending entries)
        starting_vals   --  candidate values for the intermediate variables that are introduced
                            at the time points given in starting_times
        time_points     --  times for possible lifting points
        x_dim           --  dimension of the states
        verbose         --  print additional information
        return_state_indx   --  return the index of the best new candidate for a lifting point
                                (if there are multiple candidates at one time point)
    """
    x_dim = problem.s_dim
    q_dim = problem.q_dim
    ode = problem.get_ode()
    states = []
    eps = 1.e-16
    time_points = grid["time"]

    # Assume that the ODE is autonomous and that the control initialization is constant.
    # Then we only have to integrate over the interval once and shift the states for the
    # lifting points.
    temp_grid = {"time": [el for el in time_points if el >= starting_times[0]],
                 "control": grid["control"]}
    temp_init = {"s_start": starting_vals[:x_dim], "s_dim": x_dim,
                 "controls": controls, "q_dim": q_dim}
    int_vals = initialization.initialize(temp_init, temp_grid, ode, "auto")
    int_vals = np.reshape(int_vals, (-1, x_dim))
    int_vals = remove_nan(int_vals)

    for i in range(len(starting_times)):
        num_inits_rem = len([el for el in time_points if el >= starting_times[i]])
        states += [int_vals[:num_inits_rem]]

    # The same approach can be applied to the partial evaluation of the states and controls
    J_list, punish_state_list = [], []
    for j in range(len(time_points) - 1):
        partial_grid = {
            "part_time": [time_points[j], time_points[j + 1]],
            "control": grid["control"]
        }
        J_curr, punish_state_curr = partial_eval(problem, partial_grid, states[0][j], controls)
        J_list += [J_curr]
        punish_state_list += [punish_state_curr]

    num_nodes = 1
    incidence = np.zeros((num_nodes, num_nodes))

    times = [0]
    candidate_times = [0]
    curr_candidates = 0
    num_lift_points = len(time_points) - 1
    num_curr_candidates = 0

    if return_state_indx:
        state_indices = [0]
        candidate_indices = [0]

    for curr_lift_point in range(num_lift_points):
        curr_time = time_points[curr_lift_point]
        num_curr_candidates += get_num_items(starting_times, curr_time)

        next_time = time_points[curr_lift_point + 1]
        num_new_nodes = get_num_items(starting_times, next_time)
        num_candidates_next_layer = num_curr_candidates + num_new_nodes

        candidate_times += [next_time] * num_new_nodes
        times += candidate_times
        if return_state_indx:
            candidate_indices += [i for i in range(num_new_nodes)]
            state_indices += candidate_indices

        incidence = np.pad(incidence, ((0, num_candidates_next_layer),
                                       (0, num_candidates_next_layer)),
                           'constant', constant_values=0)

        for j in range(curr_candidates, curr_candidates + num_curr_candidates):
            # No lifting edge (identity edge)
            start_index = curr_candidates + 2 * num_curr_candidates
            stop_index = curr_candidates + 2 * num_curr_candidates + num_new_nodes
            s_old = states[j - curr_candidates][-(num_lift_points - curr_lift_point)]

            # base index for current lifting point
            s_old_ind = curr_lift_point - (j - curr_candidates)

            # add the number of points that have been passed so far
            J = J_list[s_old_ind]
            punish_state = punish_state_list[s_old_ind]
            incidence[j, num_curr_candidates + j] = J + punish_state * mu + eps

            for k in range(start_index, stop_index):
                indx = k - curr_candidates - num_curr_candidates
                s_new = states[indx][0]

                continuity_violation = cs.norm_1(s_old - s_new)
                if np.isnan(continuity_violation):
                    continuity_violation = np.inf

                total_cost = float(J + punish_state * mu + continuity_violation * mu + eps)
                incidence[j, k] = total_cost

        curr_candidates += num_curr_candidates

    num_candidates_last_layer = len(starting_times)
    incidence = np.pad(incidence, ((0, 1), (0, 1)), 'constant', constant_values=0)

    start_index = len(incidence) - 1 - num_candidates_last_layer
    for m in range(num_candidates_last_layer):
        end_val = states[m][-1]
        partial_grid = {
            "part_time": [time_points[-1], time_points[-1]],
            "control": grid["control"]
        }
        J_f, punish_state_f = partial_eval(problem, partial_grid, end_val, partial_grid["control"])
        err = J_f + punish_state_f * mu
        if np.isnan(err):
            err = np.inf
        incidence[start_index + m, -1] = err + eps

    graph = csr_matrix(incidence)
    best_lifting_points = get_shortest_path(graph, times, verbose)

    if return_state_indx:
        best_state_indices = [0]
        for i in range(1, len(best_lifting_points)):
            if best_lifting_points[i] != best_lifting_points[i - 1]:
                best_state_indices.append(state_indices[i])
        return np.unique(best_lifting_points), best_state_indices

    best_lifting_points = np.unique(best_lifting_points)
    return best_lifting_points

