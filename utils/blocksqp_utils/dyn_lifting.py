import casadi as cs
import numpy as np
from .. import penalty, ode_solver, initialization
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


def partial_eval(curr_problem, partial_grid, s_init, controls):
    """Evaluate constraint violations and the objective of a problem on a given time interval.

    Keyword arguments:
        curr_problem  -- instance of the OC problem class
        partial_grid  -- dict containing discretization points of current time interval
        s_init  -- state value at the start of the time interval
        controls  -- values of discretized control values for the original time interval
    """

    q_dim = curr_problem.q_dim

    ode = curr_problem.get_ode()
    end_time = curr_problem.get_grid_details()

    J = 0
    punish_state = 0

    # input for integrator
    init = {}
    init["controls"] = controls
    init["q_dim"] = q_dim
    init["s"] = s_init

    curr_time_points = partial_grid["part_time"]
    control_points = partial_grid["control"]

    for k in range(len(curr_time_points) - 1):
        s_init, Jk = ode_solver.integrate_interval(init, control_points, ode,
                                                   curr_time_points[k], curr_time_points[k + 1])
        J += Jk
        state, ubs, lbs = curr_problem.state_bounds(s_init)
        punish_state += penalty.l1_penalty(state, ubs, lbs)
        init["s"] = s_init

    if (end_time in partial_grid):
        state, ubs, lbs = curr_problem.end_bounds(s_init)
        punish_state += penalty.l1_penalty(state, ubs, lbs)
        J += curr_problem.objective_end(s_init)

    return J, punish_state


def get_num_items(arr, item):
    """Return how often a certain item appears in a given array.

    Keyword arguments:
        arr -- list of items
        item -- element for which to compute the number of appearances
    """
    counter = 0
    for el in arr:
        if (np.isclose(el, item)):
            counter += 1
    return counter


def remove_nan(arr):
    """Replace all appearances of not a number in an array by infinity.

    Keyword arguments:
        arr -- list of items
    """
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if (np.isnan(arr[i, j])):
                arr[i, j] = np.inf
    return arr


def best_graph_lift(problem, starting_times, starting_vals, controls, grid, mu=1,
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

    # integrate all states to the end of the time interval
    for i in range(len(starting_times)):
        temp_grid = {"time": [el for el in time_points if el >= starting_times[i]],
                     "control": grid["control"]}
        temp_init = {"s_start": starting_vals[i * x_dim:(i + 1) * x_dim], "s_dim": x_dim,
                     "controls": controls, "q_dim": q_dim}
        int_vals = initialization.initialize(temp_init, temp_grid, ode, "auto")
        int_vals = np.reshape(int_vals, (-1, x_dim))
        int_vals = remove_nan(int_vals)
        states += [int_vals]

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
        # print("number of new nodes: ", num_new_nodes)
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

            partial_grid = {
                "part_time": [curr_time, next_time],
                "control": grid["control"]
            }
            J, punish_state = partial_eval(problem, partial_grid, s_old, controls)
            # print("current J and viol: ", J, punish_state)
            incidence[j, num_curr_candidates + j] = J + punish_state * mu + eps

            # print("distance without lifting: ", J + punish_state + eps)

            for k in range(start_index, stop_index):
                indx = k - curr_candidates - num_curr_candidates
                s_new = states[indx][0]

                continuity_violation = cs.norm_1(s_old - s_new)
                if np.isnan(continuity_violation):
                    continuity_violation = np.inf

                total_cost = float(J + punish_state * mu + continuity_violation * mu + eps)
                incidence[j, k] = total_cost
                # print("distance with lifting: ", J + punish_state + continuity_violation + eps)

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
    # best_lifting_points = convert_lifting(best_lifting_points, grid["time"])
    return best_lifting_points


def get_shortest_path(graph, times, verbose=False):
    """Compute the shortest path in the graph and return the corresponding lifting times.

    Keyword arguments:
        graph   -- graph for which we want to compute a shortest path
        times   -- time points that correspond to the candidates for intermediate variables
    """
    total_times = times + [1]
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=True, indices=[0],
                                              return_predecessors=True)
    predecessors = list(predecessors.flatten())
    curr_ind = predecessors[-1]
    lifting_points = []
    while (curr_ind >= 0):
        lifting_points = [total_times[curr_ind]] + lifting_points
        curr_ind = predecessors[curr_ind]
    return lifting_points


def convert_lifting(input_lift, time_points):
    """Converts a lifting given by the time points into a list of binary decisions for every
    lifting point.

    Keyword arguments:
        input_lift  -- list containing the lifting points as time points
        time_points -- list of all possible lifting points as time points
    """
    output_lift = []
    for el1 in time_points:
        flag = False
        for el2 in input_lift:
            if (np.isclose(el1, el2)):
                flag = True
                break
        if (flag):
            output_lift += [1]
        else:
            output_lift += [0]
    return output_lift
