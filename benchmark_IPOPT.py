import casadi as cs
import numpy as np
import utils.initialization as initialization
import utils.create_nlp as create_nlp
import Tkinter_plots.nlp_callback as cb
from utils.get_problem import get_problem, get_oed_problem
import timeit
import os
import argparse

parser = argparse.ArgumentParser(
    description="Defines the problem name and algorithmic settings."
)
parser.add_argument('-n', "--problem_name", default="Bioreactor",
                    help="Name of the problem to be solved.")
parser.add_argument('-hess', "--exact_hessian", default="n",
                    help="Indicates whether to use the exact Hessian (y/n).")
parser.add_argument('-cond', "--auto_condense", default="n",
                    help="Indicates whether to use the automatic condensing algorithm (y/n).")

# parse the input settings
args = parser.parse_args()
problem_name = args.problem_name
exact_hessian = ("y" == args.exact_hessian)
auto_condense = ("y" == args.auto_condense)


# file where the results will be saved
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "logs/ipopt_lists.log")

# get the current mode (used for some algorithms)
if problem_name[-3:] == "OED":
    mode = "OED"
else:
    mode = "default"

# general settings
lifting_type = "all"
init_type = "auto"
num_lifting_points = 64
num_control_points = 64
max_iter = 200
num_reps = 5

# file where the results will be saved
log_name = "logs/default_multiple_shooting/" + "IPOPT_"
log_name += exact_hessian * "exact"
log_name += (not exact_hessian) * "quasi_newton"
iter_file = os.path.join(dirname, log_name + "_iters.log")
time_file = os.path.join(dirname, log_name + "_times.log")


if mode == "OED":
    curr_problem = get_oed_problem(problem_name)
else:
    curr_problem = get_problem(problem_name)


curr_init_type = init_type
ode = curr_problem.get_ode()
num_controls = num_control_points
num_lifts = num_lifting_points
max_t = curr_problem.get_grid_details()

# log the required number of iterations
with open(iter_file, 'a') as f:
    output = "\n" + problem_name + ", "
    f.write(output)

# log the required real time
with open(time_file, 'a') as f:
    output = "\n" + problem_name + ", "
    f.write(output)

# log average time and iterations for current lifting
curr_time_log = []
curr_iter_log = []

for i in range(num_reps):
    if sum(curr_iter_log) == cs.inf:
        break

    # create grids
    control_points = [i * max_t / num_controls for i in range(num_controls)]
    time_points = [i * max_t / num_lifts for i in range(num_lifts + 1)]
    lifting_points = list(np.zeros(len(time_points)))
    grid = {}
    grid["time"] = time_points
    grid["control"] = control_points

    # initialize starting values
    init_vals = curr_problem.get_init()
    s_dim = init_vals["s_dim"]
    q_dim = init_vals["q_dim"]
    q_start = init_vals["q_start"]
    q_init = cs.DM(q_start * num_controls)

    if problem_name != "Quadrotor":
        print("Add random noise to control")
        # add random noise to controls for default mode
        np.random.seed(i)
        noise = np.random.rand(q_init.shape[0]) - 0.5
        q_init += 1.e-2 * noise

    start = init_vals["s_start"]

    init_vals["sol"] = cs.vertcat(q_init, start)
    init_vals["controls"] = q_init
    init_vals["s_end"] = start

    # initialize intermediate states
    s_init = initialization.initialize(init_vals, grid, ode, curr_init_type)
    init_vals["s_init"] = s_init

    curr_lifting_type = lifting_type

    match curr_lifting_type:
        case "all":
            lifting_points = [1 for i in range(num_lifts + 1)]
        case _:
            lifting_points = [0 for i in range(num_lifts + 1)]

    # Use the inital values as candidates for the lifting points
    s_init = initialization.select_states(s_init, s_dim, lifting_points)
    grid["lift"] = lifting_points

    init = cs.vertcat(q_init, s_init)
    input_size = init.shape[0]
    # Input for Newton's method
    default_init = {}
    default_init["sol"] = init
    default_init["s_dim"] = s_dim
    default_init["q_dim"] = q_dim

    w, lbw, ubw, g, lbg, ubg, J = create_nlp.create_nlp(curr_problem, grid)

    plot_details = {}
    plot_details["grid"] = grid
    plot_details["ode"] = ode
    plot_details["problem"] = curr_problem
    plot_details["init"] = default_init
    plot_details["log_results"] = False
    plot_details["time_scale"] = curr_problem.time_scale_ind
    plot_details["plot_iter"] = False
    plot_details["condense"] = auto_condense
    plot_details['lbg'] = lbg
    plot_details['ubg'] = ubg
    plot_details['lbw'] = lbw
    plot_details['ubw'] = ubw

    mycallback = cb.MyCallback('mycallback', cs.vertcat(*w).shape[0],
                               cs.vertcat(*g).shape[0], 0, plot_details)

    prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
    opts = {}

    opts['ipopt.print_level'] = 5
    opts['iteration_callback'] = mycallback

    if not exact_hessian:
        print("Using approximate Hessian!")
        opts['ipopt.hessian_approximation'] = "limited-memory"

    opts["ipopt.tol"] = 1.e-6
    opts['ipopt.max_iter'] = 200
    solver = cs.nlpsol('solver', 'ipopt', prob, opts)

    start_time = timeit.default_timer()
    sol = solver(x0=init, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    diff_time = timeit.default_timer() - start_time
    stats = solver.stats()

    success = stats["success"]
    condense_success = mycallback.condense_success

    if success or condense_success:
        num_iter = mycallback.iter
    else:
        num_iter = float(cs.inf)

    curr_time_log += [diff_time]
    curr_iter_log += [num_iter]

avg_time = sum(curr_time_log) / num_reps
avg_iters = sum(curr_iter_log) / num_reps

# log the required number of iterations
with open(iter_file, 'a') as f:
    output = str(avg_iters) + ", "
    f.write(output)

# log the required real time
with open(time_file, 'a') as f:
    output = str(avg_time) + ", "
    f.write(output)
