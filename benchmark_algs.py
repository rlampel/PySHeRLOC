import numpy as np
import casadi as cs
import utils.initialization as initialization
import utils.get_problem as get_problem
import utils.blocksqp_utils.create_blocksqp_problem as blockSQP2
import os
import timeit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-n', "--problem_name", default="Bioreactor",
                    help="Name of the problem to be solved.")
parser.add_argument('-hess', "--exact_hessian", default=False,
                    help="Indicates whether to use the exact Hessian (y/n).")
parser.add_argument('-fs', "--always_auto", default=False,
                    help="Indicates whether to use the FSInit algorithm (y/n).")
parser.add_argument('-cond', "--auto_condense", default=False,
                    help="Indicates whether to use the automatic condensing algorithm (y/n).")

# parse the input settings
args = parser.parse_args()
problem_name = args.problem_name
exact_hessian = ("y" == args.exact_hessian)
always_auto = ("y" == args.always_auto)
auto_condense = ("y" == args.auto_condense)

print("hess: ", exact_hessian, "\nfs: ", always_auto, "\ncond: ", auto_condense)

# get the current mode (used for some algorithms)
if problem_name[-3:] == "OED":
    mode = "OED"
else:
    mode = "default"

# general settings
solvers = ["BlockSQP 2"]
lifting_type = "all"
init_type = "auto"
num_lifting_points = 64
num_control_points = 64
max_iter = 200
num_reps = 5

# set the name of the output file
output_name = ""
output_name += "fsinit_" * always_auto
output_name += "condense_" * auto_condense
output_name += "exact" * exact_hessian
output_name += "quasi_newton" * (not exact_hessian)

# append the currect path to the log file
dirname = os.path.dirname(__file__)
iter_file = os.path.join(
    dirname,
    "logs/algorithm_results/" + output_name + "_iters.log"
)
time_file = os.path.join(
    dirname,
    "logs/algorithm_results/" + output_name + "_times.log"
)


curr_problem = get_problem.get_problem(problem_name)

curr_init_type = init_type
ode = curr_problem.get_ode()
num_controls = num_control_points
num_lifts = num_lifting_points
max_t = curr_problem.get_grid_details()

# log the required numer of iterations
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

    opts = {}
    opts["exact_hess"] = exact_hessian
    opts['plot_iter'] = False
    opts["always_auto"] = always_auto
    opts["auto_condense"] = auto_condense
    opts["max_iter"] = max_iter

    start_time = timeit.default_timer()
    num_iter, _, _ = blockSQP2.create_blocksqp_problem(
        curr_problem, grid, init, [],
        opts, condense_mode=mode
    )
    diff_time = timeit.default_timer() - start_time

    if num_iter == cs.inf or num_iter >= max_iter:
        num_iter = cs.inf
        diff_time = cs.inf

    curr_time_log += [diff_time]
    curr_iter_log += [num_iter]

avg_time = sum(curr_time_log) / num_reps
avg_iters = sum(curr_iter_log) / num_reps

# log the required numer of iterations
with open(iter_file, 'a') as f:
    output = str(avg_iters) + ", "
    f.write(output)

# log the required real time
with open(time_file, 'a') as f:
    output = str(avg_time) + ", "
    f.write(output)

