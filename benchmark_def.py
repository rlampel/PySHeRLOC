import casadi as cs
import numpy as np
import utils.initialization as initialization
import utils.create_nlp as create_nlp
from utils.get_problem import get_problem, get_oed_problem
import os
import timeit
import argparse

parser = argparse.ArgumentParser(
    description="Defines the problem name and algorithmic settings."
)
parser.add_argument('-n', "--problem_name", default="Bioreactor",
                    help="Name of the problem to be solved.")
parser.add_argument('-hess', "--exact_hessian", default=False,
                    help="Indicates whether to use the exact Hessian (y/n).")
parser.add_argument('-solver', "--solver_name", default="blockSQP2",
                    help="Name of the solver to be used.")

# parse the input settings
args = parser.parse_args()
problem_name = args.problem_name
exact_hessian = ("y" == args.exact_hessian)
solver_name = args.solver_name

print("hess: ", exact_hessian, "\nsolver: ", solver_name)

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

# turn off all algorithms
l1_refinement = False
optimize_lamb = False
log_results = False
always_auto = False
auto_condense = False

# file where the results will be saved
dirname = os.path.dirname(__file__)

# read list of problems
with open('benchmark_problems.txt', 'r') as problem_file:
    oc_problems = problem_file.readlines()


# file where the results will be saved
log_name = "logs/default_multiple_shooting/" + solver_name + "_"
log_name += exact_hessian * "exact"
log_name += (not exact_hessian) * "quasi_newton"
iter_file = os.path.join(dirname, log_name + "_iters.log")
time_file = os.path.join(dirname, log_name + "_times.log")

if mode == "OED":
    curr_problem = get_oed_problem(problem_name)
else:
    curr_problem = get_problem(problem_name)

# log the required number of iterations
with open(iter_file, 'a') as f:
    output = "\n" + problem_name + ", "
    f.write(output)

# log the required real time
with open(time_file, 'a') as f:
    output = "\n" + problem_name + ", "
    f.write(output)

# loop over the number of lifting points
for num_part_lifts in range(0, 7):
    curr_init_type = init_type
    ode = curr_problem.get_ode()
    num_controls = num_control_points
    num_lifts = num_lifting_points
    max_t = curr_problem.get_grid_details()

    # log average time and iterations for current lifting
    curr_time_log = []
    curr_iter_log = []

    for rep in range(num_reps):
        # if there is already one sample that did not converge, skip the rest
        if sum(curr_iter_log) == cs.inf:
            break

        # create grids
        control_points = [i * max_t / num_controls for i in range(num_controls)]
        time_points = [i * max_t / num_lifts for i in range(num_lifts + 1)]
        lifting_points = [0] * len(time_points)
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
            np.random.seed(rep)
            noise = np.random.rand(q_init.shape[0]) - 0.5
            q_init += 1.e-2 * noise

        start = init_vals["s_start"]

        init_vals["sol"] = cs.vertcat(q_init, start)
        init_vals["controls"] = q_init

        # initialize intermediate states
        s_init = initialization.initialize(init_vals, grid, ode, curr_init_type)
        init_vals["s_init"] = s_init

        # set the discretization so that there are num_lifts lifting points
        if num_part_lifts != 0:
            lift_interval = num_lifting_points // 2**num_part_lifts
            for i in range(num_lifting_points + 1):
                if i % lift_interval == 0:
                    lifting_points[i] = 1

        print(lifting_points)

        s_init = initialization.select_states(s_init, s_dim, lifting_points)
        init_vals["sol"] = cs.vertcat(q_init, s_init)
        grid["lift"] = lifting_points

        init = cs.vertcat(q_init, s_init)
        input_size = init.shape[0]

        w, lbw, ubw, g, lbg, ubg, J = create_nlp.create_nlp(curr_problem, grid)

        match solver_name:
            case "IPOPT":
                prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
                opts = {}
                if not exact_hessian:
                    opts['ipopt.hessian_approximation'] = "limited-memory"
                    opts['ipopt.limited_memory_max_history'] = 1000

                opts['ipopt.print_level'] = 5
                opts["ipopt.tol"] = 1.e-6
                opts['ipopt.max_iter'] = max_iter
                solver = cs.nlpsol('solver', 'ipopt', prob, opts)

                start_time = timeit.default_timer()
                sol = solver(x0=init, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
                diff_time = timeit.default_timer() - start_time

                stats = solver.stats()
                num_iter = stats["iter_count"]
                if not stats["success"] or num_iter >= max_iter:
                    print("Solver failed!")
                    num_iter = cs.inf
                    diff_time = cs.inf

            case _:
                refine = l1_refinement
                if refine:
                    refine = 5
                else:
                    refine = -1
                import utils.blocksqp_utils.create_blocksqp_problem as create_blocksqp
                opts = {}
                opts["max_iter"] = 200
                opts["exact_hess"] = exact_hessian
                opts['plot_iter'] = False
                opts["refinement"] = refine
                opts["optim_lamb"] = optimize_lamb
                opts["log_results"] = log_results
                opts["always_auto"] = always_auto
                opts["auto_condense"] = auto_condense
                opts["max_iter"] = max_iter

                start_time = timeit.default_timer()
                num_iter, _, accepted_hess = create_blocksqp.create_blocksqp_problem(
                    curr_problem, grid, init, [], opts, condense_mode=mode
                )
                diff_time = timeit.default_timer() - start_time

                if num_iter == cs.inf or num_iter >= max_iter:
                    print("Solver failed!")
                    num_iter = cs.inf
                    diff_time = cs.inf

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

