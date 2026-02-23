import numpy as np
import casadi as cs
import utils.initialization as initialization
import utils.create_nlp as create_nlp
import utils.get_problem as get_problem
from utils.blocksqp_utils.blocksqp_init_heuristics import refine_lifting
import os
import timeit


# file where the results will be saved
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "logs/benchmark_iterations.log")


problems = [
    "Van der Pol",
    "Van der Pol Mayer",
    "Lotka Volterra",
    "Lotka Volterra Mayer",
    "Batch Reactor",
    "Bioreactor",
    "Bioreactor Mayer",
    "Bryson Denham",
    "Bryson Denham Mayer",
    "Catalyst Mixing",
    "Cushioned Oscillation",
    "Dielectrophoretic Particle",
    "Double Oscillator",
    "Ducted Fan",
    "Egerstedt",
    "Egerstedt Mayer",
    "Electric Car",
    "F8 Aircraft",
    "F8 Aircraft Mayer",
    "Fuller",
    "Fuller Mayer",
    "Hang Glider",
    "Hanging Chain Lagrange",
    "Hanging Chain",
    "Lotka Competitive",
    "Lotka Competitive Mayer",
    "Lotka Shared",
    "Lotka Shared Mayer",
    "LQR",
    "LQR Mayer",
    "Mountain Car",
    "Ocean",
    "Ocean Mayer",
    "Particle Steering Mayer",
    "Rao Mease",
    "Rao Mease Mayer",
    "Three Tank",
    "Tubular Reactor",
    "Tubular Reactor Mayer",
    "Quadrotor"
]

problems = [
    # "Catalyst Mixing",
    # "Egerstedt",
    # "Egerstedt Mayer",
    # "Hang Glider",
    # "Hanging Chain Lagrange",
    # "Hanging Chain",
    "Particle Steering",
]

oed_problems = [
    "Lotka OED",
    "Dielectr Particle",
    "Jackson OED",
    "Van der Pol OED"
]

init_types = ["auto",
              "lin",
              "rand"]

lift_options = ["all",
                "none",
                "adaptive",
                "sensitivity"]

solvers = ["BlockSQP 2",
           "Old BlockSQP",
           "IPOPT",
           "fatrop"]

problem_names = problems
solver_name = solvers[0]
lifting_type = lift_options[0]
init_type = "auto"
num_lifting_points = 64
num_control_points = 64
l1_refinement = False
exact_hessian = False
optimize_lamb = False
optimize_init = False
log_results = False
always_auto = False
auto_condense = True
max_iter = 200

log_mode = "iter"
# log_mode = "time"

mode = "default"
# mode = "OED"

# solving loop
with open(filename, "a") as f:
    f.write("\n\n")

if mode == "OED":
    problem_names = oed_problems
else:
    problem_names = problems

for problem_name in problem_names:  # problem_names:
    print("SOLVING " + problem_name)

    num_reps = 5
    seeds = [1, 2, 3, 4, 5]
    iters = []

    for i in range(num_reps):
        if mode == "OED":
            curr_problem = get_problem.get_oed_problem(problem_name)
        else:
            curr_problem = get_problem.get_problem(problem_name)
        curr_init_type = init_type
        ode = curr_problem.get_ode()
        num_controls = num_control_points
        num_lifts = num_lifting_points
        max_t = curr_problem.get_grid_details()

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

        if mode == "default" and problem_name != "Quadrotor":
            print("Add random noise to control")
            # add random noise to controls for default mode
            np.random.seed(seeds[i])
            noise = np.random.rand(q_init.shape[0]) - 0.5
            q_init += 0.1 * noise

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

        if optimize_init and solver_name != "BlockSQP 2":
            # only consider the current time points
            sel_time_points = [time_points[i] for i in range(len(time_points)) if lifting_points[i]]
            s_init, _ = refine_lifting(
                curr_problem, grid, time_points, s_init, q_init
            )
        elif optimize_init:
            s_init = initialization.initialize(init_vals, grid, ode, "lin")
            s_init = initialization.select_states(s_init, s_dim, lifting_points)

        init = cs.vertcat(q_init, s_init)
        input_size = init.shape[0]

        w, lbw, ubw, g, lbg, ubg, J = create_nlp.create_nlp(curr_problem, grid)

        refine = l1_refinement
        if refine:
            refine = 5
        else:
            refine = -1
        import utils.blocksqp_utils.create_blocksqp_problem as better_ipopt
        opts = {}
        opts["exact_hess"] = exact_hessian
        opts['plot_iter'] = False
        opts["refinement"] = refine
        opts["optim_lamb"] = optimize_lamb
        opts["optim_init"] = optimize_init
        opts["log_results"] = log_results
        opts["always_auto"] = always_auto
        opts["auto_condense"] = auto_condense
        opts["max_iter"] = max_iter

        start_time = timeit.default_timer()
        num_iter, _, _ = better_ipopt.create_blocksqp_problem(
            curr_problem, grid, init, [],
            opts, condense_mode=mode
        )
        diff_time = timeit.default_timer() - start_time
        iters += [num_iter]

        print("Total time: ", diff_time)

    avg_iter = sum(iters) / len(iters)

    with open(filename, "a") as f:
        # output = problem_name + " took \t" + str(num_iter) + " iterations\n"
        if log_mode != "time" or avg_iter == np.inf:
            output = str(avg_iter) + " " + str(iters) + ", "
        else:
            output = str(diff_time) + ", "
        f.write(output)

