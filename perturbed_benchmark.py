import numpy as np
import casadi as cs
import utils.initialization as initialization
import utils.create_nlp as create_nlp
import utils.get_problem as get_problem
import os
import timeit


# file where the results will be saved
dirname = os.path.dirname(__file__)
iter_file = os.path.join(dirname, "logs/benchmark_iters.log")
time_file = os.path.join(dirname, "logs/benchmark_times.log")


# read list of problems
with open('benchmark_problems.txt', 'r') as problem_file:
    problem_names = problem_file.readlines()

problem_names = [el.strip() for el in problem_names]

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

solvers = ["BlockSQP 2"]
moolver_name = solvers[0]
lifting_type = lift_options[0]
init_type = "auto"
num_lifting_points = 64
num_control_points = 64
l1_refinement = False
exact_hessian = False
optimize_lamb = False
optimize_init = False
log_results = False
always_auto = True
auto_condense = True
max_iter = 200
num_reps = 1


for mode in ["default"]:  # ["default", "OED"]:

    if mode == "OED":
        problem_names = oed_problems

    problem_names = ["Dielectrophoretic Particle Mayer"]

    for problem_name in problem_names:
        print("SOLVING " + problem_name)

        if mode == "OED":
            curr_problem = get_problem.get_oed_problem(problem_name)
        else:
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
        heur_activation = []  # check whether the heuristic activated at all

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

            '''
            # if mode == "default" and problem_name != "Quadrotor":
                # print("Add random noise to control")
                # add random noise to controls for default mode
                # np.random.seed(i)
                # noise = np.random.rand(q_init.shape[0]) - 0.5
                # q_init += 0.1 * noise
            '''

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

            w, lbw, ubw, g, lbg, ubg, J = create_nlp.create_nlp(curr_problem, grid)

            refine = l1_refinement
            if refine:
                refine = 5
            else:
                refine = -1
            import utils.blocksqp_utils.create_blocksqp_problem as blockSQP2
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
            num_iter, _, _ = blockSQP2.create_blocksqp_problem(
                curr_problem, grid, init, [],
                opts, condense_mode=mode
            )
            diff_time = timeit.default_timer() - start_time

            if num_iter == cs.inf or num_iter >= max_iter:
                num_inter = cs.inf
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

