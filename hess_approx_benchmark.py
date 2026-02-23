import casadi as cs
import utils.initialization as initialization
# import utils.sensitivity_lifting as sensitivity_lifting
import utils.create_nlp as create_nlp
from utils.get_problem import get_problem, get_oed_problem
import os
import timeit


# file where the results will be saved
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "logs/hess_lists.log")


problems = [
    "Van der Pol",
    "Van der Pol Mayer",
    "Lotka Volterra",
    "Lotka Volterra Mayer",
    "Batch Reactor",
    "Bioreactor",
    "Bioreactor Mayer",
    "Bryson Denham",
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
    "Particle Steering",
    "Rao Mease",
    "Rao Mease Mayer",
    "Stirred Tank Reactor",
    "Three Tank",
    "Tubular Reactor",
    "Tubular Reactor Mayer",
    "Quadrotor"
]

init_types = ["auto",
              "lin",
              "rand"]

lift_options = ["all",
                "none",
                "adaptive"]

problem_names = problems
lifting_type = lift_options[0]
init_type = "auto"
num_lifting_points = 64
num_control_points = 64
l1_refinement = False
exact_hessian = True
optimize_lamb = False
log_results = False
always_auto = False
auto_condense = False

time_logs = []

problem_names = [
    "Jackson OED",
    "Lotka OED",
    "Van der Pol OED"
]

for problem_name in problem_names:
    # curr_problem = get_problem(problem_name)
    curr_problem = get_oed_problem(problem_name, criterion="A")
    print("SOLVING " + problem_name)

    f = open(filename, "a")
    output = "HessList(\n\t\""
    output += problem_name
    output += " ex. Hess." * exact_hessian
    output += " FSInit" * always_auto
    output += "\",\n\t[\n"
    f.write(output)
    f.close()

    curr_time_log = []

    # loop over the number of lifting points
    for num_part_lifts in range(0, 7):
        curr_init_type = init_type
        ode = curr_problem.get_ode()
        num_controls = num_control_points
        num_lifts = num_lifting_points
        max_t = curr_problem.get_grid_details()

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

        start = init_vals["s_start"]

        init_vals["sol"] = cs.vertcat(q_init, start)
        init_vals["controls"] = q_init
        init_vals["s_end"] = start

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

        refine = l1_refinement
        if refine:
            refine = 5
        else:
            refine = -1
        import utils.blocksqp_utils.create_blocksqp_problem as better_ipopt
        opts = {}
        opts["max_iter"] = 200
        opts["exact_hess"] = exact_hessian
        opts['plot_iter'] = False
        opts["refinement"] = refine
        opts["optim_lamb"] = optimize_lamb
        opts["log_results"] = log_results
        opts["always_auto"] = always_auto
        opts["auto_condense"] = auto_condense

        start_time = timeit.default_timer()
        num_iter, _, accepted_hess = better_ipopt.create_blocksqp_problem(
            curr_problem, grid, init, [], opts, accepted_hess_init=[]
        )
        diff_time = timeit.default_timer() - start_time

        curr_time_log += [diff_time]

        f = open(filename, "a")
        output = "\t\t"
        if num_iter == cs.inf:
            output += "[],\n"
        else:
            output += str(accepted_hess) + ",\n"
        f.write(output)
        f.close()

    time_logs += [[problem_name] + curr_time_log]
    f = open(filename, "a")
    output = "\t]),\n,\n"
    output += str(time_logs) + ",\n"
    f.write(output)
    f.close()

