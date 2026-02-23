import casadi as cs
import utils.initialization as initialization
import utils.sensitivity_lifting as sensitivity_lifting
import utils.create_nlp as create_nlp
import Tkinter_plots.nlp_callback as cb
from utils.get_problem import get_problem, get_oed_problem
import os


# file where the results will be saved
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "logs/ipopt_lists.log")


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

solvers = [
    "BlockSQP",
    "Old BlockSQP",
    "fatrop",
    "IPOPT",
]


nlp_solver = solvers[3]
problem_names = problems
lifting_type = lift_options[0]
init_type = "auto"
num_lifting_points = 64
num_control_points = 64
exact_hessian = True
optimize_lamb = False
log_results = False
always_auto = False

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
    output = "[\"" + problem_name
    output += "\", ["
    f.write(output)
    f.close()
    # solving loop

    for num_part_lifts in range(7):
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

        # Input for Newton's method
        default_init = {}
        default_init["sol"] = init
        default_init["s_dim"] = s_dim
        default_init["q_dim"] = q_dim

        w, lbw, ubw, g, lbg, ubg, J = create_nlp.create_nlp(curr_problem, grid)

        match nlp_solver:
            case "IPOPT":
                plot_details = {}
                plot_details["grid"] = grid
                plot_details["ode"] = ode
                plot_details["problem"] = curr_problem
                plot_details["init"] = default_init
                plot_details["log_results"] = True
                plot_details["time_scale"] = curr_problem.time_scale_ind
                plot_details["plot_iter"] = False

                mycallback = cb.MyCallback('mycallback', cs.vertcat(*w).shape[0],
                                           cs.vertcat(*g).shape[0], 0, plot_details)

                prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
                opts = {}

                opts['ipopt.print_level'] = 5
                # opts['iteration_callback'] = mycallback

                if not exact_hessian:
                    print("Using approximate Hessian!")
                    opts['ipopt.hessian_approximation'] = "limited-memory"
                    # opts['ipopt.limited_memory_max_history'] = 1000

                opts["ipopt.tol"] = 1.e-6
                opts['ipopt.max_iter'] = 200
                solver = cs.nlpsol('solver', 'ipopt', prob, opts)

                sol = solver(x0=init, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
                stats = solver.stats()

            case "fatrop":
                prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}

                fatrop_opt = {}
                solver = cs.nlpsol('solver', 'fatrop', prob, fatrop_opt)

                sol = solver(x0=init, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
                stats = solver.stats()

        success = stats["success"]
        if success:
            num_iter = stats["iter_count"]
        else:
            num_iter = float(cs.inf)

        f = open(filename, "a")
        output = str(num_iter)
        if num_part_lifts < 6:
            output += ", "
        f.write(output)
        f.close()

    f = open(filename, "a")
    output = "]],\n"
    f.write(output)
    f.close()
