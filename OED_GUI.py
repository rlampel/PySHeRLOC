import numpy as np
import casadi as cs
import matplotlib.pyplot as plt
import tkinter as tk
import Tkinter_plots.plot_gui as plot_gui
import Tkinter_plots.nlp_callback as cb
import utils.initialization as initialization
import utils.sensitivity_lifting as sensitivity_lifting
import utils.create_nlp as create_nlp
from utils.blocksqp_utils.blocksqp_init_heuristics import refine_lifting
from utils.get_problem import get_oed_problem
from Tkinter_plots import GUIBaseClass
import os


# file where the results will be saved
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, "logs/benchmark.log")
subdir = "oed."


class OEDGUI(GUIBaseClass.GUI):
    def __init__(self):
        super().__init__()

    def set_opt_entries(self):
        self.options = [
            "BASF Example OED",
            "Batch Reactor OED",
            "Catalytic Reaction",
            "CSTR OED",
            "Dielectr Particle",
            "Diels Alder OED",
            "Horse OED",
            "Jackson OED",
            "Lotka OED",
            "LQR OED",
            "Nonlinear Toy OED",
            "Quadrotor OED",
            "Rao Mease OED",
            "Three Tank OED",
            "Toy OED",
            "Yeast OED",
            "Van der Pol OED",
        ]

        self.oed_criteria = ["A", "D", "M", "-1/(A*A)"]
        self.init_types = [
            "automatic",
            "linear",
            "random"
        ]
        self.lift_options = [
            "all",
            "none",
            "adaptive",
            "last",
            "sensitivity"
        ]
        self.solvers = [
            "BlockSQP 2",
            "BlockSQP",
            "fatrop",
            "IPOPT"
        ]

    def define_vars(self):
        # ---------- Variables ----------
        # Define GUI variables
        self.problem_name = tk.StringVar()
        self.solver_name = tk.StringVar()
        self.init_type = tk.StringVar()
        self.lifting_type = tk.StringVar()
        self.optimize_init = tk.BooleanVar()
        self.oed_criterion = tk.StringVar()
        self.function_running = tk.BooleanVar()
        self.num_lifting_points = tk.IntVar()
        self.num_control_points = tk.IntVar()
        self.l1_refinement = tk.BooleanVar()
        self.exact_hessian = tk.BooleanVar()
        self.cond_init = tk.BooleanVar()
        self.optimize_lamb = tk.BooleanVar()
        self.log_results = tk.BooleanVar()
        self.always_auto = tk.BooleanVar()
        self.auto_condense = tk.BooleanVar()

        # initial values of GUI variables
        self.problem_name.set(self.options[0])
        self.solver_name.set(self.solvers[0])
        self.lifting_type.set(self.lift_options[0])
        self.optimize_init.set(False)
        self.oed_criterion.set(self.oed_criteria[0])
        self.init_type.set("automatic")
        self.num_lifting_points.set(64)
        self.num_control_points.set(64)
        self.function_running.set(False)
        self.l1_refinement.set(False)
        self.exact_hessian.set(False)
        self.cond_init.set(False)
        self.optimize_lamb.set(False)
        self.log_results.set(False)
        self.always_auto.set(False)
        self.auto_condense.set(False)

    def insert_options(self):
        ###########################################
        # Add options to menu
        ###########################################
        row_counter = 0

        # Select Problem
        tk.Label(self.left_frame, text="Select problem:"
                 ).grid(row=row_counter, column=0,
                        sticky="NSEW", padx=5, pady=5)
        row_counter += 1
        drop = tk.OptionMenu(self.left_frame, self.problem_name, *self.options)
        drop.grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Select number of controls
        tk.Label(self.left_frame, text="Select number of controls:"
                 ).grid(row=row_counter, column=0,
                        sticky="NSEW", padx=5, pady=5)
        row_counter += 1
        control_scale = tk.Scale(master=self.left_frame, variable=self.num_control_points,
                                 from_=1, to=100, orient="horizontal")
        control_scale.grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        control_scale.set(self.num_control_points.get())
        row_counter += 1

        # Select number of lifting points
        tk.Label(self.left_frame,
                 text="Select max. number of lifting points:").grid(row=row_counter,
                                                                    column=0, sticky="NSEW",
                                                                    padx=5, pady=5)
        row_counter += 1
        lift_scale = tk.Scale(master=self.left_frame, variable=self.num_lifting_points,
                              from_=1, to=100,
                              orient="horizontal")
        lift_scale.grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        lift_scale.set(self.num_lifting_points.get())
        row_counter += 1

        # Select OED criterion
        tk.Label(self.left_frame, text="Select lifting approach:"
                 ).grid(row=row_counter, column=0,
                        sticky="NSEW", padx=5, pady=5)
        row_counter += 1
        drop = tk.OptionMenu(self.left_frame, self.oed_criterion, *self.oed_criteria)
        drop.grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Select Lifting type
        tk.Label(self.left_frame,
                 text="Select lifting approach:").grid(row=row_counter, column=0,
                                                       sticky="NSEW", padx=5, pady=5)
        row_counter += 1
        drop = tk.OptionMenu(self.left_frame, self.lifting_type, *self.lift_options)
        drop.grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Select Solver
        tk.Label(self.left_frame, text="Select solver:").grid(row=row_counter, column=0,
                                                              sticky="NSEW", padx=5, pady=5)
        row_counter += 1
        drop = tk.OptionMenu(self.left_frame, self.solver_name, *self.solvers)
        drop.grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Callback with refinement
        tk.Checkbutton(master=self.left_frame,
                       text='refinement callback',
                       variable=self.l1_refinement,
                       onvalue=True,
                       offvalue=False
                       ).grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Use exact Hessian matrix
        tk.Checkbutton(master=self.left_frame,
                       text='exact Hessian',
                       variable=self.exact_hessian,
                       onvalue=True,
                       offvalue=False
                       ).grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Improve conditioning
        tk.Checkbutton(master=self.left_frame,
                       text='improve Hess. condition',
                       variable=self.cond_init,
                       onvalue=True,
                       offvalue=False
                       ).grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Compute better initial Lagrange multipliers
        tk.Checkbutton(master=self.left_frame,
                       text='optimize Lagr. multipliers',
                       variable=self.optimize_lamb,
                       onvalue=True,
                       offvalue=False
                       ).grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Plot convergence graph
        tk.Checkbutton(master=self.left_frame,
                       text='plot convergence',
                       variable=self.log_results,
                       onvalue=True,
                       offvalue=False
                       ).grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Always initialize automatically
        tk.Checkbutton(master=self.left_frame,
                       text='initialize auto.',
                       variable=self.always_auto,
                       onvalue=True,
                       offvalue=False
                       ).grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Enable heuristic for automatic condensing
        tk.Checkbutton(master=self.left_frame,
                       text='auto condensing',
                       variable=self.auto_condense,
                       onvalue=True,
                       offvalue=False
                       ).grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Select initialization
        tk.Label(self.left_frame,
                 text="Select type of initialization:").grid(row=row_counter, column=0,
                                                             sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        init_menu = tk.OptionMenu(self.left_frame, self.init_type, *self.init_types)
        init_menu.grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Optimize initialization
        tk.Checkbutton(master=self.left_frame,
                       text='optimize initialization',
                       variable=self.optimize_init,
                       onvalue=True,
                       offvalue=False
                       ).grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Show initialization
        tk.Button(master=self.left_frame,
                  command=self.plot,
                  height=2,
                  width=10,
                  text="Preview").grid(row=row_counter, column=0, sticky="NSEW", padx=5, pady=5)
        row_counter += 1

        # Solve OCP
        self.last_row = row_counter
        tk.Button(master=self.left_frame,
                  command=self.solve,
                  height=2,
                  width=10,
                  text="Solve").grid(row=self.last_row, column=0, sticky="NSEW", padx=5, pady=5)

    def plot(self):
        # initialize problem
        curr_criterion = self.oed_criterion.get()
        curr_problem = get_oed_problem(self.problem_name.get(), curr_criterion)
        curr_ode = curr_problem.get_ode()
        curr_init_type = self.adapt_init_type(self.init_type.get())
        num_controls = self.num_control_points.get()
        num_lifts = self.num_lifting_points.get()
        max_t = curr_problem.get_grid_details()

        # create grids
        control_points = [i * max_t / num_controls for i in range(num_controls)]
        time_points = [i * max_t / num_controls for i in range(num_controls + 1)]
        grid = {}
        grid["time"] = time_points
        grid["control"] = control_points

        # initialize starting values
        init_vals = curr_problem.get_init()
        q_start = init_vals["q_start"]
        q_init = cs.DM(q_start * num_controls)
        s_dim = init_vals["s_dim"]
        start = init_vals["s_start"]

        init_vals["sol"] = cs.vertcat(q_init, start)
        init_vals["controls"] = q_init
        init_vals["s_end"] = start

        # compute intermediate states
        s_init = initialization.initialize(init_vals, grid, curr_ode, curr_init_type)
        init_vals["s_init"] = s_init

        curr_lifting_type = self.lifting_type.get()

        match curr_lifting_type:
            case "all":
                # lifting_points = [1 for i in range(num_lifts + 1)]
                lifting_points = [0] * len(time_points)

                lift_interval = num_controls // num_lifts
                for i in range(num_controls + 1):
                    if i % lift_interval == 0:
                        lifting_points[i] = 1
                print("number of lifting points: ", sum(lifting_points))
            case "adaptive":
                lifting_points = sensitivity_lifting.refine_lifting(curr_problem, init_vals, grid)
            case "last":
                lifting_points = [0 for i in range(num_lifts)] + [1]
            case "sensitivity":
                lifting_points = [1 for i in range(num_lifts + 1)]
                grid["lift"] = lifting_points
                sens_choice = sensitivity_lifting.get_grid_sens(curr_problem, init_vals, grid)
                if sens_choice:
                    lifting_points = [1 for i in range(num_lifts + 1)]
                else:
                    lifting_points = [0 for i in range(num_lifts + 1)]
            case _:
                lifting_points = [0 for i in range(num_lifts + 1)]

        print("lifting points: ", lifting_points)
        s_init = initialization.select_states(s_init, s_dim, lifting_points)
        init_vals["sol"] = cs.vertcat(q_init, s_init)
        grid["lift"] = lifting_points

        if self.optimize_init.get() and curr_lifting_type != "none":
            sel_time_points = [time_points[i] for i in range(len(time_points)) if lifting_points[i]]
            s_init, info_lift = refine_lifting(
                curr_problem, grid, sel_time_points, s_init, q_init
            )
            print("FSInit with points ", info_lift)
            init_vals["sol"] = cs.vertcat(q_init, s_init)

        # get labels if available
        s_labels = curr_problem.state_labels
        c_labels = curr_problem.control_labels
        s_indices = curr_problem.state_indices
        s_scales = curr_problem.state_scales

        plt.clf()
        plot_gui.plot_segmented([self.toolbar, self.canvas, self.fig], init_vals, grid, curr_ode,
                                state_labels=s_labels, control_labels=c_labels,
                                state_indices=s_indices, state_scales=s_scales)
        self.canvas.draw()
        self.canvas.flush_events()

    def solve(self):
        self.start_function()
        # get gui variables
        curr_criterion = self.oed_criterion.get()
        curr_problem = get_oed_problem(self.problem_name.get(), curr_criterion)
        curr_init_type = self.adapt_init_type(self.init_type.get())
        ode = curr_problem.get_ode()
        num_controls = self.num_control_points.get()
        num_lifts = self.num_lifting_points.get()
        max_t = curr_problem.get_grid_details()

        # create grids
        control_points = [i * max_t / num_controls for i in range(num_controls)]
        time_points = [i * max_t / num_controls for i in range(num_controls + 1)]
        lifting_points = list(np.zeros(len(time_points)))
        grid = {}
        grid["time"] = time_points
        grid["control"] = control_points

        # initialize starting values
        init_vals = curr_problem.get_init()
        s_dim = init_vals["s_dim"]
        q_start = init_vals["q_start"]
        q_init = cs.DM(q_start * num_controls)

        start = init_vals["s_start"]

        init_vals["sol"] = cs.vertcat(q_init, start)
        init_vals["controls"] = q_init
        init_vals["s_end"] = start

        # initialize intermediate states
        s_init = initialization.initialize(init_vals, grid, ode, curr_init_type)
        init_vals["s_init"] = s_init

        curr_lifting_type = self.lifting_type.get()

        match curr_lifting_type:
            case "all":
                # lifting_points = [1 for i in range(num_lifts + 1)]
                lifting_points = [0] * len(time_points)

                lift_interval = num_controls // num_lifts
                for i in range(num_controls + 1):
                    if i % lift_interval == 0:
                        lifting_points[i] = 1
                print("number of lifting points: ", sum(lifting_points))
            case "adaptive":
                lifting_points = sensitivity_lifting.refine_lifting(curr_problem, init_vals, grid)
            case "last":
                lifting_points = [0 for i in range(num_lifts)] + [1]
            case "sensitivity":
                lifting_points = [1 for i in range(num_lifts + 1)]
                grid["lift"] = lifting_points
                sens_choice = sensitivity_lifting.get_grid_sens(curr_problem, init_vals, grid)
                if not sens_choice:
                    lifting_points = [0 for i in range(num_lifts + 1)]
            case _:
                lifting_points = [0 for i in range(num_lifts + 1)]

        s_init = initialization.select_states(s_init, s_dim, lifting_points)
        init_vals["sol"] = cs.vertcat(q_init, s_init)
        grid["lift"] = lifting_points

        if self.optimize_init.get() and curr_lifting_type != "none":
            sel_time_points = [time_points[i] for i in range(len(time_points)) if lifting_points[i]]
            s_init, info_lift = refine_lifting(
                curr_problem, grid, sel_time_points, s_init, q_init
            )
            print("FSInit with points ", info_lift)
            init_vals["sol"] = cs.vertcat(q_init, s_init)

        init = cs.vertcat(q_init, s_init)

        # plotting details for callback
        plot_details = {}
        plot_details["GUI"] = [self.toolbar, self.canvas, self.fig]
        plot_details["grid"] = grid
        plot_details["ode"] = ode
        plot_details["problem"] = curr_problem
        plot_details["init"] = init_vals
        plot_details["log_results"] = self.log_results.get()
        plot_details["log_name"] = filename
        plot_details["time_scale"] = curr_problem.time_scale_ind
        plot_details["condense"] = self.auto_condense.get()

        curr_solver = self.solver_name.get()

        w, lbw, ubw, g, lbg, ubg, J = create_nlp.create_nlp(curr_problem, grid)

        plot_details["lbg"] = lbg
        plot_details["ubg"] = ubg

        match curr_solver:
            case "fatrop":
                prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}

                fatrop_opt = {}
                solver = cs.nlpsol('solver', 'fatrop', prob, fatrop_opt)

                sol = solver(x0=init, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
                init_vals["sol"] = sol["x"]
                print(solver.stats())

                plt.clf()
                plot_gui.plot_segmented([self.toolbar, self.canvas, self.fig],
                                        init_vals, grid, ode)
                self.canvas.draw()
                self.canvas.flush_events()

            case "IPOPT":
                mycallback = cb.MyCallback('mycallback', cs.vertcat(*w). shape[0],
                                           cs.vertcat(*g).shape[0], 0, plot_details)

                prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
                opts = {}

                opts['ipopt.print_level'] = 5
                opts['iteration_callback'] = mycallback
                # opts['ipopt.accept_every_trial_step'] = "yes"
                if not self.exact_hessian.get():
                    opts['ipopt.hessian_approximation'] = "limited-memory"
                    # opts['ipopt.limited_memory_max_history'] = 1000
                opts["ipopt.tol"] = 1.e-6
                opts['ipopt.max_iter'] = 100
                solver = cs.nlpsol('solver', 'ipopt', prob, opts)

                sol = solver(x0=init, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
                # print("solution for x: ", sol["x"])
                # print("total solution: ", sol)

                if self.log_results.get():
                    plt.clf()
                    plot_gui.plot_conv([self.toolbar, self.canvas, self.fig], [], [],
                                       mycallback.constr_viol, mycallback.objective)
                    self.canvas.draw()
                    self.canvas.flush_events()

            case "BlockSQP":
                mycallback = cb.MyCallback('mycallback',
                                           cs.vertcat(*w).shape[0],
                                           cs.vertcat(*g).shape[0],
                                           0, plot_details)
                prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
                opts = {}
                opts['max_iter'] = 100
                opts['iteration_callback'] = mycallback
                opts['print_time'] = 0
                # opts['hess_lim_mem'] = 1  # full memory approximation
                opts["opttol"] = 1.e-6
                # opts["linsol"] = "mumps"
                # opts['hess_update'] = 2 # BFGS update
                solver = cs.nlpsol('solver', 'blocksqp', prob, opts)

                sol = solver(x0=init, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
                init_vals["sol"] = sol["x"]
                plt.clf()
                plot_gui.plot_segmented([self.toolbar, self.canvas, self.fig], init_vals, grid, ode)
                self.canvas.draw()
                self.canvas.flush_events()

            case _:
                refine = self.l1_refinement.get()
                if refine:
                    refine = 5
                else:
                    refine = -1
                import utils.blocksqp_utils.create_blocksqp_problem as better_ipopt
                opts = {}
                opts["exact_hess"] = self.exact_hessian.get()
                opts["cond_init"] = self.cond_init.get()
                opts["refinement"] = refine
                opts["optim_lamb"] = self.optimize_lamb.get()
                opts["log_results"] = self.log_results.get()
                opts["always_auto"] = self.always_auto.get()
                opts["auto_condense"] = self.auto_condense.get()
                better_ipopt.create_blocksqp_problem(curr_problem, grid, init,
                                                     [self.toolbar, self.canvas, self.fig],
                                                     opts, condense_mode="OED")
        self.stop_function()


# create GUI
OEDGUI()
