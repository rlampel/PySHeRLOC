import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
# import time
from . import plot_gui
from . import condense_ipopt
from utils import initialization, penalty
from utils.blocksqp_utils import dyn_lifting


class MyCallback(cs.Callback):
    def __init__(self, name, nx, ng, np, plot_details):
        cs.Callback.__init__(self)
        self.iter = plot_details.get("iter", 0)
        self.prim_vars = cs.DM([])
        self.nx = nx
        self.ng = ng
        self.np = np
        self.plot_details = plot_details
        self.plot_iter = plot_details.get("plot_iter", True)

        if self.plot_iter:
            self.GUI = plot_details["GUI"]

        self.condense = plot_details.get("condense", False)
        self.grid = plot_details["grid"]
        self.problem = plot_details["problem"]
        self.ode = plot_details["ode"]
        self.curr_init = plot_details["init"]
        self.log_results = plot_details.get("log_results", False)
        # self.log_name = plot_details["log_name"]
        self.time_scale_ind = plot_details["time_scale"]
        self.lbg = plot_details.get("lbg", [-cs.inf] * ng)
        self.ubg = plot_details.get("ubg", [cs.inf] * ng)
        self.objective = []
        self.constr_viol = []
        # Initialize internal objects
        self.construct(name, {})

    def get_n_in(self):
        return cs.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return cs.nlpsol_out(i)

    def get_name_out(self, i):
        return "ret"

    def get_sparsity_in(self, i):
        n = cs.nlpsol_out(i)
        if n == 'f':
            return cs.Sparsity.scalar()
        elif n in ('x', 'lam_x'):
            return cs.Sparsity.dense(self.nx)
        elif n in ('g', 'lam_g'):
            return cs.Sparsity.dense(self.ng)
        else:
            return cs.Sparsity(0, 0)

    def eval(self, arg):
        # Create dictionary
        darg = {}
        for (i, s) in enumerate(cs.nlpsol_out()):
            darg[s] = arg[i]
        self.curr_init["sol"] = darg['x']

        if self.log_results:
            # violation of constraints g
            self.constr_viol += [float(penalty.l1_penalty(darg['g'], self.ubg, self.lbg))]
            # violation of variable bounds
            # objective
            self.objective += [float(darg['f'])]

        # scale time interval if the index of the scaling factor is given
        if not (np.isnan(float(self.time_scale_ind))):
            time_scale = float(darg['x'][float(self.time_scale_ind)])
        else:
            time_scale = 1

        if self.plot_iter:
            # get labels if available
            s_labels = self.problem.state_labels
            c_labels = self.problem.control_labels
            s_indices = self.problem.state_indices
            s_scales = self.problem.state_scales
            # print("time scale: ", time_scale)
            plt.clf()
            ax = plot_gui.plot_segmented(self.GUI, self.curr_init, self.grid, self.ode,
                                         state_labels=s_labels,
                                         control_labels=c_labels,
                                         state_indices=s_indices,
                                         state_scales=s_scales,
                                         time_scale=time_scale)
            ax.set_title(f"Iteration {self.iter}", fontsize=22)
            canvas = self.GUI[1]
            canvas.draw()
            canvas.flush_events()

        self.iter += 1

        # Switch to single shooting and start over
        curr_viol = float(penalty.l1_penalty(darg['g'], self.ubg, self.lbg))
        small_violation = (curr_viol < 1.e-3)
        if small_violation and sum(self.grid["lift"][1:]) > 0:
            if self.condense:
                self.plot_details["iter"] = self.iter
                condense_ipopt.condense_ipopt_nlp(darg,
                                                  self.plot_details
                                                  )
                return [1]

        if False:
            s_dim = self.problem.s_dim
            q_dim = self.problem.q_dim
            num_lifts = len(self.grid["time"])
            s_init = self.curr_init["sol"][:s_dim * num_lifts]
            controls = self.curr_init["sol"][s_dim * num_lifts:]
            lifting_points = dyn_lifting.best_graph_lift(self.problem, self.grid["time"],
                                                         s_init, controls,
                                                         self.grid)
            temp_grid = self.grid.copy()
            temp_grid["lift"] = lifting_points
            s_init = initialization.compute_all_states({"sol": s_init,
                                                        "s_dim": s_dim,
                                                        "q_dim": q_dim},
                                                       temp_grid, self.ode)
            print("Determined Lifting Points: ", lifting_points)

        return [0]

