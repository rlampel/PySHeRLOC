import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
# import time
from . import plot_gui
from . import condense_ipopt
from utils import penalty


class MyCallback(cs.Callback):
    def __init__(self, name, nx, ng, np, plot_details):
        cs.Callback.__init__(self)
        self.iter = plot_details.get("iter", 0)
        self.prim_vars = cs.DM([])
        self.condense_success = True
        self.nx = nx
        self.ng = ng
        self.np = np
        self.plot_details = plot_details
        self.plot_iter = plot_details.get("plot_iter", True)
        self.condense = plot_details.get("condense", False)
        self.problem = plot_details["problem"]

        self.grid = plot_details["grid"]
        if self.plot_iter:
            self.GUI = plot_details["GUI"]
            self.ode = plot_details["ode"]
            self.time_scale_ind = plot_details["time_scale"]
            self.curr_init = plot_details["init"]

        self.log_results = plot_details.get("log_results", False)
        if self.log_results or self.condense:
            self.lbg = plot_details.get("lbg", [-cs.inf] * ng)
            self.ubg = plot_details.get("ubg", [cs.inf] * ng)
            self.lbx = plot_details.get("lbx", [-cs.inf] * nx)
            self.ubx = plot_details.get("ubx", [cs.inf] * nx)

            # lists for logging
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

    def get_violation(self, g, x):
        g_viol = penalty.get_violation(g, self.ubg, self.lbg)
        x_viol = penalty.get_violation(x, self.ubx, self.lbx)
        return cs.vertcat(g_viol, x_viol)

    def eval(self, arg):
        # Create dictionary
        darg = {}
        for (i, s) in enumerate(cs.nlpsol_out()):
            darg[s] = arg[i]

        if self.log_results:
            # add variable bounds and constraint violation
            viol = self.get_violation(darg['g'], darg['x'])
            self.constr_viol += [
                float(cs.norm_inf(viol))
            ]
            # objective
            self.objective += [float(darg['f'])]

        if self.plot_iter:
            self.curr_init["sol"] = darg['x']
            # scale time interval if the index of the scaling factor is given
            if not (np.isnan(float(self.time_scale_ind))):
                time_scale = float(darg['x'][float(self.time_scale_ind)])
            else:
                time_scale = 1

            # get labels if available
            s_labels = self.problem.state_labels
            c_labels = self.problem.control_labels
            s_indices = self.problem.state_indices
            s_scales = self.problem.state_scales
            plt.clf()
            ax = plot_gui.plot_segmented(
                self.GUI, self.curr_init, self.grid, self.ode,
                state_labels=s_labels,
                control_labels=c_labels,
                state_indices=s_indices,
                state_scales=s_scales,
                time_scale=time_scale
            )
            ax.set_title(f"Iteration {self.iter}", fontsize=22)
            canvas = self.GUI[1]
            canvas.draw()
            canvas.flush_events()

        self.iter += 1

        # Switch to single shooting and start over
        if self.condense:
            viol = self.get_violation(darg['g'], darg['x'])
            curr_viol = float(cs.norm_inf(viol))
            small_violation = (curr_viol < 1.e-3)

            if small_violation and sum(self.grid["lift"][1:]) > 0:
                print("#" * 20 + "\nSTART CONDENSED NLP\n" + "#" * 20)
                self.plot_details["iter"] = self.iter
                stats = condense_ipopt.condense_ipopt_nlp(
                    darg,
                    self.plot_details
                )
                self.iter += stats["iter_count"]
                success = stats["success"]
                self.condense_success = success
                return [1]

        return [0]

