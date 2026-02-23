import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 3
    q_dim = 2
    global_controls = [1]
    time_scale_ind = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.state_labels = [r"$h$", r"$v$", r"$m$"]
        self.control_labels = [r"$T$", r"$t$"]

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x', self.s_dim)
        h, v, m = cs.vertsplit(x)
        u = cs.MX.sym('u', self.q_dim)
        T, t = cs.vertsplit(u)

        # Model equations
        xdot = cs.vertcat(
            v,
            -1 + T / m,
            -T / 2.349
        )

        # Objective term
        L = 0
        ode = {'x': x, 'p': u, 'ode': xdot * t, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([1., -0.783, 1.])
        init["q_start"] = [0.5, 0.2]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 10
        return max_t

    def start_bounds(self, start):
        bs = [1., -0.783, 1.]
        return start, bs, bs

    def control_bounds(self, control):
        lbu = [0., 0.]
        ubu = [1.227, cs.inf]
        return control, ubu, lbu

    def end_bounds(self, state):
        lbs = [0., 0., -cs.inf]
        ubs = [0., 0., cs.inf]
        return state, ubs, lbs

    def objective_end(self, state):
        return -state[-1]
