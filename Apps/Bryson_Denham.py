import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 2
    q_dim = 1

    state_labels = [r"$x$", r"$v$"]
    control_labels = [r"$w$"]

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        x1 = cs.MX.sym('x1')
        x2 = cs.MX.sym('x2')
        x = cs.vertcat(x1, x2)
        u = cs.MX.sym('u')

        # Model equations
        xdot = cs.vertcat(x2, u)
        # Objective term
        L = 0.5 * u**2
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0., 1.])
        init["q_start"] = [1.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 1
        return max_t

    def start_bounds(self, start):
        lbs = [0, 1]
        ubs = [0, 1]
        return start, ubs, lbs

    def state_bounds(self, state):
        lbs = [-cs.inf, -cs.inf]
        ubs = [1 / 9, cs.inf]
        return state, ubs, lbs

    def end_bounds(self, state):
        lbs = [0, -1]
        ubs = [0, -1]
        return state, ubs, lbs

