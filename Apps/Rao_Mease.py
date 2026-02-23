import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 1
    q_dim = 1

    state_labels = [r"$x$"]
    control_labels = [r"$w$"]

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x')
        u = cs.MX.sym('u')

        # Model equations
        xdot = -x**3 + u
        # Objective term
        L = x**2 + u**2
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([1.])
        init["q_start"] = [0.0]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 10
        return max_t

    def start_bounds(self, start):
        lbs = [1]
        ubs = [1]
        return start, ubs, lbs

    def end_bounds(self, state):
        lbs = [1.5]
        ubs = [1.5]
        return state, ubs, lbs

