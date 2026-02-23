import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 2
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        a = -1
        b = 1
        x1 = cs.MX.sym('x1')
        x2 = cs.MX.sym('x2')
        x = cs.vertcat(x1, x2)
        u = cs.MX.sym('u')

        # Model equations
        x1dot = a * x1 + b * u
        x2dot = 10 * (x1 - 3)**2 + 0.1 * u**2
        xdot = cs.vertcat(x1dot, 0.1 * x2dot)

        # Objective term
        L = 0
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([1, 0])
        init["q_start"] = [3]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 10
        return max_t

    def start_bounds(self, start):
        lbs = [1, 0]
        ubs = [1, 0]
        return start, ubs, lbs

    def state_bounds(self, state):
        lbs = [-cs.inf, 0]
        ubs = [cs.inf] * 2
        return state, ubs, lbs

    def objective_end(self, state):
        return state[-1]
