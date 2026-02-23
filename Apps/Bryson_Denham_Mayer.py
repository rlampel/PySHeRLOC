import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 3
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        x1 = cs.MX.sym('x1')
        x2 = cs.MX.sym('x2')
        Mayer = cs.MX.sym('Mayer')
        x = cs.vertcat(x1, x2, Mayer)
        u = cs.MX.sym('u')

        # Model equations
        xdot = cs.vertcat(x2, u, 0.5 * u**2)
        # Objective term
        L = 0
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0., 1., 0])
        init["q_start"] = [1.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 1
        return max_t

    def start_bounds(self, start):
        lbs = [0, 1, 0]
        ubs = [0, 1, 0]
        return start, ubs, lbs

    def state_bounds(self, state):
        lbs = [-cs.inf, -cs.inf, 0]
        ubs = [1 / 9, cs.inf, cs.inf]
        return state, ubs, lbs

    def end_bounds(self, state):
        lbs = [0, -1, -cs.inf]
        ubs = [0, -1, cs.inf]
        return state, ubs, lbs

    def objective_end(self, state):
        return state[-1]
