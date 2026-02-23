import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 3
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x', self.s_dim)
        u = cs.MX.sym('u', self.q_dim)

        # Model equations
        x0dot = x[1]
        x1dot = 1 - 2 * u
        x2dot = x[0]**2
        xdot = cs.vertcat(x0dot, x1dot, x2dot)

        # Objective term
        L = 0
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([1.e-2, 0., 0.])
        init["q_start"] = [0.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 1
        return max_t

    def start_bounds(self, start):
        lbs = [1.e-2, 0., 0.]
        ubs = [1.e-2, 0., 0.]
        return start, ubs, lbs

    def end_bounds(self, states):
        lbs = [1.e-2, 0., -cs.inf]
        ubs = [1.e-2, 0., cs.inf]
        return states, ubs, lbs

    def control_bounds(self, control):
        lbu = [0]
        ubu = [1]
        return control, ubu, lbu

    def objective_end(self, state):
        return state[-1] * 1.e2

