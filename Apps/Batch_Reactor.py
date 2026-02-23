import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 2
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x', self.s_dim)
        T = cs.MX.sym('T', self.q_dim)

        # rescale temperature
        u = T * 100

        # auxiliary equations
        k1 = 4.e3 * cs.exp(-2500 / u)
        k2 = 62.e4 * cs.exp(-5000 / u)

        # Model equations
        x1dot = -k1 * x[0]**2
        x2dot = k1 * x[0]**2 - k2 * x[1]
        xdot = cs.vertcat(x1dot, x2dot)

        # Objective term
        L = 0
        ode = {'x': x, 'p': T, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([1, 0])
        init["q_start"] = [2.98]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 1
        return max_t

    def start_bounds(self, start):
        lbs = [1, 0]
        ubs = [1, 0]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [2.98]
        ubu = [3.98]
        return control, ubu, lbu

    def objective_end(self, state):
        return -state[1]

