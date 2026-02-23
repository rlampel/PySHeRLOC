import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 2
    q_dim = 3

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x', self.s_dim)
        u = cs.MX.sym('u', self.q_dim)
        # Equations
        x1dot = -x[0] * u[0] + (x[0] + x[1]) * u[1] + (x[0] - x[1]) * u[2]
        x2dot = (x[0] + 2 * x[1]) * u[0] + (x[0] - 2 * x[1]) * u[1] + (x[0] + x[1]) * u[2]

        xdot = cs.vertcat(x1dot, x2dot)

        # Objective term
        L = x[0]**2 + x[1]**2
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0.5, 0.7])
        init["q_start"] = [1 / 3, 1 / 3, 1 / 3]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 1
        return max_t

    def start_bounds(self, start):
        lbs = [0.5, 0.5]
        ubs = [0.5, 0.5]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [0] * 3
        ubu = [1] * 3
        return control, ubu, lbu

    def control_cond(self, control):
        control_sum = control[0] + control[1] + control[2]
        lbu = [1]
        ubu = [1]
        return control_sum, ubu, lbu

    def state_bounds(self, state):
        lbs = [-cs.inf, 0.4]
        ubs = [cs.inf, cs.inf]
        return state, ubs, lbs

