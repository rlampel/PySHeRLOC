import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 2
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        x0 = cs.MX.sym('x0')
        x1 = cs.MX.sym('x1')
        x = cs.vertcat(x0, x1)
        u = cs.MX.sym('w')

        c0, c1 = 0.4, 0.2
        # Model equations
        xdot = cs.vertcat(
            x0 - x0 * x1 - c0 * x0 * u,
            -x1 + x0 * x1 - c1 * x1 * u
        )

        # Objective term
        L = (x0 - 1)**2 + (x1 - 1)**2
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0.5, 0.7])
        init["q_start"] = [0.75]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 12
        return max_t

    def start_bounds(self, start):
        lbs = [0.5, 0.7]
        ubs = [0.5, 0.7]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [0]
        ubu = [1]
        return control, ubu, lbu

    def state_bounds(self, state):
        lbs = [0, 0]
        ubs = [cs.inf, cs.inf]
        return state, ubs, lbs

