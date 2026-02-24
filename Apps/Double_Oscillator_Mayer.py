import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 5
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.control_labels = [r"$u$"]

    def get_ode(self):
        # given model constants
        m1, m2 = 100, 2
        k1, k2 = 100, 3
        c = 0.5
        T = 2 * cs.pi

        # Declare model variables
        x = cs.MX.sym('x', self.s_dim)
        x1, x2, x3, x4, x5 = cs.vertsplit(x)
        u = cs.MX.sym('u')
        t = cs.MX.sym('t')

        # Model equations
        F = cs.sin(2 * cs.pi / T * t)

        xdot = cs.vertcat(
            x3, x4,
            - (k1 + k2) / m1 * x1 + k2 / m1 * x2 + 1 / m1 * F,
            k2 / m2 * x1 - k2 / m2 * x2 - c * (1 - u) / m2 * x4,
            0.5 * (x1**2 + x2**2 + u**2)
        )
        # Objective term
        L = 0.
        ode = {'x': x, 't': t, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0., 0., 0., 0., 0.])
        init["q_start"] = [0.5]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 2 * cs.pi
        return max_t

    def start_bounds(self, start):
        lbs = [0, 0, -cs.inf, -cs.inf, 0.]
        ubs = [0, 0, cs.inf, cs.inf, 0.]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [-1]
        ubu = [1]
        return control, ubu, lbu

    def objective_end(self, state):
        return state[-1]

