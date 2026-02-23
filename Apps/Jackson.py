import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 3
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.state_labels = [r"$x_1$", r"$x_2$", r"$x_3$"]
        self.control_labels = [r"$u$"]

    def get_ode(self):
        # given model constants
        k1, k2, k3 = 1, 10, 1

        # Declare model variables
        x = cs.MX.sym('x', self.s_dim)
        x1, x2, x3 = cs.vertsplit(x)
        u = cs.MX.sym('u')

        # Model equations
        xdot = cs.vertcat(
            -u * (k1 * x1 - k2 * x2),
            u * (k1 * x1 - k2 * x2) - (1 - u) * k3 * x2,
            (1 - u) * k3 * x2
        )

        # Objective term
        L = 0
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([1., 0., 0.])
        init["q_start"] = [0.5]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 1
        return max_t

    def start_bounds(self, start):
        bs = [1., 0., 0.]
        return start, bs, bs

    def state_bounds(self, state):
        ubs = [1.1] * self.s_dim
        lbs = [0.] * self.s_dim
        return state, ubs, lbs

    def control_bounds(self, control):
        lbu = [0.]
        ubu = [1.]
        return control, ubu, lbu

    def objective_end(self, state):
        return state[-1]
