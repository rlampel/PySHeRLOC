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
        # Declare model variables
        a = 3
        b = 0
        c = 0.5
        x = cs.MX.sym('x', 3)
        u = cs.MX.sym('u')

        # Model equations
        xdot = cs.vertcat(
            x[1], x[2], u
        )

        # Objective term
        L = (a * x[0] + b * x[1]**2 + c * u**2)
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([1., -2., 0.])
        init["q_start"] = [0.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 10
        return max_t

    def start_bounds(self, start):
        bs = [1., -2., 0.]
        return start, bs, bs

    def state_bounds(self, state):
        ubs = [cs.inf] * self.s_dim
        lbs = [0, -cs.inf, -cs.inf]
        return state, ubs, lbs

    def end_bounds(self, state):
        bs = [0] * self.s_dim
        return state, bs, bs
