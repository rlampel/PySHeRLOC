import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 1
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.state_labels = [r"$x$"]
        self.control_labels = [r"$u$"]

    def get_ode(self):
        # Declare model variables
        a = -1
        b = 1
        x = cs.MX.sym('x')
        u = cs.MX.sym('u')

        # Model equations
        xdot = a * x + b * u

        # Objective term
        L = 10 * (x - 3)**2 + 0.1 * u**2
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([1])
        init["q_start"] = [0]
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

