import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 4
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        x0 = cs.MX.sym('x0')
        x1 = cs.MX.sym('x1')
        x2 = cs.MX.sym('x2')
        y = cs.MX.sym('y')
        x = cs.vertcat(x0, x1, x2, y)
        u = cs.MX.sym('u')

        c1, c2 = 0.1, 0.4
        # Model equations
        xdot = cs.vertcat(x0 - x0 * x1 - x0 * x2,
                          -x1 + x0 * x1 - c1 * x1 * u,
                          -x2 + 1.2 * x0 * x2 - c2 * x2 * u,
                          (x1 - 1)**2 + (x2 - 1)**2 + 1e-3 * u**2 + (x0 - 1.7)**2)

        xdot[-1] *= 1.e-2
        # Objective term
        L = 0
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([1.5, 0.5, 1., 0.])
        init["q_start"] = [0.75]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 40
        return max_t

    def start_bounds(self, start):
        bs = [1.5, 0.5, 1., 0.]
        return start, bs, bs

    def control_bounds(self, control):
        lbu = [0]
        ubu = [1]
        return control, ubu, lbu

    def state_bounds(self, state):
        lbs = [0., 0., 0., 0.]
        ubs = [cs.inf, cs.inf, cs.inf, cs.inf]
        return state, ubs, lbs

    def objective_end(self, state):
        return state[-1]

