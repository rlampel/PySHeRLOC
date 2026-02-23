import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 4
    q_dim = 1
    d, t_f = 1., 2.

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Model parameters
        m1, m2, g = 1., 0.3, 9.81
        ell = 0.5

        # Declare model variables
        x = cs.MX.sym('x', self.s_dim)
        x1, x2, theta1, theta2 = cs.vertsplit(x)
        F = cs.MX.sym('F')

        x1dot = x2
        theta1dot = theta2
        x2dot = -m2 * g * cs.sin(theta1) * cs.cos(theta1)
        x2dot -= (F + m2 * ell * theta2**2 * cs.sin(theta1))
        x2dot /= (m2 * cs.cos(theta1)**2 - (m1 + m2))
        theta2dot = (m1 + m2) * g * cs.sin(theta1)
        theta2dot += cs.cos(theta1) * (F + m1 * ell * theta2**2 * cs.sin(theta1))
        theta2dot /= (m2 * ell * cs.cos(theta1)**2 - (m1 + m2) * ell)

        # Model equations
        xdot = cs.vertcat(x1dot, x2dot, theta1dot, theta2dot)

        # Objective term
        L = F**2
        ode = {'x': x, 'p': F, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([self.d, 0., cs.pi, 0.])
        init["q_start"] = [0.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = self.t_f
        return max_t

    def control_bounds(self, control):
        lb = [-20]
        ub = [20]
        return control, ub, lb

    def start_bounds(self, start):
        bs = [0.] * self.s_dim
        return start, bs, bs

    def state_bounds(self, state):
        lb = [-2] + [-cs.inf] * (self.s_dim - 1)
        ub = [2] + [cs.inf] * (self.s_dim - 1)
        return state, ub, lb

    def end_bounds(self, state):
        ubs = [self.d, 0., cs.pi, 0.]
        lbs = [self.d, 0., cs.pi, 0.]
        return state, ubs, lbs

