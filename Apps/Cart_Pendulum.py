import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 4
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Model parameters
        M = 1.
        m = .1
        g = 9.81
        lambda_u = 0.5

        # Declare model variables
        x = cs.MX.sym('x', 4)
        u = cs.MX.sym('u')

        x0dot = x[1]
        x1dot = (u + m * g * cs.sin(x[2]) * cs.cos(x[2]) + m * x[3]**2 * cs.sin(x[2]))
        x1dot /= (M + m * (1 - cs.cos(x[2])**2))
        x2dot = x[3]
        x3dot = (-g * cs.sin(x[2]) - ((u + m * g * cs.sin(x[2]) * cs.cos(x[2]) + m * x[3]**2 * cs.sin(x[2])) / (M + m * (1 - cs.cos(x[2])**2))) * cs.cos(x[2]))

        # Model equations
        xdot = cs.vertcat(x0dot, x1dot, x2dot, x3dot)

        # Objective term
        L = 10. * (x[0])**2 + 50. * (x[2] - cs.pi)**2 + lambda_u * u**2
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0.] * 4)
        init["q_start"] = [0.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def state_bounds(self, state):
        lbs = [-2., -cs.inf, -cs.inf, -cs.inf]
        ubs = [2., cs.inf, cs.inf, cs.inf]
        return state, ubs, lbs

    def get_grid_details(self):
        max_t = 4
        return max_t

    def start_bounds(self, start):
        bs = [0.] * 4
        return start, bs, bs

    def control_bounds(self, control):
        lbu = [-30.]
        ubu = [30.]
        return control, ubu, lbu

