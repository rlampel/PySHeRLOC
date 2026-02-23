import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 4
    q_dim = 3

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # model parameter
        k1, k2, k3, k4 = 2., 3., 1., 3.
        c1, c2, c3 = 1., 2., 0.8
        # Declare model variables
        x = cs.MX.sym('x', self.s_dim)
        w = cs.MX.sym('w', self.q_dim)

        dx1 = -cs.sqrt(x[0]) + c1 * w[0] + c2 * w[1] - w[2] * cs.sqrt(c3 * x[0])
        dx2 = cs.sqrt(x[0]) - cs.sqrt(x[1])
        dx3 = cs.sqrt(x[1]) - cs.sqrt(x[2]) + w[2] * cs.sqrt(c3 * x[0])
        dx4 = k1 * (x[1] - k2)**2 + k3 * (x[2] - k4)**2
        dx4 *= 5.e-2
        # Model equations
        xdot = cs.vertcat(dx1, dx2, dx3, dx4)
        L = 0
        ode = {'x': x, 'p': w, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([2., 2., 2., 0.])
        init["q_start"] = [1., 0., 0.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 12
        return max_t

    def start_bounds(self, start):
        lbs = [2., 2., 2., 0.]
        ubs = [2., 2., 2., 0.]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [0., 0., 0.]
        ubu = [1., 1., 1.]
        return control, ubu, lbu

    def control_cond(self, control):
        control_sum = control[0] + control[1] + control[2]
        lbu = [1]
        ubu = [1]
        return control_sum, ubu, lbu

    def objective_end(self, state):
        return state[-1]
