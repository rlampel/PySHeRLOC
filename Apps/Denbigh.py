import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 2
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.state_labels = [r"$x_1$", r"$x_2$"]
        self.control_labels = [r"$T$"]

    def get_ode(self):

        # Declare model variables
        x1 = cs.MX.sym('x1')
        x2 = cs.MX.sym('x2')
        x = cs.vertcat(x1, x2)
        u = cs.MX.sym('u')

        # auxiliary equations
        k_star = [1.e3, 1.e7, 1.e1, 1.e-3]
        E = [3.e3, 6.e3, 3.e3, 0.]
        R = 1.
        k = [k_star[i] * cs.exp(-E[i] / (R * u)) for i in range(4)]

        # Model equations
        xdot = cs.vertcat(-k[0] * x1 - k[1] * x1,
                          k[0] * x1 - (k[2] + k[3]) * x2)
        # Objective term
        L = -k[2] * x2
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([1., 0.])
        init["q_start"] = [273.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def state_bounds(self, state):
        lbs = [-1.e-5, 0.]
        ubs = [1., 1.]
        return state, ubs, lbs

    def get_grid_details(self):
        max_t = 1000
        return max_t

    def start_bounds(self, start):
        bs = [1., 0.]
        return start, bs, bs

    def control_bounds(self, control):
        lbu = [273.]
        ubu = [415.]
        return control, ubu, lbu

