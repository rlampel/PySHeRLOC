import casadi as cs
from . import oed_utils
from .. import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 11
    q_dim = 3
    p_dim = 2
    is_inverse = False  # the Fisher Matrix is given by an ode
    reg_init = [1.e-1, 0., 1.e-1]

    def __init__(self, criterion="A"):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.criterion = criterion

    def get_ode(self):
        # given constants
        p1, p3 = 1, 1
        p5, p6 = 0.4, 0.2

        # Declare model variables
        x = cs.MX.sym('x', 2)
        p = cs.MX.sym('p', 2)  # interested in p2 and p4
        p_fix = cs.DM([1., 1.])
        G = cs.MX.sym('G', 4)
        F = cs.MX.sym('F', 3)
        z = cs.MX.sym('z', 2)
        u = cs.MX.sym('u')
        w = cs.MX.sym('w', 2)

        # Model equations
        x_dot = cs.vertcat(p1 * x[0] - p[0] * x[0] * x[1] - p5 * u * x[0],
                           -p3 * x[1] + p[1] * x[0] * x[1] - p6 * u * x[1])

        # f equations
        f = cs.Function('f', [x, p, u], [x_dot])
        h = cs.Function('h', [x], [x])
        G_dot = oed_utils.get_sens_der(G, f, x, p, p_fix, u)
        F_dot = oed_utils.get_fisher_info(G, h, x, p, p_fix, w)

        # z equations
        z_dot = w

        control = cs.vertcat(u, w)
        s = cs.vertcat(x, G, F, z)
        s_dot = cs.vertcat(f(x, p_fix, u),
                           G_dot,
                           F_dot,
                           z_dot)
        # Objective term
        L = 0
        ode = {'x': s, 'p': control, 'ode': s_dot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0.5, 0.7] + [0.] * 4 + self.reg_init + [0., 0.])
        init["q_start"] = [0.5] * 3
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 12
        return max_t

    def start_bounds(self, start):
        lbs = [0.5, 0.7] + [0] * 4 + self.reg_init + [0] * 2
        ubs = [0.5, 0.7] + [0] * 4 + self.reg_init + [0] * 2
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [0] * 3
        ubu = [1] * 3
        return control, ubu, lbu

    def end_bounds(self, state):
        end_points = cs.vertcat(state[-1], state[-2])
        ubs = [4] * 2
        lbs = [0] * 2
        return end_points, ubs, lbs

    def objective_end(self, state):
        F_vec = state[6:9]
        F_full = oed_utils.vector_to_symmetric_matrix(F_vec, self.p_dim)
        return oed_utils.oed_criterion(F_full, self.p_dim, self.criterion, self.is_inverse)

