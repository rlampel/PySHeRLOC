import casadi as cs
from .. import BaseOCClass
from . import oed_utils


class problem(BaseOCClass.super_problem):
    s_dim = 27
    q_dim = 4
    p_dim = 4
    is_inverse = False  # the Fisher Matrix is given by an ode
    reg_init = 1.e-2

    def __init__(self, criterion="A"):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.criterion = criterion

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x', 3)
        G = cs.MX.sym('G', self.p_dim * 3)
        F = cs.MX.sym('F', self.p_dim * (1 + self.p_dim) // 2)
        p = cs.MX.sym('p', self.p_dim)
        p_fix = cs.DM([0.357, 0.153, 0.633, 0.043])
        # p_fix = cs.DM([0.527, 0.054, 0.935, 0.015])
        z = cs.MX.sym('z', 2)
        w = cs.MX.sym('w', 2)
        u = cs.MX.sym('u', 2)

        # Model equations
        x1, x2, r = x[0], x[1], x[2]
        x1dot = (r - u[0] - p_fix[3]) * x1
        x2dot = -r * x1 / p_fix[2] + u[0] * (u[1] - x2)
        # reformulate dae via chain rule
        rdot = (p[0] * x2dot) * (p[1] + x2)
        rdot -= p[0] * x2 * x2dot
        rdot /= (p[1] + x2)**2
        x_dot_p = cs.vertcat(x1dot, x2dot, rdot)

        # f equations
        f = cs.Function('f', [x, p, u], [x_dot_p])
        h = cs.Function('h', [x], [cs.vertcat(x[0], x[1])])
        x_dot = f(x, p_fix, u)
        G_dot = oed_utils.get_sens_der(G, f, x, p, p_fix, u)
        F_dot = oed_utils.get_fisher_info(G, h, x, p, p_fix, w)

        # z equations
        z_dot = w

        control = cs.vertcat(u, w)
        s = cs.vertcat(x, G, F, z)
        s_dot = cs.vertcat(x_dot,
                           G_dot,
                           F_dot,
                           z_dot)
        # Objective term
        L = 0
        ode = {'x': s, 'p': control, 'ode': s_dot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        x_start = cs.DM([1., 0., 0.])
        G_start = cs.DM([0.] * self.p_dim * 3)
        reg_eye = cs.DM_eye(self.p_dim) * self.reg_init
        F_start = oed_utils.lower_triangular_to_vector(cs.tril(reg_eye))
        z_start = cs.DM([0.] * 2)
        init["s_start"] = cs.vertcat(x_start, G_start, F_start, z_start)
        init["q_start"] = [0.1, 6.] + [0.5] * 2
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 40
        return max_t

    def start_bounds(self, start):
        x_lb, x_ub = [1., 0., 0.], [10., 0., 0.]
        G_b = [0.] * self.p_dim * 3
        reg_eye = cs.DM_eye(self.p_dim) * self.reg_init
        F_reg = oed_utils.lower_triangular_to_vector(cs.tril(reg_eye))
        F_b = [F_reg[i].__float__() for i in range(F_reg.numel())]
        z_b = [0.] * 2
        lbs = x_lb + G_b + F_b + z_b
        ubs = x_ub + G_b + F_b + z_b
        return start, ubs, lbs

    def control_bounds(self, control):
        ubu = [0.2, 35.] + [1.] * 2
        lbu = [0.05, 5.] + [0.] * 2
        return control, ubu, lbu

    def end_bounds(self, state):
        end_points = state[-2:]
        ubs = [2.] * 2
        lbs = [0.] * 2
        return end_points, ubs, lbs

    def objective_end(self, state):
        start_ind = 3 + 3 * self.p_dim
        end_ind = 3 + 3 * self.p_dim + self.p_dim * (self.p_dim + 1) // 2
        F_vec = state[start_ind:end_ind]
        F_full = oed_utils.vector_to_symmetric_matrix(F_vec, self.p_dim)
        return oed_utils.oed_criterion(F_full, self.p_dim, self.criterion, self.is_inverse)

