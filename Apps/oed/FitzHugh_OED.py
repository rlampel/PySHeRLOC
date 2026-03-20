import casadi as cs
from . import oed_utils
from .. import BaseOCClass


class problem(BaseOCClass.super_problem):
    q_dim = 3
    p_dim = 4
    o_dim = 2
    x_dim = 2
    s_dim = x_dim + p_dim * x_dim + p_dim * (p_dim + 1) // 2 + o_dim
    is_inverse = False  # the Fisher Matrix is given by an ode
    reg_init = 1.e-1

    def __init__(self, criterion="A"):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.criterion = criterion

    def get_ode(self):
        # given constants
        z, a, b, c = 0.25, 0.02, 0.7, -0.8

        # Declare model variables
        x = cs.MX.sym('x', 2)
        p = cs.MX.sym('p', 4)  # interested in p2 and p4
        p_fix = cs.DM([z, a, b, c])
        G = cs.MX.sym('G', self.p_dim * self.x_dim)
        F = cs.MX.sym('F', self.p_dim * (self.p_dim + 1) // 2)
        z = cs.MX.sym('z', self.o_dim)
        u = cs.MX.sym('u', 1)
        w = cs.MX.sym('w', self.o_dim)

        # Model equations
        x_dot = cs.vertcat(
            x[0] - p[0] * x[0]**3 - x[1] + u,
            p[1] * (x[0] + p[2] + p[3] * x[1])
        )

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
        x_start = cs.DM([5., -5.])
        G_start = cs.DM([0.] * self.p_dim * self.x_dim)
        F_start = oed_utils.lower_triangular_to_vector(
            cs.tril(cs.DM_eye(self.p_dim) * self.reg_init)
        )
        z_start = cs.DM([0.] * self.o_dim)
        init["s_start"] = cs.vertcat(x_start, G_start, F_start, z_start)
        init["q_start"] = [0.1] * 3
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 10
        return max_t

    def start_bounds(self, start):
        G_b = [0.] * self.p_dim * self.x_dim
        F_reg = oed_utils.lower_triangular_to_vector(
            cs.tril(cs.DM_eye(self.p_dim) * self.reg_init)
        )
        F_b = [F_reg[i].__float__() for i in range(F_reg.numel())]
        z_b = [0.] * self.o_dim

        lbs = [-5., -5.] + G_b + F_b + z_b
        ubs = [5., 5.] + G_b + F_b + z_b
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [-1] + [0.] * self.o_dim
        ubu = [0.5] + [1.] * self.o_dim
        return control, ubu, lbu

    def end_bounds(self, state):
        end_points = cs.vertcat(state[-1], state[-2])
        ubs = [2] * 2
        lbs = [0] * 2
        return end_points, ubs, lbs

    def objective_end(self, state):
        start_ind = self.x_dim + self.p_dim * self.x_dim
        end_ind = start_ind + self.p_dim * (1 + self.p_dim) // 2
        F_vec = state[start_ind:end_ind]
        F_full = oed_utils.vector_to_symmetric_matrix(F_vec, self.p_dim)
        return oed_utils.oed_criterion(F_full, self.p_dim, self.criterion, self.is_inverse)

