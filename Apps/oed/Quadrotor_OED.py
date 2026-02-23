import casadi as cs
from .. import BaseOCClass
from . import oed_utils


class problem(BaseOCClass.super_problem):
    s_dim = 23
    q_dim = 6
    p_dim = 2
    is_inverse = False  # the Fisher Matrix is given by an ode
    reg_init = 1.e-1

    def __init__(self, criterion="A"):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.criterion = criterion

    def get_ode(self):
        g, M, L, Iv = 9.8, 1.3, 0.305, 0.0605

        # Declare model variables
        x = cs.MX.sym('x', 6)
        G = cs.MX.sym('G', self.p_dim * 6)
        F = cs.MX.sym('F', self.p_dim * (self.p_dim + 1) // 2)
        p = cs.MX.sym('p', 2)
        p_fix = cs.DM([M, L])
        u = cs.MX.sym('u', 4)
        w = cs.MX.sym('w', 2)
        z = cs.MX.sym('z', 2)

        # Model equations
        x_dot_p = cs.vertcat(x[1],
                             g * cs.sin(x[4]) + u[1] * u[0] * cs.sin(x[4]) / p[0],
                             x[3],
                             g * cs.cos(x[4]) - g + u[1] * u[0] * cs.cos(x[4]) / p[0],
                             x[5],
                             - u[2] * p[1] * u[0] / Iv + u[3] * p[1] * u[0] / Iv)

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
        x_start = cs.DM([0, 0, 1, 0, 0, 0])
        G_start = cs.DM.zeros(self.p_dim * 6)
        F_start = oed_utils.lower_triangular_to_vector(
            cs.tril(cs.DM_eye(2) * self.reg_init)
        )
        z_start = cs.DM.zeros(2)
        init["s_start"] = cs.vertcat(x_start, G_start, F_start, z_start)
        init["q_start"] = [1.e-3, 1 / 3, 1 / 3, 1 / 3] + [0.5] * 2
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 7.5
        return max_t

    def start_bounds(self, start):
        x_b = [0, 0, 1, 0, 0, 0]
        G_b = [0.] * self.p_dim * 6
        F_reg = oed_utils.lower_triangular_to_vector(
            cs.tril(cs.DM_eye(2) * self.reg_init)
        )
        F_b = [F_reg[i].__float__() for i in range(F_reg.numel())]
        z_b = [0.] * 2
        lbs = x_b + G_b + F_b + z_b
        return start, lbs, lbs

    def control_bounds(self, control):
        lbu = [0] * 4 + [0.] * 2
        ubu = [1.e-3, 1, 1, 1] + [1] * 2
        return control, ubu, lbu

    def control_cond(self, control):
        qsum = control[1] + control[2] + control[3]
        lbu = [1]
        ubu = [1]
        return qsum, ubu, lbu

    def state_bounds(self, state):
        lbs = [-cs.inf, -cs.inf, 0] + [-cs.inf] * (self.s_dim - 3)
        ubs = [cs.inf] * self.s_dim
        return state, ubs, lbs

    def end_bounds(self, state):
        end_points = state[-2:]
        lb = [0] * 2
        ub = [1] * 2
        return end_points, ub, lb

    def objective_end(self, state):
        F_vec = state[18:21]
        F_full = oed_utils.vector_to_symmetric_matrix(F_vec, self.p_dim)
        return oed_utils.oed_criterion(F_full, self.p_dim, self.criterion, self.is_inverse)

