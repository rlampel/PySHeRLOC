import casadi as cs
from .. import BaseOCClass
from . import oed_utils


class problem(BaseOCClass.super_problem):
    s_dim = 11
    q_dim = 4
    p_dim = 2
    is_inverse = False  # the Fisher Matrix is given by an ode
    reg_init = [1.e-1, 0., 1.e-1]

    def __init__(self, criterion="A"):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.criterion = criterion

    def get_ode(self):
        # given constants
        F_in, F_out = 100, 100
        L = 6.6
        T_in = 350.
        r = 2.19
        E = 72740.
        R = 8.314463
        rho = 1.
        C_p = 239.
        delta_H = -5.e4
        A_r = cs.pi * r**2

        # Declare model variables
        x = cs.MX.sym('x', 2)
        p = cs.MX.sym('p', 2)  # interested in U and kr0
        p_fix = cs.DM([549.36, 7.2e10])
        G = cs.MX.sym('G', 4)
        F = cs.MX.sym('F', 3)
        z = cs.MX.sym('z', 2)
        u = cs.MX.sym('u', 2)
        w = cs.MX.sym('w', 2)

        # Model equations
        c_dot = (F_in * u[0] - F_out * x[0]) / (A_r * L)
        c_dot -= p[1] * cs.exp(-E / (R * x[1])) * x[0]
        T_dot = (F_in * T_in - F_out * x[1]) / (A_r * L)
        T_dot -= delta_H / (rho * C_p) * p[1] * cs.exp(-E / (R * x[1])) * x[0]
        T_dot += 2 * p[0] / (r * rho * C_p) * (u[1] - x[1])

        x_dot = cs.vertcat(c_dot,
                           T_dot)

        # f equations
        f = cs.Function('f', [x, p, u], [x_dot])
        h = cs.Function('h', [x], [x])
        G_dot = oed_utils.get_sens_der(G, f, x, p, p_fix, u)
        F_dot = oed_utils.get_fisher_info(G, h, x, p, p_fix, w)

        # z equations
        z_dot = w

        control = cs.vertcat(w, u)
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
        init["s_start"] = cs.DM([0.877, 323.] + [0.] * 4 + self.reg_init + [0., 0.])
        init["q_start"] = [1.] * 2 + [0.9, 300.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 20
        return max_t

    def start_bounds(self, start):
        lbs = [0.877, 323.] + [0.] * 4 + self.reg_init + [0., 0.]
        ubs = [0.877, 323.] + [0.] * 4 + self.reg_init + [0., 0.]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [0.] * 2 + [0.8, 288.]
        ubu = [1.] * 2 + [1., 353.]
        return control, ubu, lbu

    def state_bounds(self, state):
        lbs = [0.8, 298.] + [-cs.inf] * 9
        ubs = [1., 333.] + [cs.inf] * 9
        return state, ubs, lbs

    def end_bounds(self, state):
        end_points = cs.vertcat(state[-1], state[-2])
        ubs = [1.25] * 2
        lbs = [0.] * 2
        return end_points, ubs, lbs

    def objective_end(self, state):
        F_vec = state[6:9]
        F_full = oed_utils.vector_to_symmetric_matrix(F_vec, self.p_dim)
        return oed_utils.oed_criterion(F_full, self.p_dim, self.criterion, self.is_inverse)

