import casadi as cs
from .. import BaseOCClass
from . import oed_utils


class problem(BaseOCClass.super_problem):
    # model parameter
    num_meas = 1
    num_pars = 5

    s_dim = 4 + 4 * num_pars + num_pars * 3 + num_meas + 1
    q_dim = 3
    is_inverse = False
    reg_init = 1.e-4

    # State variables
    na1 = 1.
    na2 = 1.
    na3 = 0.
    na4 = 2.
    ckat = 1.
    # fixed parameters
    k1 = 0.01
    k_kat = 0.1
    E1 = 6.e4
    E_kat = 4.e4
    lamb = 0.25
    # constants
    M1 = 0.1362
    M2 = 0.09806
    M3 = 0.23426
    M4 = 0.236
    R = 8.314
    T_ref = 293.
    sigma = 1.
    theta = 20.

    def __init__(self, criterion="A"):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.criterion = criterion

    def get_ode(self):
        # Declare model variables
        # differential states
        x = cs.MX.sym('x', 4)
        p = cs.MX.sym('p', self.num_pars)  # param. k1, k_kat, E1, E_kat, lamb
        G = cs.MX.sym('G', 4 * self.num_pars)
        F = cs.MX.sym('F', self.num_pars * 3)  # (5 * (5 + 1) / 2)
        z = cs.MX.sym('z', self.num_meas)  # constraints for measuring time
        w = cs.MX.sym('w', self.num_meas)  # measurement functions
        u = cs.MX.sym('u', 2)  # other controls (ckat and theta)
        t = cs.MX.sym('t')

        p_fix = cs.DM([self.k1, self.k_kat, self.E1, self.E_kat, self.lamb])

        # Model equations
        m_total = x[0] * self.M1 + x[1] * self.M2
        m_total += x[3] * self.M4

        T = u[1] + 273.
        k = p[0] * cs.exp(-p[2] * (1 / T - 1 / self.T_ref))
        k += p[1] * u[0] * cs.exp(-p[4] * t) * cs.exp(-p[3] / self.R * (1 / T - 1 / self.T_ref))
        x_dot_p = cs.vertcat(-k * (x[0] * x[1]) / m_total,
                             -k * (x[0] * x[1]) / m_total,
                             k * (x[0] * x[1]) / m_total,
                             0)
        f = cs.Function('f', [x, p, cs.vertcat(t, u)], [x_dot_p])
        x_dot = f(x, p_fix, cs.vertcat(t, u))
        h = cs.Function('h', [x], [x[2] * self.M3 * 100 / m_total])
        G_dot = oed_utils.get_sens_der(G, f, x, p, p_fix, cs.vertcat(t, u))
        F_dot = oed_utils.get_fisher_info(G, h, x, p, p_fix, w)

        z_dot = w
        t_dot = 1

        control = cs.vertcat(w, u)
        # summarize all differential states
        s = cs.vertcat(x, G, F, z, t)
        s_dot = cs.vertcat(x_dot,
                           G_dot,
                           F_dot,
                           z_dot,
                           t_dot)
        # Objective term
        L = 0
        ode = {'x': s, 'p': control, 'ode': s_dot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        x_start = [self.na1, self.na2, self.na3, self.na4]
        G_start = cs.DM([0.] * 4 * self.num_pars)
        F_start = oed_utils.lower_triangular_to_vector(cs.tril(cs.DM_eye(5) * self.reg_init))
        z_start = cs.DM.zeros(self.num_meas)
        t_start = cs.DM.zeros(1)

        init["s_start"] = cs.vertcat(x_start, G_start, F_start, z_start, t_start)
        init["q_start"] = [0.5] * self.num_meas + [0., 20.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 20
        return max_t

    def start_bounds(self, start):
        x_ub = [10., 10., 9., 10.]
        x_lb = [0., 0., 0.4, 0.]
        G_b = [0] * 4 * 5
        F_reg = oed_utils.lower_triangular_to_vector(cs.tril(cs.DM_eye(5) * self.reg_init))
        F_b = [F_reg[i].__float__() for i in range(F_reg.numel())]
        z_b = [0] * self.num_meas
        t_b = [0]

        lbs = x_lb + G_b + F_b + z_b + t_b
        ubs = x_ub + G_b + F_b + z_b + t_b
        return start, ubs, lbs

    def custom_start_constraints(self, start):
        lbs = [0.1, 0.1]
        ubs = [6., 0.7]
        mass_sum = start[0] * self.M1 + start[1] * self.M2
        mass_sum += start[3] * self.M4
        active_mass = start[0] * self.M1 + start[1] * self.M2
        active_mass /= mass_sum
        return cs.vertcat(mass_sum, active_mass), ubs, lbs

    def control_bounds(self, control):
        lbu = [0.] * self.num_meas + [0., 20.]
        ubu = [1.] * self.num_meas + [10., 100.]
        return control, ubu, lbu

    def end_bounds(self, state):
        end_points = state[-2]
        ubs = [4.] * self.num_meas
        lbs = [0.] * self.num_meas
        return end_points, ubs, lbs

    def objective_end(self, state):
        F_vec = state[24:39]
        F_full = oed_utils.vector_to_symmetric_matrix(F_vec, self.num_pars)
        return oed_utils.oed_criterion(F_full, self.num_pars, self.criterion, self.is_inverse)

