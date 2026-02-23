import casadi as cs
from .. import BaseOCClass
from . import oed_utils


class problem(BaseOCClass.super_problem):
    # model parameter
    num_meas = 1
    num_pars = 1

    s_dim = 1 + 1 * num_pars + num_pars + num_meas
    q_dim = 1
    is_inverse = False
    reg_init = 1.e-4

    def __init__(self, criterion="A"):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.criterion = criterion
        self.state_labels = [r"$x_1$"]
        self.state_labels += [r"$G_{1}$"]
        for j in range(1, self.num_pars + 1):
            self.state_labels += [r"$F_{" + str(i) + "," + str(j) + "}$"
                                  for i in range(1, j + 1)]
        self.state_labels += [r"$z_1$"]
        self.control_labels = [r"$w_1$"]

    def get_ode(self):
        # Declare model variables
        # differential states
        x = cs.MX.sym('x', 1)
        p = cs.MX.sym('p', self.num_pars)  # param. k1, k_kat, E1, E_kat, lamb
        G = cs.MX.sym('G', 1 * self.num_pars)
        F = cs.MX.sym('F', self.num_pars)
        z = cs.MX.sym('z', self.num_meas)  # constraints for measuring time
        w = cs.MX.sym('w', self.num_meas)  # measurement functions
        t = cs.MX.sym('t')

        p_fix = cs.DM([15.])

        # Model equations

        x_dot_p = 0.2 + 0.8 * t + 0.3 * (cs.sin(p * t) + cs.cos(p * t) * p * t)
        x_dot_p -= 2.5 * (cs.sin(50 * t))
        f = cs.Function('f', [x, p, cs.vertcat(t)], [x_dot_p])
        x_dot = f(x, p_fix, cs.vertcat(t))
        h = cs.Function('h', [x], [x])
        G_dot = oed_utils.get_sens_der(G, f, x, p, p_fix, cs.vertcat(t))
        F_dot = oed_utils.get_fisher_info(G, h, x, p, p_fix, w)

        z_dot = w

        control = cs.vertcat(w)
        # summarize all differential states
        s = cs.vertcat(x, G, F, z)
        s_dot = cs.vertcat(x_dot,
                           G_dot,
                           F_dot,
                           z_dot
                           )
        # Objective term
        L = 0
        ode = {'x': s, 'p': control, 'ode': s_dot, 'quad': L, 't': t}
        return ode

    def get_init(self):
        init = {}
        x_start = [0.1]
        G_start = cs.DM([0.])
        F_start = oed_utils.lower_triangular_to_vector(cs.tril(cs.DM_eye(1) * self.reg_init))
        z_start = cs.DM.zeros(self.num_meas)

        init["s_start"] = cs.vertcat(x_start, G_start, F_start, z_start)
        init["q_start"] = [0.5]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 2
        return max_t

    def start_bounds(self, start):
        x_b = [0.1]
        G_b = [0]
        F_reg = oed_utils.lower_triangular_to_vector(cs.tril(cs.DM_eye(1) * self.reg_init))
        F_b = [F_reg[i].__float__() for i in range(F_reg.numel())]
        z_b = [0] * self.num_meas

        lbs = x_b + G_b + F_b + z_b
        ubs = x_b + G_b + F_b + z_b
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [0.] * self.num_meas
        ubu = [1.] * self.num_meas
        return control, ubu, lbu

    def end_bounds(self, state):
        end_points = state[-1]
        ubs = [0.2] * self.num_meas
        lbs = [0.] * self.num_meas
        return end_points, ubs, lbs

    def objective_end(self, state):
        F_vec = state[-2]
        F_full = oed_utils.vector_to_symmetric_matrix(F_vec, self.num_pars)
        return oed_utils.oed_criterion(F_full, self.num_pars, self.criterion, self.is_inverse)

