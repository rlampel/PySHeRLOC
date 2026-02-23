import casadi as cs
from .. import BaseOCClass
from . import oed_utils


class problem(BaseOCClass.super_problem):
    s_dim = 12
    q_dim = 1
    p_dim = 3
    is_inverse = False
    reg_init = [1.e-1, 0., 0, 1.e-1, 0, 1.e-1]
    state_indices = [i for i in range(1, 4)]
    state_scales = [1.] * s_dim
    state_scales[1] = 0.01

    def __init__(self, criterion="A"):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.criterion = criterion
        self.state_labels = []  # [r"$x_1$"]
        self.state_labels += [r"$G_{" + str(i) + "," + str(j) + "}$"
                              for i in range(1, 2)
                              for j in range(1, self.p_dim + 1)]
        '''
        for j in range(1, self.p_dim + 1):
            self.state_labels += [r"$F_{" + str(i) + "," + str(j) + "}$"
                                  for i in range(1, j + 1)]
        self.state_labels += [r"$z_1$", r"$z_2$", r"$z_3$"]
        '''
        self.control_labels = [r"$w_1$", r"$w_2$", r"$w_3$"]

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x')
        p = cs.MX.sym('p', 3)
        p_fix = cs.DM([0.05884, 4.298, 21.8])
        G = cs.MX.sym('G', self.p_dim * 1)
        F = cs.MX.sym('F', self.p_dim * (self.p_dim + 1) // 2)
        z = cs.MX.sym('z', 1)
        w = cs.MX.sym('w', 1)
        t = cs.MX.sym('t')

        # Model equations
        # theta3 * (-theta1 * exp(-theta1 * t) + theta2 * exp(-theta2 * t))
        x_dot_p = p[2] * (-p[0] * cs.exp(-p[0] * t) + p[1] * cs.exp(-p[1] * t))

        # f equations
        f = cs.Function('f', [x, p, t], [x_dot_p])
        h = cs.Function('h', [x], [x])
        x_dot = f(x, p_fix, t)
        G_dot = oed_utils.get_sens_der(G, f, x, p, p_fix, t)
        F_dot = oed_utils.get_fisher_info(G, h, x, p, p_fix, w)

        # z equations
        z_dot = w
        t_dot = 1

        control = cs.vertcat(w)
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
        init["s_start"] = cs.DM([0.] + [0.] * self.p_dim * 1 + self.reg_init + [0., 0.])
        init["q_start"] = [0.5] * 1
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 40
        return max_t

    def start_bounds(self, start):
        lbs = [0.] + [0.] * self.p_dim * 1 + self.reg_init + [0., 0.]
        ubs = [0.] + [0.] * self.p_dim * 1 + self.reg_init + [0., 0.]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [0.] * 1
        ubu = [1.] * 1
        return control, ubu, lbu

    def end_bounds(self, state):
        end_points = state[-2]
        ubs = [5] * 1
        lbs = [0] * 1
        return end_points, ubs, lbs

    def objective_end(self, state):
        F_vec = state[4:10]
        F_full = oed_utils.vector_to_symmetric_matrix(F_vec, self.p_dim)
        return oed_utils.oed_criterion(F_full, self.p_dim, self.criterion, self.is_inverse)

