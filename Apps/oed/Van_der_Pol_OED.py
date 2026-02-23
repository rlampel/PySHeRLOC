import casadi as cs
from .. import BaseOCClass
from . import oed_utils


class problem(BaseOCClass.super_problem):
    s_dim = 11
    q_dim = 3
    p_dim = 2
    is_inverse = False
    reg_init = [1.e-3, 0., 1.e-3]
    state_indices = [i for i in range(2)]
    state_scales = [1.] * s_dim
    state_scales[1] = 0.1

    def __init__(self, criterion="A"):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.criterion = criterion
        self.state_labels = [r"$x_1$", r"$x_2$"]
        '''
        self.state_labels += [r"$G_{" + str(i) + "," + str(j) + "}$"
                              for i in range(1, self.p_dim + 1)
                              for j in range(1, 2 + 1)]
        for j in range(1, self.p_dim + 1):
            self.state_labels += [r"$F_{" + str(i) + "," + str(j) + "}$"
                                  for i in range(1, j + 1)]
        self.state_labels += [r"$z_1$", r"$z_2$"]
        '''
        self.control_labels = [r"$u$", r"$w_1$", r"$w_2$"]

    def get_ode(self):
        # Declare model variables
        x1 = cs.MX.sym('x1')
        x2 = cs.MX.sym('x2')
        p = cs.MX.sym('p', 2)
        p_fix = cs.DM([1., 1.])
        x = cs.vertcat(x1, x2)
        G = cs.MX.sym('G', self.p_dim * 2)
        F = cs.MX.sym('F', self.p_dim * (self.p_dim + 1) // 2)
        z = cs.MX.sym('z', 2)
        u = cs.MX.sym('u')
        w = cs.MX.sym('w', 2)

        # Model equations
        x_dot_p = cs.vertcat((p[0] - x2**2) * x1 - x2 + u, p[1] + x1)

        # f equations
        f = cs.Function('f', [x, p, u], [x_dot_p])
        h = cs.Function('h', [x], [x])
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
        init["s_start"] = cs.DM([0., 1.] + [0.] * self.p_dim * 2 + self.reg_init + [0., 0.])
        init["q_start"] = [0.5] * 3
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 10
        return max_t

    def start_bounds(self, start):
        lbs = [0., 1.] + [0.] * self.p_dim * 2 + self.reg_init + [0., 0.]
        ubs = [0., 1.] + [0.] * self.p_dim * 2 + self.reg_init + [0., 0.]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [-1.] + [0.] * 2
        ubu = [1.] * 3
        return control, ubu, lbu

    def end_bounds(self, state):
        end_points = state[-2:]
        ubs = [2] * 2
        lbs = [0] * 2
        return end_points, ubs, lbs

    def state_bounds(self, state):
        lbs = [-0.5, -cs.inf] + [-cs.inf] * 9
        ubs = [cs.inf, cs.inf] + [cs.inf] * 9
        return state, ubs, lbs

    def objective_end(self, state):
        F_vec = state[6:9]
        F_full = oed_utils.vector_to_symmetric_matrix(F_vec, self.p_dim)
        return oed_utils.oed_criterion(F_full, self.p_dim, self.criterion, self.is_inverse)

