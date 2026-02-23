import casadi as cs
from .. import BaseOCClass
from . import oed_utils


class problem(BaseOCClass.super_problem):
    x_dim = 3
    q_dim = 6
    p_dim = 3
    o_dim = 3
    s_dim = x_dim + p_dim * x_dim + p_dim * (p_dim + 1) // 2 + o_dim
    is_inverse = False
    reg_init = 1.e-2

    def __init__(self, criterion="A"):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.criterion = criterion
        self.state_labels = [r"$x_{" + str(i) + "}$" for i in range(self.x_dim)]
        self.state_labels += [r"$G_{" + str(i) + "," + str(j) + "}$"
                              for i in range(1, self.p_dim + 1)
                              for j in range(1, self.x_dim + 1)]
        for j in range(1, self.p_dim + 1):
            self.state_labels += [r"$F_{" + str(i) + "," + str(j) + "}$"
                                  for i in range(1, j + 1)]
        self.state_labels += [r"$z_{" + str(i) + "}$" for i in range(self.o_dim)]
        self.control_labels = [r"$u_{" + str(i) + "}$" for i in range(3)]
        self.control_labels += [r"$w_{" + str(i) + "}$" for i in range(self.o_dim)]

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x', self.x_dim)
        x1, x2, x3 = cs.vertsplit(x)
        p = cs.MX.sym('p', self.p_dim)
        p_fix = cs.DM([1., 2., 0.8])
        G = cs.MX.sym('G', self.p_dim * self.x_dim)
        F = cs.MX.sym('F', self.p_dim * (self.p_dim + 1) // 2)
        z = cs.MX.sym('z', self.o_dim)
        u = cs.MX.sym('u', 3)
        w = cs.MX.sym('w', self.o_dim)

        # Model equations
        c1, c2, c3 = p[0], p[1], p[2]

        x_dot_p = cs.vertcat(
            -cs.sqrt(x[0]) + c1 * u[0] + c2 * u[1] - u[2] * cs.sqrt(c3 * x[0]),
            cs.sqrt(x[0]) - cs.sqrt(x[1]),
            cs.sqrt(x[1]) - cs.sqrt(x[2]) + u[2] * cs.sqrt(c3 * x[0])
        )

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
        x_start = [2., 2., 2.]
        G_start = [0.] * self.p_dim * self.x_dim
        F_reg = oed_utils.lower_triangular_to_vector(
            cs.tril(cs.DM_eye(self.p_dim) * self.reg_init)
        )
        F_start = [F_reg[i].__float__() for i in range(F_reg.numel())]
        z_start = [0.] * self.o_dim
        init["s_start"] = cs.DM(x_start + G_start + F_start + z_start)
        init["q_start"] = [0.5] * self.q_dim
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 12
        return max_t

    def start_bounds(self, start):
        x_start = [2., 2., 2.]
        G_start = [0.] * self.p_dim * self.x_dim
        F_reg = oed_utils.lower_triangular_to_vector(
            cs.tril(cs.DM_eye(self.p_dim) * self.reg_init)
        )
        F_b = [F_reg[i].__float__() for i in range(F_reg.numel())]
        z_start = [0.] * self.o_dim
        bs = x_start + G_start + F_b + z_start
        return start, bs, bs

    def control_bounds(self, control):
        lbu = [0.] * self.q_dim
        ubu = [1.] * self.q_dim
        return control, ubu, lbu

    def control_cond(self, control):
        control_sum = control[0] + control[1] + control[2]
        lbu = [1]
        ubu = [1]
        return control_sum, ubu, lbu

    def state_bounds(self, state):
        lbs = [-cs.inf] * self.s_dim
        ubs = [cs.inf] * self.s_dim
        lbs[:self.x_dim] = [1.e-6] * self.x_dim
        return state, ubs, lbs

    def end_bounds(self, state):
        end_points = state[-self.o_dim:]
        ubs = [2.] * self.o_dim
        lbs = [0] * self.o_dim
        return end_points, ubs, lbs

    def objective_end(self, state):
        start_ind = self.x_dim + self.p_dim * self.x_dim
        end_ind = self.x_dim + self.p_dim * self.x_dim + self.p_dim * (1 + self.p_dim) // 2
        F_vec = state[start_ind:end_ind]
        F_full = oed_utils.vector_to_symmetric_matrix(F_vec, self.p_dim)
        return oed_utils.oed_criterion(F_full, self.p_dim, self.criterion, self.is_inverse)

