import casadi as cs
from .. import BaseOCClass
from . import oed_utils


class problem(BaseOCClass.super_problem):
    s_dim = 4
    q_dim = 2
    p_dim = 1
    is_inverse = False  # the Fisher Matrix is given by an ode
    reg_init = 1.e-2

    def __init__(self, criterion="A"):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.criterion = criterion
        self.state_labels = [r"$x$", r"$G$", r"$F$", r"$z$"]
        self.control_labels = [r"$u$", r"$w$"]

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x')
        G = cs.MX.sym('G')
        F = cs.MX.sym('F')
        p = cs.MX.sym('p')
        p_fix = cs.DM([0.8])
        z = cs.MX.sym('z')
        u = cs.MX.sym('u')
        w = cs.MX.sym('w')

        # Model equations
        x_dot = -x**3 + u * p

        # f equations
        f = cs.Function('f', [x, p, u], [x_dot])
        h = cs.Function('h', [x], [x])
        G_dot = oed_utils.get_sens_der(G, f, x, p, p_fix, u)
        F_dot = oed_utils.get_fisher_info(G, h, x, p, p_fix, w)

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
        init["s_start"] = cs.DM([1.] + [0., self.reg_init, 0])
        init["q_start"] = [0.0] + [0.25]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 10
        return max_t

    def start_bounds(self, start):
        lbs = [1.] + [0., self.reg_init, 0]
        ubs = [1.] + [0., self.reg_init, 0]
        return start, ubs, lbs

    def control_bounds(self, control):
        ubs = [10, 1]
        lbs = [0, 0]
        return control, ubs, lbs

    def end_bounds(self, state):
        end_points = cs.vertcat(state[0], state[-1])
        lbs = [1.5] + [0.]
        ubs = [1.5] + [2.5]
        return end_points, ubs, lbs

    def objective_end(self, state):
        F = state[2]
        return oed_utils.oed_criterion(F, self.p_dim, self.criterion, self.is_inverse)

