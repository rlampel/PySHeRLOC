import casadi as cs
from .. import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 2
    q_dim = 1
    p_dim = 1
    is_inverse = False  # the Fisher Matrix is given by an ode
    reg_init = 1.e-3

    # model parameter
    p = 1
    x0 = 1.23

    def __init__(self, criterion="A"):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.criterion = criterion
        self.state_labels = [r"$x$", r"$G$"]
        self.control_labels = [r"$w$"]

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x')
        G = cs.MX.sym('G')
        w = cs.MX.sym('w')  # dummy variable

        # Model equations
        xdot = x * (x + self.p) + w

        # G equations
        G = cs.MX.sym('G')
        Gdot = (self.p + 2 * x) * G + x

        x = cs.vertcat(x, G)
        xdot = cs.vertcat(xdot,
                          Gdot,
                          )
        # Objective term
        L = 0
        ode = {'x': x, 'p': w, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([self.x0, self.reg_init])
        init["q_start"] = [0.1]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 0.6
        return max_t

    def control_bounds(self, control):
        cb = [0.]
        return control, cb, cb

    def start_bounds(self, start):
        lbs = [0., self.reg_init]
        ubs = [200., self.reg_init]
        return start, ubs, lbs

    def state_bounds(self, state):
        lbs = [0., -cs.inf]
        ubs = [200., cs.inf]
        return state, ubs, lbs

    def objective_end(self, state):
        return 1. / state[-1]**2
