import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 4
    q_dim = 2
    global_controls = [1]
    time_scale_ind = 1

    # define global model parameters
    x0, y0 = 0, 1000
    yf = 900
    vx0, vxf = 13.23, 13.23
    vy0, vyf = -1.288, -1.288
    uc, rc = 2.5, 100
    c0, c1 = 0.034, 0.069662
    S, rho = 14, 1.13
    m, g = 100, 9.81
    state_indices = [i for i in range(s_dim)]
    state_scales = [1.] * s_dim
    state_scales[:2] = [1.e-2, 1.e-2]

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.state_labels = [r"$x$", r"$y$", r"$v_x$", r"$v_y$"]
        self.control_labels = [r"$c_L$", r"$t_f$"]

    def get_ode(self):
        # Declare model variables
        s = cs.MX.sym('x', self.s_dim)
        x, y, vx, vy = cs.vertsplit(s)
        u = cs.MX.sym('u', self.q_dim)
        cl, t = cs.vertsplit(u)

        # auxiliary functions
        r = (x / self.rc - 2.5)**2
        U_up = self.uc * (1 - r) * cs.exp(-r)
        w = vy - U_up
        v = cs.sqrt(vx**2 + w**2)
        D = 0.5 * (self.c0 + self.c1 * cl**2) * self.rho * self.S * v**2
        L = 0.5 * self.rho * self.S * cl * v**2

        # Model equations
        sdot = cs.vertcat(
            vx, vy,
            - (L * w + D * vx) / (self.m * v),
            (L * vx - D * w) / (self.m * v) - self.g
        )
        ode = {'x': s, 'p': u, 'ode': t * sdot, 'quad': 0}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([self.x0, self.y0, self.vx0, self.vy0])
        init["q_start"] = [0.7, 1]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 100
        return max_t

    def start_bounds(self, start):
        bs = [self.x0, self.y0, self.vx0, self.vy0]
        return start, bs, bs

    def state_bounds(self, state):
        lbs = [0, -cs.inf, 0, -cs.inf]
        ubs = [cs.inf] * self.s_dim
        return state, ubs, lbs

    def control_bounds(self, control):
        lbu = [0., 0.]
        ubu = [1.4, cs.inf]
        return control, ubu, lbu

    def end_bounds(self, state):
        lbs = [-cs.inf, self.yf, self.vxf, self.vyf]
        ubs = [cs.inf, self.yf, self.vxf, self.vyf]
        return state, ubs, lbs

    def objective_end(self, state):
        return -state[0]

