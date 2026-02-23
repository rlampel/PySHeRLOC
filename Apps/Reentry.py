import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    global_controls = [1]
    time_scale_ind = 1
    s_dim = 3
    q_dim = 2

    R = 209
    beta = 4.26
    rho_0 = 2.704e-3
    g = 3.2172e-4
    Sm = 53200

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x', 3)
        v, gamma, xi = cs.vertsplit(x, 1)
        u = cs.MX.sym('u')
        t = cs.MX.sym('t')

        # auxiliary equations
        C_W = 1.175 - 0.9 * cs.cos(u)
        C_A = 0.6 * cs.sin(u)
        rho = self.rho_0 * cs.exp(-self.beta * self.R * xi)

        vdot = -self.Sm * 0.5 * rho * v**2 * C_W
        vdot -= self.g * cs.sin(gamma) / (1 + xi)**2
        gammadot = self.Sm * 0.5 * rho * v * C_A + v * cs.cos(gamma) / (self.R * (1 + xi))
        gammadot -= self.g * cs.cos(gamma) / (v + (1 + xi)**2)
        xidot = v * cs.sin(gamma) / self.R

        # Model equations
        xdot = cs.vertcat(
            vdot, gammadot, xidot
        )
        w = cs.vertcat(u, t)

        # Objective term
        L = 10 * v**3 * cs.sqrt(rho)
        ode = {'x': x, 'p': w, 'ode': xdot * t * 100, 'quad': L * t * 100}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0.36, -8.1 * (cs.pi / 180), 4. / self.R])
        init["q_start"] = [0., 2.25]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 1
        return max_t

    def start_bounds(self, start):
        bs = [0.36, -8.1 * (cs.pi / 180), 4. / self.R]
        return start, bs, bs

    def control_bounds(self, control):
        lbu = [-cs.pi / 2, 2.]
        ubu = [cs.pi / 2, 3.]
        return control, ubu, lbu

    def end_bounds(self, state):
        bs = [0.27, 0., 2.5 / self.R]
        return state, bs, bs

