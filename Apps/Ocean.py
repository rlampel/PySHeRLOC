import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 2
    q_dim = 2

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        S = cs.MX.sym('S')
        R = cs.MX.sym('R')
        t = cs.MX.sym('t')
        x = cs.vertcat(S, R)
        u1 = cs.MX.sym('u1')
        u2 = cs.MX.sym('u2')
        u = cs.vertcat(u1, u2)

        # constants
        rho, gamma, omega, b, mu = 0.03, 1.e-3, 0.1, 50., 0.5
        a1, a2, nu, c1, c2 = 2., 2., 1., 50., 4.e-3
        S_pre, S_0, R_0, D_L0 = 6.e2, 2.e3, 1.e4, 2.3e4

        # auxiliary equations
        U = b * u1 - mu * u1**2
        A = a1 * u2 + a2 * u2**2
        C = c1 - c2 * R
        D = nu * (0.3 * S - S_pre)**2
        D_L = D_L0 + R_0 + S_0 - R - S

        # Model equations
        Sdot = u1 - u2 - gamma * (S - omega * D_L)
        Rdot = -u1
        xdot = cs.vertcat(Sdot, Rdot)
        # Objective term
        L = -cs.exp(-rho * t) * (U - A - u1 * C - D)
        ode = {'x': x, 'p': u, 'ode': xdot, 't': t, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([2.e3, 1.e4])
        init["q_start"] = [10., 10.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 400
        return max_t

    def start_bounds(self, start):
        bs = [2.e3, 1.e4]
        return start, bs, bs

    def control_bounds(self, control):
        lbu = [0., 0.]
        ubu = [40., 40.]
        return control, ubu, lbu

    def state_bounds(self, state):
        lbs = [0., 0.]
        ubs = [1.e5, 1.e5]
        return state, ubs, lbs

