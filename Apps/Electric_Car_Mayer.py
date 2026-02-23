import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 4
    q_dim = 1

    # Define parameters
    Kr = 10
    rho = 1.293
    Cx = 0.4
    S = 2
    r = 0.33
    Kf = 0.03
    Km = 0.27
    Rm = 0.03
    Lm = 0.05
    M = 250
    g = 9.81
    Valim = 150
    Rbat = 0.05

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        Valim, Rm, Km, Lm, S = self.Valim, self.Rm, self.Km, self.Lm, self.S
        Kr, M, Kf, r, g, rho = self.Kr, self.M, self.Kf, self.r, self.g, self.rho
        Cx, Rbat = self.Cx, self.Rbat
        # Declare model variables
        x0 = cs.MX.sym('x0')
        x1 = cs.MX.sym('x1')
        x2 = cs.MX.sym('x2')
        x3 = cs.MX.sym('x3')
        x = cs.vertcat(x0, x1, x2, x3)
        u = cs.MX.sym('u')

        # Model equations
        dx0 = (Valim * u - Rm * x0 - Km * x1) / Lm
        dx1 = (Kr**2) / (M * r**2)
        dx1 = dx1 * (Km * x0 - r / Kr * (M * g * Kf + 0.5 * rho * S * Cx * r**2 / Kr**2 * x1**2))
        dx2 = r / Kr * x1
        dx3 = Valim * u * x0 + Rbat * x0**2
        xdot = cs.vertcat(dx0,
                          dx1,
                          dx2,
                          dx3 * 1.e-3
                          )
        L = 0
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0., 0., 0., 0.])
        init["q_start"] = [0.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 10
        return max_t

    def start_bounds(self, start):
        lbs = [0., 0., 0., 0.]
        ubs = [0., 0., 0., 0.]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [-1]
        ubu = [1]
        return control, ubu, lbu

    def state_bounds(self, state):
        lbs = [-150] + [-cs.inf] * 3
        ubs = [150] + [cs.inf] * 3
        return state, ubs, lbs

    def end_bounds(self, state):
        lbs = [-cs.inf, -cs.inf, 100, -cs.inf]
        ubs = [cs.inf, cs.inf, 100, cs.inf]
        return state, ubs, lbs

    def objective_end(self, state):
        return state[-1]

