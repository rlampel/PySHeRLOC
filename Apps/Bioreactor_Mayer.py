
import casadi as cs

from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 4
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # model constants
        D = 0.15
        Ki = 22
        Km = 1.2

        Pm = 50
        Yxs = 0.4
        alpha = 2.2
        beta = 0.2
        mym = 0.48

        # Declare model variables
        X = cs.MX.sym('X')
        S = cs.MX.sym('S')
        P = cs.MX.sym('P')
        Mayer = cs.MX.sym('Mayer')
        u = cs.MX.sym('u', self.q_dim)

        # auxiliary equations
        my = mym * (1 - P / Pm) * S / (Km + S + S**2 / Ki)

        # Model equations
        Xdot = -D * X + my * X
        Sdot = D * (u - S) - my / Yxs * X
        Pdot = -D * P + (alpha * my + beta) * X
        Mayerdot = D * (u - P)**2

        x = cs.vertcat(X, S, P, Mayer)
        xdot = cs.vertcat(Xdot, Sdot, Pdot, Mayerdot)

        # Objective term
        L = 0
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([6.5, 12, 22, 0])
        init["q_start"] = [30]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 48
        return max_t

    def start_bounds(self, start):
        lbs = [6.5, 12, 22, 0]
        ubs = [6.5, 12, 22, 0]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [28.7]
        ubu = [40.0]
        return control, ubu, lbu

    def objective_end(self, state):
        return state[-1]

