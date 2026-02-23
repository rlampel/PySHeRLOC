import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 6
    q_dim = 4

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        g, M, L, Iv = 9.8, 1.3, 0.305, 0.0605

        # Declare model variables
        x = cs.MX.sym('x', self.s_dim)
        u = cs.MX.sym('u', self.q_dim)

        # Model equations
        x1dot = x[1]
        x2dot = g * cs.sin(x[4]) + u[1] * u[0] * cs.sin(x[4]) / M
        x3dot = x[3]
        x4dot = g * cs.cos(x[4]) - g + u[1] * u[0] * cs.cos(x[4]) / M
        x5dot = x[5]
        x6dot = - u[2] * L * u[0] / Iv + u[3] * L * u[0] / Iv

        xdot = cs.vertcat(x1dot, x2dot, x3dot, x4dot, x5dot, x6dot)

        # Objective term
        L = 5 * (u[0])**2
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0, 0, 1, 0, 0, 0])
        init["q_start"] = [1.e-3, 0.3, 0.3, 0.3]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 7.5
        return max_t

    def start_bounds(self, start):
        lbs = [0, 0, 1, 0, 0, 0]
        ubs = [0, 0, 1, 0, 0, 0]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [0] * 4
        ubu = [1.e-3, 1, 1, 1]
        return control, ubu, lbu

    def control_cond(self, control):
        qsum = control[1] + control[2] + control[3]
        lbu = [1]
        ubu = [1]
        return qsum, ubu, lbu

    def state_bounds(self, state):
        lbs = [-cs.inf, -cs.inf, 0, -cs.inf, -cs.inf, -cs.inf]
        ubs = [cs.inf] * self.s_dim
        return state, ubs, lbs

    def objective_end(self, state):
        curr_obj = 5 * (state[0] - 6)**2
        curr_obj += 5 * (state[2] - 1)**2
        curr_obj += (cs.sin(0.5 * state[4]))**2
        return curr_obj
