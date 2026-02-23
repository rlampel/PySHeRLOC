import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 3
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x', self.s_dim)
        u = cs.MX.sym('u', self.q_dim)

        # Model equations
        x1dot = u
        x2dot = x[0] * (1 + u**2)**0.5
        x3dot = (1 + u**2)**0.5

        xdot = cs.vertcat(x1dot, x2dot, x3dot)
        # Objective term
        L = 0
        ode = {'x': x, 'p': u, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([1, 0, 0])
        init["q_start"] = [0]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 1
        return max_t

    def start_bounds(self, start):
        a = 1
        lbs = [a] + [0] * 2
        ubs = [a] + [0] * 2
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [-10]
        ubu = [20]
        return control, ubu, lbu

    def state_bounds(self, state):
        lbs = [0] * 3
        ubs = [10] * 3
        return state, ubs, lbs

    def end_bounds(self, state):
        b = 3
        Lp = 4
        temp_state = cs.vertcat(state[0], state[2])
        lbs = [b, Lp]
        ubs = [b, Lp]
        return temp_state, ubs, lbs

    def objective_end(self, state):
        return state[1]

