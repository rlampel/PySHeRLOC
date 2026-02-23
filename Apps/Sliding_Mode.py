import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 1
    q_dim = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x')
        w = cs.MX.sym('w')

        # Model equations
        xdot = cs.vertcat(
            1 - 2 * w
        )

        # Objective term
        L = x**2 + (1 - (-1 + 2 * w)**2)**2
        ode = {'x': x, 'p': w, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0.])
        init["q_start"] = [0.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = float(cs.sqrt(2))
        return max_t

    def start_bounds(self, start):
        lbs = [0.]
        ubs = [0.]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [0]
        ubu = [1]
        return control, ubu, lbu

    def end_bounds(self, state):
        lbs = [0.]
        ubs = [0.]
        return state, ubs, lbs

