import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 2
    q_dim = 2
    time_scale_ind = 1
    global_controls = [1]

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x')
        v = cs.MX.sym('v')
        s = cs.vertcat(x, v)
        u = cs.MX.sym('u')
        t = cs.MX.sym('t')
        control = cs.vertcat(u, t)

        # Model equations
        xdot = cs.vertcat(v,
                          1.e-3 * u - 2.5e-3 * cs.cos(3 * x))

        # Objective term
        L = 0
        ode = {'x': s, 'p': control, 'ode': t * xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([-0.5, 0.])
        init["q_start"] = [0., 1.2]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 100
        return max_t

    def start_bounds(self, start):
        bs = [-0.5, 0.]
        return start, bs, bs

    def control_bounds(self, control):
        lbu = [-1, 0.01]
        ubu = [1, 100]
        return control, ubu, lbu

    def end_bounds(self, state):
        lbs = [0.5, 0]
        ubs = [0.5, cs.inf]
        return state, ubs, lbs

    def objective_end_control(self, state, control):
        return control[-1]

