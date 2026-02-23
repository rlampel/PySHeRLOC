import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 2
    q_dim = 2
    global_controls = [1]
    time_scale_ind = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # model parameter
        m = 5
        c = 10
        # Declare model variables
        x = cs.MX.sym('x')
        v = cs.MX.sym('v')
        s = cs.vertcat(x, v)
        u = cs.MX.sym('u')
        t = cs.MX.sym('t')

        # Model equations
        sdot = cs.vertcat(
            v,
            1 / m * (u - c * x),
        )
        L = 0
        u = cs.vertcat(u, t)
        ode = {'x': s, 'p': u, 'ode': t * sdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([2., 5.])
        init["q_start"] = [0., 1]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 10
        return max_t

    def start_bounds(self, start):
        lbs = [2., 5.]
        ubs = [2., 5.]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [-5., 0.]
        ubu = [5., cs.inf]
        return control, ubu, lbu

    def end_bounds(self, state):
        bs = [0.] * self.s_dim
        return state, bs, bs

    def objective_end_control(self, state, control):
        return control[-1]
