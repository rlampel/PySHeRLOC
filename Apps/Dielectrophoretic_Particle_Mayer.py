import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 2
    q_dim = 2
    global_controls = [1]
    time_scale_ind = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.control_labels = [r"$u$", r"$t_f$"]

    def get_ode(self):
        # model parameters
        alpha = -0.75
        c = 1
        # Declare model variables
        x1 = cs.MX.sym('x1')
        x2 = cs.MX.sym('x2')
        x = cs.vertcat(x1, x2)
        u = cs.MX.sym('u')
        t = cs.MX.sym('t')

        # Model equations
        xdot = cs.vertcat(
            x[1] * u + alpha * u**2,
            -c * x[1] + u
        )
        L = 0
        u = cs.vertcat(u, t)
        ode = {'x': x, 'p': u, 'ode': t * xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([1., 0.])
        init["q_start"] = [0.5, 1.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 10
        return max_t

    def start_bounds(self, start):
        bs = [1., 0.]
        return start, bs, bs

    def control_bounds(self, control):
        lbu = [-1., 1.e-3]
        ubu = [1., cs.inf]
        return control, ubu, lbu

    def end_bounds(self, state):
        lbs = [2., -cs.inf]
        ubs = [2., cs.inf]
        return state, ubs, lbs

    def objective_end_control(self, state, control):
        return control[-1]
