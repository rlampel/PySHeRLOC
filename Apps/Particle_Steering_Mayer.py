import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 4
    q_dim = 2
    global_controls = [1]
    time_scale_ind = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # model parameter
        a = 100
        # Declare model variables
        x1 = cs.MX.sym('x1')
        x2 = cs.MX.sym('x2')
        # first derivatives
        dx1 = cs.MX.sym('dx1')
        dx2 = cs.MX.sym('dx2')
        x = cs.vertcat(x1, x2, dx1, dx2)
        u = cs.MX.sym('u')
        t = cs.MX.sym('t')

        # Model equations
        xdot = cs.vertcat(dx1,
                          dx2,
                          a * cs.cos(u),
                          a * cs.sin(u)
                          )
        L = 0
        u = cs.vertcat(u, t)
        ode = {'x': x, 'p': u, 'ode': t * xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0.] * self.s_dim)
        init["q_start"] = [0., 1]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 10
        return max_t

    def start_bounds(self, start):
        lbs = [0.] * self.s_dim
        ubs = [0.] * self.s_dim
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [-cs.pi / 2, 0]
        ubu = [cs.pi / 2, cs.inf]
        return control, ubu, lbu

    def end_bounds(self, state):
        lbs = [-cs.inf, 5, 45, 0]
        ubs = [cs.inf, 5, 45, 0]
        return state, ubs, lbs

    def objective_end_control(self, state, control):
        return control[-1]

