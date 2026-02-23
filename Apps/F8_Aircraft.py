import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 3
    q_dim = 2
    global_controls = [1]
    time_scale_ind = 1

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Declare model variables
        x0 = cs.MX.sym('x0')
        x1 = cs.MX.sym('x1')
        x2 = cs.MX.sym('x2')
        x = cs.vertcat(x0, x1, x2)
        w = cs.MX.sym('w')
        t = cs.MX.sym('t')
        xi = 0.05236

        # Model equations
        dx0 = -0.877 * x0 + x2 - 0.088 * x0 * x2 + 0.47 * x0**2 - 0.019 * x1**2
        dx0 += - x0**2 * x2 + 3.846 * x0**3
        dx0 += 0.215 * xi - 0.28 * x0**2 * xi + 0.47 * x0 * xi**2 - 0.63 * xi**3
        dx0 += - (0.215 * xi - 0.28 * x0**2 * xi - 0.63 * xi**3) * 2 * w
        dx1 = x2
        dx2 = -4.208 * x0 - 0.396 * x2 - 0.47 * x0**2 - 3.564 * x0**3
        dx2 += 20.967 * xi - 6.265 * x0**2 * xi + 46. * x0 * xi**2 - 61.4 * xi**3
        dx2 += -(20.967 * xi - 6.265 * x0**2 * xi - 61.4 * xi**3) * 2 * w

        xdot = cs.vertcat(dx0, dx1, dx2)
        L = t
        u = cs.vertcat(w, t)
        ode = {'x': x, 'p': u, 'ode': t * xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0.4655, 0., 0.])
        init["q_start"] = [0., 1.]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 10
        return max_t

    def start_bounds(self, start):
        lbs = [0.4655, 0., 0.]
        ubs = [0.4655, 0., 0.]
        return start, ubs, lbs

    def control_bounds(self, control):
        # lbu = [-0.05236, 0.]
        lbu = [0, 1.e-3]
        ubu = [1, cs.inf]
        # ubu = [0.05236, cs.inf]
        return control, ubu, lbu

    def end_bounds(self, state):
        lbs = [0., 0., 0.]
        ubs = [0., 0., 0.]
        return state, ubs, lbs

