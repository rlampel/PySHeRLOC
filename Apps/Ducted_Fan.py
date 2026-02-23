import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 6
    q_dim = 3
    global_controls = [2]
    time_scale_ind = 2

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.state_labels = [r"$x_1$", r"$v_1$", r"$x_2$", r"$v_2$",
                             r"$\alpha$", r"$v_\alpha$"]
        self.control_labels = [r"$u_1$", r"$u_2$", r"$t_\mathrm{f}$"]

    def get_ode(self):
        # given model constants
        m = 2.2
        J = 0.05
        r = 0.2
        mg = 4

        # Declare model variables
        x = cs.MX.sym('x', self.s_dim)
        x1, v1, x2, v2, a, va = cs.vertsplit(x)
        u = cs.MX.sym('u', 2)
        u1, u2 = cs.vertsplit(u)
        t = cs.MX.sym('t')

        # Model equations
        xdot = cs.vertcat(
            v1,
            1 / m * (u1 * cs.cos(a) - u2 * cs.sin(a)),
            v2,
            1 / m * (-mg + u1 * cs.sin(a) + u2 * cs.cos(a)),
            va,
            r / J * u1
        )

        u = cs.vertcat(u, t)
        # Objective term
        L = 1 / t * (2 * u1**2 + u2**2)
        ode = {'x': x, 'p': u, 'ode': t * xdot, 'quad': t * L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([0.] * self.s_dim)
        init["q_start"] = [0., 0., 1]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 1
        return max_t

    def start_bounds(self, start):
        bs = [0] * self.s_dim
        return start, bs, bs

    def end_bounds(self, state):
        bs = [1] + [0] * (self.s_dim - 1)
        return state, bs, bs

    def control_bounds(self, control):
        lbu = [-5, 0, 0]
        ubu = [5, 17, cs.inf]
        return control, ubu, lbu

    def objective_end_control(self, state, control):
        mu = 1
        return mu * control[-1]
