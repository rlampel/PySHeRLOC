import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 3
    q_dim = 2
    global_controls = [1]
    time_scale_ind = 1

    # Define fixed parameters
    r0 = 1
    v0 = 0
    m0 = 1
    rT = 1.01
    b = 7
    T_max = 3.5
    A = 310
    k = 500
    C = 0.6

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.state_labels = [r"$r$", r"$v$", r"$m$"]

    def Drag(self, r, v):
        A = 310
        k = 500
        r0 = 1
        rho = cs.exp(-k * (r - r0))
        return A * v**2 * rho

    def get_ode(self):
        T_max = self.T_max
        b = self.b
        r0 = self.r0
        k = self.k
        A = self.A
        T_max = self.T_max

        # Declare model variables
        r = cs.MX.sym('r')
        v = cs.MX.sym('v')
        m = cs.MX.sym('m')
        x = cs.vertcat(r, v, m)
        u = cs.MX.sym('u')
        t = cs.MX.sym('t')

        # Model equations

        xdot = cs.vertcat(v,
                          -1 / (r**2) + (1 / m) * (T_max * u - A * (v**2) * cs.exp(-k * (r - r0))),
                          -b * u)

        """
        rdot = v
        vdot = - 1 / r**2 + 1 / m * (T_max * u - self.Drag(r, v))
        mdot = -b * u
        xdot = cs.vertcat(rdot,
                          vdot,
                          mdot
                          )        # Objective term
        """
        L = 0
        w = cs.vertcat(u, t)
        ode = {'x': x, 'p': w, 'ode': t * xdot, 'quad': L}
        return ode

    def get_init(self):
        r0, v0, m0 = self.r0, self.v0, self.m0
        init = {}
        init["s_start"] = cs.DM([r0, v0, m0])
        init["q_start"] = [0.5, 0.1]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 1
        return max_t

    def start_bounds(self, start):
        r0, v0, m0 = self.r0, self.v0, self.m0
        lbs = [r0, v0, m0]
        ubs = [r0, v0, m0]
        return start, ubs, lbs

    def control_bounds(self, control):
        T_max = self.T_max
        lbu = [0, 1.e-2]
        ubu = [1, T_max]
        return control, ubu, lbu

    def state_bounds(self, state):
        lbs = [1, 0, 0]
        ubs = [cs.inf] * 3
        return state, ubs, lbs

    def custom_state_constraints(self, state):
        lbg = [0]
        ubg = [self.C]
        gc = self.Drag(state[0], state[1])
        return gc, ubg, lbg

    def end_bounds(self, state):
        rT = self.rT
        lbs = [rT, -cs.inf, -cs.inf]
        ubs = [rT, cs.inf, cs.inf]
        return state, ubs, lbs

    def objective_end(self, state):
        return -state[-1]

