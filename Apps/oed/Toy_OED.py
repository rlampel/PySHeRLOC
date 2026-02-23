import casadi as cs
from .. import BaseOCClass
from . import oed_utils


class problem(BaseOCClass.super_problem):
    s_dim = 4
    q_dim = 1
    p_dim = 1
    is_inverse = False  # the Fisher Matrix is given by an ode
    reg_init = 1.e-2

    # model parameter
    p = -2

    def __init__(self, criterion="A"):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)
        self.criterion = criterion

    def get_ode(self):
        # Declare model variables
        x = cs.MX.sym('x')
        G = cs.MX.sym('G')
        F = cs.MX.sym('F')
        z = cs.MX.sym('z')
        w = cs.MX.sym('w')

        # Model equations
        xdot = self.p * x

        # G equations
        G = cs.MX.sym('G')
        Gdot = self.p * G + x

        # F equations
        Fdot = w * G**2

        # z equations
        z = cs.MX.sym('z')
        zdot = w

        control = cs.vertcat(w)
        x = cs.vertcat(x, G, F, z)
        xdot = cs.vertcat(xdot,
                          Gdot,
                          Fdot,
                          zdot)
        # Objective term
        L = 0
        ode = {'x': x, 'p': control, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([1., 0., self.reg_init, 0.])
        init["q_start"] = [0.5]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 1
        return max_t

    def start_bounds(self, start):
        lbs = [1., 0., self.reg_init, 0.]
        ubs = [1., 0., self.reg_init, 0.]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [0.]
        ubu = [1.]
        return control, ubu, lbu

    def end_bounds(self, state):
        end_points = state[-1]
        ubs = [0.2]
        lbs = [0.]
        return end_points, ubs, lbs

    def objective_end(self, state):
        F = state[2]
        return oed_utils.oed_criterion(F, self.p_dim, self.criterion, self.is_inverse)

