import casadi as cs
from . import BaseOCClass


class problem(BaseOCClass.super_problem):
    s_dim = 4
    q_dim = 2

    def __init__(self):
        BaseOCClass.super_problem.__init__(self, self.s_dim, self.q_dim)

    def get_ode(self):
        # Define constants
        const_k1 = 1.287e12
        const_k2 = const_k1
        const_k3 = 9.403e9
        E1 = -9758.3
        E2 = E1
        E3 = -8560
        H1 = 4.2
        H2 = -11
        H3 = -41.85
        rho = 0.9342
        Cp = 3.01
        kw = 4032
        AR = 0.215
        VR = 10
        mK = 5
        CPK = 2
        cA0 = 5.1
        theta0 = 104.9

        # Declare model variables
        cA = cs.MX.sym('cA')
        cB = cs.MX.sym('cB')
        theta = cs.MX.sym('theta')
        thetaK = cs.MX.sym('thetaK')
        V = cs.MX.sym('V')
        Q = cs.MX.sym('Q')

        var_k1 = const_k1 * cs.exp(E1 / (theta + 273.15))
        var_k2 = const_k2 * cs.exp(E2 / (theta + 273.15))
        var_k3 = const_k3 * cs.exp(E3 / (theta + 273.15))

        cAdot = V / VR * (cA0 - cA) - var_k1 * cA - var_k3 * cA**2
        cBdot = -V / VR * cB + var_k1 * cA - var_k2 * cB
        aux0 = V / VR * (theta0 - theta) + (kw * AR) / (rho * Cp * VR) * (thetaK - theta)
        aux1 = - 1 / (rho * Cp) * (var_k1 * cA * H1 + var_k2 * cB * H2 + var_k3 * cA**2 * H3)
        thetadot = aux0 + aux1
        thetaKdot = 1 / (mK * CPK) * (Q + kw * AR * (theta - thetaK))

        # time unit correction
        corr = 1 / 3600
        cAdot = cAdot * corr
        cBdot = cBdot * corr
        thetadot = thetadot * corr
        thetaKdot = thetaKdot * corr

        # Model equations

        control = cs.vertcat(V, Q)
        x = cs.vertcat(cA, cB, theta, thetaK)
        xdot = cs.vertcat(cAdot, cBdot, thetadot, thetaKdot)

        # Objective term
        L = 0
        ode = {'x': x, 'p': control, 'ode': xdot, 'quad': L}
        return ode

    def get_init(self):
        init = {}
        init["s_start"] = cs.DM([2.14, 1.09, 114.2, 112.9])
        init["q_start"] = [0, 0]
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        max_t = 1500
        return max_t

    def start_bounds(self, start):
        lbs = [-0.02, -0.02, 50, 50]
        ubs = [6., 4., 160, 160]
        return start, ubs, lbs

    def control_bounds(self, control):
        lbu = [3, -9000]
        ubu = [35, 0]
        return control, ubu, lbu

    def state_bounds(self, state):
        lbs = [-0.02, -0.02, 50, 50]
        ubs = [6., 4., 160, 160]
        return state, ubs, lbs

    def objective_end(self, state):
        return -state[1]

