import casadi as cs


class super_problem:
    global_controls = []
    time_scale_ind = cs.DM_nan()
    # data for least squares data fitting problems
    meas_times = []
    meas_data = []
    obs_ind = []
    state_indices = None
    state_scales = None

    def __init__(self, s_dim, q_dim):    # label names for plotting
        self.state_labels = []
        self.control_labels = []
        self.s_dim = s_dim
        self.q_dim = q_dim

    def get_ode(self):
        """Return the differential equation.
        """
        raise NotImplementedError("ODE is not defined")

    def get_init(self):
        """Return states at time 0, initial constant control, state dimension,
        and control dimension.
        """
        init = {}
        init["s_start"] = cs.DM([0] * self.s_dim)
        init["q_start"] = [0] * self.q_dim
        init["s_dim"] = self.s_dim
        init["q_dim"] = self.q_dim
        return init

    def get_grid_details(self):
        """Return the length of the time interval.
        """
        max_t = 10
        return max_t

    def start_bounds(self, start):
        """Return upper and lower bounds for the states at time 0.
        """
        lbs = [-cs.inf] * self.s_dim
        ubs = [cs.inf] * self.s_dim
        return start, ubs, lbs

    def control_bounds(self, control):
        """Return upper and lower bounds for the controls.
        """
        lbu = [-cs.inf] * self.q_dim
        ubu = [cs.inf] * self.q_dim
        return control, ubu, lbu

    def control_cond(self, control):
        """Return custom constraints for the controls.
        """
        return cs.DM([]), [], []

    def state_bounds(self, state):
        """Return upper and lower bounds for the states at times greater 0.
        """
        lbs = [-cs.inf] * self.s_dim
        ubs = [cs.inf] * self.s_dim
        return state, ubs, lbs

    def custom_start_constraints(self, state):
        """Return custom constraints for the states at time 0.
        """
        return cs.DM([]), [], []

    def custom_state_constraints(self, state):
        """Return custom constraints for the states at times greater 0.
        """
        return cs.DM([]), [], []

    def end_bounds(self, state):
        """Return custom constraints for the states at the end of the time interval.
        """
        return cs.DM([]), [], []

    def objective_end(self, state):
        """Return the Mayer objective that depends on the states at the end of the time interval.
        """
        return 0

    def objective_end_control(self, state, control):
        """Return the objective that depends on the states and controls at
        the end of the time interval.
        """
        return 0

    def transform_quad(self, quad):
        """Transform the quadrature term computed by the ODE.
        """
        return quad

