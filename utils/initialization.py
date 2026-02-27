import casadi as cs
import numpy as np
from . import ode_solver


def initialize(init_vals, grid, ode, init_type="auto", bounds=None):
    """Determine initial values for the intermediate variables.

    Keyword arguments:
        init_vals   -- dict containing independent variable, dimension and
                       values for given lifting points
        grid    -- dict containing discretization and lifting points
        ode     -- casadi function
        init_type   -- type of initialization ("auto" or "lin", else random)
        bounds  -- bounds for the intermediate variables
    """
    if (init_type == "auto"):
        s_init = initialize_auto(init_vals, grid, ode)
    elif (init_type == "lin"):
        s_init = initialize_lin(init_vals, grid)
    else:
        s_init = initialize_random(init_vals, grid, bounds)
    if (bounds is not None):
        s_init = project_bounds(s_init, bounds)
    return s_init


def initialize_lin(init_vals, grid):
    """Determine initial values for the intermediate variables by linear interpolation.

    Keyword arguments:
        init_vals   -- dict containing independent variable, dimension and
                       values for given lifting points
        grid    -- dict containing discretization and lifting points
    """
    s_start = init_vals["s_start"]
    s_end = init_vals.get("s_end", s_start)
    time_points = grid["time"]

    start_time = time_points[0]
    end_time = time_points[-1]  # end point
    incline = (s_end - s_start) / (end_time - start_time)
    s_init = cs.DM([])

    for t in time_points:
        curr_val = s_start + incline * (t - start_time)
        s_init = cs.vertcat(s_init, curr_val)

    return s_init


def initialize_random(init_vals, grid, bounds):
    """Determine initial values for the intermediate variables randomly.

    Keyword arguments:
        init_vals   -- dict containing independent variable, dimension and
                       values for given lifting points
        grid    -- dict containing discretization and lifting points
        bounds  -- bounds for the intermediate variables
    """
    s_dim = init_vals["s_dim"]
    time_points = grid["time"]
    s_list = []

    if (bounds is None):
        up_bounds = cs.DM([1] * s_dim)
        low_bounds = cs.DM([-1] * s_dim)
    else:
        up_bounds = bounds["upper"]
        low_bounds = bounds["lower"]

    for t in range(len(time_points)):
        rand_vals = np.random.uniform(low_bounds, up_bounds)
        s_list += list(rand_vals.flatten())

    print(s_list)
    return cs.DM(s_list)


def initialize_auto(init_vals, grid, ode):
    """Determine initial values for the intermediate variables via FSInit.

    Keyword arguments:
        init_vals   -- dict containing independent variable, dimension and
                       values for given lifting points
        grid    -- dict containing discretization and lifting points
        ode     -- casadi function
    """
    curr_s = init_vals["s_start"]
    controls = init_vals["controls"]
    grid["lift"] = grid.get("lift", [0 for el in grid["time"]])
    init_vals["sol"] = init_vals.get("sol", cs.vertcat(controls, curr_s))
    return compute_all_states(init_vals, grid, ode)


def compute_all_states(init, grid, ode):
    """Determine initial values for all remaining intermediate variables via FSInit.

    Keyword arguments:
        init_vals   -- dict containing independent variable, dimension and
                       values for given lifting points
        grid    -- dict containing discretization and lifting points
        ode     -- casadi function
    """
    q_dim = init["q_dim"]
    s_dim = init["s_dim"]
    sol = init["sol"]

    control_points = grid["control"]
    time_points = grid["time"]
    lifting_points = grid["lift"]

    num_control_points = len(control_points)

    q_temp = sol[:q_dim * num_control_points]
    s_temp = sol[q_dim * num_control_points:]
    curr_lift_point = 0
    all_states = cs.DM([])

    curr_init = {}
    curr_init["q_dim"] = q_dim
    curr_init["controls"] = q_temp

    curr_s = s_temp[curr_lift_point * s_dim:(curr_lift_point + 1) * s_dim]
    all_states = cs.vertcat(all_states, curr_s)

    for j in range(len(time_points) - 1):
        curr_init["s"] = curr_s

        if (lifting_points[j + 1] == 1):
            curr_lift_point += 1
            curr_s = s_temp[curr_lift_point * s_dim:(curr_lift_point + 1) * s_dim]
        else:
            curr_s, _ = ode_solver.integrate_interval(curr_init, control_points, ode,
                                                      time_points[j], time_points[j + 1])

        all_states = cs.vertcat(all_states, curr_s)

    return all_states


def project_bounds(s_init, bounds):
    """Project the given initialization onto given bounds.

    Keyword arguments:
        s_init  -- initialization of intermediate variables
        bounds  -- bounds for the intermediate variables
    """
    up_bounds = bounds["upper"]
    low_bounds = bounds["lower"]
    s_dim = up_bounds.shape[0]
    init_length = s_init.shape[0]
    for j in range(1, init_length):
        curr_comp = j % s_dim
        if (s_init[j] > up_bounds[curr_comp]):
            s_init[j] = up_bounds[curr_comp]
        if (s_init[j] < low_bounds[curr_comp]):
            s_init[j] = low_bounds[curr_comp]
    return s_init


def select_states(s_init, s_dim, lifting_points):
    """Select the intermediate variables that are contained in a lifting.

    Keyword arguments:
        s_init  -- initialization of intermediate variables
        s_dim   -- dimension of the variables
        lifting_points -- points at which to select the states
    """
    sel_states = cs.DM([])
    for i in range(len(lifting_points)):
        if (lifting_points[i] or i == 0):
            curr_s = s_init[i * s_dim:(i + 1) * s_dim]
            sel_states = cs.vertcat(sel_states, curr_s)
    return sel_states


def random_control(problem, num_control_points, seed=42):
    """Randomly initialize the vector of control discretizations within the correct bounds.

    Keyword arguments:
        problem  -- instance of the problem class
        num_control_points  -- number of control discretizations
        seed  -- random seed

    """
    q_dim = problem.q_dim
    u = cs.MX.sym('u', q_dim)
    _, ubc, lbc = problem.control_bounds(u)
    np.random.seed(seed)
    lbc = cs.DM(lbc)
    diff = cs.DM(ubc) - lbc

    q_init = cs.DM([])
    for i in range(num_control_points):
        q_init = cs.vertcat(
            q_init, lbc + np.random.rand() * diff
        )

    return q_init

