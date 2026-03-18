import casadi as cs
from .. import ode_solver


# standard version for casadi and nlp_sol
def fsinit_nlp(curr_problem, grid, S0, u):
    """Return state variables and controls together with bounds as well as
    the objective function to construct the problem.

    Keyword arguments:
        curr_problem    -- instance of the problem class
        grid    -- dict containing discretization and lifting points
    """
    init = curr_problem.get_init()
    ode = curr_problem.get_ode()
    s_dim = init["s_dim"]
    q_dim = init["q_dim"]

    try:
        global_control_indc = curr_problem.global_controls
    except AttributeError:
        global_control_indc = []

    control_points = grid["control"]
    lifting_points = grid["lift"]
    time_points = grid["time"]

    Sk = S0

    N = len(time_points) - 1

    J = 0

    # create lists for variables
    s = [Sk]
    g = []
    cg = []
    lg = []

    # additional constraints for start values
    start_cond, _, _ = curr_problem.custom_start_constraints(Sk)
    cg += [start_cond]

    # Add control variables
    Qk_old = 0
    for m in range(len(control_points)):
        Qk = u[m * q_dim:(m + 1) * q_dim]
        cond, ubc, lbc = curr_problem.control_cond(Qk)
        cg += [cond]
        if (m > 0):
            for g_indx in global_control_indc:
                cg += [Qk_old[g_indx] - Qk[g_indx]]
        Qk_old = Qk

    # input for integrator
    init = {}
    init["controls"] = u
    init["q_dim"] = q_dim

    Sk_temp = Sk
    # Formulate the NLP
    for k in range(N):
        init["s"] = Sk_temp
        Sk_end, Jk = ode_solver.integrate_interval(
            init, control_points, ode,
            time_points[k], time_points[k + 1]
        )

        J += Jk
        Sk_temp = Sk_end

        if (lifting_points[k + 1]):
            # New NLP variable for state at end of interval
            state, ub, lb = curr_problem.state_bounds(Sk_temp)
            s += [state]
            # Add equality constraint
            g += [cs.DM.zeros(s_dim)]
        else:
            state, ub, lb = curr_problem.state_bounds(Sk_temp)
            lg += [state]
        # store custom constraints for later to get a good cblock structure
        cust_c, ub, lb = curr_problem.custom_state_constraints(Sk_temp)
        cg += [cust_c]

    state, ub, lb = curr_problem.end_bounds(Sk_temp)
    cg += [state]

    # transform quadrature in case it is higher-dimensional
    J = curr_problem.transform_quad(J)

    J += curr_problem.objective_end_control(Sk_temp, Qk)
    J += curr_problem.objective_end(Sk_temp)

    # add custom bounds back
    g += lg + cg

    # combine controls and states
    w = [u] + s

    return cs.vertcat(*w), cs.vertcat(*g), J
