import casadi as cs
from . import ode_solver


# standard version for casadi and nlp_sol
def create_nlp(curr_problem, grid, fix_controls=False, ret_cblocks=False):
    """Return state variables and controls together with bounds as well as
    the objective function to construct the problem.

    Keyword arguments:
        curr_problem    -- instance of the problem class
        grid    -- dict containing discretization and lifting points
        fix_controls    -- binary decision whether to fix the controls
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

    Sk = cs.MX.sym('X0', s_dim)

    N = len(time_points) - 1

    # create lists for variables
    u = cs.DM([])
    s = []
    g = []
    cg = []
    lg = []

    # constraint indices of matching conditions
    cblocks = []
    cblocks_match = []

    # create lists for lower and upper bounds
    lbu, ubu = [], []
    lbs, ubs = [], []
    lbg, ubg = [], []
    clbg, cubg = [], []
    llbg, lubg = [], []

    J = 0

    # "Lift" initial conditions
    start, ub, lb = curr_problem.start_bounds(Sk)
    s += [start]
    ubs += ub
    lbs += lb

    # additional constraints for start values
    start_cond, ub, lb = curr_problem.custom_start_constraints(Sk)
    cg += [start_cond]
    clbg += lb
    cubg += ub

    # Add control variables
    Qk_old = 0
    for m in range(len(control_points)):
        Qk = cs.MX.sym('U_' + str(m), q_dim)
        if (fix_controls):
            u = cs.vertcat(u, Qk)
            ubu += [cs.inf] * q_dim
            lbu += [-cs.inf] * q_dim
        else:
            control, ub, lb = curr_problem.control_bounds(Qk)
            u = cs.vertcat(u, control)
            ubu += ub
            lbu += lb
            cond, ubc, lbc = curr_problem.control_cond(Qk)
            cg += [cond]
            cubg += ubc
            clbg += lbc
            if (m > 0):
                for g_indx in global_control_indc:
                    cg += [Qk_old[g_indx] - Qk[g_indx]]
                    clbg += [0]
                    cubg += [0]
            Qk_old = Qk

    # input for integrator
    init = {}
    init["controls"] = u
    init["q_dim"] = q_dim

    # check whether there is a first cblock
    if len(lbg) > 0:
        cblocks += [0]
        cblocks_match += [False]

    Sk_temp = Sk
    # Formulate the NLP
    for k in range(N):
        init["s"] = Sk_temp
        Sk_end, Jk = ode_solver.integrate_interval(
            init, control_points, ode,
            time_points[k], time_points[k + 1]
        )
        J += Jk

        if (lifting_points[k + 1]):
            # New NLP variable for state at end of interval
            Sk = cs.MX.sym('X_' + str(k + 1), s_dim)
            state, ub, lb = curr_problem.state_bounds(Sk)
            s += [state]
            ubs += ub
            lbs += lb
            # Add equality constraint
            # g += [Sk_end - Sk]
            cblocks += [len(lbg)]
            cblocks_match += [True]
            g += [Sk - Sk_end]  # required for BlockSQP2 condensing
            ubg += [0] * s_dim
            lbg += [0] * s_dim
            Sk_temp = Sk
        else:
            Sk_temp = Sk_end
            state, ub, lb = curr_problem.state_bounds(Sk_temp)
            # only add conditions that are not +/- infinity
            for c in range(state.shape[0]):
                # if not (ub[c] == cs.inf and lb[c] == -cs.inf):
                lg += [state[c]]
                lubg += [ub[c]]
                llbg += [lb[c]]
        # store custom constraints for later to get a good cblock structure
        cust_c, ub, lb = curr_problem.custom_state_constraints(Sk_temp)
        cg += [cust_c]
        cubg += ub
        clbg += lb

    state, ub, lb = curr_problem.end_bounds(Sk_temp)
    cg += [state]
    cubg += ub
    clbg += lb

    # transform quadrature in case it is higher-dimensional
    J = curr_problem.transform_quad(J)

    J += curr_problem.objective_end_control(Sk_temp, Qk)
    J += curr_problem.objective_end(Sk_temp)

    # add final cblock
    if len(clbg) + len(llbg) > 0:
        cblocks += [len(lbg)]
        cblocks_match += [False]

    # add custom bounds back
    g += lg + cg
    ubg += lubg + cubg
    lbg += llbg + clbg

    # combine controls and states
    lbw = lbu + lbs
    ubw = ubu + ubs
    w = [u] + s

    if ret_cblocks:
        return w, lbw, ubw, g, lbg, ubg, J, cblocks, cblocks_match
    else:
        return w, lbw, ubw, g, lbg, ubg, J

