import casadi as cs
import numpy as np
from .. import initialization, penalty
from .sort_vars import sort_back, sort_vars_by_time
from . import dyn_lifting
from . import fast_init_lift


def get_kkt_error(prim_vars, dual_vars, lag_der):
    """Compute the current KKT error for given primal and dual variables.

    Keyword arguments:
        prim_vars  -- primary variables
        dual_vars  -- Lagrange multipliers
        lag_der -- derivative of Lagrange function
    """
    return cs.norm_2(lag_der(prim_vars, dual_vars))


def fsinit_heuristic(xi_temp, sort_grid, grid, ode, s_dim, q_dim,
                     num_control_points, num_time_points):
    """Compute all states using FSInit based on the state at time 0.

    Keyword arguments:
        xi_temp  -- primary variables
        sorting_grid -- positions of shooting variables obtained by sorting
        grid    -- dict containing discretization and lifting points
        ode     -- casadi function
        s_dim  -- number of differential states
        q_dim  -- number of controls
        num_control_points -- number of control discretizations
        num_time_points -- number of time discretizations
    """
    # sort back
    xi_temp = np.array(sort_back(xi_temp, sort_grid, s_dim, q_dim))
    # fsinit
    q_temp = xi_temp[:q_dim * num_control_points]
    s_temp = xi_temp[q_dim * num_control_points:q_dim * num_control_points + s_dim]

    temp_grid = grid.copy()
    temp_grid["lift"] = [1] + [0] * (num_time_points - 1)
    s_plot = initialization.compute_all_states({"sol": cs.vertcat(q_temp, s_temp),
                                                "s_dim": s_dim,
                                                "q_dim": q_dim},
                                               temp_grid, ode)
    s_init = initialization.select_states(s_plot, s_dim, grid["lift"])
    xi_temp = np.array(cs.vertcat(q_temp, s_init))

    xi_temp = np.array(sort_vars_by_time(xi_temp, grid, s_dim, q_dim)[0])
    return xi_temp.reshape(-1)


def fsinit_merit(xi_temp, xi_fsinit, lam_temp, lbg, ubg, lbx, ubx, func_f, func_g,
                 opt_err=1., old_point=None, exact_hess=True):
    """Replace the current states by FSInit if merit and KKT do not get too much worse.

    Keyword arguments:
        xi_temp  -- primal variables
        xi_fsinit  -- primal variables obtained by FSInit
        lam_temp  -- Lagrange multipliers
        lbg, ubg -- lower and upper bounds for g
        lbx, ubx -- lower and upper bounds for primal variables
        func_f -- objective function
        func_g -- constraint function
        opt_err -- value of current optimality error
        old_point  -- primal variables of previous step
    """

    # for Quasi-Newton, return the original input
    # if not exact_hess:
    #     return xi_temp

    # violation of constraints g
    constr_viol = penalty.get_violation(func_g(xi_temp), ubg, lbg)
    constr_viol_fs = penalty.get_violation(func_g(xi_fsinit), ubg, lbg)

    # violation of variable bounds
    constr_viol = cs.vertcat(constr_viol, penalty.get_violation(xi_temp, ubx, lbx))
    constr_viol_fs = cs.vertcat(constr_viol_fs, penalty.get_violation(xi_fsinit, ubx, lbx))
    feas_norm = cs.norm_inf(constr_viol)

    if feas_norm > 1.e-1 or opt_err < 1.e-1:
        print(f"Violation {feas_norm} too large or opt error too small {opt_err}")
        return xi_temp

    # objective
    # mu = np.max(lam_temp)
    mu = np.linalg.norm(lam_temp, np.inf)
    objective = func_f(xi_temp)
    objective_fs = func_f(xi_fsinit)

    constr_viol_norm = cs.norm_1(constr_viol)
    merit = float(cs.norm_1(objective) + mu * constr_viol_norm)
    merit_fs = float(cs.norm_1(objective_fs) + mu * cs.norm_1(constr_viol_fs))

    if merit_fs > merit:
        print("Merit does not decrase")
        return xi_temp
    else:
        print("Replace by FSInit")
        return xi_fsinit


def refine_lifting(curr_problem, grid, starting_times, s_temp, q_temp, mu=1):
    """Replace the current states by FSInit if merit and KKT do not get too much worse.

    Keyword arguments:
        curr_problem  -- instance of the the current OC problem class
        grid  -- dict containing discretization and lifting points
        starting_times  -- time points corresponding to lifting points
        s_temp  -- state variables at the shooting points
        q_temp  -- control variables
        mu  -- penalty parameter
    """
    s_dim = curr_problem.s_dim
    q_dim = curr_problem.q_dim
    ode = curr_problem.get_ode()
    lift_grid = grid.copy()
    lift_grid["time"] = starting_times

    '''
    lifting_points = dyn_lifting.best_graph_lift(curr_problem, starting_times,
                                                 s_temp, q_temp,
                                                 lift_grid, mu)
    '''
    lifting_points = fast_init_lift.best_init_lift(curr_problem, starting_times,
                                                   s_temp, q_temp,
                                                   lift_grid, mu)

    red_lift_points = dyn_lifting.convert_lifting(lifting_points, starting_times)
    lifting_points = dyn_lifting.convert_lifting(lifting_points, grid["time"])

    temp_grid = grid.copy()
    temp_grid["lift"] = lifting_points
    s_select = initialization.select_states(s_temp, s_dim, red_lift_points)
    s_plot = initialization.compute_all_states({"sol": cs.vertcat(q_temp, s_select),
                                                "s_dim": s_dim,
                                                "q_dim": q_dim},
                                               temp_grid, ode)
    s_init = initialization.select_states(s_plot, s_dim, grid["lift"])
    return s_init, lifting_points

