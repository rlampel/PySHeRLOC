import casadi as cs
import numpy as np
from .sort_vars import sort_back


def get_matching_violation(cblocks, cblock_match, constr_viol, x_dim):
    """Return the vector containing the evaluated matching conditions.

    Keyword arguments:
        cblocks  -- indices of constraint blocks
        cblock_match  -- booleans indicating whether a block is a matching condition
        costr_viol  -- vector containing the evaluated constraint function
        x_dim  -- number of differential states
    """
    # get vector that contains the violations for all of g
    match_viol = cs.DM([])

    # filter the constraints to only consider the matching conditions
    start_ind = int(not cblock_match[0])
    end_ind = len(cblocks) - int(not cblock_match[-1])

    # add matching violations
    for i in range(start_ind, end_ind):
        c_ind = cblocks[i]
        match_viol = cs.vertcat(match_viol, constr_viol[c_ind:c_ind + x_dim])

    return match_viol


def get_relative_match_viol(prim_vars, sort_grid, x_dim, q_dim, num_control_points,
                            match_viol):
    """Compute the relative error of the matching conditions.

    Keyword arguments:
        prim_vars  -- primary variables
        sort_grid -- positions of shooting variables obtained by sorting
        x_dim  -- number of differential states
        q_dim -- number of controls
        num_control_points  -- number of control points
        match_viol  -- vector containing the evaluated matching conditions
    """
    match_viol = np.array(match_viol).flatten()
    # sort variables to get the state variables
    prim_vars = np.array(sort_back(prim_vars, sort_grid, x_dim, q_dim))
    s_temp = prim_vars[q_dim * num_control_points:]

    # for the relative error we can neglect the state at time 0
    s_temp = s_temp[x_dim:].flatten()
    # get the other state value at the lifting point by adding the error

    eps = 1.e-16
    rel_err = np.linalg.norm(match_viol) / (eps + np.linalg.norm(s_temp))

    return rel_err


def monitor_kkt_conv(kkt_norm_list, accepted_hess):
    """Check whether the convergence of the KKT errors deteriorates.

    Keyword arguments:
        kkt_norm_list  -- list of previous KKT error norms
    """
    # compare current contraction with the previous one

    if len(kkt_norm_list) < 3:
        # too few past data points to compare the contraction
        return False

    # print("current accepted hess: ", accepted_hess[-1])
    # if accepted_hess[-1] < 4:
    #     return False

    avg_kkt = float(kkt_norm_list[-2])
    curr_kkt = kkt_norm_list[-1]

    # print("prev contr: ", prev_contr, " curr contr: ", curr_contr)
    print("avg: ", avg_kkt, " current: ", curr_kkt)

    if avg_kkt < curr_kkt:
        # contraction is getting worse
        return True

    return False


def trigger_auto_condensing(grid, sort_grid, num_control_points,
                            cblocks, cblock_match,
                            prim_vars,
                            kkt_norm_list,
                            s_dim, q_dim, constr_viol, accepted_hess,
                            curr_opt, step_norm, rel_step_norm, constr_viol_norm,
                            exact_hess,
                            mode="default"
                            ):
    """Decide whether to switch from multiple to single shooting.

    Keyword arguments:
        grid    -- dict containing discretization and lifting points
        sort_grid -- positions of shooting variables obtained by sorting
        num_control_points  -- number of control points
        cblocks  -- indices of constraint blocks
        cblock_match  -- booleans indicating whether a block is a matching condition
        prim_vars  -- primary variables
        kkt_norm_list  -- list of previous KKT error norms
        s_dim  -- number of differential states
        q_dim -- number of controls
        costr_viol  -- vector containing the evaluated constraint function
    """
    # if there are no lifting points to be eliminated, return
    print("CURRENT MODE: ", mode)
    if not (1 in grid["lift"][1:]):
        return False

    # check whether the current point is close to the solution
    # print("Check the norms:\n\tviol: ", constr_viol_norm,
    #       "\n\tstep: ", step_norm,
    #       "\n\trel. step: ", rel_step_norm)
    if mode == "OED_transformed":
        opt_cond = (curr_opt > 5.e-3)
    else:
        opt_cond = (curr_opt > 5.e-3)

    viol_cond = (constr_viol_norm > 1.e-3)
    step_cond = (step_norm > 1.e-3)
    rel_step_cond = (rel_step_norm > 5.e-2)

    if exact_hess or mode != "default":
        viol_cond = False
        step_cond = False
        rel_step_cond = False

    if opt_cond or viol_cond or step_cond or rel_step_cond:
        print(" CONDITION NOT SATISFIED")
        return False

    match_viol = get_matching_violation(
        cblocks, cblock_match, constr_viol, s_dim
    )
    rel_err = get_relative_match_viol(
        prim_vars, sort_grid, s_dim, q_dim, num_control_points, match_viol
    )

    print("current relative error: ", rel_err)

    # check whether the 'relative' error is below a certain threshold
    if rel_err >= 0.005:
        return False

    # change lifting if the current contraction deteriorates
    condense_now = monitor_kkt_conv(kkt_norm_list, accepted_hess)

    return condense_now

