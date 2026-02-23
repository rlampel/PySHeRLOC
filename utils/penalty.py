import casadi as cs


def get_violation(var, upper_bounds, lower_bounds):
    """Evaluate the violation for given variables and bounds as a vector.

    Keyword arguments:
        var -- variables
        upper_bounds    -- upper bounds of var
        lower_bounds    -- lower bounds of var
    """
    P = cs.DM([])
    dim_var = var.shape[0]

    for i in range(dim_var):
        # lower bound
        curr_viol = 0
        if (lower_bounds[i] != -cs.inf):
            curr_viol += (cs.fmin(lower_bounds[i], var[i]) - lower_bounds[i])
        # upper bound
        if (upper_bounds[i] != cs.inf):
            curr_viol += (cs.fmax(upper_bounds[i], var[i]) - upper_bounds[i])
        # add current violation to vector
        P = cs.vertcat(P, curr_viol)
    return P


def penalty(var, upper_bounds, lower_bounds, lam=1):
    """Create an L2 penalty function.

    Keyword arguments:
        var -- variables
        upper_bounds    -- upper bounds of var
        lower_bounds    -- lower bounds of var
        lam -- penalty parameter
    """
    P = get_violation(var, upper_bounds, lower_bounds)
    err = lam * P.T @ P
    return err


def l1_penalty(var, upper_bounds, lower_bounds, lam=1):
    """Create an L1 penalty function.

    Keyword arguments:
        var -- variables
        upper_bounds    -- upper bounds of var
        lower_bounds    -- lower bounds of var
        lam -- penalty parameter
    """
    P = get_violation(var, upper_bounds, lower_bounds)
    err = lam * cs.norm_1(P)
    return err

