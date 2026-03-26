import numpy as np
import casadi as cs


def get_best_lam(x_curr, lbx, ubx, g, lbg, ubg, grad_f, grad_g):
    """Compute the best initial Lagrange multipliers that minimize the KKT error.

    Keyword arguments:
        x_curr -- primary variables
        lbx, ubx -- lower and upper bounds for primal variables
        g -- vector of additional constraints
        lbg, ubg -- lower and upper bounds for g
        grad_f -- derivative of objective function
        grad_g -- derivative of constraint function
    """
    active_ind = get_active_set(x_curr, lbx, ubx, g, lbg, ubg)

    grad_bounds = cs.DM_eye(len(lbx))
    grad_comb = cs.horzcat(grad_bounds, grad_g.T)
    grad_red = select_columns(grad_comb, active_ind)
    lam_red = least_squares_init(grad_f.T, grad_red)
    lam_opt = []

    active_count = 0
    for i in range(len(active_ind)):
        if active_ind[i]:
            lam_opt += [float(lam_red[active_count])]
            active_count += 1
        else:
            lam_opt += [0]
    return np.array(lam_opt).reshape(-1)


def get_active_set(x_curr, lbx, ubx, g, lbg, ubg):
    """Return the indices of the constraints that are currently active, i.e., at their bounds.

    Keyword arguments:
        x_curr -- primary variables
        lbx, ubx -- lower and upper bounds for primal variables
        g -- vector of additional constraints
        lbg, ubg -- lower and upper bounds for g
    """
    num_prim_vars = len(lbx)
    num_constr = len(lbg)
    # create binary vector corresponding to active constraints
    active_ind = [0] * (num_prim_vars + num_constr)

    for i in range(num_prim_vars):
        if x_curr[i] <= lbx[i] or x_curr[i] >= ubx[i]:
            active_ind[i] = 1
    y_curr = g(x_curr)
    for i in range(num_constr):
        if y_curr[i] <= lbg[i] or y_curr[i] >= ubg[i] or lbg[i] == ubg[i]:
            active_ind[num_prim_vars + i] = 1
    return active_ind


def select_columns(matrix, binary_mask):
    """Select columns from a CasADi matrix based on a binary mask.

    Args:
        matrix (casadi.MX or casadi.SX): Input CasADi matrix.
        binary_mask (list of int): List of 0s and 1s indicating which columns to keep.

    Returns:
        casadi.MX or casadi.SX: Matrix with selected columns.
    """
    if matrix.shape[1] != len(binary_mask):
        raise ValueError("Length of binary_mask must match number of columns in matrix")

    selected = [matrix[:, i] for i, b in enumerate(binary_mask) if b]
    if not selected:
        # Return an empty matrix with correct number of rows and 0 columns
        return matrix[:, :0]
    return cs.horzcat(*selected)


def least_squares_init(a, B):
    """Compute the vector x that minimizes the squared error of |a - B * x|^2.

    Keyword arguments:
        a -- vector in R^n
        B -- matrix of dimension n times m
    """
    lam = np.linalg.lstsq(B, a, rcond=None)[0]
    return lam

