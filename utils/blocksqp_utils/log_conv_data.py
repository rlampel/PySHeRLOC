from .. import penalty
import casadi as cs
import numpy as np
from tabulate import tabulate
import os
from . import get_blocksqp_path
# Path to BlockSQP installation
blockSQP_path = get_blocksqp_path.get_path()
os.sys.path.append(blockSQP_path)
import blockSQP2 as blockSQP


def init_logs():
    """Create an empty dictionary that contains entries for all quantities that shall be logged.
    """
    log = {}
    log["conv"] = []
    log["merit"] = []
    log["obj"] = []
    log["viol"] = []
    log["step"] = []
    log["kappa"] = []
    return log


def add_log_entry(log, prim_vars, dual_vars,
                  ubx, lbx, ubg, lbg,
                  func_f, func_g, full_lag_der,
                  old_prim_vars=None, old_dual_vars=None):
    """Compute the relevant quantities for the current iteration and add them to the dictionary.

    Keyword arguments:
        log  -- dictionary to store the quantities in
        prim_vars  -- primal variables
        dual_vars  -- dual variables
        ubx, lbx  -- upper and lower bounds for primal variables
        ubg, lbg  -- upper and lower bounds for constrain function
        func_f  -- objective function
        func_g  -- constraint function
        full_lag_der  -- derivative of Lagrange function
    """
    dual_vars = np.array(dual_vars).reshape(-1)

    # violation of constraints g
    constr_viol = penalty.get_violation(func_g(prim_vars), ubg, lbg)
    # violation of variable bounds
    constr_viol = cs.vertcat(constr_viol, penalty.get_violation(prim_vars, ubx, lbx))
    # objective
    objective = func_f(prim_vars)
    # error of Newton step
    kkt_func = full_lag_der(prim_vars, dual_vars).T
    kkt_func = cs.vertcat(kkt_func, constr_viol)
    rhs_res = kkt_func.T @ kkt_func
    rhs_res = cs.norm_inf(kkt_func) / (1 + cs.norm_inf(dual_vars))
    mu = np.linalg.norm(dual_vars, np.inf)

    # if there are old steps available, monitor the step sizes
    if old_dual_vars is not None and old_prim_vars is not None:
        step_norm = cs.norm_2(prim_vars - old_prim_vars)**2
        step_norm += cs.norm_2(dual_vars - old_dual_vars)**2
        step_norm = float(step_norm)
    else:
        step_norm = np.inf

    log["conv"] += [float(rhs_res)]
    log["merit"] += [float(cs.norm_1(objective) + mu * cs.norm_1(func_g(prim_vars)))]
    log["viol"] += [float(cs.norm_inf(constr_viol))]
    log["obj"] += [float(objective)]
    log["step"] += [step_norm]

    return log


def get_kappa(vars, lag_hess, lag_der, jac_g, x, lambd, x_old, sparsity_pattern, hess_type=1):
    '''Criterion taken from Numerical Optimization by Nocedal, equation (18.63).'''
    dx = x - x_old
    x_dim = x.shape[0]
    hess_eval = np.array(lag_hess(x, lambd[x_dim:]).full(), dtype=np.float64)
    # iterate over all hessian blocks
    for i in range(len(sparsity_pattern) - 1):
        if hess_type == 1:
            # compute kappa for sr1 matrix
            hess_approx_block = vars.get_hess1_block(i)
        else:
            # compute kappa for BFGS matrix
            hess_approx_block = vars.get_hess2_block(i)
        hess_approx_block = np.array(blockSQP.Matrix(hess_approx_block))
        hess_eval[sparsity_pattern[i]:sparsity_pattern[i + 1],
                  sparsity_pattern[i]:sparsity_pattern[i + 1]] -= hess_approx_block

    grad_bounds = cs.DM_eye(x_dim)
    grad_comb = cs.horzcat(grad_bounds, jac_g(x_old).T)
    A = np.array(grad_comb)
    P = np.eye(x_dim) - A @ cs.inv(A.T @ A) @ A.T
    kappa = cs.norm_2(P @ (hess_eval) @ dx) / cs.norm_2(dx)
    print("computed kappa: ", kappa)

    return float(kappa)


def add_kappa(log, prim_vars, old_prim_vars, dual_vars, meth_vars,
              lag_hess, lag_der, jac_g, sparsity_pattern, hess_type=2):
    """Compute the current kappa value and add it to the log dictionary.

    Keyword arguments:
        log  -- dictionary to store the quantities in
        prim_vars  -- primal variables
        old_prim_vars  -- primal variables of previous step
        dual_vars  -- dual variables
        meth_vars  -- var entries of the blockSQP SQPmethod instance
        lag_hess  -- Hessian of the Lagrange function
        lag_der  -- derivative of the Lagrange function
        jac_g  -- derivative of the constraint function
        sparsity_pattern  -- indices of the individual blocks for the vector of primal variables
        hess_type  -- indicates whether to compute kappa for the SR1 (1) or BFGS (2) matrix
    """
    curr_kappa = get_kappa(
        meth_vars, lag_hess, lag_der, jac_g,
        prim_vars, dual_vars,
        old_prim_vars,
        sparsity_pattern,
        hess_type=hess_type
    )
    log["kappa"] += [curr_kappa]
    return log


def print_logs_table(log):
    """Print the entries of the dictionary in table format.

    Keyword arguments:
        log  -- dictionary to store the quantities in
    """
    table_data = []
    headers = list(log.keys())
    table_data += [["iter"] + headers]

    entry_sizes = [len(log[key]) for key in headers]
    num_rows = max(entry_sizes)

    for i in range(num_rows):
        row_data = [i]
        # check whether the current key has an entry, else add an empty string
        for j in range(len(headers)):
            curr_key = headers[j]
            if i < entry_sizes[j]:
                row_data += [log[curr_key][i]]
            else:
                row_data += [" "]
        table_data += [row_data]

    print(tabulate(table_data, headers="firstrow", tablefmt="fancy_outline"))


def print_logs(log):
    """Print the entries of the dictionary in table format.

    Keyword arguments:
        log  -- dictionary to store the quantities in
    """
    table_data = []
    headers = ["conv", "step"]
    table_data += [["iter"] + headers]

    entry_sizes = [len(log[key]) for key in headers]
    num_rows = max(entry_sizes)

    for i in range(num_rows):
        row_data = [str(i) + " "]
        # check whether the current key has an entry, else add an empty string
        for j in range(len(headers)):
            curr_key = headers[j]
            if i < entry_sizes[j]:
                row_data += [log[curr_key][i]]
            else:
                row_data += [" "]
        table_data += [row_data]

    for row in table_data:
        for el in row:
            print(el, end=" ")
        print()

