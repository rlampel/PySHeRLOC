import numpy as np
import casadi as cs
import os
from . import get_blocksqp_path
# Path to BlockSQP installation
blockSQP_path = get_blocksqp_path.get_path()
os.sys.path.append(blockSQP_path)
import py_blockSQP as blockSQP


def get_block_cond(x, lambd, lag_hess, sparsity_pattern):
    '''Compute the condition number of a block matrix.

    Keyword arguments:
        x  -- primal variables
        lambd  -- dual variables
        lag_hess  -- Hessian of the Lagrange function
        sparsity_pattern  -- indices of the individual blocks for the vector of primal variables
    '''
    hess_eval = lag_hess(x, lambd)
    blocks = []
    block_conds = []

    for j in range(len(sparsity_pattern) - 1):
        blocks.append(
            np.array(hess_eval[sparsity_pattern[j]:sparsity_pattern[j + 1],
                               sparsity_pattern[j]:sparsity_pattern[j + 1]].full(),
                     dtype=np.float64)
        )
    for i in range(len(blocks)):
        b = blocks[i]
        curr_cond = np.linalg.cond(b, 'fro')
        print("Eigenvalues: ", np.linalg.eig(b)[0])
        # curr_cond = np.linalg.norm(b, 'fro')
        block_conds += [curr_cond]
        print("condition number: ", curr_cond, "\n")

    return block_conds


def get_better_cond_init(x, x_dim, block_conds, sparsity_pattern, threshold=10, upper_lim=1.e8):
    '''Assume that all blocks have the same size and replace the states of the blocks
    with the worst condition number by the ones with the best.'''
    # get block with best condition number
    best_ind = np.argmin(block_conds)
    best_cond = block_conds[best_ind]
    best_x_vals = x[sparsity_pattern[best_ind]:sparsity_pattern[best_ind] + x_dim]

    if best_cond >= upper_lim:
        return x

    # iterate over all blocks and replace the states with bad condition numbers
    # (i.e. the ones that are greater than the threshold)
    for i in range(len(block_conds)):
        curr_cond = block_conds[i]
        if curr_cond > threshold and curr_cond > best_cond:
            print(f"Change block entry {i} to entry {best_ind}")
            block_start = sparsity_pattern[i]
            x[block_start:block_start + x_dim] = best_x_vals

    return x


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

