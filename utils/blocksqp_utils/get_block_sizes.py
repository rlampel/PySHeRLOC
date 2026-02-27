import casadi as cs
import numpy as np


def get_vblock_sizes(x_dim, q_dim, lifting_points):
    """Compute the sizes of the blocks corresponding to dependent and independent
    variables.

    Keyword arguments:
        x_dim  -- number of differential states
        q_dim  -- number of controls
        lifting_points  -- zero-one vector indicating where to introduce lifting points
    """
    vblock_sizes = []
    vblock_dependencies = []

    # the first states are independent
    curr_size = x_dim + q_dim

    for i in lifting_points[1:-1]:
        if i:
            if curr_size > 0:
                vblock_sizes += [curr_size]
                vblock_dependencies += [False]
            vblock_sizes += [x_dim]
            vblock_dependencies += [True]
            curr_size = 0

        curr_size += q_dim

    # add variables for the last shooting interval
    if curr_size > 0:
        vblock_sizes += [curr_size]
        vblock_dependencies += [False]

    if lifting_points[-1]:
        # add last states
        vblock_sizes += [x_dim]
        vblock_dependencies += [True]

    return vblock_sizes, vblock_dependencies


def get_cblock_sizes(x_dim, g, cblocks, cblocks_match):
    """Compute the sizes of the constraint blocks for normal and lifting conditions.

    Keyword arguments:
        x_dim  -- number of differential states
        q_dim  -- number of controls
        cblocks  -- indices of lifting conditions in the constraints g
    """
    cblock_sizes = []
    start_ind = int(not cblocks_match[0])
    end_ind = len(cblocks) - int(not cblocks_match[-1])

    # add potential first block
    if not cblocks_match[0]:
        # add size of first block
        cblock_sizes += [cblocks[1] - cblocks[0]]

    # add blocks for matching conditions
    for i in range(start_ind, end_ind):
        c_ind = cblocks[i]
        # add previous different block and new cblock
        cblock_sizes += [x_dim]

    # add potential last block
    if not cblocks_match[-1]:
        c_ind = cblocks[-1]
        cblock_sizes += [g.shape[0] - c_ind]

    return cblock_sizes, start_ind, end_ind


def get_cont_viol(prim_vars, x_dim, func_g, cblocks, norm=2):
    """Compute the current KKT error for given primal and dual variables.

    Keyword arguments:
        prim_vars  -- primary variables
        x_dim  -- number of differential states
        func_g  -- constraint function
        cblocks  -- indices of continuity constraints
        norm  -- norm in which to compute the violation (1, 2, or inf)
    """
    all_viol = func_g(prim_vars)
    cont_viol = 0

    for c_ind in cblocks:
        curr_constr = all_viol[c_ind:c_ind + x_dim]

        match norm:
            case 1:
                curr_viol = cs.norm_1(curr_constr)
            case 2:
                curr_viol = cs.norm_2(curr_constr)
            case _:
                curr_viol = cs.norm_inf(curr_constr)

        cont_viol += float(curr_viol)
    return cont_viol


def condense_dual_vars(dual_vars, prim_dim, cblocks, cblock_match):
    """Delete all Lagrange multipliers that correspond to matching conditions

    Keyword arguments:
        dual_vars  -- Lagrange multipliers
        prim_dim  -- total number of primal variables
        cblocks  -- indices of constraint blocks
        cblock_match  -- indicates whether a cblock is a matching condition
    """
    # add all multipliers up to the first index in cblocks
    condensed_dual_vars = dual_vars[:prim_dim]

    for i in range(len(cblocks)):
        if not cblock_match[i]:
            # not a matching condition, add to condensed dual variables
            if i == len(cblocks) - 1:
                end_index = dual_vars.shape[0]
            else:
                end_index = prim_dim + cblocks[i + 1]
            condensed_dual_vars = cs.vertcat(condensed_dual_vars,
                                             dual_vars[prim_dim + cblocks[i]:end_index])

    return np.array(condensed_dual_vars).reshape(-1)


def get_hessblock_sizes(sparsity_pattern):
    """Compute the sizes of the individual Hessian blocks.

    Keyword arguments:
        sparsity_pattern  -- indices of the individual blocks for the vector of primal variables
    """
    hessblock_sizes = []
    for i in range(len(sparsity_pattern) - 1):
        hessblock_sizes += [sparsity_pattern[i + 1] - sparsity_pattern[i]]

    return hessblock_sizes

