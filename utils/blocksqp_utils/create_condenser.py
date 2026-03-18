import numpy as np
import os
from . import get_blocksqp_path
# Path to BlockSQP installation
blockSQP_path = str(get_blocksqp_path.get_path())
os.sys.path.append(blockSQP_path)
import blockSQP2 as blockSQP
# from blockSQP_pyProblem import blockSQP_pyProblem as Problemspec
from blockSQP2 import Problemspec


def condense_hessian(
        vblock_sizes, cblock_sizes, hessblock_sizes, vblock_dependencies,
        ub_var, lb_var, ub_con, lb_con, grad_obj, jac_g, meth_vars,
        primal_vars, dual_vars, grid,
        hess_type=1, c_start=None, c_end=None):
    """Condense the current Hessian approximation.

    Keyword arguments:
        vblock_sizes  -- sizes of blocks for dependent and independent variables
        cblock_sizes  -- sizes of constraint blocks
        hessblock_sizes  -- sizes of individual Hessian blocks
        ub_var, lb_var  -- upper and lower bounds for primal variables
        ub_con, lb_con  -- upper and lower bounds for the constraint function
        grad_obj  -- gradient of the objective function
        jac_g  -- Jacobian of the constraint function
        meth_vars  -- instance of SQPmethod containing the current Hessian approximation
        primal_vars  -- primal variables
        dual_vars  -- dual variables
        grid    -- dict containing discretization and lifting points
        hess_type  -- 1 to condense the SR1 matrix, 2 for the BFGS matrix
        c_start  -- index of first cblock that contains a matching condition
        c_end  -- index of last cblock that contains a matching condition
    """

    # Create condenser, get corresponding block indices
    vblocks = blockSQP.vblock_array(len(vblock_sizes))
    cblocks = blockSQP.cblock_array(len(cblock_sizes))
    hblocks = blockSQP.int_array(len(hessblock_sizes))
    num_lift_stages = sum(grid["lift"][1:])  # + 1
    targets = blockSQP.condensing_targets(1)

    if c_start is None:
        c_start = 0
    if c_end is None:
        c_end = len(cblock_sizes)
    targets[0] = blockSQP.condensing_target(
        num_lift_stages, 0, len(vblock_sizes), c_start, c_end
    )
    for i in range(len(vblock_sizes)):
        vblocks[i] = blockSQP.vblock(vblock_sizes[i], vblock_dependencies[i])
    for i in range(len(cblock_sizes)):
        cblocks[i] = blockSQP.cblock(cblock_sizes[i])
    for i in range(len(hessblock_sizes)):
        hblocks[i] = hessblock_sizes[i]

    cond_bounds = blockSQP.Condenser(vblocks, cblocks, hblocks, targets, 2)

    lb_var = np.array(lb_var).reshape(-1)
    ub_var = np.array(ub_var).reshape(-1)
    lb_con = np.array(lb_con).reshape(-1)
    ub_con = np.array(ub_con).reshape(-1)
    # grad obj - evaluated gradient objective
    grad_obj = np.array(grad_obj).reshape(-1)

    M_lb_var = blockSQP.Matrix(len(lb_var))
    M_ub_var = blockSQP.Matrix(len(ub_var))
    M_lb_con = blockSQP.Matrix(len(lb_con))
    M_ub_con = blockSQP.Matrix(len(ub_con))
    M_grad_obj = blockSQP.Matrix(len(grad_obj))

    np.array(M_lb_var, copy=False)[:, 0] = lb_var
    np.array(M_ub_var, copy=False)[:, 0] = ub_var
    np.array(M_lb_con, copy=False)[:, 0] = lb_con
    np.array(M_ub_con, copy=False)[:, 0] = ub_con
    np.array(M_grad_obj, copy=False)[:, 0] = grad_obj

    # get Jacobian of constraint function
    Jacobian = jac_g(primal_vars)
    nnz = int(Jacobian.nnz())
    m = int(Jacobian.shape[0])
    n = int(Jacobian.shape[1])
    nz = np.array(Jacobian.nz[:]).reshape(-1)
    row = np.array(Jacobian.row())
    colind = np.array(Jacobian.colind())

    A_nz = blockSQP.double_array(nnz)
    A_row = blockSQP.int_array(nnz)
    A_colind = blockSQP.int_array(n + 1)
    np.array(A_nz, copy=False)[:] = nz
    np.array(A_row, copy=False)[:] = row
    np.array(A_colind, copy=False)[:] = colind

    SM_Jacobian = blockSQP.Sparse_Matrix(m, n, A_nz, A_row, A_colind)

    hess = blockSQP.SymMat_array(len(hessblock_sizes))
    for i in range(len(hessblock_sizes)):
        if hess_type == 1:
            hess[i] = meth_vars.get_hess1_block(i)
        else:
            hess[i] = meth_vars.get_hess2_block(i)

    cond_args_bounds = blockSQP.condensing_args()
    cond_args_bounds.grad_obj = M_grad_obj
    cond_args_bounds.con_jac = SM_Jacobian
    cond_args_bounds.hess = hess
    cond_args_bounds.lb_var = M_lb_var
    cond_args_bounds.ub_var = M_ub_var
    cond_args_bounds.lb_con = M_lb_con
    cond_args_bounds.ub_con = M_ub_con

    # Condense with condenser that includes dependent variable bounds
    cond_bounds.condense_args(cond_args_bounds)

    return cond_args_bounds.condensed_hess

