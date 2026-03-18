from . import get_blocksqp_path
import os
# Path to BlockSQP installation
blockSQP_path = str(get_blocksqp_path.get_path())
os.sys.path.append(blockSQP_path)
import py_blockSQP as blockSQP


def get_blocksqp_options(exact_hess=False):
    """Return the options for blockSQP.

    Keyword arguments:
        exact_hess -- boolean that indicates whether the exact Hessian shall be used.
    """
    opts = blockSQP.SQPoptions()
    opts.max_QP_it = 10000
    opts.max_QP_secs = 5.0
    opts.max_conv_QPs = 4  # 1
    opts.sparse = True
    opts.conv_strategy = 1  # 2 for vblocks
    # line search
    # opts.enable_linesearch = False

    # use exact Hessian
    if exact_hess:
        opts.exact_hess = 2  # 0, (2 für exakt)
    else:
        opts.exact_hess = 0

    opts.hess_approx = 1
    opts.sizing = 2
    opts.fallback_approx = 2
    opts.fallback_sizing = 4
    # set to false to enable condensing
    opts.lim_mem = False
    opts.mem_size = 20
    opts.opt_tol = 1e-6
    opts.feas_tol = 1e-6
    opts.automatic_scaling = 0
    opts.max_extra_steps = 0
    opts.enable_premature_termination = True
    opts.max_filter_overrides = 0
    opts.par_QPs = False
    opts.qpsol = 'qpOASES'

    return opts

