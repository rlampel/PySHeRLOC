import casadi as cs
import numpy as np
from . import nlp_callback as cb
import utils.create_nlp as create_nlp


def condense_ipopt_nlp(darg, plot_details):
    x = darg['x']
    lam_x = darg['lam_x']
    lam_g = darg['lam_g']
    # remove lifting variables from x
    grid = plot_details["grid"]
    num_control_points = len(grid["control"])
    num_lifting_points = sum(grid["lift"][1:])
    grid["lift"] = [0] * len(grid["lift"])
    curr_problem = plot_details["problem"]
    s_dim = curr_problem.s_dim
    q_dim = curr_problem.q_dim
    init = x[:num_control_points * q_dim + s_dim]

    w, lbw, ubw, g, lbg, ubg, J, cblocks, cblock_match = create_nlp.create_nlp(
        curr_problem, grid, ret_cblocks=True
    )

    # default options
    plot_details["grid"]["lift"] = grid["lift"]
    mycallback = cb.MyCallback('mycallback',
                               cs.vertcat(*w).shape[0],
                               cs.vertcat(*g).shape[0],
                               0, plot_details)

    prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
    opts = {}

    opts['ipopt.print_level'] = 5
    opts['iteration_callback'] = mycallback

    # only works with exact Hessian
    opts["ipopt.tol"] = 1.e-6
    opts["ipopt.warm_start_init_point"] = "yes"
    opts['ipopt.max_iter'] = 200
    solver = cs.nlpsol('solver', 'ipopt', prob, opts)

    lam_x_cond, lam_g_cond = condense_duals_ipopt(lam_x, lam_g,
                                                  num_control_points, num_lifting_points,
                                                  s_dim, q_dim, len(g))

    solver(x0=init, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg,
           lam_g0=lam_g_cond, lam_x0=lam_x_cond)
    return solver.stats()


def condense_duals_ipopt(lam_x, lam_g, num_controls, num_lifts, x_dim, q_dim, cond_g_shape):
    """Delete all Lagrange multipliers that correspond to matching conditions

    Keyword arguments:
        dual_vars  -- Lagrange multipliers
        prim_dim  -- total number of primal variables
        cblocks  -- indices of constraint blocks
        cblock_match  -- indicates whether a cblock is a matching condition
    """
    # eliminate variable bounds for lifted states
    lam_x_cond = lam_x[:num_controls * q_dim + x_dim]
    # add former variable bounds as constraints eliminate matching conditions
    lam_g_cond = lam_x[num_controls * q_dim + x_dim:]
    lam_g_cond = np.append(lam_g_cond, lam_g[num_lifts * x_dim:])

    return lam_x_cond, lam_g_cond

