import casadi as cs
import numpy as np
import timeit
import matplotlib.pyplot as plt
from .. import create_nlp, adapt_init, penalty
from . import get_blocksqp_path
from . import log_conv_data
from . import blocksqp_options, get_block_sizes
from .sort_vars import sort_back, sort_vars_by_time
from .blocksqp_init_heuristics import fsinit_merit
from .blocksqp_init_heuristics import refine_lifting, fsinit_heur_new
from .auto_condensing_heuristic import trigger_auto_condensing
import os
import time
from Tkinter_plots import plot_gui
# Path to BlockSQP installation
blockSQP_path = str(get_blocksqp_path.get_path())
os.sys.path.append(blockSQP_path)
import blockSQP2 as blockSQP
# from blockSQP_pyProblem import blockSQP_pyProblem as Problemspec
from blockSQP2 import Problemspec

# global variable to plot lifting points
plot_lift = []


def graph_lift_heuristic(xi_temp, lam_temp, sort_grid, grid, ode, s_dim, q_dim,
                         starting_times, curr_problem,
                         num_control_points, num_time_points, iter=0):
    global plot_lift  # only for plotting

    if iter > 3:
        plot_lift = [0] * num_time_points
        return xi_temp

    print(np.linalg.norm(lam_temp, np.inf))
    # sort back
    xi_temp = np.array(sort_back(xi_temp, sort_grid, s_dim, q_dim))

    q_temp = xi_temp[:q_dim * num_control_points]
    s_temp = xi_temp[q_dim * num_control_points:]
    mu = np.linalg.norm(np.array(lam_temp).reshape(-1, 1), np.inf)

    s_init, plot_lift = refine_lifting(curr_problem, grid, starting_times,
                                       s_temp, q_temp, mu)
    xi_temp = np.array(cs.vertcat(q_temp, s_init))

    xi_temp = np.array(sort_vars_by_time(xi_temp, grid, s_dim, q_dim)[0])
    return xi_temp.reshape(-1)


def to_blocks_LT(sparse_hess: cs.DM, hessBlock_index):
    blocks = []
    for j in range(len(hessBlock_index) - 1):
        blocks.append(
            np.array(cs.tril(sparse_hess[hessBlock_index[j]:hessBlock_index[j + 1],
                             hessBlock_index[j]:hessBlock_index[j + 1]].full()).nz[:],
                     dtype=np.float64).reshape(-1)
        )
    return blocks


def create_blocksqp_problem(curr_problem, grid, start_point, GUI, input_opts,
                            i_start=0, log=None, sr1_init=None, bfgs_init=None,
                            lam_init=None, accepted_hess_init=[], condense_mode="default"):
    global plot_lift
    max_iter = input_opts.get("max_iter", 100)
    refinement = input_opts.get("refinement", -1)
    exact_hess = input_opts.get("exact_hess", False)
    optim_lamb = input_opts.get("optim_lamb", False)
    optim_init = input_opts.get("optim_init", False)
    always_auto = input_opts.get("always_auto", False)
    auto_condense = input_opts.get("auto_condense", False)
    verbose = input_opts.get("verbose", True)
    plot_iter = input_opts.get("plot_iter", True)
    log_results = input_opts.get("log_results", True)

    opts = blocksqp_options.get_blocksqp_options(exact_hess)

    # allow the solver to use indefinite Hessian (approximations) immediately
    if sr1_init is not None or (exact_hess and i_start > 0):
        opts.indef_delay = 0

    x, lbx, ubx, g, lbg, ubg, J, cblocks, cblock_match = create_nlp.create_nlp(
        curr_problem, grid, ret_cblocks=True
    )
    if plot_iter:
        # get control and stated labels if available
        state_labels, control_labels = curr_problem.state_labels, curr_problem.control_labels
        state_indices = curr_problem.state_indices
        state_scales = curr_problem.state_scales
        toolbar, canvas, fig = GUI

    lbg, ubg = np.array(lbg).reshape(-1), np.array(ubg).reshape(-1)
    lbx, ubx = np.array(lbx).reshape(-1), np.array(ubx).reshape(-1)

    x = cs.vertcat(*x)
    g = cs.vertcat(*g)
    s_dim = curr_problem.s_dim
    q_dim = curr_problem.q_dim
    ode = curr_problem.get_ode()

    if verbose:
        print("INITIAL LIFTING: ", grid["lift"])

    num_time_points = len(grid["time"])
    starting_times = [grid["time"][i] for i in range(num_time_points) if grid["lift"][i] or i == 0]

    lift_grid = grid.copy()
    lift_grid["time"] = starting_times

    # separate the supplied point into control and state variables
    num_control_points = len(grid["control"])
    q_temp = start_point[:q_dim * num_control_points]
    s_temp = start_point[q_dim * num_control_points:]

    s_init = s_temp
    xi_temp = np.array(cs.vertcat(q_temp, s_init))

    # sort the variables and obtain the sparsity pattern
    start_point = cs.vertcat(q_temp, s_init)
    start_point, sparsity_pattern, sort_grid = sort_vars_by_time(start_point, grid, s_dim, q_dim)

    # create resorting filter
    Y = cs.MX.sym("Y", x.shape[0])
    Lam = cs.MX.sym("Lam", g.shape[0])
    Y_sort = sort_back(Y, sort_grid, s_dim, q_dim)
    lbx = list(np.array(sort_vars_by_time(lbx, grid, s_dim, q_dim)[0]).flatten())
    ubx = list(np.array(sort_vars_by_time(ubx, grid, s_dim, q_dim)[0]).flatten())
    sort_func = cs.Function("sort", [Y], [Y_sort])

    # unsorted functions
    func_g_u = cs.Function("g", [x], [g])
    func_f_u = cs.Function("F", [x], [J])

    # sorted functions
    func_g = cs.Function("g", [Y], [func_g_u(sort_func(Y))])
    func_f = cs.Function("F", [Y], [func_f_u(sort_func(Y))])

    problem = Problemspec()

    lag_func = cs.Function("lag", [Y, Lam], [func_f(Y) - Lam.T @ func_g(Y)])
    lag_der = cs.Function("lag_der", [Y, Lam], [cs.jacobian(lag_func(Y, Lam), Y)])
    lag_hess = cs.Function("lag_hess", [Y, Lam], [cs.jacobian(lag_der(Y, Lam), Y)])

    def lag_hess_print(xi, lam):
        hess_creation_time = timeit.default_timer()
        res = lag_hess(xi, lam)
        hess_time = timeit.default_timer() - hess_creation_time
        print("Computing the exact Hessian took: ", hess_time)
        return res

    if exact_hess:
        # create exact Hessian
        problem.hess = lambda xi, lambd: to_blocks_LT(lag_hess_print(xi, lambd), sparsity_pattern)

    problem.g = lambda arg_x: np.array(func_g(arg_x)).reshape(-1)
    problem.f = lambda arg_x: np.array(func_f(arg_x)).reshape(-1)
    problem.nVar = x.shape[0]
    problem.nCon = g.shape[0]
    problem.set_blockIndex(sparsity_pattern)
    problem.set_bounds(lbx, ubx, lbg, ubg)

    # sorted functions
    problem.grad_f = cs.Function('grad_f', [Y], [cs.jacobian(func_f(Y), Y)])
    problem.jac_g = cs.Function('jac_g', [Y], [cs.jacobian(func_g(Y), Y)])
    jac_g_start = problem.jac_g(start_point)
    problem.make_sparse(jac_g_start.nnz(), jac_g_start.row(), jac_g_start.colind())
    problem.jac_g_nz = lambda x: np.array(problem.jac_g(x).nz[:]).reshape(-1)

    start_point = np.array(start_point).reshape(-1)

    if optim_lamb:
        lam_opt = adapt_init.get_best_lam(start_point, lbx, ubx,
                                          func_g, lbg, ubg,
                                          problem.grad_f(start_point), jac_g_start)
        print("L1 Norm: ", cs.norm_inf(lam_opt))
        problem.lam_start = lam_opt
    elif lam_init is not None:
        # initialize lambda with given values (e.g. obtained via condensing)
        problem.lam_start = lam_init
    else:
        problem.lam_start = np.ones(problem.nVar + problem.nCon,
                                    dtype=np.float64).reshape(-1)

    if optim_init:
        print(f"Optimize Init with optimal Lambda: {cs.norm_inf(problem.lam_start)}")
        start_point = graph_lift_heuristic(
            start_point, problem.lam_start, sort_grid, grid, ode, s_dim, q_dim,
            starting_times, curr_problem,
            num_control_points, num_time_points
        )

    problem.x_start = start_point

    # determine optimal Lagrange multipliers for exact hessian
    Lam_full = cs.MX.sym("Lam_full", x.shape[0] + g.shape[0])
    full_lag_func = cs.Function("lag_full", [Y, Lam_full],
                                [func_f(Y) - Lam_full.T @ cs.vertcat(Y, func_g(Y))])
    full_lag_der = cs.Function("lag_full_der", [Y, Lam_full],
                               [cs.jacobian(full_lag_func(Y, Lam_full), Y)])

    accepted_hess = accepted_hess_init

    if always_auto:
        def fsinit(xi_temp):
            return fsinit_heur_new(
                xi_temp, sort_grid, grid, curr_problem
            )

        def step_modifier(xi_temp, lam_temp_in):
            xi_temp[:] = fsinit_merit(
                xi_temp, fsinit, problem.lam_start, lbg, ubg, lbx, ubx, func_f, func_g,
                lam_new=lam_temp_in,
                exact_hess=exact_hess, lag_der=full_lag_der,
                mode=condense_mode
            )
            return 0
        problem.set_stepModification(step_modifier)

    elif refinement != -1:
        def step_modifier(xi_temp, lam_temp):
            xi_temp[:] = graph_lift_heuristic(
                xi_temp, problem.lam_start, sort_grid, grid, ode, s_dim, q_dim,
                starting_times, curr_problem,
                num_control_points, num_time_points)
            return 0
        problem.set_stepModification(step_modifier)

    if log_results and log is None:
        log = log_conv_data.init_logs()
        log = log_conv_data.add_log_entry(
            log, start_point, problem.lam_start,
            ubx, lbx, ubg, lbg, func_f, func_g, full_lag_der,
        )

    problem.complete()

    stats = blockSQP.SQPstats("./")
    meth = blockSQP.SQPmethod(problem, opts, stats)

    meth.init()

    if sr1_init is not None:
        meth.vars.set_hess1(sr1_init)
    if bfgs_init is not None:
        meth.vars.set_hess2(bfgs_init)

    ret = 0
    i = i_start

    default_init = {}
    default_init["s_dim"] = curr_problem.s_dim
    default_init["q_dim"] = curr_problem.q_dim
    num_control_points = len(grid["control"])

    old_point = np.array(start_point).reshape(-1)
    old_lambd = np.array(problem.lam_start).reshape(-1)

    # used for switching to single shooting
    kkt_norm_list = []

    while ret == 0 and i < max_iter:
        ret = int(meth.run(1, 1).value)  # second argument: 0 restart, 1 keeps previous approximation
        i += 1
        xi_temp = np.array(meth.get_xi()).reshape(-1)
        lam_temp = meth.get_lambda()

        if False:
            # compute lagrange multipliers which minimze the KKT error
            jac_g_curr = problem.jac_g(xi_temp)
            lam_temp_np = np.array(lam_temp).reshape(-1, 1)
            lam_temp_np = adapt_init.get_best_lam(xi_temp, lbx, ubx,
                                                  func_g, lbg, ubg,
                                                  problem.grad_f(xi_temp), jac_g_curr)
            np.array(lam_temp, copy=False)[:] = lam_temp_np.reshape(-1, 1)
            meth.set_iterate_(meth.get_xi(), lam_temp, False)  # third argument to reset QN-Hessian
            lam_temp = meth.get_lambda()

        # accepted_hess += [meth.vars.hess_num_accepted]
        accepted_hess += [meth.vars.QP_num_accepted]

        # get norm of rhs
        if log_results:
            log = log_conv_data.add_log_entry(
                log, xi_temp, np.array(lam_temp).reshape(-1, 1),
                ubx, lbx, ubg, lbg, func_f, func_g, full_lag_der,
                old_point, old_lambd
            )

            '''
            # compute kappa values of Hessian approximation
            log = log_conv_data.add_kappa(
                log, xi_temp, old_point, np.array(lam_temp).reshape(-1, 1),
                meth.vars, lag_hess, lag_der, problem.jac_g, sparsity_pattern,
                hess_type=1
            )
            '''

        default_init["sol"] = np.array(sort_back(xi_temp, sort_grid, s_dim, q_dim))

        # store current iterate as old point
        prim_step = xi_temp - old_point
        dual_step = np.array(lam_temp).reshape(-1) - old_lambd
        old_point = xi_temp
        old_lambd = np.array(lam_temp).reshape(-1)

        change_lift = False
        if refinement != -1:
            change_lift = True
            # update heuristic with new Lagrange multipliers

            def step_modifier(xi_temp, lam_temp_in):
                xi_temp[:] = graph_lift_heuristic(
                    xi_temp, lam_temp, sort_grid, grid, ode, s_dim, q_dim,
                    starting_times, curr_problem,
                    num_control_points, num_time_points, i)
                return 0
            problem.set_stepModification(step_modifier)

        elif always_auto:
            # update the old point
            def step_modifier(xi_temp, lam_temp_in):
                xi_temp[:] = fsinit_merit(
                    xi_temp, fsinit, np.array(lam_temp).reshape(-1, 1),
                    lbg, ubg, lbx, ubx, func_f, func_g,
                    lam_new=lam_temp_in,
                    opt_err=meth.vars.tol, exact_hess=exact_hess,
                    lag_der=full_lag_der, mode=condense_mode

                )
                return 0
            problem.set_stepModification(step_modifier)

        if plot_iter:
            plt.clf()

            # get time scale for plots
            if plot_iter:
                time_scale_ind = curr_problem.time_scale_ind
                if not (np.isnan(float(time_scale_ind))):
                    time_scale = float(default_init["sol"][time_scale_ind])
                    print("current scale: ", time_scale)
                else:
                    time_scale = 1

            ax = plot_gui.plot_segmented([toolbar, canvas, fig],
                                         default_init, grid, ode,
                                         state_labels,
                                         control_labels,
                                         state_indices=state_indices,
                                         state_scales=state_scales,
                                         time_scale=time_scale)

            # plot vertical lines
            if change_lift:
                yl, yu = ax.get_ylim()
                ax.vlines([grid["time"][k] for k in range(len(grid["time"])) if plot_lift[k] == 1],
                          yl, yu, color="gray", linestyles="dashed", alpha=0.5)

            ax.set_title(f"Iteration {i}", fontsize=22)
            canvas.draw()
            canvas.flush_events()

        if auto_condense:
            # violation of constraints g
            curr_g_eval = np.array(meth.vars.constr).reshape(-1)
            g_viol = penalty.get_violation(curr_g_eval, ubg, lbg)
            # violation of variable bounds
            constr_viol = cs.vertcat(g_viol, penalty.get_violation(xi_temp, ubx, lbx))
            # current scaled optimality error
            curr_opt = meth.vars.tol
            kkt_norm_list += [float(curr_opt)]

            # ensure that there are only small Hessian updates left
            prim_step_norm = float(cs.norm_2(prim_step)**2)
            dual_step_norm = float(cs.norm_2(dual_step)**2)
            step_norm = prim_step_norm + dual_step_norm

            curr_prim_norm = cs.norm_2(xi_temp)**2
            curr_dual_norm = cs.norm_2(np.array(lam_temp).reshape(-1))**2
            curr_norm = curr_prim_norm + curr_dual_norm

            constr_viol_norm = cs.norm_2(constr_viol)
            rel_step_norm = cs.sqrt(step_norm / curr_norm)

            condense_now = trigger_auto_condensing(
                grid, sort_grid, num_control_points, cblocks, cblock_match,
                xi_temp, kkt_norm_list,
                s_dim, q_dim, g_viol, accepted_hess,
                curr_opt, step_norm, rel_step_norm, constr_viol_norm,
                exact_hess, mode=condense_mode
            )

            if ret == 0 and condense_now:
                vblock_sizes, vblock_dependencies = get_block_sizes.get_vblock_sizes(
                    s_dim, q_dim, grid["lift"]
                )
                cblock_sizes, c_start, c_end = get_block_sizes.get_cblock_sizes(
                    s_dim, g, cblocks, cblock_match
                )
                hessblock_sizes = get_block_sizes.get_hessblock_sizes(sparsity_pattern)

                from . import create_condenser
                if exact_hess:
                    condensed_sr1 = None
                else:
                    condensed_sr1 = create_condenser.condense_hessian(
                        vblock_sizes, cblock_sizes, hessblock_sizes, vblock_dependencies,
                        ubx, lbx, ubg, lbg, problem.grad_f(xi_temp), problem.jac_g, meth.vars,
                        xi_temp, lam_temp, grid, hess_type=1, c_start=c_start, c_end=c_end
                    )

                condensed_bfgs = None

                print("#" * 30 + "\nSTART OVER WITH SMALLER QP\n" + "#" * 30)
                num_lift_points = sum(grid["lift"][1:])
                grid["lift"] = [0] * num_time_points
                xi_temp_sort = sort_back(xi_temp, sort_grid, s_dim, q_dim)
                # just the controls and the state values at time 0
                xi_cond = xi_temp_sort[:q_dim * num_control_points + s_dim]

                if num_lift_points + 1 == len(grid["lift"]):
                    lam_cond = get_block_sizes.condense_dual_vars(
                        np.array(lam_temp).reshape(-1), xi_temp.shape[0], cblocks, cblock_match
                    )
                else:
                    lam_cond = None

                # disable auto condensing and FSInit for the resulting single shooting problem
                input_opts["auto_condense"] = False
                input_opts["always_auto"] = False
                return create_blocksqp_problem(curr_problem, grid, xi_cond, GUI,
                                               input_opts, i_start=i, log=log,
                                               sr1_init=condensed_sr1,
                                               bfgs_init=condensed_bfgs,
                                               lam_init=lam_cond,
                                               accepted_hess_init=accepted_hess)

    if log_results:
        log_conv_data.print_logs(log)
        merit_arr = [np.abs(el - log["merit"][-1]) for el in log["merit"]]
        if plot_iter:
            time.sleep(2)
            plt.clf()
            ax = plot_gui.plot_conv(
                [toolbar, canvas, fig],
                [log["conv"]] + [merit_arr] + [log["step"]],
                ["rhs conv", "merit conv", "step sizes"],
                log["viol"], log["obj"])
            canvas.draw()
            canvas.flush_events()

    # meth.finish()
    print("Return Value: ", ret)
    if ret < 0:
        i = cs.inf

    return i, meth, accepted_hess

