import casadi as cs
from . import ode_solver, penalty


def get_sensitivity_old(curr_problem, partial_grid, controls, multipliers={}):
    # mu_s = multipliers.get("state", 0)
    # mu_t = multipliers.get("objective", 1)
    s_dim = curr_problem.s_dim
    Sk = cs.MX.sym('Sk', s_dim)
    q_dim = curr_problem.q_dim

    ode = curr_problem.get_ode()
    end_time = curr_problem.get_grid_details()

    J = 0
    punish_state = 0

    # input for integrator
    init = {}
    # init["controls"] = controls
    controls = cs.MX.sym('c', controls.shape[0])
    init["controls"] = controls
    init["q_dim"] = q_dim

    Sk_temp = Sk
    curr_time_points = partial_grid["part_time"]
    control_points = partial_grid["control"]

    for k in range(len(curr_time_points) - 1):
        init["s"] = Sk_temp
        Sk_temp, Jk = ode_solver.integrate_interval(init, control_points, ode,
                                                    curr_time_points[k], curr_time_points[k + 1])
        J += Jk
        state, ubs, lbs = curr_problem.state_bounds(Sk_temp)
        punish_state += penalty.penalty(state, ubs, lbs)

    if (end_time in partial_grid["part_time"]):
        state, ubs, lbs = curr_problem.end_bounds(Sk_temp)
        punish_state += penalty.penalty(state, ubs, lbs)
        J += curr_problem.objective_end(Sk_temp)

    comb = cs.vertcat(Sk, controls)
    Integral_Func = cs.Function("Int", [comb], [cs.vertcat(J, Sk_temp)])
    return Integral_Func


def get_sensitivity(curr_problem, partial_grid, controls, multipliers={}, mayer=True):
    s_dim = curr_problem.s_dim
    Sk = cs.MX.sym('Sk', s_dim)
    q_dim = curr_problem.q_dim

    ode = curr_problem.get_ode()
    end_time = curr_problem.get_grid_details()

    # input for integrator
    init = {}
    controls = cs.MX.sym('c', controls.shape[0])
    init["controls"] = controls
    init["q_dim"] = q_dim

    J_total = 0
    Sk_temp = Sk
    curr_time_points = partial_grid["part_time"]
    control_points = partial_grid["control"]

    for k in range(len(curr_time_points) - 1):
        init["s"] = Sk_temp
        Sk_temp, J_temp = ode_solver.integrate_interval(
            init, control_points, ode,
            curr_time_points[k], curr_time_points[k + 1]
        )
        J_total += J_temp

    # print("end time: ", end_time, ", partial grid: ", partial_grid)
    if (end_time == partial_grid["part_time"][-1]) and mayer:
        J_total = curr_problem.objective_end(Sk_temp) + J_total
        # Sk_temp = 0  # cs.DM([])
        # print("end point included")

    comb = cs.vertcat(Sk, controls)
    Integral_Func = cs.Function("Int", [comb], [Sk_temp])
    J_Func = cs.Function("Int", [comb], [J_total])
    return Integral_Func, J_Func


def eval_norm(problem, grid, controls, s_init, start, stop, mayer=True):
    s_dim = problem.s_dim

    # get current partial time interval
    grid["part_time"] = grid["time"][start:stop]
    Integral_Func, J_Func = get_sensitivity(problem, grid, controls, mayer=mayer)

    # create symbolic variables for states and controls
    S = cs.MX.sym('S', s_dim)
    C = cs.MX.sym('C', controls.shape[0])
    comb = cs.vertcat(S, C)

    # compute the derivative
    DInt = cs.Function("DInt", [comb], [cs.jacobian(Integral_Func(comb), comb)])
    DJ_total = cs.Function("DInt", [comb], [cs.jacobian(J_Func(comb), comb)])

    # evaluate for current states and controls
    curr_s = s_init[start * s_dim:(start + 1) * s_dim]
    comb_input = cs.vertcat(curr_s, controls)
    d_int_comb = DInt(comb_input)
    d_j_comb = DJ_total(comb_input)
    d_int_norm = cs.norm_fro(d_int_comb)**2
    d_j_norm = cs.norm_fro(d_j_comb)**2
    # print("Norm is ", dcomb_norm)
    return d_int_norm, d_j_norm


def refine_lifting(problem, init, grid):
    time_points = grid["time"]
    s_init = init["s_init"]
    controls = init["controls"]
    num_time_points = len(time_points)
    curr_end = num_time_points
    mayer = True

    # determine sensitivity for single shooting
    d_int_norm_s, d_j_norm_s = eval_norm(problem, grid, controls, s_init, 0, num_time_points)
    # min_norm_s = cs.fmin(d_int_norm_s, d_j_norm_s)
    print(f"single shooting norms: {d_int_norm_s, d_j_norm_s}")

    # assume that the ODE can be evaluated solved over the entire time interval
    lifting_points = [0] * num_time_points
    # iterate backwards over all time points
    for i in reversed(range(1, num_time_points)):
        d_int_norm, d_j_norm = eval_norm(problem, grid, controls, s_init, i, curr_end,
                                         mayer=mayer)
        contains_end = (curr_end == num_time_points)
        print(f"Interval: {time_points[i:curr_end]}")
        print(f" current norms: {d_int_norm, d_j_norm}")

        # current interval contains the end, only lift if the total sensitivity goes down
        if contains_end and mayer:
            d_int_norm_compl, d_j_norm_compl = eval_norm(problem, grid, controls, s_init,
                                                         0, i + 1, mayer=False)
            curr_max = float(cs.fmax(d_int_norm, d_j_norm))
            comp_max = float(cs.fmax(d_int_norm_compl, d_j_norm_compl))
            print(f" complement norms: {d_int_norm_compl, d_j_norm_compl}\n")
            if float(cs.fmax(curr_max, comp_max)) <= d_j_norm_s and float(d_j_norm) > 0:
                # if (d_int_norm_compl < d_int_norm_s) and d_j_norm_compl < d_j_norm_s:
                print(" -> Add lifting with end included")
                lifting_points[i] = 1
                mayer = False
                curr_end = i + 1

        # else lift if both sensitivities decrease individually
        elif d_int_norm <= d_int_norm_s and d_j_norm <= d_j_norm_s:
            # check complement of current interval
            d_int_norm_compl, d_j_norm_compl = eval_norm(problem, grid, controls, s_init,
                                                         0, i + 1)
            print(f" complement norms: {d_int_norm_compl, d_j_norm_compl}\n")
            if (d_int_norm_compl <= d_int_norm_s) and d_j_norm_compl <= d_j_norm_s:
                # d_int_norm_s, d_j_norm_s = d_int_norm_compl, d_j_norm_compl
                print(" -> Add lifting point")
                lifting_points[i] = 1
                curr_end = i + 1
    return lifting_points


def get_grid_sens(problem, init, grid):
    time_points = grid["time"]
    s_init = init["s_init"]
    controls = init["controls"]
    num_time_points = len(time_points)
    norm_list = []

    # determine sensitivity for single shooting
    d_int_norm_s, d_j_norm_s = eval_norm(
        problem, grid, controls, s_init, 0, num_time_points
    )
    # min_norm_s = cs.fmin(d_int_norm_s, d_j_norm_s)
    print(f"single shooting norms: {d_int_norm_s, d_j_norm_s}")

    # assume that the ODE can be evaluated solved over the entire time interval
    lift_indc = [i for i in range(num_time_points) if grid["lift"][i] or i == 0]
    # print("lifting indices: ", lift_indc)
    # iterate backwards over all time points
    for i in range(len(lift_indc) - 1):
        start, end = lift_indc[i], lift_indc[i + 1]
        d_int_norm, d_j_norm = eval_norm(
            problem, grid, controls, s_init, start, end + 1
        )
        # print(f"Interval: {time_points[start:end + 1]}")
        # print(f" current norms: {d_int_norm, d_j_norm}")
        norm_list += [[d_int_norm, d_j_norm]]

    max_state_sens = max([float(norm_list[i][0]) for i in range(len(norm_list))])
    max_obj_sens = max([float(norm_list[i][1]) for i in range(len(norm_list))])
    print(f"Multiple shooting norms: {max_state_sens, max_obj_sens}")
    print(f"Norm list: {norm_list}")
    # the total sensitivity of the system decreases
    if max_state_sens + max_obj_sens <= d_int_norm_s + d_j_norm_s:
        print("Sensitivity decreased by lifting")
        return True

    return False
