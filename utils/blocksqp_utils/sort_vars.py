import casadi as cs


def sort_vars_by_time(prim_vars, grid, s_dim, q_dim):
    """Sort the primary variables by time so that the Hessian has a block structure.

    Keyword arguments:
        prim_vars -- primal variables
        grid    -- dict containing discretization and lifting points
        s_dim -- number of differential states
        q_dim -- number of controls
    """
    time_points = grid["time"]
    control_points = grid["control"]
    lifting_points = grid["lift"]
    num_control_points = len(control_points)

    q_temp = prim_vars[:q_dim * num_control_points]
    s_temp = prim_vars[q_dim * num_control_points:]

    sparsity_pattern = []
    sorting_grid = []
    sorted_vars = cs.DM([])

    curr_state_ind = 0
    curr_contr_ind = 0
    total_var_count = 0
    # iterate over all lifting points except the last one
    for i in range(len(time_points) - 1):
        # add state if there is a lifting point
        if lifting_points[i] or i == 0:
            sorting_grid += [1]
            sparsity_pattern += [total_var_count]
            curr_state = s_temp[curr_state_ind * s_dim:(curr_state_ind + 1) * (s_dim)]
            sorted_vars = cs.vertcat(sorted_vars, curr_state)
            curr_state_ind += 1
            total_var_count += s_dim
            # print(f"add lifting point number {i}")
        # iterate over all control points
        for j in range(curr_contr_ind, num_control_points):
            # stop if the control point belongs to the next interval
            if control_points[j] >= time_points[i + 1]:
                break
            sorting_grid += [0]
            curr_contr = q_temp[curr_contr_ind * q_dim:(curr_contr_ind + 1) * (q_dim)]
            sorted_vars = cs.vertcat(sorted_vars, curr_contr)
            curr_contr_ind += 1
            total_var_count += q_dim
            # print(f"add control point number {j}")

    # add last state at end of time interval
    if lifting_points[-1]:
        sorting_grid += [1]
        sparsity_pattern += [total_var_count]
        curr_state = s_temp[-s_dim:]
        sorted_vars = cs.vertcat(sorted_vars, curr_state)
        total_var_count += s_dim
    sparsity_pattern += [total_var_count]
    # print("Sorted with sparsity pattern: ", sparsity_pattern)
    # print("Sorting grid: ", sorting_grid)
    return sorted_vars, sparsity_pattern, sorting_grid


def sort_back(prim_vars, sort_grid, s_dim, q_dim):
    """Sort the primal variables back to their original order.

    Keyword arguments:
        prim_vars -- primal variables
        sorting_grid -- positions of shooting variables obtained by sorting
        s_dim -- number of differential states
        q_dim -- number of controls
    """
    q_temp = cs.DM([])
    s_temp = cs.DM([])

    total_var_count = 0
    # iterate over all items in the sorting_grid
    for s in sort_grid:
        if s:
            curr_state = prim_vars[total_var_count:total_var_count + s_dim]
            s_temp = cs.vertcat(s_temp, curr_state)
            total_var_count += s_dim
        else:
            curr_contr = prim_vars[total_var_count:total_var_count + q_dim]
            q_temp = cs.vertcat(q_temp, curr_contr)
            total_var_count += q_dim

    return cs.vertcat(q_temp, s_temp)

