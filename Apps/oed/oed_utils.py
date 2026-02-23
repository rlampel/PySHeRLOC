import casadi as cs


def lower_triangular_to_vector(S):
    """
    Extract the lower triangular part of a symmetric matrix S (CasADi SX/DM)
    and return it as a vector.
    """
    # Ensure S is square
    n_rows, n_cols = S.shape
    assert n_rows == n_cols, "Matrix must be square."

    # Use CasADi tril to get a mask of the lower triangular part
    mask = cs.tril(cs.DM.ones(S.shape))  # 1s in lower triangle, 0s elsewhere

    # Flatten and extract the non-zero entries (i.e., the lower triangle)
    vec = [S[i, j] for i in range(n_rows) for j in range(n_cols) if mask[i, j]]
    vec_out = cs.DM([])
    for k in range(len(vec)):
        vec_out = cs.vertcat(vec_out, vec[k])
    return vec_out


def vector_to_symmetric_matrix(v, n):
    """
    Convert a vector `v` of length n(n+1)/2 representing the lower triangular
    part of a symmetric matrix into the full symmetric CasADi matrix.

    Args:
        v: CasADi vector (SX or DM) of length n(n+1)/2
        n: size of the symmetric matrix (n x n)

    Returns:
        CasADi SX or DM matrix of shape (n, n)
    """
    assert v.shape[0] == n * (n + 1) // 2, "Invalid vector length for matrix reconstruction"

    S = cs.MX.zeros(n, n) if isinstance(v, cs.MX) else cs.SX.zeros(n, n)
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            S[i, j] = v[idx]
            S[j, i] = v[idx]  # Ensure symmetry
            idx += 1
    return S


def get_sens_der(G, f, x, p, p_fix, add_vars=cs.DM([])):
    dfx = cs.Function('dfx', [x, p, add_vars], [cs.jacobian(f(x, p, add_vars), x)])
    dfp = cs.Function('dfp', [x, p, add_vars], [cs.jacobian(f(x, p, add_vars), p)])

    G_matrix = cs.reshape(G, p.shape[0], -1).T
    rhs = dfx(x, p, add_vars) @ G_matrix + dfp(x, p, add_vars)
    rhs_func = cs.Function('rhs', [G, x, p, add_vars], [rhs])

    # turn matrix shaped differential equation into a vector
    G_dot = cs.reshape(rhs_func(G, x, p_fix, add_vars).T, -1, 1)

    return G_dot


def get_fisher_info(G, h, x, p, p_fix, w):
    num_pars = p.shape[0]
    num_meas = w.shape[0]
    F = cs.DM.zeros(num_pars, num_pars)

    dhx_func = cs.Function('dhx', [x], [cs.jacobian(h(x), x)])
    dhx = dhx_func(x)
    G_matrix = cs.reshape(G, p.shape[0], -1).T

    for i in range(num_meas):      # number of measurement variables
        ref = dhx[i, :]
        F += w[i] * (ref @ G_matrix).T @ (ref @ G_matrix)

    F_rhs = cs.Function('DF', [G, x, p, w], [F])
    F_vec = lower_triangular_to_vector(F_rhs(G, x, p_fix, w))
    return F_vec


# Criteria for OED
def oed_criterion(M, num_pars, name, is_inverse):
    # M can be the Fisher matrix or its inverse
    if name == "A":
        if not is_inverse:
            M = cs.inv(M)
        return cs.trace(M) / num_pars
    elif name == "D":
        if not is_inverse:
            # det(M^-1) = 1 / det(M)
            return (1 / cs.det(M))**(1 / num_pars)
        else:
            return cs.det(M)**(1 / num_pars)
    elif name == "M":
        if not is_inverse:
            M = cs.inv(M)
        diag = cs.diag(M)
        max_entry = cs.mmax(diag)
        return cs.sqrt(max_entry)
    else:
        if not is_inverse:
            M = cs.inv(M)
        denom = cs.trace(M) / num_pars
        return -1 / (denom**2)


def oed_criterion_lagr(M, num_pars, name, is_inverse):
    # M can be the Fisher matrix or its inverse
    if name == "A":
        if not is_inverse:
            M = cs.inv(M)
        return cs.trace(M) / num_pars
    elif name == "D":
        if not is_inverse:
            # det(M^-1) = 1 / det(M)
            return (1 / cs.det(M))**(1 / num_pars)
        else:
            return cs.det(M)**(1 / num_pars)
    elif name == "M":
        if not is_inverse:
            M = cs.inv(M)
        diag = cs.diag(M)
        max_entry = cs.mmax(diag)
        return cs.sqrt(max_entry)
    else:
        if not is_inverse:
            M = cs.inv(M)
        denom = cs.trace(M) / num_pars
        return -1 / (denom**2)
