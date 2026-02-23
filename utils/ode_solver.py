import numpy as np
import casadi as cs


def my_rk(ode, start_time, end_time):
    """Create a fixed step Runge-Kutta 4 integrator

    Keyword arguments:
        ode -- struct containing values for 'x', 'p', 'ode', and 'quad'
        start_time  -- beginning of integration interval
        end_time  -- end of integration interval
    """
    x, u = ode['x'], ode['p']
    xdot, L = ode['ode'], ode['quad']
    # add explicit time dependence
    t = ode.get('t', cs.MX.sym('t'))
    x_dim = x.shape[0]
    u_dim = u.shape[0]
    # Fixed step Runge-Kutta 4 integrator
    h = end_time - start_time
    M = 4  # RK4 steps per interval
    DT = h / M
    f = cs.Function('f', [x, u, t], [xdot, L])
    X0 = cs.MX.sym('X0', x_dim)
    U = cs.MX.sym('U', u_dim)
    X = X0
    Q = 0
    for j in range(M):
        k1, k1_q = f(X, U, start_time)
        k2, k2_q = f(X + DT / 2 * k1, U, start_time + DT / 2)
        k3, k3_q = f(X + DT / 2 * k2, U, start_time + DT / 2)
        k4, k4_q = f(X + DT * k3, U, start_time + DT)
        X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
    F = cs.Function('F', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])
    return F


def get_control(curr_time, control_points, controls, c_dim):
    """Return the value of the control variable for the current time.

    Keyword arguments:
        curr_time   -- current time point
        control_points  -- discretization points of the control
        controls    -- values of the control discretization
        c_dim   -- dimension of the control variable
    """
    num_controls = len(control_points)
    for i in range(num_controls):
        if (control_points[i] > curr_time):
            return controls[(i - 1) * c_dim:i * c_dim]
    return controls[(num_controls - 1) * c_dim:num_controls * c_dim]


def get_next_control_time(control_points, curr_time):
    """Return the next time point when the control changes.

    Keyword arguments:
        curr_time   -- current time point
        control_points  -- discretization points of the control
    """
    for el in control_points:
        if (el > curr_time):
            return el
    return np.inf


# integrate for given control and starting value
def integrate_const_control(init, ode, start_time, end_time):
    """Integrate the ODE on an interval with constant control.

    Keyword arguments:
        curr_time   -- current time point
        control_points  -- discretization points of the control
    """
    curr_s = init["s"]
    curr_q = init["q"]
    Int = cs.integrator('F', 'rk', ode, start_time, end_time)
    Ik = Int(x0=curr_s, p=curr_q)
    # Int = my_rk(ode, start_time, end_time)
    # xf, qf = Int(curr_s, curr_q)
    # Ik = {"xf": xf, "qf": qf}
    return Ik


def integrate_interval(init, control_points, ode, start_time, end_time):
    """Integrate the ODE on a given time interval.

    Keyword arguments:
        init    -- dict containing information about the initial value
        control_points  -- discretization points of the control
        ode -- casadi function that depends on the control
        start_time  -- initial time point
        end_time    -- final time point
    """
    curr_time = start_time
    curr_s = init["s"]
    controls = init["controls"]
    q_dim = init["q_dim"]
    J = 0

    while (curr_time < end_time):
        next_time = get_next_control_time(control_points, curr_time)
        if (next_time > end_time):
            next_time = end_time
        curr_q = get_control(curr_time, control_points, controls, q_dim)
        curr_init = {}
        curr_init["q"] = curr_q
        curr_init["s"] = curr_s
        Ik = integrate_const_control(curr_init, ode, curr_time, next_time)
        curr_s = Ik["xf"]
        J += Ik["qf"]
        curr_time = next_time
    return curr_s, J

