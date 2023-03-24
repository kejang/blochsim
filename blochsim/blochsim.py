import numpy as np


def blochsim_t(
    b1_t,
    g_t,
    dt_t,
    r_t,
    df,
    t1,
    t2,
    gamma,
    m0,
    a0,
    b0,
    dtype='float32',
):
    """Bloch simulator at instant t.

    Args:
        b1_t (float | complex): b1 in G, can be complex
        g_t (list): (3,) gradient amplitude in G/cm (x,y,z)
        dt_t (float): time-step in sec
        r_t (list): (3,) position vector in cm (x,y,z)
        df (float): off-resonance in Hz
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G
        m0 (list): initial magnetization vector (x,y,z)
        a0 (`ndarray`): (3, 3) initial propagation matrix
        b0 (`ndarray`): (3,) initial propagation vector
        dtype (str): dtype of real type ('float32', 'float64', ...)

    Returns:
        tuple: m, a, b
    """

    m = np.array(m0, dtype=dtype)

    # rotation due to RF pulse, gradient, and off-resonance

    rotang_x, rotang_y = rotang_b1(b1_t, dt_t, gamma)
    rotang_z = rotang_offres(g_t, r_t, df, dt_t, gamma)

    # convert rotation angles to rotation around arbitrary axis

    rotax = np.array([rotang_x, rotang_y, rotang_z], dtype=dtype)
    rotang = np.linalg.norm(rotax)

    if np.isclose(rotang, 0):
        rotmat = np.eye(3, dtype=dtype)
    else:
        rotmat = rotmat_around_arbitrary_axis(rotax, rotang, dtype=dtype)
        m = np.matmul(rotmat, m)

    # T1, T2 decay and T1 recovery

    decay, recov = decay_and_recovery(t1, t2, dt_t, dtype=dtype)
    m = np.matmul(decay, m) + recov

    # update propagation equation

    a = np.linalg.multi_dot([decay, rotmat, a0])
    b = np.linalg.multi_dot([decay, rotmat, b0]) + recov

    return m, a, b


def blochsim(
    b1,
    g,
    dt,
    r,
    df,
    t1,
    t2,
    gamma=4257.59,
    m0=[0, 0, 1],
    dtype='float32',
):
    """Bloch simulator

    Args:
        b1 (list): (n,) RF pulse in G, can be complex
        g (list): (n, 3) gradient amplitude in G/cm (x,y,z)
        dt (list): (n,) time steps in sec
        r (list): (n, 3) (or (3,) if static) position vector in cm (x,y,z)
        df (float): off-resonance in Hz
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G
        m0 (list): (3,) initial magnetization vector (x,y,z)
        dtype (str): dtype of real type ('float32', 'float64', ...)

    Returns:
        tuple: ms, a, b
    """

    n = len(b1)
    a = np.eye(3, dtype=dtype)
    b = np.zeros((3,), dtype=dtype)
    m = np.array(m0, dtype=dtype)
    ms = []

    is_static = not hasattr(r[0], '__iter__')

    for i in range(n):
        if is_static:
            r_t = r
        else:
            r_t = r[i]

        m, a, b = blochsim_t(
            b1[i], g[i], dt[i], r_t, df, t1, t2, gamma, m, a, b, dtype
        )

        ms.append(m)

    return ms, a, b


def blochsim_const_flow(
    b1,
    g,
    dt,
    r0,
    v,
    df,
    t1,
    t2,
    gamma=4257.59,
    m0=[0, 0, 1],
    dtype='float32',
    ignore_flow=None,
):
    """Bloch simulator

    Args:
        b1 (list): (n,) RF pulse in G, can be complex
        g (list): (n, 3) gradient amplitude in G/cm (x,y,z)
        dt (list): (n,) time steps in sec
        r0 (list): (3,) initial position vector in cm (x,y,z)
        v (list): (3,) velocity vector in cm/sec (x,y,z)
        df (float): off-resonance in Hz
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G
        m0 (list): (3,) initial magnetization vector (x,y,z)
        dtype (str): dtype of real type ('float32', 'float64', ...)
        ignore_flow (None | list): ignore the flow at instances

    Returns:
        tuple: ms, a, b
    """

    n = len(b1)
    a = np.eye(3, dtype=dtype)
    b = np.zeros((3,), dtype=dtype)
    m = np.array(m0, dtype=dtype)
    ms = []
    r_t = np.array(r0, dtype=dtype)
    v_c = np.array(v, dtype=dtype)

    for i in range(n):
        m, a, b = blochsim_t(
            b1[i], g[i], dt[i], r_t, df, t1, t2, gamma, m, a, b, dtype
        )
        ms.append(m)

        if (ignore_flow is None) or (not ignore_flow[i]):
            r_t += dt[i]*v_c

    return ms, a, b


def blochsim_const_acc(
    b1,
    g,
    dt,
    r0,
    v0,
    acc,
    df,
    t1,
    t2,
    gamma=4257.59,
    m0=[0, 0, 1],
    dtype='float32',
    ignore_flow=None,
):
    """Bloch simulator

    Args:
        b1 (list): (n,) RF pulse in G, can be complex
        g (list): (n, 3) gradient amplitude in G/cm (x,y,z)
        dt (list): (n,) time steps in sec
        r0 (list): (3,) initial position vector in cm (x,y,z)
        v0 (list): (3,) initial velocity vector in cm/sec (x,y,z)
        acc (list): (3,) constant acceleration vector in cm/sec (x,y,z)
        df (float): off-resonance in Hz
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G
        m0 (list): (3,) initial magnetization vector (x,y,z)
        dtype (str): dtype of real type ('float32', 'float64', ...)
        ignore_flow (None | list): ignore the flow at instances

    Returns:
        tuple: ms, a, b
    """

    n = len(b1)
    a = np.eye(3, dtype=dtype)
    b = np.zeros((3,), dtype=dtype)
    m = np.array(m0, dtype=dtype)
    ms = []

    r_t = np.array(r0, dtype=dtype)
    v0 = np.array(v0, dtype=dtype)
    a0 = np.array(acc, dtype=dtype)

    for i in range(n):
        m, a, b = blochsim_t(
            b1[i], g[i], dt[i], r_t, df, t1, t2, gamma, m, a, b, dtype
        )
        ms.append(m)

        if (ignore_flow is None) or (not ignore_flow[i]):
            r_t += v0*dt[i] + 0.5*a0*(dt[i]**2)

    return ms, a, b


def rotang_offres(g, r, df, dt, gamma):
    """Returns rotation angle around z-axis due to off-resonance.

    Args:
        g (list): (3,) gradient amplitude in G/cm
        r (list): (3,) position vector in cm
        df (float): off-resonance in Hz
        dt (float): time-step in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G

    Returns:
        float: rotation angle in radian
    """

    rotang = ((-1)                               # left-hand rotation
              * dt                               # time step
              * (gamma * np.inner(g, r) + df)    # gradient and off-res
              * 2.0 * np.pi)                     # Hz -> radian

    return rotang


def rotang_b1(b1, dt, gamma):
    """Returns rotation angles around x- and y-axis due to b1

    Args:
        b1 (float | complex): RF waveform in Gauss, can be complex
        dt (float): time-step in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G

    Returns:
        tuple: rotation angles around x- and y-axis
    """
    rotang_x = ((-1)    # left-hand rotation
                * np.real(b1)
                * gamma * 2.0 * np.pi * dt)

    rotang_y = (np.imag(b1)
                * gamma * 2.0 * np.pi * dt)

    return rotang_x, rotang_y


def decay_and_recovery(t1, t2, dt, dtype='float32'):
    """Returns decay matrix and recovery vector.

    Args:
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        dt (float): time-step in sec
        dtype (str): dtype of real type ('float32', 'float64', ...)

    Returns:
        tuple: (decay, recov)

    Notes:
        - decay: (3, 3)-matrix
        - recovery: (3,)-vector
        - m_new = decay * m_old + recovery
    """

    decay = np.zeros((3, 3), dtype=dtype)
    decay[0, 0] = np.exp(-dt/t2)
    decay[1, 1] = np.exp(-dt/t2)
    decay[2, 2] = np.exp(-dt/t1)

    recov = np.zeros((3,), dtype=dtype)
    recov[2] = 1 - np.exp(-dt/t1)

    return decay, recov


def rotmat_around_arbitrary_axis(rotax, th, dtype='float32'):
    """Returns rotation matrix around an arbitrary axis.

    Args:
        rotax (list): (3,) rotation axis
        th (float): angle in radian
        dtype (str): dtype of real type ('float32', 'float64', ...)

    Returns:
        numpy array: (3, 3) rotation matrix

    Notes:
        - See http://scipp.ucsc.edu/~haber/ph216/rotation_12.pdf
    """

    rotmat = np.zeros((3, 3), dtype=dtype)
    n = rotax / np.linalg.norm(rotax)

    rotmat[0, 0] = np.cos(th) + n[0]*n[0]*(1 - np.cos(th))
    rotmat[1, 0] = n[0]*n[1]*(1 - np.cos(th)) + n[2]*np.sin(th)
    rotmat[2, 0] = n[0]*n[2]*(1 - np.cos(th)) - n[1]*np.sin(th)

    rotmat[0, 1] = n[0]*n[1]*(1 - np.cos(th)) - n[2]*np.sin(th)
    rotmat[1, 1] = np.cos(th) + n[1]*n[1]*(1 - np.cos(th))
    rotmat[2, 1] = n[1]*n[2]*(1 - np.cos(th)) + n[0]*np.sin(th)

    rotmat[0, 2] = n[0]*n[2]*(1 - np.cos(th)) + n[1]*np.sin(th)
    rotmat[1, 2] = n[1]*n[2]*(1 - np.cos(th)) - n[0]*np.sin(th)
    rotmat[2, 2] = np.cos(th) + n[2]*n[2]*(1 - np.cos(th))

    return rotmat
