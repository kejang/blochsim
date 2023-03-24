import numpy as np
import numba as nb
from numba.core.errors import (
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
    NumbaPerformanceWarning
)
import warnings


warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)


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
    is_static=True,
    precision='single'
):
    """Bloch simulator

    Args:
        b1 (list | `ndarray`): (n,) RF pulse in G, can be complex
        g (list | `ndarray`): (n, 3) gradient amplitude in G/cm (x,y,z)
        dt (list | `ndarray`): (n,) time steps in sec
        r (list | `ndarray`): (n, 3) (or (3,)) position vector in cm (x,y,z)
        df (float): off-resonance in Hz
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G
        m0 (list | `ndarray`): (3,) initial magnetization vector (x,y,z)
        is_static (bool): static or moving object
        precision (str): 'single' or 'double'

    Returns:
        tuple: ms, a, b
    """

    n = len(b1)

    if precision == 'double':
        ms = np.zeros((n, 3), dtype='float64')
        b1 = np.array(b1, dtype='complex128')
        g = np.array(g, dtype='float64')
        r = np.array(r, dtype='float64')
        m = np.array(m0, dtype='float64')
        a = np.eye(3, dtype='float64')
        b = np.zeros((3,), dtype='float64')
        blochsim_t = blochsim_t_64
    else:
        ms = np.zeros((n, 3), dtype='float32')
        b1 = np.array(b1, dtype='complex64')
        g = np.array(g, dtype='float32')
        r = np.array(r, dtype='float32')
        m = np.array(m0, dtype='float32')
        a = np.eye(3, dtype='float32')
        b = np.zeros((3,), dtype='float32')
        blochsim_t = blochsim_t_32

    for i in range(n):
        if is_static:
            r_t = r
        else:
            r_t = r[i]

        m, a, b = blochsim_t(
            b1[i], g[i], dt[i], r_t, df, t1, t2, gamma, m, a, b
        )

        ms[i] = m

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
    ignore_flow=None,
    precision='single'
):
    """Bloch simulator

    Args:
        b1 (list | `ndarray`): (n,) RF pulse in G, can be complex
        g (list | `ndarray`): (n, 3) gradient amplitude in G/cm (x,y,z)
        dt (list | `ndarray`): (n,) time steps in sec
        r0 (list | `ndarray`): (3,) initial position vector in cm (x,y,z)
        v (list | `ndarray`): (3,) velocity vector in cm/sec (x,y,z)
        df (float): off-resonance in Hz
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G
        m0 (list | `ndarray`): (3,) initial magnetization vector (x,y,z)
        ignore_flow (None | list): ignore the flow at instances
        precision (str): 'single' or 'double'

    Returns:
        tuple: ms, a, b
    """

    n = len(b1)

    if precision == 'double':
        ms = np.zeros((n, 3), dtype='float64')
        b1 = np.array(b1, dtype='complex128')
        g = np.array(g, dtype='float64')
        r_t = np.array(r0, dtype='float64')
        v_c = np.array(v, dtype='float64')
        m = np.array(m0, dtype='float64')
        a = np.eye(3, dtype='float64')
        b = np.zeros((3,), dtype='float64')
        blochsim_t = blochsim_t_64
    else:
        ms = np.zeros((n, 3), dtype='float32')
        b1 = np.array(b1, dtype='complex64')
        g = np.array(g, dtype='float32')
        r_t = np.array(r0, dtype='float32')
        v_c = np.array(v, dtype='float32')
        m = np.array(m0, dtype='float32')
        a = np.eye(3, dtype='float32')
        b = np.zeros((3,), dtype='float32')
        blochsim_t = blochsim_t_32

    for i in range(len(b1)):
        m, a, b = blochsim_t(
            b1[i], g[i], dt[i], r_t, df, t1, t2, gamma, m, a, b
        )

        ms[i] = m

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
    ignore_flow=None,
    precision='single'
):
    """Bloch simulator

    Args:
        b1 (list | `ndarray`): (n,) RF pulse in G, can be complex
        g (list) | `ndarray`: (n, 3) gradient amplitude in G/cm (x,y,z)
        dt (list | `ndarray`): (n,) time steps in sec
        r0 (list | `ndarray`): (3,) initial position vector in cm (x,y,z)
        v0 (list | `ndarray`): (3,) initial velocity vector in cm/sec (x,y,z)
        acc (list | `ndarray`): (3,) constant acceleration vector in cm/sec^2
        df (float): off-resonance in Hz
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G
        m0 (list | `ndarray`): (3,) initial magnetization vector (x,y,z)
        ignore_flow (None | list): ignore the flow at instances
        precision (str): 'single' or 'double'

    Returns:
        tuple: ms, a, b
    """

    n = len(b1)

    if precision == 'double':
        ms = np.zeros((n, 3), dtype='float64')
        b1 = np.array(b1, dtype='complex128')
        g = np.array(g, dtype='float64')
        r_t = np.array(r0, dtype='float64')
        v0 = np.array(v0, dtype='float64')
        a0 = np.array(acc, dtype='float64')
        m = np.array(m0, dtype='float64')
        a = np.eye(3, dtype='float64')
        b = np.zeros((3,), dtype='float64')
        blochsim_t = blochsim_t_64
    else:
        ms = np.zeros((n, 3), dtype='float32')
        b1 = np.array(b1, dtype='complex64')
        g = np.array(g, dtype='float32')
        r_t = np.array(r0, dtype='float32')
        v0 = np.array(v0, dtype='float32')
        a0 = np.array(acc, dtype='float32')
        m = np.array(m0, dtype='float32')
        a = np.eye(3, dtype='float32')
        b = np.zeros((3,), dtype='float32')
        blochsim_t = blochsim_t_32

    for i in range(len(b1)):
        m, a, b = blochsim_t(
            b1[i], g[i], dt[i], r_t, df, t1, t2, gamma, m, a, b
        )

        ms[i] = m

        if (ignore_flow is None) or (not ignore_flow[i]):
            r_t += v0*dt[i] + 0.5*a0*(dt[i]**2)

    return ms, a, b


@nb.jit(nb.float64[:, :](nb.int64), nopython=True)
def eye_f8(n):
    """numpy.identity() (double)."""
    out = np.zeros((n, n), dtype='float64')
    for i in range(n):
        out[i, i] = 1
    return out


@nb.jit(nb.float32[:, :](nb.int64), nopython=True)
def eye_f4(n):
    """numpy.identity() (single)."""
    out = np.zeros((n, n), dtype='float32')
    for i in range(n):
        out[i, i] = 1
    return out


@nb.jit(nb.float64(nb.float64[:], nb.float64[:], nb.float64, nb.float64,
                   nb.float64),
        nopython=True)
def rotang_offres_64(g, r, df, dt, gamma):
    """Returns rotation angle around z-axis due to off-resonance (double).

    Args:
        g (`ndarray`): (3,) gradient amplitude in G/cm
        r (`ndarray`): (3,) position vector in cm
        df (float): off-resonance in Hz
        dt (float): time-step in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G

    Returns:
        float: rotation angle in radian
    """

    rotang = ((-1)                               # left-hand rotation
              * dt                               # time step
              * (gamma * np.sum(g*r) + df)       # gradient and off-res
              * 2.0 * np.pi)                     # Hz -> radian

    return rotang


@nb.jit(nb.float32(nb.float32[:], nb.float32[:], nb.float32, nb.float32,
                   nb.float32),
        nopython=True)
def rotang_offres_32(g, r, df, dt, gamma):
    """Returns rotation angle around z-axis due to off-resonance (single).

    Args:
        g (`ndarray`): (3,) gradient amplitude in G/cm
        r (`ndarray`): (3,) position vector in cm
        df (float): off-resonance in Hz
        dt (float): time-step in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G

    Returns:
        float: rotation angle in radian
    """

    rotang = ((-1)                               # left-hand rotation
              * dt                               # time step
              * (gamma * np.sum(g*r) + df)       # gradient and off-res
              * 2.0 * np.pi)                     # Hz -> radian

    return rotang


@nb.jit(
    nb.types.UniTuple(nb.float64, 2)(nb.complex128, nb.float64, nb.float64),
    nopython=True
)
def rotang_b1_64(b1, dt, gamma):
    """Returns rotation angles around x- and y-axis due to b1 (double).

    Args:
        b1 (complex): RF waveform in Gauss, can be complex
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


@nb.jit(
    nb.types.UniTuple(nb.float32, 2)(nb.complex64, nb.float32, nb.float32),
    nopython=True
)
def rotang_b1_32(b1, dt, gamma):
    """Returns rotation angles around x- and y-axis due to b1 (single).

    Args:
        b1 (complex): RF waveform in Gauss, can be complex
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


@nb.jit(nb.float64[:, :](nb.float64, nb.float64, nb.float64), nopython=True)
def get_decay_matrix_64(t1, t2, dt):
    """Returns decay matrix (double).

    Args:
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        dt (float): time-step in sec

    Returns:
        ndarray: decay matrix (3, 3)
    """

    decay = np.zeros((3, 3), dtype='float64')
    decay[0, 0] = np.exp(-dt/t2)
    decay[1, 1] = np.exp(-dt/t2)
    decay[2, 2] = np.exp(-dt/t1)

    return decay


@nb.jit(nb.float32[:, :](nb.float32, nb.float32, nb.float32), nopython=True)
def get_decay_matrix_32(t1, t2, dt):
    """Returns decay matrix (single).

    Args:
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        dt (float): time-step in sec

    Returns:
        ndarray: decay matrix (3, 3)
    """

    decay = np.zeros((3, 3), dtype='float32')
    decay[0, 0] = np.exp(-dt/t2)
    decay[1, 1] = np.exp(-dt/t2)
    decay[2, 2] = np.exp(-dt/t1)

    return decay


@nb.jit(nb.float64[:](nb.float64, nb.float64, nb.float64), nopython=True)
def get_recovery_vector_64(t1, t2, dt):
    """Returns recovery vector (double).

    Args:
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        dt (float): time-step in sec

    Returns:
        ndarray: recovery vector (3,)
    """

    recov = np.zeros((3,), dtype='float64')
    recov[2] = 1 - np.exp(-dt/t1)

    return recov


@nb.jit(nb.float32[:](nb.float32, nb.float32, nb.float32), nopython=True)
def get_recovery_vector_32(t1, t2, dt):
    """Returns recovery vector (single).

    Args:
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        dt (float): time-step in sec

    Returns:
        `ndarray`: recovery vector (3,)
    """

    recov = np.zeros((3,), dtype='float32')
    recov[2] = 1 - np.exp(-dt/t1)

    return recov


@nb.jit(nb.float64[:, :](nb.float64[:], nb.float64), nopython=True)
def get_rotmat_around_arbitrary_axis_64(rotax, th):
    """Returns rotation matrix around an arbitrary axis (double).

    Args:
        rotax (`ndarray`): (3,) rotation axis
        th (float): angle in radian

    Returns:
        `ndarray`: (3, 3) rotation matrix

    Notes:
        - See http://scipp.ucsc.edu/~haber/ph216/rotation_12.pdf
    """

    rotmat = np.zeros((3, 3), dtype='float64')
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


@nb.jit(nb.float32[:, :](nb.float32[:], nb.float32), nopython=True)
def get_rotmat_around_arbitrary_axis_32(rotax, th):
    """Returns rotation matrix around an arbitrary axis (single).

    Args:
        rotax (`ndarray`): (3,) rotation axis
        th (float): angle in radian

    Returns:
        `ndarray`: (3, 3) rotation matrix

    Notes:
        - See http://scipp.ucsc.edu/~haber/ph216/rotation_12.pdf
    """

    rotmat = np.zeros((3, 3), dtype='float32')
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


@nb.jit(nb.types.Tuple((nb.float64[:], nb.float64[:, :], nb.float64[:]))(
        nb.complex128,     # b1_t
        nb.float64[:],     # g_t
        nb.float64,        # dt_t
        nb.float64[:],     # r_t
        nb.float64,        # df
        nb.float64,        # t1
        nb.float64,        # t2
        nb.float64,        # gamma
        nb.float64[:],     # m
        nb.float64[:, :],  # a0
        nb.float64[:],     # b0
        ),
        nopython=True)
def blochsim_t_64(
    b1_t,
    g_t,
    dt_t,
    r_t,
    df,
    t1,
    t2,
    gamma,
    m,
    a0,
    b0,
):
    """Bloch simulator at instant t (double).

    Args:
        b1_t (complex): b1 in G, can be complex
        g_t (`ndarray`): (3,) gradient amplitude in G/cm (x,y,z)
        dt_t (float): time-step in sec
        r_t (`ndarray`): (3,) position vector in cm (x,y,z)
        df (float): off-resonance in Hz
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G
        m (`ndarray`): initial magnetization vector (x,y,z)
        a0 (`ndarray`): (3, 3) initial propagation matrix
        b0 (`ndarray`): (3,) initial propagation vector

    Returns:
        tuple: m, a, b
    """

    # rotation due to RF pulse, gradient, and off-resonance

    rotang_x, rotang_y = rotang_b1_64(b1_t, dt_t, gamma)
    rotang_z = rotang_offres_64(g_t, r_t, df, dt_t, gamma)

    # convert rotation angles to rotation around arbitrary axis

    rotax = np.array([rotang_x, rotang_y, rotang_z], dtype='float64')
    rotang = np.linalg.norm(rotax)

    if abs(rotang) < 1e-6:
        rotmat = eye_f8(3)
    else:
        rotmat = get_rotmat_around_arbitrary_axis_64(rotax, rotang)
        m = np.dot(rotmat, m)

    # T1, T2 decay and T1 recovery

    decay = get_decay_matrix_64(t1, t2, dt_t)
    recov = get_recovery_vector_64(t1, t2, dt_t)

    m = np.dot(decay, m) + recov

    # update propagation equation

    # a = np.linalg.multi_dot([decay, rotmat, a0])
    # b = np.linalg.multi_dot([decay, rotmat, b0]) + recov
    decay_and_rotate = np.dot(decay, rotmat)
    a = np.dot(decay_and_rotate, a0)
    b = np.dot(decay_and_rotate, b0) + recov

    return m, a, b


@nb.jit(nb.types.Tuple((nb.float32[:], nb.float32[:, :], nb.float32[:]))(
        nb.complex64,     # b1_t
        nb.float32[:],     # g_t
        nb.float32,        # dt_t
        nb.float32[:],     # r_t
        nb.float32,        # df
        nb.float32,        # t1
        nb.float32,        # t2
        nb.float32,        # gamma
        nb.float32[:],     # m
        nb.float32[:, :],  # a0
        nb.float32[:],     # b0
        ),
        nopython=True)
def blochsim_t_32(
    b1_t,
    g_t,
    dt_t,
    r_t,
    df,
    t1,
    t2,
    gamma,
    m,
    a0,
    b0,
):
    """Bloch simulator at instant t (single).

    Args:
        b1_t (complex): b1 in G, can be complex
        g_t (`ndarray`): (3,) gradient amplitude in G/cm (x,y,z)
        dt_t (float): time-step in sec
        r_t (`ndarray`): (3,) position vector in cm (x,y,z)
        df (float): off-resonance in Hz
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G
        m (`ndarray`): initial magnetization vector (x,y,z)
        a0 (`ndarray`): (3, 3) initial propagation matrix
        b0 (`ndarray`): (3,) initial propagation vector

    Returns:
        tuple: m, a, b
    """

    # rotation due to RF pulse, gradient, and off-resonance

    rotang_x, rotang_y = rotang_b1_32(b1_t, dt_t, gamma)
    rotang_z = rotang_offres_32(g_t, r_t, df, dt_t, gamma)

    # convert rotation angles to rotation around arbitrary axis

    rotax = np.array([rotang_x, rotang_y, rotang_z], dtype='float32')
    rotang = np.linalg.norm(rotax)

    if abs(rotang) < 1e-6:
        rotmat = eye_f4(3)
    else:
        rotmat = get_rotmat_around_arbitrary_axis_32(rotax, rotang)
        m = np.dot(rotmat, m)

    # T1, T2 decay and T1 recovery

    decay = get_decay_matrix_32(t1, t2, dt_t)
    recov = get_recovery_vector_32(t1, t2, dt_t)

    m = np.dot(decay, m) + recov

    # update propagation equation

    # a = np.linalg.multi_dot([decay, rotmat, a0])
    # b = np.linalg.multi_dot([decay, rotmat, b0]) + recov
    decay_and_rotate = np.dot(decay, rotmat)
    a = np.dot(decay_and_rotate, a0)
    b = np.dot(decay_and_rotate, b0) + recov

    return m, a, b
