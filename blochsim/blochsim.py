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
        df (list | `ndarray`): (n,) off-resonance in Hz
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
        b1_amp = np.array(np.abs(b1), dtype='float64')
        b1_phs = np.array(np.angle(b1), dtype='float64')
        g = np.array(g, dtype='float64')
        r = np.array(r, dtype='float64')
        df = np.array(df, dtype='float64')
        m = np.array(m0, dtype='float64')
        a = np.eye(3, dtype='float64')
        b = np.zeros((3,), dtype='float64')
        blochsim_t = blochsim_t_64
    else:
        ms = np.zeros((n, 3), dtype='float32')
        b1_amp = np.array(np.abs(b1), dtype='float32')
        b1_phs = np.array(np.angle(b1), dtype='float32')
        g = np.array(g, dtype='float32')
        r = np.array(r, dtype='float32')
        df = np.array(df, dtype='float32')
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
            b1_amp[i], b1_phs[i], g[i], dt[i], r_t, df[i], t1, t2, gamma, m,
            a, b
        )

        ms[i] = m

    return ms, a, b


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
    decay[0, 0] = np.exp(-dt / t2)
    decay[1, 1] = np.exp(-dt / t2)
    decay[2, 2] = np.exp(-dt / t1)

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
    decay[0, 0] = np.exp(-dt / t2)
    decay[1, 1] = np.exp(-dt / t2)
    decay[2, 2] = np.exp(-dt / t1)

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
    recov[2] = 1 - np.exp(-dt / t1)

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
    recov[2] = 1 - np.exp(-dt / t1)

    return recov


@nb.jit(nb.float32[:, :](nb.float32, nb.float32, nb.float32[:], nb.float32[:],
                         nb.float32, nb.float32, nb.float32),
        nopython=True)
def get_rotation_matrix_32(b1_amp, phi, g, r, df, dt, gamma):
    """Returns rotation angle around z-axis due to off-resonance (single).

    Args:
        b1_amp (float): B1 amplitude in G
        phi (float): B1 phase in radian
        g (`ndarray`): (3,) gradient amplitude in G/cm
        r (`ndarray`): (3,) position vector in cm
        df (float): off-resonance in Hz
        dt (float): time-step in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G

    Returns:
        `ndarray`: (3, 3)-rotation matrix

    Notes:
        - Right-hand notation
        - full matrix: Rz(phase) * (RF pulse rotation matrix) * Rz(-phase)
        - RF pulse rotation matrix: Ry(theta) * Rx(-w) * Ry(-theta)
        - Reference: Multidimensional NMR in Liquid by Frank F. M. van de Ven.
          But in this book, the matrix is Ry(-theta) * Rx(-w) * Ry(theta).
          (See the equation above 1.16.)
          I think the matrix should be Ry(theta) * Rx(-w) * Ry(-theta),
          as implemented in this code, because the left-hand rotation changes
          the coordinate. (Figure 1.5.)
    """

    # field offset (negative), "Omega = w0 - w"

    offset = -(np.sum(g * r) + df / gamma)

    # effective B-field

    b_eff = np.sqrt(b1_amp ** 2 + offset ** 2)    # (1.15)

    # prepare rotation of the reference frame
    # (new x-axis is parallel to the effective B-field.)

    if np.isclose(b_eff, 0):
        s = 0
        c = 1
    else:
        s = offset / b_eff    # sin(theta), (1.16)
        c = b1_amp / b_eff     # cos(theta), (1.16)

    # rotation angle in radian

    w = 2.0 * np.pi * gamma * b_eff * dt

    # left-hand rotation around x (effective B-field)

    rmtx = np.zeros((3, 3), dtype='float32')
    rmtx[0, 0] = 1
    rmtx[0, 1] = 0
    rmtx[0, 2] = 0

    rmtx[1, 0] = 0
    rmtx[1, 1] = np.cos(w)
    rmtx[1, 2] = np.sin(w)

    rmtx[2, 0] = 0
    rmtx[2, 1] = -np.sin(w)
    rmtx[2, 2] = np.cos(w)

    # change the reference coordinate: new x-axis == effective B-field

    rmtx_y_pos = np.zeros((3, 3), dtype='float32')
    rmtx_y_neg = np.zeros((3, 3), dtype='float32')

    rmtx_y_pos[0, 0] = c
    rmtx_y_pos[0, 1] = 0
    rmtx_y_pos[0, 2] = s

    rmtx_y_pos[1, 0] = 0
    rmtx_y_pos[1, 1] = 1
    rmtx_y_pos[1, 2] = 0

    rmtx_y_pos[2, 0] = -s
    rmtx_y_pos[2, 1] = 0
    rmtx_y_pos[2, 2] = c

    rmtx_y_neg[0, 0] = c
    rmtx_y_neg[0, 1] = 0
    rmtx_y_neg[0, 2] = -s

    rmtx_y_neg[1, 0] = 0
    rmtx_y_neg[1, 1] = 1
    rmtx_y_neg[1, 2] = 0

    rmtx_y_neg[2, 0] = s
    rmtx_y_neg[2, 1] = 0
    rmtx_y_neg[2, 2] = c

    # phase of B1

    rmtx_z_pos = np.zeros((3, 3), dtype='float32')
    rmtx_z_neg = np.zeros((3, 3), dtype='float32')

    rmtx_z_pos[0, 0] = np.cos(phi)
    rmtx_z_pos[0, 1] = -np.sin(phi)
    rmtx_z_pos[0, 2] = 0

    rmtx_z_pos[1, 0] = np.sin(phi)
    rmtx_z_pos[1, 1] = np.cos(phi)
    rmtx_z_pos[1, 2] = 0

    rmtx_z_pos[2, 0] = 0
    rmtx_z_pos[2, 1] = 0
    rmtx_z_pos[2, 2] = 1

    rmtx_z_neg[0, 0] = np.cos(phi)
    rmtx_z_neg[0, 1] = np.sin(phi)
    rmtx_z_neg[0, 2] = 0

    rmtx_z_neg[1, 0] = -np.sin(phi)
    rmtx_z_neg[1, 1] = np.cos(phi)
    rmtx_z_neg[1, 2] = 0

    rmtx_z_neg[2, 0] = 0
    rmtx_z_neg[2, 1] = 0
    rmtx_z_neg[2, 2] = 1

    # complete RF pulse rotation matrix

    rotmat = rmtx_z_pos @ rmtx_y_pos @ rmtx @ rmtx_y_neg @ rmtx_z_neg

    return rotmat.astype('float32')


@nb.jit(nb.float64[:, :](nb.float64, nb.float64, nb.float64[:], nb.float64[:],
                         nb.float64, nb.float64, nb.float64),
        nopython=True)
def get_rotation_matrix_64(b1_amp, phi, g, r, df, dt, gamma):
    """Returns rotation angle around z-axis due to off-resonance (single).

    Args:
        b1_amp (float): B1 amplitude in G
        phi (float): B1 phase in radian
        g (`ndarray`): (3,) gradient amplitude in G/cm
        r (`ndarray`): (3,) position vector in cm
        df (float): off-resonance in Hz
        dt (float): time-step in sec
        gamma (float): gyromagnetic ratio over 2*PI in Hz/G

    Returns:
        `ndarray`: (3, 3)-rotation matrix

    Notes:
        - Right-hand notation
        - full matrix: Rz(phase) * (RF pulse rotation matrix) * Rz(-phase)
        - RF pulse rotation matrix: Ry(theta) * Rx(-w) * Ry(-theta)
        - Reference: Multidimensional NMR in Liquid by Frank F. M. van de Ven.
          But in this book, the matrix is Ry(-theta) * Rx(-w) * Ry(theta).
          (See the equation above 1.16.)
          I think the matrix should be Ry(theta) * Rx(-w) * Ry(-theta),
          as implemented in this code, because the left-hand rotation changes
          the coordinate. (Figure 1.5.)
    """

    # field offset (negative), "Omega = w0 - w"

    offset = -(np.sum(g * r) + df / gamma)

    # effective B-field

    b_eff = np.sqrt(b1_amp ** 2 + offset ** 2)    # (1.15)

    # prepare rotation of the reference frame
    # (new x-axis is parallel to the effective B-field.)

    if np.isclose(b_eff, 0):
        s = 0
        c = 1
    else:
        s = offset / b_eff    # sin(theta), (1.16)
        c = b1_amp / b_eff     # cos(theta), (1.16)

    # rotation angle in radian

    w = 2.0 * np.pi * gamma * b_eff * dt

    # left-hand rotation around x (effective B-field)

    rmtx = np.zeros((3, 3), dtype='float64')
    rmtx[0, 0] = 1
    rmtx[0, 1] = 0
    rmtx[0, 2] = 0

    rmtx[1, 0] = 0
    rmtx[1, 1] = np.cos(w)
    rmtx[1, 2] = np.sin(w)

    rmtx[2, 0] = 0
    rmtx[2, 1] = -np.sin(w)
    rmtx[2, 2] = np.cos(w)

    # change the reference coordinate: new x-axis == effective B-field

    rmtx_y_pos = np.zeros((3, 3), dtype='float64')
    rmtx_y_neg = np.zeros((3, 3), dtype='float64')

    rmtx_y_pos[0, 0] = c
    rmtx_y_pos[0, 1] = 0
    rmtx_y_pos[0, 2] = s

    rmtx_y_pos[1, 0] = 0
    rmtx_y_pos[1, 1] = 1
    rmtx_y_pos[1, 2] = 0

    rmtx_y_pos[2, 0] = -s
    rmtx_y_pos[2, 1] = 0
    rmtx_y_pos[2, 2] = c

    rmtx_y_neg[0, 0] = c
    rmtx_y_neg[0, 1] = 0
    rmtx_y_neg[0, 2] = -s

    rmtx_y_neg[1, 0] = 0
    rmtx_y_neg[1, 1] = 1
    rmtx_y_neg[1, 2] = 0

    rmtx_y_neg[2, 0] = s
    rmtx_y_neg[2, 1] = 0
    rmtx_y_neg[2, 2] = c

    # phase of B1

    rmtx_z_pos = np.zeros((3, 3), dtype='float64')
    rmtx_z_neg = np.zeros((3, 3), dtype='float64')

    rmtx_z_pos[0, 0] = np.cos(phi)
    rmtx_z_pos[0, 1] = -np.sin(phi)
    rmtx_z_pos[0, 2] = 0

    rmtx_z_pos[1, 0] = np.sin(phi)
    rmtx_z_pos[1, 1] = np.cos(phi)
    rmtx_z_pos[1, 2] = 0

    rmtx_z_pos[2, 0] = 0
    rmtx_z_pos[2, 1] = 0
    rmtx_z_pos[2, 2] = 1

    rmtx_z_neg[0, 0] = np.cos(phi)
    rmtx_z_neg[0, 1] = np.sin(phi)
    rmtx_z_neg[0, 2] = 0

    rmtx_z_neg[1, 0] = -np.sin(phi)
    rmtx_z_neg[1, 1] = np.cos(phi)
    rmtx_z_neg[1, 2] = 0

    rmtx_z_neg[2, 0] = 0
    rmtx_z_neg[2, 1] = 0
    rmtx_z_neg[2, 2] = 1

    # complete RF pulse rotation matrix

    rotmat = rmtx_z_pos @ rmtx_y_pos @ rmtx @ rmtx_y_neg @ rmtx_z_neg

    return rotmat.astype('float64')


@nb.jit(nb.types.Tuple((nb.float64[:], nb.float64[:, :], nb.float64[:]))(
        nb.float64,     # b1_amp_t
        nb.float64,     # b1_phs_t
        nb.float64[:],     # g_t
        nb.float64,        # dt_t
        nb.float64[:],     # r_t
        nb.float64,        # df_t
        nb.float64,        # t1
        nb.float64,        # t2
        nb.float64,        # gamma
        nb.float64[:],     # m
        nb.float64[:, :],  # a0
        nb.float64[:],     # b0
        ),
        nopython=True)
def blochsim_t_64(
    b1_amp_t,
    b1_phs_t,
    g_t,
    dt_t,
    r_t,
    df_t,
    t1,
    t2,
    gamma,
    m,
    a0,
    b0,
):
    """Bloch simulator at instant t (double).

    Args:
        b1_amp_t (float): b1 amplitude in G
        b1_phs_t (float): b1 phase in radian
        g_t (`ndarray`): (3,) gradient amplitude in G/cm (x,y,z)
        dt_t (float): time-step in sec
        r_t (`ndarray`): (3,) position vector in cm (x,y,z)
        df_t (float): off-resonance in Hz
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

    rotmat = get_rotation_matrix_64(
        b1_amp_t, b1_phs_t, g_t, r_t, df_t, dt_t, gamma
    )
    m = np.dot(rotmat, m)

    # T1, T2 decay and T1 recovery

    decay = get_decay_matrix_64(t1, t2, dt_t)
    recov = get_recovery_vector_64(t1, t2, dt_t)

    m = np.dot(decay, m) + recov

    # update propagation equation

    decay_and_rotate = np.dot(decay, rotmat)
    a = np.dot(decay_and_rotate, a0)
    b = np.dot(decay_and_rotate, b0) + recov

    return m, a, b


@nb.jit(nb.types.Tuple((nb.float32[:], nb.float32[:, :], nb.float32[:]))(
        nb.float32,     # b1_amp_t
        nb.float32,     # b1_phs_t
        nb.float32[:],     # g_t
        nb.float32,        # dt_t
        nb.float32[:],     # r_t
        nb.float32,        # df_t
        nb.float32,        # t1
        nb.float32,        # t2
        nb.float32,        # gamma
        nb.float32[:],     # m
        nb.float32[:, :],  # a0
        nb.float32[:],     # b0
        ),
        nopython=True)
def blochsim_t_32(
    b1_amp_t,
    b1_phs_t,
    g_t,
    dt_t,
    r_t,
    df_t,
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
        df_t (float): off-resonance in Hz
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

    rotmat = get_rotation_matrix_32(
        b1_amp_t, b1_phs_t, g_t, r_t, df_t, dt_t, gamma
    )
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
