import numpy as np


def blochsim(b1, g, dt, r, df, t1, t2, m0=[0, 0, 1], gamma=4257.59):
    """Bloch simulator: m = a*m0 + b

    Args:
        m0 (list): (3,) initial magnetization vector (x,y,z)
        b1 (`numpy array`): (n,) RF pulse in G, can be complex
        g (`numpy array`): (n, 3) gradient amplitude in G/cm (x,y,z)
        dt (float): time-step in sec
        r (list): (3,) position vector in cm (x,y,z)
        df (float): off-resonance in Hz
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        gamma (float): gyromagnetic ratio in Hz/G

    Returns:
        tuple: ms, a, b

    Notes:
        - ms (`numpy array`): (n, 3) stack of magnetization vectors over time,
            in rotating frame
        - a (`numpy array`): (3, 3)-matrix. (propagation over one TR)
        - b (`numpy array`): (3,)-vector. (propagation over one TR)
        - ms[-1]: a*m0 + b
    """

    n = np.size(b1)
    ms = np.zeros((n, 3))
    a = np.eye(3)
    b = np.zeros((3,))

    m = m0

    for i in range(n):

        # rotation due to RF pulse, gradient, and off-resonance

        rotang_x, rotang_y = rotang_b1(b1[i], dt[i], gamma)
        rotang_z = rotang_offres(g[i], r, df, dt[i], gamma)

        # convert rotation angles to rotation around arbitrary axis

        rotax = np.array([rotang_x, rotang_y, rotang_z])
        rotang = np.linalg.norm(rotax)

        if np.abs(rotang) < 1e-15:
            rotmat = np.eye(3)
        else:
            rotmat = rotmat_around_arbitrary_axis(rotax, rotang)

        m = np.matmul(rotmat, m)

        # T1, T2 decay and T1 recovery

        decay, recov = decay_and_recovery(t1, t2, dt[i])

        m = np.matmul(decay, m) + recov

        ms[i] = m

        # update propagation equation

        a = np.linalg.multi_dot([decay, rotmat, a])
        b = np.linalg.multi_dot([decay, rotmat, b]) + recov

    return ms, a, b


def rotang_offres(g, r, df, dt, gamma):
    """Returns rotation angle around z-axis due to off-resonance.

    Args:
        g: (3,) gradient amplitude in G/cm
        r: (3,) position vector in cm
        df: off-resonance in Hz
        dt: time-step in sec
        gamma: gyromagnetic ratio in Hz/G

    Returns: 
        float: rotation angle in radian
    """

    rotang = ((-1)                               # left-hand rotation
              * (gamma * np.inner(g, r) + df)    # gradient and off-res
              * 2.0 * np.pi * dt)

    return rotang


def rotang_b1(b1, dt, gamma):
    """Returns rotation angles around x- and y-axis due to b1

    Args:
        b1: RF waveform in Gauss, can be complex
        dt: time-step in sec
        gamma: gyromagnetic ratio in Hz/G

    Returns:
        tuple: (rotang_x, rotang_y) rotation angles around x- and y-axis
    """
    rotang_x = ((-1)    # left-hand rotation
                * np.real(b1)
                * gamma * 2.0 * np.pi * dt)

    rotang_y = (np.imag(b1)
                * gamma * 2.0 * np.pi * dt)

    return rotang_x, rotang_y


def decay_and_recovery(t1, t2, dt):
    """Returns decay matrix and recovery vector. 

    Args:
        t1: T1 in sec
        t2: T2 in sec
        dt: time-step in sec

    Returns:
        tuple: (decay, recov)

    Notes: 
        - decay: (3, 3)-matrix
        - recovery: (3,)-vector
        - m_new = decay * m_old + recovery
    """

    decay = np.zeros((3, 3))
    decay[0, 0] = np.exp(-dt/t2)
    decay[1, 1] = np.exp(-dt/t2)
    decay[2, 2] = np.exp(-dt/t1)

    recov = np.zeros((3,))
    recov[2] = 1 - np.exp(-dt/t1)

    return decay, recov


def rotmat_around_arbitrary_axis(rotax, th):
    """Returns rotation matrix around an arbitrary axis.

    Args:
        rotax (list): (3,) rotation axis
        th (float): angle in radian

    Returns:
        numpy array: (3, 3) rotation matrix

    Notes:
        - See http://scipp.ucsc.edu/~haber/ph216/rotation_12.pdf
    """

    rotmat = np.zeros((3, 3))
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
