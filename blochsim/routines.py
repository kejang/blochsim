import numpy as np
from .blochsim import blochsim


def estimate_inversion_time(
    t1,
    t2,
    time_res=1e-4,
    df=0,
    r=[0, 0, 0]
):
    """Estimate Inversion Time (TI).

    Args:
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        time_res (float): time resolution in sec
        df (float): off-resonance in Hz
        r (list): 3 by 1 Cartesian coordinate in cm

    Returns:
        float: inversion time in sec    
    """

    b1 = np.zeros(1, dtype='complex64')
    g = np.zeros((1, 3), dtype='float32')
    dt = [time_res]

    m = [0, 0, -1]
    ti = 0
    while m[-1] < 0:
        m = blochsim(b1, g, dt, r, df, t1, t2, m0=m)[0][-1]
        ti += time_res

    return ti
