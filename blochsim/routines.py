import numpy as np
from .blochsim import blochsim


def estimate_inversion_time(
    t1,
    t2,
    time_res=1e-4,
    df=0,
    r=[0, 0, 0],
    dtype='float32',
):
    """Estimate Inversion Time (TI).

    Args:
        t1 (float): T1 in sec
        t2 (float): T2 in sec
        time_res (float): time resolution in sec
        df (float): off-resonance in Hz
        r (list): 3 by 1 Cartesian coordinate in cm
        dtype (str): dtype of real type ('float32', 'float64', ...)

    Returns:
        float: inversion time in sec
    """

    _, a, b = blochsim(
        b1=np.zeros(1, dtype='complex64'),
        g=np.zeros((1, 3), dtype='float32'),
        dt=[time_res],
        r=r,
        df=df,
        t1=t1,
        t2=t2,
        dtype=dtype,
    )

    m = np.array([0, 0, -1], dtype='float32')
    ti = 0
    while m[-1] < 0:
        m = np.matmul(a, m) + b
        ti += time_res

    return ti
