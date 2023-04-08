import numpy as np


def rfscaleg(rf, t, gamma=4257.58):
    """Convert scaled rf to that in Gauss.

    Args:
        rf (`ndarray`): scaled rf waveform (sum(rf) = flip angle)
        t (float): duration of RF pulse in ms
        gamma (float): gyromagnetic ratio in Hz/G

    Returns:
        numpy array: rf waveform scaled to Gauss
    """

    dt = t/rf.size
    return rf / (2*np.pi*gamma*1e-3*dt)


def get_hard_pulse(
    b1_max,
    flip_ang,
    RF_UPDATE_TIME=2,
    multiple_of_n=4,
    gamma=4257.58,
    precision='single'
):
    """Returns a Hard pulse.

    Args:
        b1_max (float): maximum b1 value in Gauss
        flip_ang (float): flip angle in radian
        RF_UPDATE_TIME (int | float): dt in us
        multiple_of_n (int): constraint in length, like 2 or 4
        gamma (float): gyromagnetic ratio in Hz/G
        precision (str): 'single' or 'double'

    Returns:
        `ndarray`: hard pulse scaled to Gauss

    Notes:
        - See John Pauly's rftools
    """

    n = flip_ang/b1_max/(2*np.pi*gamma*1e-3)/(RF_UPDATE_TIME/1000)
    n = int((n + multiple_of_n - 1)/multiple_of_n)*multiple_of_n

    if precision == 'double':
        rf = np.ones(n, dtype='complex128')
    else:
        rf = np.ones(n, dtype='complex64')

    rf = rf/np.sum(rf)*flip_ang

    return rfscaleg(rf, n*RF_UPDATE_TIME/1000)


def design_hyperbolic_secant_pulse(b1_max, dur, tbw, beta, RF_UPDATE_TIME=2):
    """Design Adiabatic Full-passage Pulse (Hyperbolic Secant Pulse)

    Args:
        b1_max (float): maximum rf amplitude in G
        dur (float): duration in ms
        tbw (float): time-bandwidth
        beta (float): beta (see Bernstein (6.34))
        RF_UPDATE_TIME (int): time interval in us

    Returns:
        `ndarray`: complex HS pulse
        `ndarray`: amplitude of HS pulse
        `ndarray`: phase of HS pulse

    Notes:
        df = mu*beta/pi in (6.42), so once tbw is determined, either one of
        beta and mu should be fixed. Here beta is given, and mu is determined
        using (6.42) in Bernstein:
            mu = pi/beta * (tbw / (dur * 1e-3))
    """

    n = int(2*np.round(0.5 * dur * 1e3 / RF_UPDATE_TIME))
    t = np.linspace(-0.5*dur*1e-3, 0.5*dur*1e-3, n + 1, endpoint=True)[1:]

    b1_rho = b1_max / np.cosh(beta * t)                              # (6.36)
    mu = np.pi/beta * (tbw / (dur * 1e-3))                           # (6.42)
    b1_phs = mu * -np.log(np.cosh(beta * t)) + mu * np.log(b1_max)   # (6.37)
    b1 = b1_rho * np.exp(1j * b1_phs)                                # (6.35)

    return b1, b1_rho, b1_phs
