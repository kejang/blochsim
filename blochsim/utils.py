import numpy as np


def rfscaleg(rf, t, gamma=4257.58):
    """Convert scaled rf to that in Gauss.

    Args:
        rf (numpy array): scaled rf waveform (sum(rf) = flip angle)
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
    gamma=4257.58
):
    """Returns a Hard pulse.

    Args:
        b1_max (float): maximum b1 value in Gauss
        flip_ang (float): flip angle in radian
        RF_UPDATE_TIME (int or float): dt in us
        multiple_of_n (int): constraint in length, like 2 or 4
        gamma (float): gyromagnetic ratio in Hz/G

    Returns:
        numpy array: hard pulse scaled to Gauss

    Notes: 
        - See John Pauly's rftools
    """

    n = flip_ang/b1_max/(2*np.pi*gamma*1e-3)/(RF_UPDATE_TIME/1000) + 2
    n = int(n/multiple_of_n + 0.5)*multiple_of_n

    rf = np.ones(n)
    rf[0] = 0
    rf[-1] = 0

    rf = rf/np.sum(rf)*flip_ang

    return rfscaleg(rf, n*RF_UPDATE_TIME/1000)
