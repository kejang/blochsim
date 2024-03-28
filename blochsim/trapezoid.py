import numpy as np


def round_up_length(x, round_n=2):
    """Rounding function for trapezoid design."""

    return round_n * int(np.ceil(x / round_n))


def get_trap(amp, ramp, plateau, boundary='right', dtype='float32'):
    """Returns a trapezoid.

    Args:
        amp (float): amplitude in G/cm
        ramp (int): length of ramp
        plateau (int): length of plateau
        boundary (str): {'right', 'left', 'both', 'neither'}
        dtype (str): dtype of ndarray

    Returns:
        `ndarray`: trapezoid
    """

    if boundary == 'right':
        attack = np.linspace(0, amp, ramp + 1, endpoint=True)[1:]
        decay = np.linspace(0, amp, ramp, endpoint=False)[::-1]
    elif boundary == 'left':
        attack = np.linspace(0, amp, ramp, endpoint=False)
        decay = (np.linspace(0, amp, ramp + 1, endpoint=True)[1:])[::-1]
    elif boundary == 'both':
        attack = np.linspace(0, amp, ramp, endpoint=False)
        decay = attack[::-1]
    else:    # neither
        attack = np.linspace(0, amp, ramp + 1, endpoint=False)[1:]
        decay = attack[::-1]

    trap = np.concatenate(
        [attack,
         amp * np.ones(plateau),
         decay]
    )

    return trap.astype(dtype)


def get_trap_slice_select(
    slthick,
    dur,
    bw,
    gmax=4.0,
    smax=15.0,
    interval=4,
    boundary='right',
    dtype='float32',
    gamma=4257.59,
    round_n=1,
):
    """Returns a slice-select trapezoid.

    Args:
        slthick (float): slice thickness in cm
        dur (int): RF duration in us
        bw (float): RF bandwidth in Hz
        gmax (float): maximum gradient amplitude in G/cm
        smax (float): maximum gradient slew rate in G/cm/ms
        interval (int): time interval in us
        boundary (str): {'right', 'left', 'both'}
        dtype (str): dtype of ndarray
        gamma (float): gyromagnetic ratio in Hz/G
        round_n (int): round up ramp length to multiple of this number

    Returns:
        tuple: `ndarray` (trapezoid) and int (length of ramp)
    """

    amp = bw / (slthick * gamma)

    if amp > gmax:
        # bandwidth is too large, and/or slice thickness is too thin.
        return None, None

    smax *= 1e-3    # G/cm/us

    ramp = round_up_length(amp / smax / interval, round_n)
    plateau = round_up_length(dur / interval, round_n)
    trap = get_trap(amp, ramp, plateau, boundary, dtype)

    return trap, ramp


def get_trap_given_area(
    area,
    min_plateau=2,
    gmax=4.0,
    smax=15.0,
    interval=4,
    boundary='right',
    dtype='float32',
    round_n=1,
):
    """Returns a trapezoid with given area.

    Args:
        area (float): area of trapezoid in G/cm * us
        min_plateau (int): minimum plateau length
        gmax (float): maximum gradient amplitude in G/cm
        smax (float): maximum gradient slew rate in G/cm/ms
        interval (int): time interval in us
        boundary (str): {'right', 'left', 'both', 'neither'}
        dtype (str): dtype of ndarray
        round_n (int): round up ramp length to multiple of this number

    Returns:
        tuple: `ndarray` (trapezoid) and int (length of ramp)
    """

    smax *= 1e-3    # G/cm/us
    min_ramp = round_up_length(gmax / smax / interval, round_n)
    tt = np.abs(area) / gmax - min_ramp * interval

    if tt > min_plateau * interval:
        plateau = round_up_length(tt / interval, round_n)
        amp = gmax
        ramp = min_ramp
    else:
        plateau = min_plateau
        tt = plateau * interval
        amp = (-tt * smax
               + np.sqrt((tt * smax)**2 + 4 * smax * np.abs(area))) / 2
        ramp = round_up_length(amp / smax / interval, round_n)

    trap = get_trap(amp, ramp, plateau, boundary, dtype)
    trap *= area / (np.sum(trap) * interval)

    return trap, ramp


def get_trap_largest(
    n,
    gmax=4.0,
    smax=15.0,
    min_plateau=2,
    interval=4,
    boundary='right',
    dtype='float32',
    round_n=1,
):
    """Returns the largest trapezoid for given length.

    Args:
        n (int): length (total number of sampling points) of trapezoid
        gmax (float): maximum gradient amplitude in G/cm
        smax (float): maximum gradient slew rate in G/cm/ms
        min_plateau (int): minimum plateau length
        interval (int): time interval in us
        boundary (str): {'right', 'left', 'both'}
        dtype (str): dtype of ndarray
        round_n (int): round up ramp length to multiple of this number

    Returns:
        tuple: `ndarray` (trapezoid) and int (length of ramp)
    """
    min_ramp = round_up_length(gmax / (smax * 1e-3) / interval, round_n)

    max_ramp_raw = (n - min_plateau) / 2
    max_ramp = round_up_length(max_ramp_raw, round_n)

    while (min_plateau + 2 * max_ramp > n):
        max_ramp_raw -= 0.5
        max_ramp = round_up_length(max_ramp_raw, round_n)

    if min_ramp < max_ramp:
        amp = gmax
        ramp = min_ramp
        plateau = n - 2 * ramp
    else:
        amp = (smax * 1e-3) * max_ramp * interval
        ramp = max_ramp
        plateau = n - 2 * ramp

    return get_trap(amp, ramp, plateau, boundary, dtype), ramp


def get_trap_triangle(
    amp,
    gmax=4.0,
    smax=15.0,
    min_plateau=2,
    interval=4,
    boundary='right',
    dtype='float32',
    round_n=1,
):
    """Returns a triangular trapezoid.

    Args:
        amp (float): amplitude in G/cm
        gmax (float): maximum gradient amplitude in G/cm
        smax (float): maximum gradient slew rate in G/cm/ms
        min_plateau (int): minimum plateau length
        interval (int): time interval in us
        boundary (str): {'right', 'left', 'both'}
        dtype (str): dtype of ndarray
        round_n (int): round up ramp length to multiple of this number

    Returns:
        tuple: `ndarray` (trapezoid) and int (length of ramp)
    """

    if amp > gmax:
        return None, None

    smax *= 1e-3    # G/cm/us

    ramp = round_up_length(amp / smax / interval, round_n)
    plateau = min_plateau
    trap = get_trap(amp, ramp, plateau, boundary, dtype)

    return trap, ramp
