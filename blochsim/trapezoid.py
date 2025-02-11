import numpy as np
from scipy.spatial.transform import Rotation


def round_up_length(x, round_n=2):
    """Rounding function for trapezoid design."""

    return round_n * int(np.ceil(x / round_n))


def get_nrmp(amp, smax, dt):
    """Returns the number of points of the ramp.

    Args:
        amp (float): target gradient amplitude in G/cm
        smax (float): slew rate in G/cm/ms
        dt (float | int): sampling interval in us

    Returns:
        int: number of points of the ramp.
    """
    return int(np.ceil(np.abs(amp) / dt / smax * 1000 - 1))


def get_trap_area(amp, nrmp, nplt, dt):
    """Returns the area of the trapezoid.

    Args:
        amp (float): amplitude of the plateau
        nrmp (it): number of points of the ramp
        nplt (int): number of points of the plateau
        dt (float | int): sampling interval in us

    Returns:
        _type_: _description_
    """

    # d = np.abs(amp) / (nrmp + 1)
    # area_ramp = (1 + nrmp) * nrmp * d
    # area_plateau = nplt * amp
    # return dt * (area_ramp + area_plateau)

    return dt * amp * (nrmp + nplt)


def get_trap(amp, nrmp, nplt, dtype="float32"):
    """Returns a trapezoid.

    Args:
        amp (float): amplitude of the plateau in G/cm
        nrmp (int): number of points of the ramp
        nplt (int): number of points of the plateau
        dtype (str, optional): dtype of ndarray. Defaults to "float32".

    Returns:
        ndarray: trapezoid

    Notes:
        The previous gradient amplitude is assumed to be zero. The end of this
        trapezoid is zero.
        (In other words, it follows "right-side" boundary convention.)
    """

    trap = np.zeros((2 * nrmp + nplt + 1), dtype=dtype)
    d = amp / (nrmp + 1)

    for i in range(nrmp):
        trap[i] = (i + 1) * d

    for i in range(nrmp, nrmp + nplt):
        trap[i] = amp

    for i in range(nrmp + nplt, 2 * nrmp + nplt):
        trap[i] = amp - (i - nrmp - nplt + 1) * d

    return trap


def design_trap_given_area(
    area,
    min_plateau=2,
    gmax=3.9,
    smax=14.9,
    dt=4,
):
    """Returns trapezoid parameters for the given area.

    Args:
        area (float): desired area in G/cm * us
        min_plateau (int, optional): minimum number of points of the plateau. Defaults to 2.
        gmax (float, optional): maximum gradient amplitude in G/cm. Defaults to 3.9.
        smax (float, optional): maximum slew-rate in G/cm/ms. Defaults to 14.9.
        dt (int | float, optional): sampling interval in us. Defaults to 4.

    Returns:
        tuple: (amp, nrmp, nplt)
    """

    # first, design the trapezoid using the analytic model

    nrmp = get_nrmp(gmax, smax, dt)
    nplt = int(np.ceil(np.abs(area) / dt / gmax - nrmp))

    if nplt < min_plateau:
        nplt = min_plateau
        b = (nplt - 1) * dt * (smax / 1000.0)
        amp = (-b + np.sqrt(b**2 + 4 * (smax / 1000.0) * np.abs(area))) / 2
        nrmp = get_nrmp(amp, smax, dt)
    else:
        nplt = 1 + int(np.ceil(np.abs(area) / gmax / dt - gmax / (smax / 1000.0) / dt))
        nrmp = get_nrmp(gmax, smax, dt)
        if area > 0:
            amp = gmax
        else:
            amp = -gmax

    # compensate the error of the analytic model

    area_temp = get_trap_area(amp, nrmp, nplt, dt)
    amp *= area / area_temp

    return amp, nrmp, nplt


def cart_to_sph(x, y, z):
    """
    Convert Cartesian to Spherical coordinate.

    Args:
        x (float): x-coordinate in Cartesian.
        y (float): y-coordinate in Cartesian.
        z (float): z-coordinate in Cartesian.

    Returns:
        tuple: A tuple containing the following spherical coordinates:
            rho (float): Radial distance from the origin.
            azimuthal_angle (float): Azimuthal angle.
            polar_angle (float): Polar angle.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    azimuthal_angle = np.arctan2(y, x)
    polar_angle = np.arccos(z / r)

    return r, azimuthal_angle, polar_angle


def design_concurrent_trap(
    area_x, area_y, area_z, min_plateau=2, gmax=3.9, smax=14.9, dt=4
):
    """Returns concurrent trapezoid parameters for the given area.

    Args:
        area_x (float): desired area on XGRAD in G/cm * us
        area_y (float): desired area on YGRAD in G/cm * us
        area_z (float): desired area on ZGRAD in G/cm * us
        min_plateau (int, optional): minimum number of points of the plateau. Defaults to 2.
        gmax (float, optional): maximum gradient amplitude in G/cm. Defaults to 3.9.
        smax (float, optional): maximum slew-rate in G/cm/ms. Defaults to 14.9.
        dt (int | float, optional): sampling interval in us. Defaults to 4.

    Returns:
        tuple: (amp_x, amp_y, amp_z, nrmp, nplt)
    """

    rho, theta, phi = cart_to_sph(area_x, area_y, area_z)

    amp, nrmp, nplt = design_trap_given_area(rho, min_plateau, gmax, smax, dt)

    rotmat = Rotation.from_euler("yz", [phi, theta]).as_matrix()
    amps = np.asarray(np.matmul(rotmat, [0, 0, amp]))
    amp_x, amp_y, amp_z = amps

    return amp_x, amp_y, amp_z, nrmp, nplt


def design_trap_largest(
    n, gmax=3.9, smax=14.9, kmax=1e9, dt=4, min_plateau=2, gamma=4257.58
):

    nrmp = get_nrmp(gmax, smax, dt)
    nplt = n - 2 * nrmp - 1

    if nplt > min_plateau:
        amp = gmax
    else:
        nrmp = int((n - min_plateau - 1) / 2)
        nplt = n - 2 * nrmp - 1
        amp = (nrmp + 1) * (smax * 1e-3) * dt

    k = 1e-6 * dt * gamma * amp * (nrmp + nplt)

    if k < kmax:
        return amp, nrmp, nplt
    else:
        b = -dt * n * (smax * 1e-3)
        c = kmax * smax / gamma * 1e3

        amp_0 = 0.5 * (-b - np.sqrt(b**2 - 4 * c))
        amp_1 = 0.5 * (-b + np.sqrt(b**2 - 4 * c))
        if np.abs(amp_0) < gmax:
            amp = amp_0
        else:
            amp = amp_1

        nrmp = get_nrmp(amp, smax, dt)
        nplt = n - 2 * nrmp - 1

        return amp, nrmp, nplt


def get_trap_largest(
    n,
    gmax=3.9,
    smax=14.9,
    kmax=1e9,
    dt=4,
    min_plateau=2,
    gamma=4257.58,
    dtype="float32",
):

    amp, nrmp, nplt = design_trap_largest(n, gmax, smax, kmax, dt, min_plateau, gamma)

    trap = get_trap(amp, nrmp, nplt, dtype)

    return trap, nrmp


def get_trap_triangle(
    amp,
    smax=14.9,
    min_plateau=2,
    dt=4,
    dtype="float32",
):

    nrmp = get_nrmp(amp, smax, dt)
    nplt = min_plateau
    trap = get_trap(amp, nrmp, nplt, dtype)

    return trap, nrmp


def design_refocus_waveform(bw, dur, slthick, ncyc, gmax, smax, dt, gamma=4257.58):
    """Designs the gradient waveform for the refocus pulse in spin-echo sequences.

    Args:
        bw (float): bandwidth of the refocus pulse in Hz
        dur (int): duration of the refocus pulse in us
        slthick (float): slice-thickness in cm
        ncyc (float): number of cycles of the crusher
        gmax (float): maximum gradient amplitude in G/cm
        smax (float): maximum slew-rate in G/cm/ms
        dt (int): time interval in us
        gamma (float, optional): gyromagnetic ratio. Defaults to 4257.58.

    Returns:
        tuple: amp of crusher, amp of slice-select, ramp, bridge, plateau (crusher), plateau (rf)
    """

    # initial design for crusher

    _, nrmp_0, nplt_crsh_0 = design_trap_given_area(
        ncyc / (gamma * slthick) * 1e6,
        gmax=gmax,
        smax=smax,
        dt=dt,
    )

    # use the maximum area for given time

    amp_crsh, nrmp, nplt_crsh = design_trap_largest(
        (2 * nrmp_0 + nplt_crsh_0 + 1),
        gmax=gmax,
        smax=smax,
        dt=dt,
    )

    # design slice-select

    amp_rf = bw / (slthick * gamma)
    nplt_rf = dur // dt

    # design bridge

    nbrdg = get_nrmp(np.abs(amp_crsh - amp_rf), smax, dt)

    return amp_crsh, amp_rf, nrmp, nbrdg, nplt_crsh, nplt_rf


def get_refocus_waveform(amp_crsh, amp_rf, nrmp, nbrdg, nplt_crsh, nplt_rf):

    segments = []

    # ramp: zero to crusher plateau

    d = amp_crsh / (nrmp + 1)
    segments.append(d * (1 + np.arange(nrmp)))

    # crusher plateau

    segments.append(amp_crsh * np.ones(nplt_crsh))

    # crusher plateau to slice-select

    d = np.abs(amp_crsh - amp_rf) / (nbrdg + 1)
    segments.append(amp_crsh - d * (1 + np.arange(nbrdg)))

    # slice-select

    segments.append(amp_rf * np.ones(nplt_rf))

    # slice-select to crusher plateau

    d = np.abs(amp_crsh - amp_rf) / (nbrdg + 1)
    segments.append(amp_rf + d * (1 + np.arange(nbrdg)))

    # crusher plateau

    segments.append(amp_crsh * np.ones(nplt_crsh))

    # ramp: crusher plateau to zero

    d = amp_crsh / (nrmp + 1)
    segments.append(amp_crsh - d * (1 + np.arange(nrmp)))

    # final point should be zero.

    segments.append([0])

    return np.concatenate(segments)
