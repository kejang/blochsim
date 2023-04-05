import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial
from multiprocessing import Pool


def plot_m(fig,
           m,
           axis_limit=0.75,
           color='#000000',
           marker='o',
           markersize=2.0,
           linewidth=1.0):
    """Plot m/m0 on a unit sphere.

    Args:
        fig (`plt.figure`): matplotlib.pyplot.figure
        m (list): (mx, my, mz), normalized with respect to m0
        axis_limit (int): value for set_xlim3d. should be less than one.
        color (str): color of m
        marker (str): marker of tip
        markersize (int): marker size
        linewidth (int): linewidth
    """

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.set_xlim3d([-axis_limit, axis_limit])
    ax.set_ylim3d([-axis_limit, axis_limit])
    ax.set_zlim3d([-axis_limit, axis_limit])

    ax.set_aspect('equal')
    ax.set_axis_off()

    # draw a unit sphere

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    for i in range(x.shape[0]):
        ax.plot(x[i], y[i], z[i],
                color='#C0C0C0',
                linestyle='dashed',
                linewidth=linewidth*0.25)

    for i in range(x.shape[1]):
        ax.plot(x.T[i], y.T[i], z.T[i],
                color='#C0C0C0',
                linestyle='dashed',
                linewidth=linewidth*0.25)

    # plot m

    ax.plot([0, m[0]], [0, m[1]], [0, m[2]],
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            marker=marker,
            markersize=markersize,
            markevery=[1],
            linestyle='solid',
            linewidth=linewidth)


def export_to_image(m_and_ind,
                    target_dir,
                    n_digit,
                    prefix='',
                    figsize=(2, 2),
                    dpi=300,
                    transparent=False,
                    axis_limit=0.75,
                    color='#000000',
                    marker='o',
                    markersize=2,
                    linewidth=1.0):
    """Export m/m0 to a png image.

    Args:
        m_and_ind (tuple): (m, index)
        target_dir (str): target directory name
        n_digit (int): number of digits for the suffix of image filenames
        prefix (str): prefix of image filenames
        figsize (tuple): figure size in inches
        dpi (int): dots-per-inch
        transparent (bool): option of savefig
        axis_limit (int): value for set_xlim3d. should be less than one.
        color (str): color of m
        marker (str): marker of tip
        markersize (int): marker size
        linewidth (int): linewidth
    """

    m, ind = m_and_ind

    fig = plt.figure(figsize=figsize)
    plot_m(fig,
           m,
           axis_limit=axis_limit,
           color=color,
           marker=marker,
           markersize=markersize,
           linewidth=linewidth)

    if prefix:
        fn = prefix + '-' + str(ind).zfill(n_digit) + '.png'
    else:
        fn = str(ind).zfill(n_digit) + '.png'

    filepath = Path(target_dir).joinpath(fn)

    fig.savefig(
        filepath,
        dpi=dpi,
        transparent=transparent,
        bbox_inches='tight'
    )

    plt.close(fig)


def export_to_images(target_dir,
                     ms,
                     figsize=(2, 2),
                     dpi=300,
                     transparent=False,
                     axis_limit=0.75,
                     color='#000000',
                     marker='o',
                     markersize=2,
                     linewidth=1.0,
                     max_workers=1):
    """Export a series of m/m0 to png images.

    Args:
        target_dir (str): target directory name
        ms (`ndarray`): series of m (n by 3)
        figsize (tuple): figure size in inches
        dpi (int): dots-per-inch
        transparent (bool): option of savefig
        axis_limit (int): value for set_xlim3d. should be less than one.
        color (str): color of m
        marker (str): marker of tip
        markersize (int): marker size
        linewidth (int): linewidth
        max_workers (int): number of cores for multiprocessing
    """

    n_digit = len(str(len(ms)))

    func = partial(export_to_image,
                   target_dir=target_dir,
                   n_digit=n_digit,
                   figsize=figsize,
                   dpi=dpi,
                   transparent=transparent,
                   axis_limit=axis_limit,
                   color=color,
                   marker=marker,
                   markersize=markersize,
                   linewidth=linewidth)

    args = [(m, ind) for ind, m in enumerate(ms)]

    if max_workers > 1:
        with Pool(max_workers) as pool:
            _ = pool.map(func, args)
    else:
        for arg in args:
            func(arg)
