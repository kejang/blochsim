import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from functools import partial
from multiprocessing import Pool


def plot_m_3d(fig,
              ms=[[0, 0, 1]],
              axis_limit=0.75,
              color='#000000',
              marker='o',
              markersize=2.0,
              linewidth=1.0,
              label=None,
              labelloc='upper right',
              labelsize='xx-small'):
    """Plot m/m0 on a unit sphere.

    Args:
        fig (`plt.figure`): matplotlib.pyplot.figure
        ms (list): list of m's, i.e., [[0, 0, 1]]
        axis_limit (int): value for set_xlim3d. should be less than one.
        color (str | list): color of m's (str: single, list: multiple colors)
        marker (str): marker of tip
        markersize (int): marker size
        linewidth (int): linewidth
        label (None | list): labels of m
        labelloc (str): label location
        labelsize (int | str): fontsize of legend()
    """

    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.set_xlim3d([-axis_limit, axis_limit])
    ax.set_ylim3d([-axis_limit, axis_limit])
    ax.set_zlim3d([-axis_limit, axis_limit])

    # ax.set_aspect('equal')
    ax.set_box_aspect((1, 1, 1))

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

    if isinstance(color, list):
        colors = color
    else:
        colors = [color]*len(ms)

    if isinstance(label, list):
        labels = label
    else:
        labels = [None]*len(ms)

    for m, color, label in zip(ms, colors, labels):
        ax.plot([0, m[0]], [0, m[1]], [0, m[2]],
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                marker=marker,
                markersize=markersize,
                markevery=[1],
                linestyle='solid',
                linewidth=linewidth,
                label=label)

    if any(labels):
        ax.legend(loc=labelloc, fontsize=labelsize)


def plot_m_xy(fig,
              ms=[[0, 0, 1]],
              axis_limit=1.0,
              color='#000000',
              marker='o',
              markersize=2.0,
              linewidth=1.0,
              label=None,
              labelloc='upper right',
              labelsize='xx-small'):
    """Plot m/m0 on a unit sphere.

    Args:
        fig (`plt.figure`): matplotlib.pyplot.figure
        ms (list): list of m's, i.e., [[0, 0, 1]]
        axis_limit (int): value for set_xlim. should be less than one.
        color (str | list): color of m's (str: single, list: multiple colors)
        marker (str): marker of tip
        markersize (int): marker size
        linewidth (int): linewidth
        label (None | list): labels of m
        labelloc (str): label location
        labelsize (int | str): fontsize of legend()
    """

    ax = fig.add_axes([-1, -1, 1, 1])
    ax.set_xlim([-axis_limit, axis_limit])
    ax.set_ylim([-axis_limit, axis_limit])
    ax.set_aspect('equal')

    # plot m

    if isinstance(color, list):
        colors = color
    else:
        colors = [color]*len(ms)

    if isinstance(label, list):
        labels = label
    else:
        labels = [None]*len(ms)

    for m, color, label in zip(ms, colors, labels):
        ax.plot([0, m[0]], [0, m[1]],
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                marker=marker,
                markersize=markersize,
                markevery=[1],
                linestyle='solid',
                linewidth=linewidth,
                label=label)

    if any(labels):
        ax.legend(loc=labelloc, fontsize=labelsize)


def export_to_image(ms_and_ind,
                    target_dir,
                    mode='3d',
                    n_digit=5,
                    prefix='',
                    figsize=(3, 3),
                    dpi=300,
                    transparent=False,
                    axis_limit=0.75,
                    color='#000000',
                    marker='o',
                    markersize=2,
                    linewidth=1.0,
                    label=None,
                    labelloc='upper right',
                    labelsize='xx-small'):
    """Export m/m0 to a png image.

    Args:
        ms_and_ind (tuple): (ms, index)
                            (ms (list): list of m's, i.e., [[0, 0, 1]])
        target_dir (str): target directory name
        mode (str): {'3d', 'xy}
        n_digit (int): number of digits for the suffix of image filenames
        prefix (str): prefix of image filenames
        figsize (tuple): figure size in inches
        dpi (int): dots-per-inch
        transparent (bool): option of savefig
        axis_limit (int): value for set_xlim3d. should be less than one.
        color (str | list): color of m's (str: single, list: multiple colors)
        marker (str): marker of tip
        markersize (int): marker size
        linewidth (int): linewidth
        label (None | list): labels of m
        labelloc (str): label location
        labelsize (int | str): fontsize of legend().
    """

    ms, ind = ms_and_ind

    fig = plt.figure(figsize=figsize)

    if mode == 'xy':
        plot_func = plot_m_xy
    else:
        plot_func = plot_m_3d

    plot_func(fig,
              ms,
              axis_limit=axis_limit,
              color=color,
              marker=marker,
              markersize=markersize,
              linewidth=linewidth,
              label=label,
              labelloc=labelloc,
              labelsize=labelsize)

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
                     ms_list=[[[0, 0, 1]]],
                     mode='3d',
                     figsize=(3, 3),
                     dpi=300,
                     transparent=False,
                     axis_limit=0.75,
                     color='#000000',
                     marker='o',
                     markersize=2,
                     linewidth=1.0,
                     max_workers=1,
                     label=None,
                     labelloc='upper right',
                     labelsize='xx-small'):
    """Export a series of m/m0 to png images.

    Args:
        target_dir (str): target directory name
        ms_list (list): list of a series of ms
                        [series-0, series-1, ...] where series-i is a list of
                        m's, i.e., [[0, 0, 1]].
                        In other words, the size of it is (p, q, 3) where
                        q is the length of time-series.
        mode (str): {'3d', 'xy}
        figsize (tuple): figure size in inches
        dpi (int): dots-per-inch
        transparent (bool): option of savefig
        axis_limit (int): value for set_xlim3d. should be less than one.
        color (str): color of m
        marker (str): marker of tip
        markersize (int): marker size
        linewidth (int): linewidth
        max_workers (int): number of cores for multiprocessing
        label (None | list): labels of m
        labelloc (str): label location
        labelsize (int | str): fontsize of legend().
    """

    n_sample = len(ms_list[0])
    n_digit = len(str(n_sample))

    func = partial(export_to_image,
                   target_dir=target_dir,
                   mode=mode,
                   n_digit=n_digit,
                   figsize=figsize,
                   dpi=dpi,
                   transparent=transparent,
                   axis_limit=axis_limit,
                   color=color,
                   marker=marker,
                   markersize=markersize,
                   linewidth=linewidth,
                   label=label,
                   labelloc=labelloc,
                   labelsize=labelsize)

    args = [[[], i] for i in range(n_sample)]
    for ms in ms_list:
        for i, m in enumerate(ms):
            args[i][0].append(m)

    if max_workers > 1:
        with Pool(max_workers) as pool:
            _ = pool.map(func, args)
    else:
        for arg in args:
            func(arg)
