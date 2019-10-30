"""
.. module:: features
        :synopsis: Spike clustering

.. moduleauthor:: Ole Jonas Wenzel <wenzel@multichannelsystems.com>
"""

from sklearn.mixture import GaussianMixture
import numpy as np
from typing import Callable, Dict, List, Tuple, Optional, Iterator, Any

import matplotlib.pyplot as plt
import matplotlib as mpl

import warnings

VERBOSE = True


def plotWaveforms(w: np.ndarray,
                  assignment: np.ndarray,
                  clusters: np.ndarray = np.array([-1]),
                  waveforms: int = 20,
                  figure: int = 4,
                  colors: mpl.colors.LinearSegmentedColormap = None,
                  fs: float = 1.0):
    """
    Plot waveform per cluster.
    :param w: waveforms [length(cutout), #spikes, #channels]
    :param assignment: cluster assignments [#spikes, 1]
    :param clusters: numpy array holding clusters that are to be visualized
    :param waveforms: number of sample waveforms to plot per cluster
    :param figure: figure to use for plotting
    :param colors: colors to use for clusters
    :param fs: sampling frequency of signal
    :return:
    """

    # PARAMETERS
    if clusters[0] == -1:
        clusters = np.unique(assignment)

    if colors is None:
        colors = plt.cm.jet(np.linspace(0, 1, max(assignment) + 1))

    dt = 1/fs

    num_clusters = clusters.shape[0]
    num_channels = w.shape[2]

    w_len = w.shape[0]
    start = -1 * w_len // 2
    tt = np.arange(start, start + w_len) * dt * 1000

    w_max = np.max(w)
    w_min = np.min(w)
    w_range = w_max-w_min
    w_max += w_range*0.05
    w_min -= w_range*0.05

    # VISUALIZTION
    fig = plt.figure(figure, figsize=(4 * num_channels, 2.5 * num_clusters))
    for idx, cluster in enumerate(clusters):
        for channel in range(num_channels):
            ax = plt.subplot(num_clusters, num_channels, num_channels * idx + channel + 1)
            sample = np.random.randint(low=0, high=assignment[assignment == cluster].shape[0], size=waveforms)
            ax.plot(tt, w[:, assignment == cluster, channel][:, sample], c=colors[cluster], linewidth=0.5, alpha=0.3)
            ax.plot(tt, w[:, assignment == cluster, channel][:, sample].mean(axis=1), 'k', linewidth=2)
            plt.title('Cluster {}, Channel {}'.format(cluster, channel + 1))
            plt.ylim((w_min, w_max))
            plt.xlim((tt[0], tt[-1]))
            plt.ylabel('Voltage')
            plt.xlabel('time')

    plt.subplots_adjust(hspace=0.6, wspace=0.5)
    plt.show()


def correlogram(t: np.ndarray, assignment: np.ndarray, binsize: int, maxlag: int):
    """
    Computes cross- and autocorrelation of spike trains.
    :param t: spike trains givens as spike timestamps   [#spikes, 1]
    :param assignment: cluster assignments              [#spikes, 1]
    :param binsize: binsize in ccg in ms                scalar
    :param maxlag: maximal lag in ms                    scalar
    :return: ccg, bins computed correlogram and bins
            relative to center respectively             [#bins, 3clusters, #custers], [#bins, 1]
    """

    # PARAMETERS
    clusters = np.unique(assignment)
    num_clusters = clusters.shape[0]
    num_spikes = t.shape[0]
    num_bins = 2 * maxlag // binsize + 1
    ccg = np.zeros((num_bins, num_clusters, num_clusters))
    VERBOSE = False

    # GET SPIKE TRAINS PER CLUSTER
    spike_trains = [t[assignment == cluster] for cluster in clusters]

    for reference in range(num_clusters):
        if VERBOSE:
            print('Processing cluster {}.'.format(reference))
        for target in range(reference + 1):
            # COMPUTE REFERENCE SPIKE TRAIN
            ref_spike_train = spike_trains[reference]
            if VERBOSE:
                print('\tCorrelating {} ref spikes.'.format(ref_spike_train.shape[0]))
            for ref_spike in ref_spike_train:
                # COMPUTE TARGET SPIKE TRAIN
                trgt_spike_train = spike_trains[target]
                diff = trgt_spike_train - ref_spike
                window = diff[np.abs(diff) <= maxlag]
                if VERBOSE:
                    print(trgt_spike_train.shape)
                bincount = np.bincount(((window + maxlag) // binsize).astype(int), minlength=num_bins)
                if reference == target:
                    bincount[num_bins // 2] = 0
                    ccg[:, reference, target] += bincount
                else:
                    ccg[:, reference, target] += bincount
                    ccg[:, target, reference] += np.flip(bincount)

    bins = np.arange(-((maxlag) * binsize), ((maxlag + 1) * binsize), binsize)

    return ccg, bins


def plotCorrelogram(ccg: np.ndarray,
                    bins: np.ndarray,
                    figure: int = 2,
                    axis: str = 'off',
                    colors: mpl.colors.LinearSegmentedColormap = None):
    """
    Plots cross-correlogram of all combination of clusters.
    :param ccg: computed correlograms [#bins, 3clusters, #custers]
    :param bins: bins relative to center [#bins, 1]
    :param figure: figure to use for plotting
    :param axis: should be 'on'/'off'
    :param colors: colors
    :return:
    """

    # PARAMETERS
    if colors is None:
        colors = plt.cm.jet(np.linspace(0, 1, ccg.shape[1] + 1))

    # VISUALIZATION
    fig = plt.figure(figure, figsize=(12, 10))
    plt.clf

    bg = 0.7 * np.ones(3)

    K = ccg.shape[1]

    for ix in range(K):
        for jx in range(K):
            ax = plt.subplot(K, K, K * ix + jx + 1, facecolor=bg)

            if ix == jx:
                ax.bar(bins, ccg[:, ix, jx], width=1, facecolor=colors[ix, :], edgecolor=colors[ix, :])
            else:
                ax.bar(bins, ccg[:, ix, jx], width=1, facecolor='black', edgecolor='black')
            ax.axis(axis)
            ax.set_xlim(1.2 * bins[[0, -1]])
            ylim = np.array(list(ax.get_ylim()))
            ax.set_ylim(np.array([0, 1.2]) * ylim)
            ax.set_yticks([])

            if ix != jx:
                ax.plot(0, 0, '*', c=colors[jx, :])

            if ix != K - 1:
                ax.set_xticks([])

            if ix == K - 1:
                ax.set_xlabel('ms')


def separation(b: np.ndarray,
               m: np.ndarray,
               S: np.ndarray,
               p: np.ndarray,
               assignment: np.ndarray,
               nbins: int = 50,
               figure: int = 3):
    """
    Calculate and plot cluster separation using LDA. Results are row-wise standard-normalized
    to the left/row-consistent cluster.
    :param b: features [#spikes, #features]
    :param m: mean [#clusters, #features]
    :param S: covariances [#features, #features, #clusters]
    :param p: priors [#clusters, 1]
    :param assignment: [#spikes, 1]
    :param nbins: number of bins
    :param figure: figure to use for plotting
    :return:
    """

    K = np.max(assignment) + 1
    colors = plt.cm.jet(np.linspace(0, 1, K))

    fig = plt.figure(figure, figsize=(12, 12))
    plt.clf

    bg = 0.7 * np.ones(3)

    for ix in range(K):
        for jx in np.delete(np.arange(0, K), ix):
            # LDA - determine optimal projection line for clusters ix and ij
            w = np.linalg.inv(S[:, :, ix] + S[:, :, jx]) @ (m[jx, :] - m[ix, :])

            # Project spikes(spike features)onto line w
            qi = b[assignment == ix, :] @ w
            qj = b[assignment == jx, :] @ w

            # Compute shift and scale values
            mean = np.average(qi)
            std = np.std(qi)

            # Normalize to Standard-Normal using shift and scale
            qi = (qi - mean) / std
            qj = (qj - mean) / std

            # plot histograms on optimal axis

            ax = plt.subplot(K, K, ix * K + jx + 1, facecolor=bg)
            bins = np.linspace(-3, 10, nbins)
            h = np.array([np.histogram(qj, bins)[0], np.histogram(qi, bins)[0]])
            ax.bar(bins[1:-1], h[0, 1:], 1, color=colors[jx], linestyle=None, edgecolor=colors[jx])
            ax.bar(bins[1:-1], h[1, 1:], 1, color=colors[ix], edgecolor=colors[ix])

            x_min = np.minimum(np.min(qi), np.min(qj))
            x_max = np.maximum(np.max(qi), np.max(qj))
            ax.set_ylim([0, 1.2 * np.max(h)])
            ax.set_xlim([x_min, x_max])
            ax.set_yticks([])
            ax.set_xticks([])


def scatter_projected_features(data: np.ndarray,
                               color: np.ndarray = None,
                               title: str = 'For your own sake: ADD A TITLE!',
                               *args,
                               **kwargs):
    """
    scatter plot data
    :param data: holding features projected into 2D space [#samples, 2]
    :param title: title for the plot
    :param color: array of color values (one for each datapoint) [#samples, 4]
    :return: handle to created figure and axes
    """
    #PARAMETERS
    alpha = 0.05
    if 'alpha' in kwargs.keys():
        alpha = kwargs['alpha']

    if 'lw' not in kwargs.keys():
        lw = 0

    if 's' not in kwargs.keys():
        s = 40
    if color is None:
        color = 'blue'

    #VISUALIZATION
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(data[:, 0],
                    data[:, 1],
                    lw=lw,
                    s=s,
                    alpha=alpha,
                    c=color)
    # plt.xlim()
    # plt.ylim()
    ax.axis('off')
    ax.axis('tight')
    ax.set_title(title)

    return fig, ax