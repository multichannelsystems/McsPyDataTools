"""
.. module:: detection
        :synopsis: Spike detection - extract time stamps and cutouts for every spike

.. moduleauthor:: Ole Jonas Wenzel <wenzel@multichannelsystems.com>
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Optional, Iterator, Any

import warnings


def median_spike_threshold(signal: np.ndarray, value_axis: int = 0) -> np.ndarray:
    """
    Calculates the threshold of all n channel of a n-trode to detect spikes according to the following formula:

    ..math::
        \sigma_n = \frac{median(|X|)}{0.6745}

    :param signal: data in a 2D numpy.array with the dimensions [#samples, #channels]
    :param value_axis: axis along which the signal values are organized
    :return: estimated spike threshold for all n channels of a n-trode
    """
    if value_axis not in [0, 1]:
        value_axis = 0
        warnings.warn('Parameter value_axis needs to be 0 or 1, setting default: value_axis={}.'.format(value_axis))

    return np.median(np.absolute(signal), axis=value_axis) / 0.6745


def std_spike_threshold(signal: np.array, value_axis: int = 1) -> np.array:
    """
    Calculates the threshold to detect spikes according to the following formula:

    ..math::
        \sigma_n = \sigma(|X|)

    :param signal: signal that is used to estimate the threshold
    :param value_axis: axis along which the signal values are organized
    :return: estimated spike threshold
    """
    if value_axis not in [0, 1]:
        value_axis = 1
        warnings.warn('Parameter value_axis needs to be 0 or 1, setting default: value_axis=0.')

    return np.std(np.absolute(signal), axis=value_axis)


def extract_spike_timestamps(
        signal: np.array,
        idxs_of_signal_above_threshold: Tuple[np.array, np.array],
        interspike_distance_in_samples: int,
        value_axis: int = 1) -> np.array:
    """
    Detect all time stamps that are associated with the peak of one putative spike

    :param signal: continous analog signal that contains spikes and noise in n channels
    :param signal_above_threshold_index: tuple of numpy arrays which contain the indices of the sample above the
                                         threshold. Each tuple entry corresponds to one dimension.
    :param interspike_distance_in_samples: minimal distance between two different spikes
    :param axis along which the signal values are organized
    :return: numpy array that contains the spike time stamps
    """

    start: int = 0
    spike_ts: List[List] = [[], []]

    idxs_of_signal_above_threshold = np.asarray(idxs_of_signal_above_threshold)

    for channelIdx in np.unique(idxs_of_signal_above_threshold[0]):
        idxs_of_signal_above_threshold_per_channel = (idxs_of_signal_above_threshold[0] == channelIdx).nonzero()
        diff = np.diff(idxs_of_signal_above_threshold[1, idxs_of_signal_above_threshold_per_channel])
        relevant_threshold_crossings_per_channel = np.asarray((diff >= interspike_distance_in_samples).nonzero())[1, :]
        spike_strips = idxs_of_signal_above_threshold[1, idx_relevant_threshold_crossings_per_channel]
        start = 0
        for signalIdx in spike_strips[1]:
            idx_span = idxs_of_signal_above_threshold_per_channel[start:signalIdx+1]
            max_idx = np.argmax(np.absolute(signal[channelIdx, index_span]))
            ts_index = idxs_of_signal_above_threshold[1][start + max_index]
            spike_ts[0].append(channelIdx)
            spike_ts[1].append(ts_index)
            start = signalIdx + 1



    # signal_above_threshold_index = np.asarray(idxs_of_signal_above_threshold)
    # #  compute index diff
    # diff = np.diff(idxs_of_signal_above_threshold)
    # #  retrieve all spikes that satisfy the minimum distance criterium
    # spike_strips = np.asarray(np.nonzero(diff >= interspike_distance_in_samples))
    # #  retrieve corresponding channel indices and store/match with spike strips
    # spike_strips[0] = idxs_of_signal_above_threshold[0][spike_strips[1]]
    #
    # for (channelIdx, signalIdx) in zip(*spike_strips):
    #     print(channelIdx, signalIdx)
    #     index_span = idxs_of_signal_above_threshold[channelIdx, start:(signalIdx + 1)]
    #     # signal = asignal.rescale('mV').magnitude[index_span]
    #     max_index = np.argmax(np.absolute(signal[channelIdx, index_span]))
    #     ts_index = idxs_of_signal_above_threshold[1][start + max_index]
    #     spike_ts[0].append(channelIdx)
    #     spike_ts[1].append(ts_index)
    #     start = signalIdx + 1
    return np.asanyarray(spike_ts)


def threshold_detection(
        signal: np.ndarray,
        sigma_multiplier: float = 5,
        sigma_estimator: Callable[[np.array], np.array] = median_spike_threshold) -> np.ndarray:
    """
    Detect which samples of the noisy spike signal have a magnitude (absolut value) above a threshold
    that is estimated from the signal by the help of the given estimator function

    :param signal: continous analog signal that contains spikes and noise
    :param sigma_multiplier: multiplier for the estimated sigma (default is 5)
    :param sigma_estimator: estimator function (default is the median estimator)
    :return: numpy array which contains the indices of the sample above the threshold
    """
    # times = asignal[0].rescale('ms').magnitude
    # signal = asignal.rescale('mV').magnitude
    sigma_estimate = sigma_estimator(signal)
    threshold = sigma_multiplier * sigma_estimate
    above_threshold = (np.absolute(signal) >= threshold[:, np.newaxis]).nonzero()
    return above_threshold


def detectSpikes(x: np.ndarray, fs: float, std_multiplier: float = 5):
    """
    Detects spikes in a signal.
    :param x: signal values for n channels [numSamples, numChannels]
    :param fs: sampling rate (in Hz)
    :param std_multiplier: fator to scale estimated standard deviation to a threshold value
    :return: s and t are column vectors of spike times in samples and ms, respectively. By convention the time of the
             zeroth sample is 0ms.
    """
    # PARAMETERS
    minimal_time_distance: float = 0.001  # minimal inter spike distance in seconds
    minimal_distance: int = int(minimal_time_distance*fs)  # minimal inter spike distance in samples

    # DETECT THRESHOLD CROSSINGS
    sigma: np.ndarray = median_spike_threshold(x, value_axis=0)
    thresholds = -1 * std_multiplier * sigma
    sub_threshold_mask = (x <= thresholds).astype(int)
    sup_crossings: Tuple = (np.diff(sub_threshold_mask, axis=0) > 0).nonzero()  # indices in format (array*rows*, array*columns*)
    sub_crossings: Tuple = (np.diff(sub_threshold_mask, axis=0) < 0).nonzero()

    # DETECT SPIKE PEAK TIMESTAMPS
    spike_timestamps = []
    spike_channels = []
    channels = set(sup_crossings[1]).intersection(set(sub_crossings[1]))
    for channel in channels:
        # assumes spikes are sorted by timestamp by nature of np.diff
        sup_idcs = [idx for idx, chnnl in zip(sup_crossings[0], sup_crossings[1]) if chnnl == channel]
        sub_idcs = [idx for idx, chnnl in zip(sub_crossings[0], sup_crossings[1]) if chnnl == channel]

        if sup_idcs[0] > sub_idcs[0]:
            # in case recording started in middle of spike
            del sup_idcs[0]
        min_len = np.minimum(len(sup_idcs), len(sub_idcs)) - 1
        sup_idcs, sub_idcs = sup_idcs[:min_len], sub_idcs[:min_len]

        for sup_idx, sub_idx in zip(sup_idcs, sub_idcs):
            spike_timestamps.append(
                sup_idx + np.argmin(x[sup_idx:sub_idx, channel]))  # print(channel, sup_idx, sub_idx)
            spike_channels.append(channel)

    spike_timestamps = np.array(spike_timestamps)
    spike_channels = np.array(spike_channels)

    # DELETE SPIKES THAT ARE LESS THAN minimal_distance FROM THEIR PREDECESSOR,
    # as they are, most likely, caused by the same spike on different channels
    distance_sufficient = np.diff(spike_timestamps) >= minimal_distance
    distance_sufficient = np.append(distance_sufficient, False)
    spikes = (spike_timestamps[distance_sufficient],
              spike_channels[distance_sufficient])

    s = np.unique(spikes[0]).astype(int)
    t = s / fs

    return s, t


def extractWaveforms(x: np.ndarray,
                     s: np.array,
                     fs: float,
                     window: Any = 0.001) -> np.ndarray:
    """
    extracts the waveforms at times s (given in samples) from the signal x using fixed window around the times of the
    spikes. The return value is a 3D array of dimensions [ length(window) x #spikes x #channels]
    :param x: signal values for n channels [numSamples, numChannels]
    :param s: iterable of timestamps (in samples) where the signal is to be cut
    :param fs: sampling rate (in Hz)
    :param window: length of window (in sec) or tuple (pre (in sec), post (in sec))
    :return:
    """

    if isinstance(window, tuple):
        pre, post = window
        pre = int(pre*fs)
        post = int(post*fs)
    elif isinstance(window, (int, float)):
        window = int(window*fs)
        pre = window // 2
        post = window // 2
    else:
        window = int(window * fs)
        pre = window // 2
        post = window // 2
        warnings.warn('using default values (pre, post) = ({}, {})'.format(pre, post))

    cutouts = []
    warning = False
    for timeStep in s:
        start = np.max([0, timeStep-pre])
        stop = np.min([timeStep+post, x.shape[0]])
        cutout = x[start:stop, :]
        if cutout.shape == (pre+post, x.shape[1]):
            cutouts.append(cutout)
        else:
            warning = True
            template = np.zeros((pre+post, x.shape[1]))
            if timeStep-pre < 0:  # then prepend zeros
                template[pre+post-cutout.shape[0]:]
            if timeStep+post > x.shape[0]: # then append zeros
                template[:cutout.shape[0]] = cutout
            cutouts.append(template)

    return np.stack(cutouts, axis=1)
