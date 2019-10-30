"""
.. module:: features
        :synopsis: Spike feature computation - computes features for every spike

.. moduleauthor:: Ole Jonas Wenzel <wenzel@multichannelsystems.com>
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

from typing import List, Tuple, Any

import warnings
import pywt

# GPU_ENV = False
# if GPU_ENV:
#     # GPU COMPUTING
#     import skcuda
#     import pycuda.gpuarray as gpuarray
#     import pycuda.driver as driver
#     import pycuda.autoinit
#     from pycuda.compiler import SourceModule


# CONSTANTS
EPS = np.finfo(float).eps


def extract_features(w: np.ndarray, method: str = 'pca', *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    extracts features for spike sorting from the waveforms in w

    :param w: a 3d array of dimensionality [length(window) x #spikes x #channels]
    :param method: method that is to be used for feature computation
    :param args: additional parameters that are passed on to method
    :param kwargs additional parameters that are passed on to kwargs
    :return: output is a matrix of dimensionality [#spikes x #features]
    """

    # if not hasattr(this_module, method.__name__):
    #     raise ValueError('The function provided is not valid feature computer.')

    # PARAMETERS
    num_features, num_samples, num_channels = w.shape

    # CHECK METHOD
    if method == 'pca':

        # CHECK
        # number of principle components that are to be extracted per channel
        if 'num_components_ch' in kwargs.keys():
            num_components = kwargs['num_components_ch']
        else:
            num_components = 3

        # ALLOCATION
        components = np.zeros((num_components * num_channels, num_features))  # [#components x #channels, #features]
        transformed_data = np.zeros((num_samples, num_components * num_channels))  # [#samples, #components x #channels]
        variance_explained = []

        # PERFORM PCA
        pcas: List[PCA] = []
        for channel in range(num_channels):
            w_ch = w[:, :, channel].squeeze().T  # [#samples, #features/#samples in window]
            w_std = StandardScaler().fit_transform(w_ch)  # standard normalize the data
            pcas.append(PCA(num_components=num_components).fit(w_std))

            start = channel * num_components
            stop = ((channel+1) * num_components) % (num_components * num_channels)
            if stop != 0:
                # transform and store channel data
                transformed_data[:, start:stop] = pcas[-1].transform(w_ch)
                # store principal components
                components[start:stop, :] = pcas[-1].components_
            else:
                # transform and store channel data
                transformed_data[:, start:] = pcas[-1].transform(w_ch)
                # store principal components
                components[start:, :] = pcas[-1].components_

            # store explained variance
            variance_explained.append(pcas[-1].variance_explained_ratio_)

    # elif method == 'wavelet':
    #     components = np.concatenate(pywt.dwt(w, 'haar', axis=0), axis=0)
    #     components = np.transpose(components, (0, 2, 1))
    #     components = np.reshape(components, (-1, components.shape[2]), order='C')
    #     components = np.transpose(components)
    #     variance_explained = None
    #     features = None

    else:
        raise ValueError('Provide a valid feature extraction option.')

    return transformed_data, variance_explained, components


class PCA:

    def __init__(self, num_components: int):
        """
        initializes the pca class
        :param num_components: defines the number of principle components used in the pca.
        """
        self._num_components = num_components
        self._total_variance: None
        self._eig_val = None
        self._eig_vec = None
        self._components_ = None
        self._total_variance = None

    def fit(self, w: np.ndarray) -> Any:
        """
        fits the data in w in the sense that it extracts the principle components
        :param w: 2D array holding data, dimensions [#datapoints, #features]
        :return: returns ___
        """

        # PARAMETERS & ALLOCATION

        # CHECKING
        if w.shape[1] < self._num_components:
            self._num_components = w.shape[1]
            warnings.warn('\'num_components\' is specified larger than the number of features in the data allows. Using #festure instead {}'.format(self.num_components))

        # DECOMPOSITION
        cov_mat = np.cov(w.T)
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        ind = np.argsort(np.abs(eig_val))
        ind = ind[::-1]
        eig_val, eig_vec = eig_val[ind], eig_vec[:, ind]

        # EXTRACT PRINCIPLE COMPONENTS & TRANSFORMED DATA
        self._eig_val = eig_val[:self._num_components]
        self._eig_vec = eig_vec[:, :self._num_components]

        self._components_ = self._eig_vec.T
        self._total_variance = np.var(w, axis=0).sum()

        return self

    def fit_transform(self, w):
        """
        extracts the principle components from w and transforms w to feature space
        :param w: 2D data array [#datapoints, #features]
        :return: w transformed into feature space
        """
        if (self._eig_vec is None) or (self._eig_val is None):
            self.fit(w)
        return self.transform(w)

    def transform(self, w):
        """
        transforms w into feature space given the principle components computed previously
        :param w: 2D data array [#datapoints, #features]
        :return: w transformed into feature space
        """
        if self._eig_vec is None:
            raise AttributeError('\'PCA\' object has no attribute \'transform_\', yet. Please, fit some data first.')
        return self._eig_vec.T.dot(w.T).T

    @property
    def num_components(self) -> int:
        """
        :return: number of principle components extracted in this particular PCA instance
        """
        return self._num_components

    @num_components.setter
    def num_components(self, value: int):
        """
        returns number of components extracted in this particular PCA instance
        :param value: new number of principal components
        :return:
        """
        self.num_components = value

    @property
    def features_(self) -> np.ndarray:
        """
        :return: extracted features
        """
        if self._eig_vec is None:
            raise AttributeError('\'PCA\' object has no attribute \'features_\'')
        return self._eig_vec

    @property
    def variance_explained_(self) -> np.ndarray:
        """
        :return: variance explained by the principle components
        """
        if self._components_ is None:
            raise AttributeError('\'PCA\' object has no attribute \'variance_explained_\'')
        return np.var(self.components_, axis=0)

    @property
    def variance_explained_ratio_(self) -> np.ndarray:
        """
        :return: ratio of variance explained by the principle components
        """
        if (self._total_variance is None) or (self._components_ is None):
            raise AttributeError('\'PCA\' object has no attribute \'variance_explained_ratio_\'')
        variance_explained = np.var(self.components_, axis=0)
        return variance_explained/self._total_variance

    @property
    def singular_values_(self) -> np.ndarray:
        """
        :return: eigenvalues extracted in PCA
        """
        if self._eig_val is None:
            raise AttributeError('\'PCA\' object has no attribute \'singular_values_\'')
        return self._eig_val

    @property
    def components_(self):
        """
        return: transformed data from initial fit
        """
        if self._components_ is None:
            raise AttributeError('\'PCA\' object has no attribute \'components_\'')
        return self._components_