"""
.. module:: cluster
        :synopsis: Spike clustering

.. moduleauthor:: Ole Jonas Wenzel <wenzel@multichannelsystems.com>
"""
# DATASCIENCE PACKAGES
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.special import gammaln
from scipy import sparse
import matplotlib.pyplot as plt

from typing import List

import warnings
from datetime import datetime

VERBOSE = True


def cluster(features: np.ndarray,
            timestamps=None,
            method: str = 'gmm', 
            verbose: bool = False, 
            *args, 
            **kwargs) -> np.ndarray:
    """
    clusters the 'features' according to the specified 'method'
    :param features: [#spikes, #features] hold the features with dimensionality
    :param timestamps: [#spikes] holds the timestamps for each detected spike
    :param method: method used for clustering, ['kmeans', 'gmm']
    :param verbose: run in verbose mode
    :return:
    """

    # CONSTANTS
    methods = ['gmeans', 'gmm', 'modt']
    # kmeans: KMeans algorithm
    # gmm   : Gaussian Mixture Model
    # modt  : Mixture of Drifitng t-distributions, for nu = Infinity Mixture of drifting Gaussian distributions

    # PARAMETERS
    if 'num_clusters' not in kwargs.keys():
        num_clusters = [5, 15]  # min and max # of clusters
        if 'verbose':
            warnings.warn('Using default number of clusters in interval [min #, max #]={}'.format(num_clusters))

    if method not in methods:
        raise ValueError('Method has to be one of the following specified methods: {}'.format(methods))

    if method == 'gmeans':
        pass

    if method == 'gmm':
        return gmm(features, num_clusters=np.arange(*num_clusters), *args, **kwargs)

    if method == 'modt':
        assert timestamps is not None, 'Provide timestamps for the spikes'

        return modt(features, timestamps, *args, **kwargs)


def gmeans():
    pass


class GMM:
    """

    Implements spike sorting using a Gaussian Mixture Model.
    Model fitting is done using an expectation maximization algorithm.
    Model selection is done according to Bayesian Information Criterion (BIC)

    """

    def __init__(self,  num_clusters: List[int] = list(range(1, 10)),
            num_runs: int = 5, *args, **kwargs):
        """

        Initializes the GMM model
        :param num_clusters: List(int), candidate number of clusters to fit
        :param num_runs: int, num models to fit per number of clusters
        """
        self._bic = None
        self._num_clusters_fitted = num_clusters
        self._num_runs = num_runs
        self._model = None
        self._init_args = args
        self._init_kwargs = kwargs
        self._assignment = None

        self._verbose: bool = False
        if 'verbose' in kwargs.keys():
            self._verbose = kwargs['verbose']

    def fit(self, X, *args):
        """

        fits a Gaussian Mixture Model (GMM) to the spiking data

        :param X: [N, D] spike data
        :return: self
        """

        # PARSE INPUT

        # PARAMETERS
        num_clusters = self._num_clusters_fitted
        num_runs = self._num_runs

        # ALGORITHM
        trials = np.arange(num_runs)
        bic = np.zeros((num_runs, len(num_clusters)))
        for idx, num in enumerate(num_clusters):
            print('Run {} GMM(s) with BIC on {} cluster(s).'.format(num_runs, num))
            for trial in trials:
                bic[trial][idx] = GaussianMixture(n_components=num, *self._init_args, **self._init_kwargs).fit(X).bic(X)
        self._bic = bic.min(axis=0)
        print('The GMM with {} clusters scores the lowest BIC.'.format(num_clusters[np.argmin(self._bic)]))
        if self._verbose:
            self.plot_bic()
        self._model = GaussianMixture(n_components=num_clusters[np.argmin(bic)]).fit(X)
        self._assignment = self.predict(X)

        return self

    def fit_predict(self, X):
        """

        fits a Gaussian Mixture Model (GMM) to the spiking data and predicts the outcome

        :param X: [N, D] spike data
        :param num_clusters: List(int), candidate number of clusters to fit
        :param num_runs: num models to fit per number of clusters
        :return: self
        """
        instance_of_self = self.fit(X)
        return instance_of_self.predict(X)

    def predict(self, X):
        if self._model is None:
            print('Fitting model on the provided data with default values.')
            return self.fit_predict(X)
        self._assignment = self._model.predict(X)
        return self._assignment

    def plot_bic(self):
        """

        Plots the resulting BIC value over the number of clusters to which the GMM was fitted.

        :return: handle to plot's matplotlib figure
        """

        fig = plt.figure(figsize=(5, 4))

        if self._num_clusters_fitted is not None and self.bic is not None:
            print('BIC not yet available. Fit the model first.')
            return fig

        plt.plot(self._num_clusters_fitted, np.min(self.bic, axis=0), 's')
        plt.xlabel('Number of clusters')
        plt.ylabel('BIC')
        plt.xticks(num_clusters)
        plt.xlim((1, num_clusters[-1] + 1))
        plt.show()

        return fig

    @property
    def bic(self):
        """
        :return: average bic value for each fit
        """
        if self._bic is None:
            print('BIC not yet available. Fit the model first')
            return
        return self._bic

    @property
    def num_clusters(self):
        """
        :return: List[int] of number of clusters that are fitted
        """
        return self._bic

    @property
    def model(self):
        if self._model is None:
            return GaussianMixture(n_components=3, *self._init_args, **self._init_kwargs)

        return self._model

    @property
    def assignment(self):
        if self._assignment is None:
            print('Fit the model before requesting results.')
        return self._assignment


class MoDT:
    """
    implements mixture of drifting t-distribution model for clustering spikes and measuring unit isolation

    The model and first implementation are developed by Alexander Ecker and Kevin Shan in MATLAB and are available
    on github:
        - https://github.com/aecker/moksm/blob/master/MoKsm.m
        - https://github.com/kqshan/MoDT

    This python implementation was ported and modified by Ole Jonas Wenzel.
    """

    def __init__(self,
                 nu: int = 3,
                 Q: np.ndarray = np.array([0.033]),
                 C_reg: np.ndarray = np.zeros((1)),
                 max_cond: float = 1e6,
                 use_gpu: bool = False,
                 *args,
                 **kwargs):
        """
        initializes the mixture of t-distribution instance

        :param args:
        :param kwargs:
        """

        # PROTECTED ATTRIBUTES
        self._D = None  # [1] number of feature space dimensions
        self._K = 1  # [1] number of cluster in the mixture
        self._T = None  # [1] number of time frames in the model
        self._N = None  # [1] number of spikes in training set
        self._spike_frame_id = None  # [N, 1] time frame ID (1...T) for each spike
        self._frame_spike_lim = None  # [T, 2] first and last spike id (1..N) for each frame
        self._tic = None  # cache to store time elapsed since start of fitting

        # FITTED ATTRIBUTES
        self._alphas = None  # [K, 1] mixing proportions, i.e. relative cluster sizes
        if 'alphas' in kwargs.keys():
            self._alphas = kwargs['alphas']
        self._mus = None  # [D, T, K] drifting cluster locations in feature space
        if 'mus' in kwargs.keys():
            self._mus = kwargs['mus']
        self._Cs = None  # [D, D, K] cluster scale matrices
        if 'Cs' in kwargs.keys():
            self._Cs = kwargs['Cs']

        # USER-DEFINED MODEL ATTRIBUTES
        self._mu_ts = None  # [T+1, 1] time frame boundaries for drifting cluster locations. Time frame t is defined
        # as half-closed interval  [mu_ts[t], mu_ts[t+1])
        self._nu: int = nu  # [1] degrees-of-freedom parameter for t-distribution. This may range from 1 to inf,
        # ith smalles values corresponding to heavier tails. nu=1 correponds to Cauchy distribution, and nu=Inf
        # corresponds to a Gaussian distribution
        self._Q = Q  # [D, D] cluster drift regularization matrix. If given as a scalar, it is interpreted
        # as a diagonal matrix with the given scalar along the diagonal. Smaller values correspond to more
        # regularization (producing an estimated mu that is smoother over time). The units of this quantity are
        # [(feature space units)^2/(time frame)]
        self._C_reg = C_reg  # [1] addded to diagonal of the scale matrix C during MStep. Setting C_reg>0 can help
        # ensure that C is well-condiioned, but it also means that the MStep is no longer  maximizing the expected
        # likelihood. Hence you may encounter a case where running self.EM() actually reduces the overall log-likelihood

        # length of one frame in seconds. Within one frame, the algorithm fits frame-specific values for
        # alpha, mu, and C.
        self._frame_dur = 60
        if 'frame_duration' in kwargs.keys():
            self._frame_dur = kwargs.pop('frame_duration')

        # USER-DEFINTED ALGORITHM ATTRIBUTES
        self._min_iter: int = 1  # minimum number of EM loops
        self._max_iter: int = 10  # maximum number of EM loops
        self._tol: float = 1e-4  # [1] EM is consideren converged when the absolute change in the overall
        # (data+prior) log-likelihood, divided by the number of spikes (self._spk_w_train.sum()), falls below this
        # threshold
        self._max_num_units = 25  # maximum number of units to fit

        # TRAINING DATA
        self._spk_x_train = None  # [D, N] train spike data in feature space. If use_gpu==true, this will be a
        # gpuArray
        self._spk_x_test = None  # [D, N] test spike data in feature space. If use_gpu==true, this will be a gpuArray
        self._spk_w_train = None  # [N, 1] spike weights, or scalar for uniform weighting. A single spike with
        # spk_w=2 is equivalent to two spikes at that location. If use_gpu==true, this will be a gpuArray
        self._spk_w_test = None  # [N, 1] spike weights, or scalar for uniform weighting. A single spike with
        # spk_w=2 is equivalent to two spikes at that location. If use_gpu==true, this will be a gpuArray
        self._spk_t_train = None  # [N, 1] spike times in s
        self._spk_t_test = None  # [N, 1] spike times in s

        # CACHED VALUES
        self._mahal_dist = None  # [N, K] squared Mahalanobis distances for each cluster x spike. If
        # use_gpu == True, this will be a gpu Array
        self._posterior = None  # [N, K] posterior likelihood, i.e. the probability that spike n belongs to
        # cluster k, given the observation y_n and the current model parameters. Columns of this matrix sum to one. If
        # use_gpu==True, this will be a gpuArray
        self._data_loglike: List = []  # [1] data log-likelihood, i.e. log P(spl_Y|alpha, mu, C)
        self._prior_loglike: List = []  # [1] parameter prior log-likelihood, i.e. log P(alpha, mu, C). In this model,
        # this only depends on mu, and isn't a proper probability because it doesn't integrate to one.

        # DEVELOPMENT ATTRIBUTES
        self._VERBOSE = False
        self._seed = 123

        # OTHER ATTRIBUTES
        self._max_cond = max_cond  # [1] maximum allowable condition number for the scale matrix C. If you set
        # parameters with an ill conditioned C matrix, it will throw an error. If MStep produces an ill conditioned C
        # matrix, it will add max(svd(C))/max_cond to the diagonal of C, which should reduce cond(C) below max_cond
        self._cluster_cost = 2.5e-3  # Cost for adding additional cluster
        self._use_gpu = use_gpu  # (boolean) use GRPU acceleration. If enabled, this will store the training data
        # spk_x and spk_w) and various intermediate vaues as gpuArray objects

        self.set_params(**kwargs)

    def set_params(self, **params):
        """
        sets model parameters
        :param params: dict of parameters
        :return:
        """
        # FLAGS
        update_mahal_dist = False
        update_posterior = False

        # SET SPIKE FEATURES
        if 'X_train' in params:
            self._spk_x_train = params['X_train']
            self._N, self._D = self._spk_x_train.shape
            init_model_params = True
        if 'X_test' in params:
            x_test_dim = params['X_test'].shape[1]
            assert self._D == x_test_dim, 'test and training spikes are not of the same feature dimensionality: ' \
                                          'expected: {} , got {}'.format(self._D, x_test_dim)
            self._spk_x_test = params['X_test']

        # SET SPIKE TIMES
        if 't_train' in params:
            num_timestamps = params['t_train'].shape[0]
            assert self._N == num_timestamps, 'train spike waveforms and timestamps do not match in number: ' \
                                              'expected: {} , got {}'.format(self._N, num_timestamps)
            self._spk_t_train = params['t_train']
            init_model_params = True
        if 't_test' in params:
            num_timestamps = params['t_test'].shape[0]
            num_test_spikes = self._spk_x_test.shape[0]
            assert num_test_spikes == num_timestamps, 'test spike waveforms and timestamps do not match in number: ' \
                                                      'expected: {} , got {}'.format(num_test_spikes, num_timestamps)
            self._spk_t_test = params['t_test']

        # SET SPIKE WEIGHTS
        if 'spike_weights_train' in params:
            self._spk_w_train = params['spike_weights_train']
            init_model_params = True
        if 'spike_weights_test' in params:
            self._spk_w_test = params['spike_weights_test']

        # SET TIME INTERVALS FOR MEAN COMPUTATION
        if 'time_bins' in params:
            self._mu_ts = params['time_bins']

        if 'K' in params:
            self._K = params['K']

        if 'nu' in params:
            self._nu = params['nu']

        # SET MODEL PARAMETERS
        if 'alphas' in params:
            alphas = params['alphas']
            if self._VERBOSE and self._alphas.shape != params['alphas'].shape:
                warnings.warn('the number of alphas provided does not match the shape of alphas cached and is going to '
                              'be updated: old {}, new {}'.format(self._alphas.shape, alphas.shape))
            self._alphas = alphas
            self._K = alphas.size
            update_mahal_dist = True
            update_posterior = True

        if 'mus' in params:
            mus = params['mus']
            D, T, K = mus.shape
            assert D == self._D, 'dimension-axis of mus provided does not match dimension-axis of the model: ' \
                                 'got {}, expected {}'.format(D, self._D)
            assert T == self._T, 'time-axis of mus provided does not match time-axis of the model: ' \
                                 'got {}, expected {}'.format(T, self._T)
            assert K == self._K, 'cluster-axis of mus provided does not match cluster-axis of the model: ' \
                                 'got {}, expected {}'.format(K, self._K)
            self._mus = mus
            update_mahal_dist = True
            update_posterior = True

        if 'Cs' in params:
            Cs = params['Cs']
            _, D, K = Cs.shape
            assert D == self._D, 'dimension-axis of Cs provided does not match dimension-axis of the model: ' \
                                 'got {}, expected {}'.format(D, self._D)
            assert K == self._K, 'cluster-axis of Cs provided does not match cluster-axis of the model: ' \
                                 'got {}, expected {}'.format(K, self._K)
            self._Cs = Cs
            update_mahal_dist = True
            update_posterior = True

        if 'frame_duration' in params:
            self._discretize_time(params['frame_duration'])
            update_mahal_dist = True

        if 'mahalanobis_distance' in params:
            self._mahal_dist = params['mahalanobis_distance']
            update_mahal_dist = False

        # SET ALGORITHM PARAMETERS
        if 'max_iter' in params:
            self._max_iter = params['max_iter']
        if 'verbose' in params:
            self._VERBOSE = params['verbose']

        # UPDATE FLAGGED VALUES
        if update_mahal_dist:
            self._mahal_dist = self._compute_mahalanobis_distance()
        if update_posterior:
            Z, U = self._expectation()

    def _discretize_time(self, frame_dur=None):
        """
        discretizes time into intervals of length frams_dur [s]
        :param frame_dur: length of time interval [s]
        :return:
        """
        # PARSE INPUT
        if frame_dur is None:  # case: no frame duration provided
            frame_dur = self._frame_dur

        if frame_dur is None:  # case: no frame duration stored in model -> no information on frame duration available
            frame_dur = 60
            self._frame_dur = frame_dur

        if self._VERBOSE:
            print('Discretizing time into intervals of length {}.'.format(frame_dur))

        assert (self._spk_t_train is not None and self._spk_t_test is not None), 'Provide spike-timestamp data before' \
                                                                                 'discretizing time.'

        t_0 = np.min([self._spk_t_train[0], self._spk_t_test[0]])
        t_end = np.max([self._spk_t_train[-1], self._spk_t_test[-1]])
        self._mu_ts = np.arange(t_0, t_end + frame_dur, frame_dur)  # including upper and lower edge
        num_edges = self._mu_ts.size  # compute #edges
        T = num_edges - 1  # compute #intevalls

        if self._T is None:
            self._T = T

        if self._T != T:
            warnings.warn('Discretization of time failed because provided frame duration does not match the '
                          'time axis of the model parameters: computed: '
                          '{}, time-axis of model parameters of length: {}. Continue using {}'.format(T, self._T,
                                                                                                      self._T))
            self._mus = np.linspace(t_0, t_end, self._T)

        # TODO: Make more flexible for arbitrary input tlim and frame_dur

        # UPDATE SPIKE 2 FRAME ASSIGNMENT
        self._determine_spike_frame_lim()

    def _determine_spike_frame_id(self, spike_train=None):
        """

        :param spike_train: spike train in seconds, default: take stored spike train

        :return: spike train ids, i.e. the id of the time interval in which the spikes occured
        """
        # PARAMETERS
        cache = False

        # PARSE INPUT
        if spike_train is None:
            spike_train = self._spk_t_train
            assert spike_train is not None, 'Provide the model with a train of spikes, before computing ' \
                                            'spike-frame-ids.'
            cache = True

        if self._mu_ts is None:
            self._discretize_time()
            if self._VERBOSE:
                print('Initializing time discretization to default: intervals of length 60s')

        # compute spike frame id
        spike_frame_id = np.digitize(spike_train,
                                     self._mu_ts,
                                     right=False)  # leftest spike included, but rightest spike in extra bin

        # correct results of np.digitize
        num_edges = self._mu_ts.size  # compute #edges
        spike_frame_id[spike_frame_id == num_edges] = num_edges - 1
        spike_frame_id -= 1  # -1 -> from one-based to zero-based indexing

        # RETURN AND CACHE
        if cache:
            self._spike_frame_id = spike_frame_id

        return spike_frame_id

    def _determine_spike_frame_lim(self, X=None, spike_train=None):
        """

        determine the index of the first and last spikes per interval

        :return: [T, 2] each row i holds the indices of the first and last spike in time interval i
        """
        # PARAMETERS
        cache = False

        # PARSE INPUT
        if spike_train is None:
            cache = True
        spike_frame_id = self._determine_spike_frame_id(spike_train=spike_train)

        # ids of first spikes in intervals where the spike count is non-zero
        firsts_exist = np.diff(spike_frame_id, prepend=-1).nonzero()[0]
        firsts = np.full(self._T, np.nan)
        firsts[np.unique(spike_frame_id)] = firsts_exist
        lasts = np.hstack((firsts[1:] - 1, [spike_frame_id.size - 1]))

        # fill nan values
        firsts = self._ffill(firsts[np.newaxis, :]).astype(int).flatten()
        lasts = self._bfill(lasts[np.newaxis, :]).astype(int).flatten()

        # stack first and last indices
        _frame_spike_lim = np.vstack((firsts, lasts)).T

        # CACHE AND RETURN RESULTS
        if cache:
            self._frame_spike_lim = _frame_spike_lim

        return _frame_spike_lim

    def _init_model_params(self):
        """
        initializes the model parameters:
        """

        # PARAMETERS

        print('Initializing Model Parameters...')

        # DISCRETIZE TIME
        self._discretize_time()

        # INITIALIZE MIXING PROPORTIONS
        alphas = self._alphas
        if alphas is None:
            alphas = np.ones(self._K) / self._K
        if alphas.size != self._K:
            warnings.warn('The alphas provided insinuate a change in #clusters in the model: expected: '
                          '{}, got: {}. update # of clusters to {}.'.format(self._K, alphas.size, alphas.size))
            self._K = alphas.size

        self._alphas = alphas

        # INITIALIZE CLUSTER LOCATION
        mus = self._mus
        if mus is not None:  # check model integrity
            D = mus.shape[0]
            if self._D is None:
                self._D = D
            if D != self._D:
                warnings.warn('The dimensionality of the mus provided does not match the dimensionality of the model '
                              'expected: {}, got: {}. update # of dimensions to {}.'.format(self._D, D,
                                                                                            D) + 'Fresh initialization of Cs if shape does not match.')
                self._D = D

            T = mus.shape[1]
            if self._T is None:
                self._T = T
            if T != self._T:
                raise ValueError('The # of intervals of the mus provided does not match the model: got '
                                 '{}, expected: {}.'.format(T, self._T))

        else:  # fresh initialization of mus
            mus = np.zeros((self._D, self._T, self._K))

            if self._K == 1:
                cluster_centers = np.mean(self._spk_x_train, axis=0)[np.newaxis, :]
            else:
                cluster_centers = KMeans(n_clusters=self._K,
                                         n_init=5,
                                         random_state=self._seed).fit(self._spk_x_train).cluster_centers_
            # cluster_centers = np.zeros((self._K, self._D))
            # for d in range(self._D):
            #     min_v = np.min(self._spk_x_train[:, d])
            #     max_v = np.max(self._spk_x_train[:, d])
            #     cluster_centers[:, d] = np.random.uniform(low=min_v, high=max_v, size=self._K)
            mus[:, :, :] = np.repeat(np.transpose(cluster_centers)[:, np.newaxis, :], self._T, axis=1)

        self._mus = mus

        # INITIALIZE CLUSTER VARIANCES
        Cs = self._Cs
        if Cs is None or self._D != Cs.shape[0]:
            C = np.cov(self._spk_x_train.T)
            eig_vals = np.linalg.eig(C)[0]
            if (eig_vals < 0).any:
                offset = np.abs(np.min(eig_vals)) + np.finfo(np.float32).eps
                C += (np.eye(C.shape[0]) * offset)
            Cs = np.repeat(C[:, :, np.newaxis], self._K, axis=2)

        self._Cs = Cs

        if self._VERBOSE:
            print('Model parameters successfully initialized!')

    def _split_in_train_test(self, X, t, spike_weights):
        """

        split data into training and test data, while mainting the order in time

        :param X: spike features
        :param t: spike-timestamps
        :param spike_weights: weights for each spike
        :return: X_train, X_test, t_train, t_test, w_train, w_test
        """

        if spike_weights.size == 1:
            X_train, X_test, t_train, t_test = train_test_split(X,
                                                                t,
                                                                test_size=0.1,
                                                                shuffle=True,
                                                                random_state=self._seed)
            w_train = spike_weights
            w_test = spike_weights
        else:
            X_train, X_test, t_train, t_test, w_train, w_test = train_test_split(X,
                                                                                 t,
                                                                                 spike_weights,
                                                                                 test_size=0.1,
                                                                                 shuffle=True,
                                                                                 random_state=self._seed)
        # sort training data
        train_order = np.argsort(t_train)
        X_train = X_train[train_order]
        t_train = t_train[train_order]
        if w_train.size != 1:
            w_train = w_train[train_order]

        # sort test data
        train_order = np.argsort(t_train)
        X_train = X_train[train_order]
        t_train = t_train[train_order]
        if w_train.size != 1:
            w_train = w_train[train_order]

        return X_train, X_test, t_train, t_test, w_train, w_test

    def _expectation(self):
        """
        computes the latent variables z,u from given mu, S, nu
        z   holds expected value of Z, i.e. the posterior cluster likelihood
        u   holds expected value of U, i.e. the t-distribution scaling variable
            will be assigned u=1 if nu is inf
        :return: (Z, U) latent variables of dimensionality ([N, K], [N, K])
        """
        # PARAMETERS
        D = self._D  # number of features/dimensions
        K = self._K  # number of clusters
        N = self._N  # number of spikes
        nu = self._nu

        # CALCULATE MAHALANOBIS DISTANCE
        if self._mahal_dist is None:
            # need to compute Mahalanobis distance
            delta = self._compute_mahalanobis_distance()

        else:
            delta = self._mahal_dist

        # CALCULATE Z
        Z = self._compute_posterior()

        weighted_data_loglike = self._compute_data_loglike()

        # CALCULATE U
        if nu == np.Infinity:
            U = 1
        else:
            U = (nu + D) / (nu + delta)

        # # Compute weighted data likelihood
        # # TODO: whether weighted_data_loglike is supposed to be vetor or matrix
        # if self._spk_w_train.size == 1:
        #     weighted_data_loglike = np.sum(unweighted_data_loglike) * self._spk_w_train
        # else:
        #     weighted_data_loglike = unweighted_data_loglike.T@self._spk_w_train
        #
        # if self._use_gpu:
        #     weighted_data_loglike = weighted_data_loglike.get()

        # UPDATE INSTANCE
        self._mahal_dist = delta
        if Z.ndim == 1:
            self._posterior = Z[:, np.newaxis]
        else:
            self._posterior = Z
        self._data_loglike.append(weighted_data_loglike)
        self._prior_loglike.append(self._compute_prior_loglike())

        return Z, U

    def _maximization(self, Z, U):
        """
        Perform the maximmization step on the lateny variables Z and U.
        This method updates the fitted model parameters alpha, mu, C. It also clears cached
        intermediate results and updates the cached Mahalanobis distance.
        :param Z: [N, K] expected value of Z, i.e. the posterior cluster likelihoods
        :param U: [N, K] expected value of U, i.e. the t-distribution scaling variable
        """

        # PARAMETERS
        K = self._K
        N = self._N

        # SOME PRECOMPUTATION
        if self._spk_w_train.size == 1:
            wzu = self._spk_w_train * Z
            sum_w = self._spk_w_train * N
        else:
            wzu = Z * self._spk_w_train
            sum_w = np.sum(self._spk_w_train)

        sum_wz = np.sum(wzu, axis=0)  # [K, 1] sum of w*z for each cluster
        wzu *= U

        if self._use_gpu:
            sum_wz = sum_wz.get()
            sum_w = sum_w.get()

        # UPDATE ALPHA
        alphas = sum_wz / sum_w
        alphas /= np.sum(alphas)  # just to ensure alpha sums to one

        # UPDATE MU
        mus = self._optimize_mus(wzu, self._Cs)

        # UPDATE C & COMPUTE MAHALANOBIS DISTANCES
        Cs, delta = self._optimize_Cs(wzu, mus, sum_wz)

        # GET FROM GPU
        if self._use_gpu:
            mus = mus.get()

        # RETURN OPTIMIZED PARAMETERS
        return alphas, mus, Cs, delta

    def _expectation_maximization(self, max_iter):
        """

        runs EM iterations until convergence, but at most max_iter

        :param max_iter: maximum number of EM iterations
        :return:
        """

        # PARAMETERS
        starve_thresh = 2 * self._D  # Default
        override_Z = None  # Default
        sparse_thresh = 0  # Default

        # SPECIAL CONSIDERATION FOR GPU COMPUTATION
        if self._use_gpu:
            # sparse operations are not supported for gpuArrays
            assert sparse_thresh <= 0, 'Sparse matrix operations currently not supported for GPU computation.'
            # if Z is overridden, make sure it is a gpuArray
            if override_Z is not None:
                override_Z = gpuarray.to_gpu(override_Z)

        # INITIAL E-STEP: get base log-likelihood
        Z, U = self._expectation()
        old_loglike = self._data_loglike[-1] + self._prior_loglike[-1]
        # normalize log-likelihood with number of spikes

        # REPORT
        self._tic = datetime.now()
        if self._VERBOSE:
            time_elapsed = datetime.now() - self._tic
            cell_width = 20
            out = 'Iter'.center(cell_width) \
                  + '|' \
                  + 'Time (H:M:S)'.center(cell_width) \
                  + '|' \
                  + 'Elapsed (H:M:S)'.center(cell_width) \
                  + '|' \
                  + 'Log-like/spike'.center(cell_width) \
                  + '|' \
                  + 'Improvement/spike'.center(cell_width) + '\n'
            out += '-' * cell_width + '+' + '-' * cell_width + '+' + '-' * cell_width + '+' + '-' * cell_width + '+' + '-' * cell_width + '\n'
            out += '{}'.format(0).center(cell_width) + '|' \
                   + '{}'.format(datetime.now().strftime('%H:%M:%S')).center(cell_width) + '|' \
                   + '{}'.format(time_elapsed).center(cell_width) + '|' \
                   + '{:0.4f}'.format(old_loglike[0] / self._N).center(cell_width) + '|' \
                   + '{}'.format(np.Infinity).center(cell_width) + '\n'
            print(out)

        # RUN EM
        cur_iter = 0
        improvement = np.Infinity
        num_spikes = self._spk_t_train.size
        while (cur_iter <= self._min_iter
               or (cur_iter < max_iter and np.abs(improvement) / self._N > self._tol)):
            if self._use_gpu:
                num_clusters = skcuda.misc.sum(Z, axis=0).transpose()
                num_clusters = num_spikes.get()
            else:
                num_clusters = np.sum(Z, axis=0).T
            is_starved = (num_clusters < starve_thresh)
            if is_starved.any() and override_Z is not None:
                if self._VERBOSE:
                    pass  # print cur_iter, time, idx of staved cluster, number of spikes in starved cluster
                if starveAction == 'error':
                    pass
                elif starveAction == 'stop':
                    print('Stopped due to cluster starvation')
                    break
                else:  # == remove
                    # remove the starved cluster
                    self._remove(is_starved)
                    # run a new expectation step
                    Z, U = self._expectation()
                    old_loglike = self._data_loglike[-1] + self._prior_loglike[-1]

            # sparsify Z if desired
            if sparse_thresh > 0 and override_Z is not None:
                Z = self.sparsifyZ(Z, sparse_thresh)

            # Override Z if desired
            if override_Z is not None:
                Z = override_Z

            # MStep
            alphas, mus, Cs, delta = self._maximization(Z, U)

            # Cache values
            self.set_params(**{'alphas': alphas,
                               'mus': mus,
                               'Cs': Cs,
                               'mahalanobis_distance': delta})

            # EStep
            Z, U = self._expectation()
            loglike = self._data_loglike[-1] + self._prior_loglike[-1]
            if self._VERBOSE:
                pass  # print curIter, loglike, old_loglike

            # Check for convergence
            improvement = loglike - old_loglike

            # Iterate again
            old_loglike = loglike
            cur_iter += 1

            if self._VERBOSE:
                time_elapsed = datetime.now() - self._tic
                hours = time_elapsed.seconds // 3600
                minutes = (time_elapsed.seconds // 60) % 60
                seconds = time_elapsed.seconds % 60
                cell_width = 20
                out = '{}'.format(cur_iter).center(cell_width) + '|' \
                      + '{}'.format(datetime.now().strftime('%H:%M:%S')).center(cell_width) + '|' \
                      + '{}:{}:{}'.format(hours, minutes, seconds).center(cell_width) + '|' \
                      + '{:0.4f}'.format(loglike[0] / self._N).center(cell_width) + '|' \
                      + '{:0.4f}'.format(improvement[0] / self._N).center(cell_width) + '\n'
                print(out)
                # test = self.predict(self._spk_x_train)

        if self._VERBOSE:
            if np.abs(improvement / num_spikes) < self._tol:
                print('Converged')
            elif cur_iter <= self._max_iter:
                print('stopped due to iteration limit')
            else:
                print('Stopped due to unknown cause')

    def fit(self,
            X: np.ndarray,
            t: np.ndarray,
            spike_weights: np.ndarray = np.array([1]),
            *args,
            **kwargs):
        """
        fits a MoDT to the data
        :param X: [N, D] spike data
        :param t: [N, 1] spike times
        :param spike_weights: [1] or [N, 1] spike weights
        :param frame_dur: frame duration for auto defining mu_t in s
        :return:
        """
        # PARAMETERS

        # PARSE INPUTS
        X_train, X_test, t_train, t_test, w_train, w_test = self._split_in_train_test(X, t, spike_weights)

        # SET MODEL DATA AND INITIALIZE MODEL PARAMS
        model_data = {'X_train': X_train,
                      'X_test': X_test,
                      't_train': t_train,
                      't_test': t_test,
                      'spike_weights_train': w_train,
                      'spike_weights_test': w_test}

        self.set_params(**model_data)

        self._init_model_params()

        # RUN EM WITH INITIAL NUM OF CLUSTERS
        print('Starting fit ...')
        self._expectation_maximization(self._max_iter)

        SPLIT_N_MERGE = True
        num_extra_clusters = 2
        if SPLIT_N_MERGE:
            merge_failed = True
            while True:
                self, split_failed = self._try_split()
                if split_failed:
                    self, merge_failed = self._try_merge()
                if merge_failed and split_failed:

                    if self._VERBOSE:
                        print('Splitting for {} additional clusters.'.format(num_extra_clusters))

                    for i in range(num_extra_clusters):

                        split_cands = self._get_split_candidates()

                        if split_cands.size == 0:  # no split candidates found
                            break

                        if self._VERBOSE:
                            print('splitting cluster {}'.format(split_cands[0]))

                        split_failed = self.split(split_cands[0], max_iter=self._max_iter, run_em=True)

                        if split_failed:
                            break
                    break

        print('... fit completed.')

        return self

    def predict(self, X, t):
        posterior = self._compute_posterior(X, t)
        return np.argmax(posterior, axis=1)

    def _try_split(self):
        """

        Try splitting clusters.
        Split is accepted if penalized average likelihood improved.

        :return: exitcode( 1 == no split done, 0 == succesfully split a cluster )
        """
        # PARAMETERS
        exitcode = 0

        split_cands = self._get_split_candidates()
        loglike_test = self._compute_penalized_loglike(X=self._spk_x_test, t=self._spk_t_test)
        bic_test = self._compute_bic(X=self._spk_x_test, t=self._spk_t_test)

        if split_cands.size == 0:  # no split candidates found
            exitcode = 1
            return self, exitcode

        for i, cand in enumerate(split_cands):
            if self._VERBOSE:
                print('trying to split cluster {}'.format(cand))

            new_self: MoDT = copy.deepcopy(self)

            split_successfull = new_self.split(cand, max_iter=self._max_iter, run_em=True)
            new_loglike_test = new_self._compute_penalized_loglike(X=self._spk_x_test, t=self._spk_t_test)
            new_bic_test = new_self._compute_bic(X=self._spk_x_test, t=self._spk_t_test)
            if new_bic_test <= bic_test:  # because we want to slightly overestimate the number of clusters
                # if new_loglike_test > loglike_test:
                if self._VERBOSE:
                    print('success! (likelihood changed by {:0.4f} per spike)'.format(
                        (new_loglike_test - loglike_test)[0] / self._N))
                return new_self, exitcode
            if self._VERBOSE:
                print('aborted! (likelihood changed by {:0.4f} per spike)'.format(
                    (new_loglike_test - loglike_test)[0] / self._N))

        exitcode = 1
        return self, exitcode

    def split(self,
              clustID: int,
              starve_thresh=None,
              S: int = 2,
              split_init: str = 'kmeans',
              max_iter: int = 10,
              run_em: bool = False):
        """
        This method splits a cluster by fitting a MoDT model with S clusters to the
        subset of data assigned to the selected cluster. The splitInit parameter
        controls how this model is initialized:
            'kmeans' - The data residual (Y-mu) is whitened and clustered using k-means
                with (k=S). The initial M-step uses these cluster assignments and U=1.
            [N, S] matrix of posteriors or [N, 1] vector of cluster IDs (1..S), to use
                as cluster assignments for the initial M-step. This also specifies which
                spikes to include in the data subset: any spikes assigned to cluster 0 or
                have np.sum(posterior, axis=1)==0 will not be included in the data subset.
        After initialization, EM iterations are run on the subset model, only, then the
        cluster parameters (alpha, mu, C) are substituted back into the full model. The
        other clusters of the fill model are not affected.

        :param clustID: [1] cluster index (1..K) to split
        :param S : [1] number of clusters to split his into
        :param split_init: initialization of the split
        :param max_iter: [1] maximum # of EM iterations on split model
        :param starve_thresh: [1] minimum # of spikes per cluster
        :return: exitcode (0 == successful split, 1 == no split)
        """
        # PARAMETERS
        D = self._D
        K = self._K
        N = self._N
        splitted_cluster_centers = None
        exitcode = 0

        # 1. DEAL WITH INPUTS
        # TODO: deal with inputs to split function
        if starve_thresh is None:
            starve_thresh = 2 * self._D
        min_spk_per_cluster = starve_thresh
        if isinstance(clustID, float):
            clustID = np.array([int(clustID)])
        if isinstance(clustID, int):
            clustID = np.array([clustID])

        # 2. INITIALIZE (data_mask and Z)
        # check what we got as our split_init
        if isinstance(split_init, str):
            # string specifying an initialization method
            if split_init == 'kmeans':
                # get the data points associated with this cluster
                post = self._posterior
                data_mask = (post[:, clustID] > 0.2).squeeze()
                # get the whitened residual
                L = np.linalg.cholesky(self._Cs[:, :, clustID].squeeze())
                frames = self._spike_frame_id[data_mask]
                X = np.linalg.inv(L) @ (self._spk_x_train[data_mask, :].T - self._mus[:, frames, clustID])
                # Run k-means until we get a valid initialization
                is_valid_init = False
                n_tries = 0
                while not is_valid_init and n_tries <= 10:
                    n_tries += 1
                    # run kmeans
                    kmeans = KMeans(n_clusters=S, random_state=self._seed).fit(X.T)
                    split_assign = kmeans.labels_
                    splitted_cluster_centers = kmeans.cluster_centers_
                    # see if the initialization is valid
                    spk_count = np.bincount(split_assign, minlength=S)  # , np.array([S, 1]))
                    is_valid_init = (spk_count >= starve_thresh).all()
                # convert the assignment to posterior likelihoods
                N_sub = split_assign.size
                # TODO: check whether np.arange(0, N_sub).T or np.arange(1, N_sub+1).T
                Z = sparse.csr_matrix((np.ones_like(split_assign), (np.arange(0, N_sub), split_assign)),
                                      shape=(N_sub, S))
            else:
                raise NotImplementedError('Unknown split_init method {}'.format(split_init))
        elif split_init.size == N:
            # [N, 1] vector of cluster IDs (1..S)
            split_assign = split_init
            # Well, actually some can be 0 to indicate not to use this spike
            data_mask = (split_init > 0)
            split_assign = split_assign(data_mask)
            # Convert to posterior likelihoods
            S = np.maximum(split_assign)
            N_sub = split_assign.size
            Z = sparse.csr_matrix(np.arange(0, N_sub).T, split_assign, 1, N_sub, S)
            # No sparse gpuArray support yet
            if self.use_gpu:
                Z = Z.todense()

        elif split_init.shape[1] > 1 and split_init.shape[0] == N:
            # [N, S] matrix of posterior likelihoods
            Z = split_init
            # Some columns can be 0 to indicate not to use the spike
            S = Z.shape[1]
            data_mask = Z.any(axis=1)
            Z = Z[:, data_mask]

        else:
            raise NotImplementedError('Unknown split_init method {}'.format(split_init))

        # Check if we ended up changing S
        # assert (S == prm.S | | ismember('S', ip.UsingDefaults), 'MoDT:split:BadInit', ...
        #         'You specified S=%d, but splitInit had %d clusters', prm.S, S);

        # Check that the initialization is valid
        is_valid_init = (Z.sum(axis=0) >= min_spk_per_cluster).all()
        if not is_valid_init:
            exitcode = 1
            return exitcode

        # 3. SPLIT CLUSTER
        # retrieve changing model parameters
        mus = self._mus
        Cs = self._Cs
        alphas = self._alphas

        # manipulate model parameters
        mu = np.repeat(splitted_cluster_centers[:, np.newaxis, :], self._T, axis=1).T
        alpha = alphas[clustID] / S
        C = np.zeros((D, D, S))
        for new_cluster in range(S):
            spks_in_new_cluster = self._spk_x_train[data_mask, :][split_assign == new_cluster]
            cluster_cov = np.cov(spks_in_new_cluster.T)
            C[:, :, new_cluster] = cluster_cov

        # organize model parameters
        mus[:, :, clustID] = mu[:, :, 0]
        Cs[:, :, clustID] = C[:, :, 0]
        alphas[clustID] = alpha

        mus = np.concatenate((mus, mu[:, :, 1:]), axis=2)
        Cs = np.concatenate((Cs, C[:, :, 1:]), axis=2)
        alphas = np.concatenate((alphas, alpha * np.ones((S - 1))), axis=0)

        # store results
        parameters = {'mus': mus,
                      'Cs': Cs,
                      'alphas': alphas}
        self.set_params(**parameters)
        # self._K = K + S - 1
        # self._mus = mus
        # self._Cs = Cs
        # self._alphas = alphas
        # # update mahal distance, when changing number of clusters
        # self._mahal_dist = self._compute_mahalanobis_distance()

        # FIT MODEL (RUN EM)
        if run_em:
            self._expectation_maximization(max_iter=max_iter)

        return exitcode

    def _get_split_candidates(self):

        # PARAMETERS
        D = self._D
        N = self._N

        # COMPUTATION
        p = self.likelihood()
        pk = p / self._alphas.squeeze()  # broadcast alphas over p
        if self._posterior is None:
            self._posterior = self._compute_posterior()
        post = self._posterior
        fk = post / np.sum(post, axis=1)[:, np.newaxis]
        Jsplit = np.sum(fk * (np.log(fk + (fk == 0)) - np.log(pk + (pk == 0))), axis=0)
        cand = Jsplit.argsort()[::-1]
        assignments = np.argmax(post, axis=1)
        cand = cand[np.isin(cand, assignments) & (self._alphas[cand] * N > 4 * D)]

        return cand

    def _get_merge_candidates(self):
        """

        Determines good candidates for merging.

        :return:
        """
        post = self._posterior
        K = post.shape[1]
        max_candidates = int(np.ceil(K * np.sqrt(K) / 2))
        L2 = np.sqrt((post * post).sum(axis=0))
        Jmerge = np.zeros((int(K * (K - 1) / 2), 1))
        candidates = np.zeros((int(K * (K - 1) / 2), 2))
        k = 0
        for i in range(K):
            for j in range(i + 1, K):
                Jmerge[k, :] = (post[:, i] @ post[:, j].T) / (L2[i] * L2[j])
                candidates[k, :] = np.array([i, j])
                k += 1
        order = np.argsort(Jmerge, axis=None)[::-1]
        candidates = candidates[order[:max_candidates], :]

        return candidates

    def likelihood(self, X=None, t=None, clust=None, spike=None):
        """
        computes the likelihood

        :param X: [N, D] spike data for which to compute the likelihood
        :param t: [N, 1] spike time data for which to compute the likelihood
        :param clust: (int, array_like) clusters for which to compute likelihood
        :param spike: (int, array_like) indices of spikes for which to compute likelihood
        :return: likelihood
        """
        # PARSE INPUT
        if (X is None) != (t is None):
            raise ValueError('Either provide both, spike data and spike train or none of the two for likelihood '
                             'computation.')
        if X is None:
            X = self._spk_x_train
            t = self._spk_t_train

        if spike is None:
            N = X.shape[0]
            spike = slice(N)
        elif isinstance(spike, (int, float, bool, complex)):
            spike = np.array([int(spike)])
            N = 1
        else:
            try:
                N = len(spike)
            except TypeError:
                raise TypeError('Not a valid input type for spike.')

        if clust is None:
            clust = np.arange(self._K)
            K = len(clust)
        elif isinstance(clust, (int, float, bool, complex)):
            clust = np.array([int(clust)])
            K = 1
        else:
            try:
                K = len(clust)
            except TypeError:
                raise TypeError('Not a valid input type for clust.')

        # PARAMETERS
        D = self._D
        T = self._T
        nu = self._nu
        alphas = self._alphas
        mus = self._mus
        Cs = self._Cs
        spike_frame_id = self._determine_spike_frame_id(spike_train=t)

        # ALLOCATION
        likelihood = np.zeros((K, N))
        for k, c in enumerate(clust):
            mu = mus[:, spike_frame_id[spike], c]
            C = Cs[:, :, c]
            if nu == np.Infinity:
                # Gaussian
                likelihood[k, :] = alphas[c] * self._mvn.pdf(X - mu.T, C)
            else:
                # t-Distribution
                likelihood[k, :] = alphas[c] * self._mvt(X - mu.T, C, nu)

        return likelihood.T

    def _try_merge(self):
        """

        Tries to merge clusters. Merge is accepted if penalized average likelihood
        improves.

        :return: exitcode (0 == successful merge of two clusters, 1 == no merge done)
        """
        # PARAMETER
        exit_code = 0
        VERBOSE = self._VERBOSE

        if self._K == 1:
            exit_code = 1
            return self, exit_code

        # COMPUTATION
        candidates = self._get_merge_candidates()
        loglike_test = self._compute_penalized_loglike(X=self._spk_x_test, t=self._spk_t_test)
        bic_test = self._compute_bic(X=self._spk_x_test, t=self._spk_t_test)
        for candidate in candidates:
            if self._VERBOSE:
                print('trying to merge clusters {:0.0f} and {:0.0f}'.format(*candidate))
                # print('trying to merge clusters {} and {}'.format(*(candidate.astype(int))))
            new_self: MoDT = copy.deepcopy(self)
            successful_merge = new_self.merge(candidate, max_iter=self._max_iter, run_em=True)
            new_loglike_test = new_self._compute_penalized_loglike(X=self._spk_x_test, t=self._spk_t_test)
            new_bic_test = new_self._compute_bic(X=self._spk_x_test, t=self._spk_t_test)
            # if new_loglike_test > loglike_test:
            if new_bic_test < bic_test:
                if self._VERBOSE:
                    print('success! (likelihood improved by {:0.4f} per spike)'.format(
                        (new_loglike_test - loglike_test)[0] / self._N))
                return new_self, exit_code

        exit_code = 1
        return self, exit_code

    def merge(self,
              clustIDs,
              max_iter: int = 10,
              run_em: bool = False
              ):
        """
        This replaces the specified clusters with a single cluster by fitting a single
        cluster to the robust of data currently assigned to the specified clusters.
        Other clusters in the model are not affected by this operation.

        :param clustIDs: List of cluster indices (1..K) to merge. This may also be
            specified as a [K, 1] logical vector.
        :param max_iter: [1] maximum # of EM operations on subset model
        :return: exitcode ( 0 == sucessful merge of two clusters, 1 == no merge done )
        """
        # PARAMETERS
        exitcode = 0
        K = self._K

        # 1. DEAL WITH INPUT
        # determine which clusters will be merged
        cl_merge = np.zeros((K, 1)).astype(bool)
        cl_merge[clustIDs.astype(int)] = True  # takes indices as well as boolean indexing
        assert cl_merge.any, 'No valid clusters were specified for merging.'
        assert cl_merge.sum() > 1, 'Specify at least 2 cluster for merging.'

        # Determine indices of clusters that will be merged
        clustIDs = cl_merge.nonzero()[0]
        clust_keep = clustIDs[0]
        clust_drop = clustIDs[1:]

        # 2. MERGE CLUSTERS
        mus = self._mus[:, :, clustIDs]
        Cs = self._Cs[:, :, clustIDs]
        alphas = self._alphas[clustIDs]

        # 3. COMPUTE AND STORE NEW VALUES
        p = alphas[np.newaxis, np.newaxis, :]
        self._mus[:, :, clust_keep] = (mus * p).sum(axis=2) / p.sum()
        self._Cs[:, :, clust_keep] = (Cs * p).sum(axis=2) / p.sum()
        self._alphas[clust_keep] = alphas.sum()

        # delete excess clusters
        self._drop_clusters(clust_drop)

        # 4. FIT THE MODEL
        if run_em:
            self._expectation_maximization(max_iter=max_iter)

        return exitcode

    def _drop_clusters(self, clusters):
        """

        Deletes the specified clusters from the model.
        Other clusters in the model are not affected by this operation.


        :param clusters: List of cluster indices (1..K) to merge. This may also be
            specified as a [K, 1] logical vector.
        :return: exit_code (0 == successful deletion of clusters, 1 no deletion done)
        """
        # PARAMETERS
        K = self._K
        exitcode = 0

        # 1. DEAL WITH INPUT
        # determine which clusters will be merged
        cl_keep = np.ones((K, 1)).astype(bool)
        cl_keep[clusters] = False  # takes indices as well as boolean indexing
        if cl_keep.all():
            print('No cluster specified for dropping.')
            return exitcode

        # Determine indices of clusters that will be merged
        cl_keep = np.nonzero(cl_keep)[0]
        parameters = {'alphas': self._alphas[cl_keep],
                      'mus': self._mus[:, :, cl_keep],
                      'Cs': self._Cs[:, :, cl_keep]}
        self.set_params(**parameters)

        # update cache
        # self._update_cache()

        return exitcode

    def _update_cache(self):
        """
        updates the cached values of the model
        """
        self._mahal_dist = self._compute_mahalanobis_distance()
        Z, U = self._expectation()
        self._maximization(Z, U)

    def _optimize_mus(self, wzu, Cs):
        """
        Solve the quadratic optimization to update mu
        :param wzu: [N, K] = w * z * u = weight * posterior likelihood * t-dist scaling
        :param Cs: [D, D, K] estimate for cluster scale
        :return: mus [D, T, K] optimized estimate for cluster locations
        """

        # PARAMETERS
        D = self._D
        K = self._K
        T = self._T

        # SPECIAL CASE T==1
        if T == 1:
            mus = (self._spk_x_train.T @ wzu) / np.sum(wzu, axis=0)  # [D, K]
            return mus[:, np.newaxis, :]

        # CASE T!=1
        # calculate sums for each time block
        wzuX, sum_wzu = self._calculate_weighted_sums(wzu,
                                                      self._spk_x_train,
                                                      self._spike_frame_id,
                                                      self._frame_spike_lim)
        # solve mu = H \ b

        # start with process information matrix
        # This Qinv matrix determines the banded structure of H, so we can pre-compute some structural information
        # that will be useful for our call to sparse later.
        B_Q = self._make_Qinv_matrix(self._Q, D, T)
        nSuperDiag = int((B_Q.shape[0] - 1) / 2)
        H_i = np.arange(-nSuperDiag + 1, nSuperDiag + 2, 1)[:, np.newaxis] + np.arange(0, D * T, 1)[np.newaxis, :]
        H_j = np.tile(np.arange(1, D * T + 1, 1), (2 * nSuperDiag + 1, 1))
        H_mask = (H_i > 0) & (H_i <= D * T)
        H_i = H_i.T[H_mask.T] - 1  # zero- -> one-based indexing
        H_j = H_j.T[H_mask.T] - 1  # zero- -> one-based indexing

        # memory allocation
        if self._use_gpu:
            mus = skcuda.misc.zeros((D, T, K))
        else:
            mus = np.zeros((D, T, K))

        # loop over clusters
        for k in range(K):
            C = Cs[:, :, k]
            B_C = self._make_Cinv_matrix(C, sum_wzu[:, k])

            # add matrices together to get H
            B = B_Q
            offset = int((B_Q.shape[0] - B_C.shape[0]) / 2)
            B[offset:-offset, :] = B[offset:-offset, :] + B_C

            # calculate b= C_k \ wzu[:, :, k]
            b = np.linalg.lstsq(C, wzuX[:, :, k], rcond=None)[0]
            b = b.T.flatten()

            # solve H \ b
            # potentially if self.use_mex
            # if ~use_mex
            H = sparse.csr_matrix((B.T[H_mask.T], (H_i, H_j)), shape=(D * T, D * T))
            mu = sparse.linalg.lsqr(H, b)[0]

            # Update
            mus[:, :, k] = np.reshape(mu, (D, T), order='F')

        return mus

    def _optimize_Cs(self, wzu, mus, sum_wz):
        """
        solve the optimization to update. also compute the Mahalanobis distance
        :param wzu: [N, K] w*Z*U = weight*posterior likelyhood*t-dist calsing
        :param mus: [D, T, K] estimate for cluster location
        :param sum_wz: [K, 1] sum of w(n,k)*Z(n,k) over all spikes for each component
        :return: (C, delta) ([D, D, K], [N, K]) squared Mahalanobis distance
        """
        N, K = wzu.shape
        D = mus.shape[0]
        # To make sure that the output of optimizeC() will be accepted by setParams()
        # (which checks the condition number of the C matrix it's given), we are going
        # to use a slightly more conservative condition number (by 5%) here.
        min_rcond = 1.05 / self._max_cond

        # ALLOCATE memory
        C = np.zeros((D, D, K))
        if self._use_gpu:
            delta = skcuda.misc.zeros((N, K))
        else:
            delta = np.zeros((N, K))

        # loop over clusters
        for k in range(K):
            mu = mus[:, :, k]
            X = self._spk_x_train.T - mu[:, self._spike_frame_id]

            # update covariances
            # C_k = x_scaled * diag(wzu(:,k)) * X_scaled' / sum_wz(k)
            # TODO: potentially use_mex, gpu if sparse
            # if sparse.issparse(wzu):
            # [i, ~, v] = find(wzu(:, k));
            # X_scaled = bsxfun( @ times, X(:, i), sqrt(v)');
            X_scaled = X * np.sqrt(wzu[:, k])
            C_k = (X_scaled @ X_scaled.T) / sum_wz[k]

            if self._use_gpu:
                C_k = C_k.get()

            # add diagonal ridge for regularization
            C_k = C_k + self._C_reg * np.eye(D)

            # ensure C is well-conditioned
            if 1 / np.linalg.cond(C_k) < min_rcond:  # this is 1-norm condition but close enough
                U, S, _ = np.linalg.svd(C_k)
                s = np.diag(S)
                s = np.maximum(np.max(s) * min_rcond, s)
                U = U * np.diag(np.sqrt(s))
                C_k = U @ U.T

            # update the overall matrix
            C[:, :, k] = C_k

            # compute the mahalanobis distance
            L = np.linalg.cholesky(C_k)
            # delta[:,k] = np.sum((L\X).^2, 1).transpose
            # TODO: potentially use mex and use self._compute_malanobis_distance
            if self._use_gpu:
                delta[:, k] = skcuda.misc.sum(skcuda.linalg.dot(skcuda.linalg.inv(L), X) ** 2, 1).transpose()
            else:
                L_K = np.linalg.lstsq(L, X, rcond=None)[0]  # Matlab left division
                delta[:, k] = np.sum(L_K ** 2, axis=0)

        return C, delta

    def _mvt(self, x, C, df):
        '''

        Zero mean multivariate t-student density. Returns the density
        of the density function with mean 0, scale C and degrees of freedom df
        at points specified by x.

        :param x: [N, D] spikes, will be forced to 2D
        :param C: [D, D] scale matrix
        :param df: degrees of freedom
        :return: t-distribution density values for the points specified in x

        '''

        x = np.atleast_2d(x)  # requires x as 2d
        D = C.shape[0]  # dimensionality

        # if self._mahal_dist is None:
        #     delta = self._compute_mahalanobis_distance()
        #     self._mahal_dist = delta
        # else:
        #     delta = self._mahal_dist

        Ch = np.linalg.cholesky(C)
        delta = np.sum((np.linalg.inv(Ch) @ x.T) ** 2, axis=0)  # sum out features.
        p = np.exp(gammaln((df + D) / 2)
                   - gammaln(df / 2)
                   - ((df + D) / 2) * np.log(1 + delta / df)
                   - np.sum(np.log(np.diag(Ch)))
                   - (D / 2) * np.log(df * np.pi))

        return p

        # ----------     HELPER FUNCTIONS     ---------- #

    def _mvn(self, x, C):
        '''

        Zero mean multivariate normal density. Returns the density
        of the density function with mean 0, scale C at points specified by x.

        :param x: [N, D] spikes, will be forced to 2D
        :param C: [D, D] scale matrix
        :return: t-distribution density values for the points specified in x
        '''

        # PARSE INPUT
        C = np.atleast_2d(C)

        # PARAMETERS
        D = C.shape[0]
        const = 2 * np.pi ** (-D / 2)
        Ch = np.linalg.cholesky(C)
        p = const / np.prod(np.diag(Ch)) * np.exp(-0.5 * np.sum((np.linalg.inv(Ch) @ x.T) ** 2, axis=0))

        return p

    def _calculate_weighted_sums(self, wzu, X, spk_fid, f_spklim, use_mex=False):
        """
        Calculate the weighted sums for each time block.
        More specifically this returns wzuX and sum_wzu such that
        wzuX[:,t,k] = x[:,n_t]*wzu[k,n_t].T
        and sum_wzu[t,k] = sum(wzu[k,n_t])
        where n_t=frame_spklim[t,1]:frame_spklim[t,2]
        There is some optimization for the case where wzu is sparse.
        :param wzu: [N, K] weighted sum of points in each time block
        :param X:  [N, D] spike data
        :param spk_fid: [N, 1] spike frame ID (1...T) for each spike
        :param f_spklim: [T, 2]  = [first, last] spike ID (1...N) in each time frame
        :param use_mex: wether to use MEX routines, bool
        :return: (wzuX, sum_wzu) ([D, T, K], [T, K]) sum of the weights in each time block
        """

        # PARAMETERS
        N, K = wzu.shape
        D = X.shape[1]
        T = self._T  # f_spklim.shape[0]

        # some optimization for the case f sparse weight matrices
        if sparse.issparse(wzu):
            # Sparse wzu
            [i, j, v] = find(wzu)
            # Rearrange wzu into a[N, T * K] matrix so columns are consecutive frames
            j = spk_fid(i) + T * (j - 1)
            wzu_byframe = sparse(i, j, v, N, T * K)
            # Now we can get wzuX by simply multiplying X * wzu_byframe
            wzuX = X * wzu_byframe
            # Similarly, we can get sum_wzu by summing the columns by wzu_byframe
            sum_wzu = sum(wzu_byframe, 1)
            # Finally, reshape to match our desired shape
            wzuX = np.reshape(wzuX, [D, T, K])
            sum_wzu = np.reshape(sum_wzu, [T, K])
        else:  # dense wzu
            # start by computing them as {K, T] and [D, K, T]
            if not use_mex:
                # Allocate memory
                if False:  # isinstance(X, gpuArray):
                    # sum_wzu = zeros(K, T, 'gpuArray')
                    # wzuX = zeros(D, K, T, 'gpuArray')
                    pass
                else:
                    sum_wzu = np.zeros((K, T))
                    wzuX = np.zeros((D, K, T))
                # For loop over time frames
                for t in range(T):
                    n1 = f_spklim[t, 0]
                    n2 = f_spklim[t, 1] + 1
                    if t == T - 1:
                        wzu_t = wzu[n1:, :]
                        wzuX[:, :, t] = X[n1:, :].T @ wzu_t
                    else:
                        wzu_t = wzu[n1:n2, :]
                        wzuX[:, :, t] = X[n1: n2, :].T @ wzu_t
                    sum_wzu[:, t] = np.sum(wzu_t, axis=0).T

            else:
                # Call a MEX routine
                if False:  # isinstance(X, 'gpuArray'):
                    # [wzuX, sum_wzu] = MoDT.sumFramesGpu(X, wzu, f_spklim)
                    pass
                else:
                    [wzuX, sum_wzu] = self._sum_frames(X, wzu, f_spklim)

            # Reshape to match our desired shape
            sum_wzu = sum_wzu.T  # now [T, K]
            wzuX = wzuX.transpose([0, 2, 1])  # now[D, T, K]
            # Gather
            if False:  # isinstance(sum_wzu, 'gpuArray'):
                # sum_wzu = gather(sum_wzu)
                # wzuX = gather(wzuX)
                pass

        return wzuX, sum_wzu

    def sum_frames(X, wzu, f_spklim):
        """
        Compute the weighted sum of data points for each time frame
        This performs the following for loop:

            for t in range(T):
                n1 = frame_spklim[t, 1]
                n2 = frame_spklim[t, 2]
                sum_wzu[:, t] = np.sum(wzu[n1: n2,:], axis=0).T
                wzuX[:,:, t] = X[:, n1: n2]@wzu[n1: n2,:]

        The MEX version performs the same exact operations, but reduces the overhead associated with the for loop.

        :param X: [D, N] data points (D dimensions x N points)
        :param wzu: [N, K] weights for each (data point) x (cluster)
        :param f_spklim: [T, 2] [first, last] data index (1..N) in each time frame
        :return: (wzuX, sum_wzu): ([D, K, T], [K, T]) = (weighted sums of data points in each time frame, sums of weights in each time frame)
        """

        # Get sizes
        D = X.shape[0]
        K = wzu.shape[1]
        T = f_spklim.size[0]

        # Alocate memory
        wzuX = np.zeros((D, K, T))
        sum_wzu = np.zeros((K, T))

        # For loop over time frames
        for t in range(T):
            n1 = f_spklim(t, 1)
            n2 = f_spklim(t, 2)
            sum_wzu[:, t] = np.sum(wzu[n1: n2, :], axis=0).T
            wzuX[:, :, t] = X[:, n1: n2] @ wzu[n1: n2, :]

        return wzuX, sum_wzu

    def _make_Qinv_matrix(self, Q, D, T):
        """
        construct the banded representation of the process information matrix
        This represents the [D*T, D*T] process information matrix Qinv:
           [  Q^-1  -Q^-1               ]
           [ -Q^-1  2Q^-1  -Q^-1        ]
           [        -Q^-1   ...   -Q^-1 ]
           [               -Q^-1   Q^-1 ]
        in column-major banded storage. In this storage format, the columns of B still
        correspond to the columns of Qinv, but the rows of B correspond to the diagonals of Qinv.
        In other words:
           [ 11  12   0   0 ]         [ **  12  23  34 ]
           [ 21  22  23   0 ]   ==>   [ 11  22  33  44 ]
           [  0  32  33  34 ]         [ 21  32  43  ** ]
           [  0   0  43  44 ]

        Note that this is not the same format as returned by spdiags (MATLAB)
        """

        assert T > 1, 'T must be strictly larger than 1; found {}'.format(T)

        if Q.size == 1:
            # This means a diagonal Q matrix with the scalar Q along the diagonal,
            # so the inverse is simply 1/Q along the diagonal
            B = np.zeros((2 * D + 1, D * T))
            # OFF-diagonal bands of inv(Q)
            B[0, D:] = -1 / Q
            B[-1, 0:-D + 1] = -1 / Q
            # inv(Q) and 2*inv(Q) along the main diagonal
            B[D, 0:D] = 1 / Q
            B[D, D:-D + 1] = 2 / Q
            B[D, -D + 1:] = 1 / Q
        else:
            # General case: arbitrary Q^-1
            Qinv = np.linalg.inv(Q)
            Q = Qinv + np.transpose(Qinv) / 2  # enforce symmetry
            # Make the 3x3 block case
            Q3 = np.array([[Qinv, -Qinv, np.zeros(D)],
                           [-Qinv, 2 * Qinv, -Qinv],
                           [np.zeros(D), -Qinv, Qinv]])
            B3 = np.zeros((4 * D - 1, 3 * D))
            idx_i = np.transpose(np.arange(-2 * D + 1, 2 * D - 1, 1)) + np.arange(1, 3 * D, 1)
            idx_j = np.arange(0, 3 * D, 1)
            idx_lin = idx_i + ((idx_j - 1) * (3 * D))
            mask = (idx_i > 0) * (idx_i <= 3 * D)
            B3[mask] = Q3[idx_lin[mask]]
            # trim away extraneous diagonals
            raise RuntimeError('Translate to python!!!')
            nz_offset = 3  # find(~all(B3==0,2), 1, 'first') - 1
            B = B3[nz_offset + 1:-1 - nz_offset, :]
            # Repeat this as necessary
            B = np.concatenate((B[:, 0: D], np.repeat(B[:, D + 1: 2 * D], [1, T - 2]), B[:, 2 * D + 1: 3 * D]),
                               axis=1)

        return B

    def _make_Cinv_matrix(self, C, wzu):
        """
        construct the banded representation of the observation information matrix
        This represents the [D*T, D*T] observation matrix:
           [ wzu(1)*C^-1                                     ]
           [             wzu(2)*C^-1                         ]
           [                             ...                 ]
           [                                     wzu(T)*C^-1 ]
        in the column-major banded storage format. In this format,
           [ 11  12   0   0 ]         [ **  12  23  34 ]
           [ 21  22  23   0 ]   ==>   [ 11  22  33  44 ]
           [  0  32  33  34 ]         [ 21  32  43  ** ]
           [  0   0  43  44 ]
        Note that this is not the same format as returned by spdiags (MATLAB)!!
        :param C: [D, D] observation covariance matrix
        :param wzu: [T, 1] weight for each time frame
        :return: B [2*D-1, D*T] column-major based storage of a [D*T, D*T] matrix
        """
        Cinv = np.linalg.inv(C)
        Cinv = (Cinv + Cinv.T) / 2  # enforce symmetry
        # convert the [D, D] full format into a [2*D-1, D] banded format
        D = C.shape[0]
        B1 = np.zeros((2 * D - 1, D))
        idx = np.arange(0, D, 1)[:, np.newaxis] + (np.arange(D - 1, -1, -1) + np.arange(0, D, 1) * B1.shape[0])[
                                                  np.newaxis, :]
        B1[np.unravel_index(idx, B1.shape, 'F')] = Cinv
        # repeat and scale by wzu
        B = np.kron(wzu[np.newaxis, :], B1)

        return B

    def _compute_prior_loglike(self):
        """
        Computes the log-likelihood of the parameter prior. This is not a proper
        prior over the parameters as it doesn't integrate to one. However, for
        a given model class, it is proportional to a uniform prior over the
        remaining parameters, so it is still useful for evaluating EM convergence.
        :return: log-likelihood of the model parameters given the drift model
        """

        # PARAMETERS
        T = self._T
        D = self._D

        # Calculate the log-likelihood of mu given the drift model
        if T > 1:
            drift = np.diff(self._mus, axis=1)
            # its the same Q for each cluster, so we can flatten out that dimension
            # delta = drift[:,:].t * Q^-1 * drift[:,:]
            if self._Q.size == 1:  # if is scalar
                # Q = eye(D) * self._Q OR Q^-1 = eye(D) / self._Q
                Q = np.eye(self._D) * self._Q
                delta = np.sum(np.reshape(drift, (D, -1)) ** 2, axis=0) / self._Q
                log_sqrt_det_Q = - D / 2 * np.log(self._Q)
            else:
                # Q = L * L.T; inv(Q) = inv(L)'.T* inv(L)
                L = np.linalg.cholesky(self._Q)
                L_drift, _, _, _ = np.linalg.lstsq(L, np.reshape(drift, (D, -1)), rcond=None)
                delta = np.sum(L_drift ** 2, axis=0)
                log_sqrt_det_Q = np.sum(np.log(np.diag(L)))

            # evaluate the multivariate gaussian
            # p(k, t) = 1 / ((2 * pi) ^ (D / 2) * sqrt(det(Q))) * exp(-1 / 2 * delta(k, t))
            # drift_ll = sum(log(p))
            log_norm_const = -D / 2 * np.log(2 * np.pi) - log_sqrt_det_Q
            # print(log_norm_const, np.sum(delta)/2)
            drift_ll = delta.size * log_norm_const - np.sum(delta) / 2
        else:
            drift_ll = 0

        return drift_ll

    def _compute_mahalanobis_distance(self, X=None, spike_frame_id=None):
        """
        computes the mahalanobis distance delta

        :param X: [N, D] spikes for which to compute the mahalobis distance
            dimensionality D of the spikes has to match the dimensionality of the data
        :param spike_frame_id: [N, 1] spike frame id 1..T for each spike
        :param C: [D, D] covariance matrix
        :return [N,K] malanobis distance for each spike to each cluster
        """
        # PARSE INPUT
        if X is None and spike_frame_id is None:
            X = self._spk_x_train
            spike_frame_id = self._spike_frame_id
        if (X is None) ^ (spike_frame_id is None):
            raise TypeError('Invalid input. Provide either both, X and spike_frame_id, or none of the two.')
        try:
            if len(X) != len(spike_frame_id):
                raise ValueError('X and spike_frame_id have to be of the same length')
        except TypeError as e:
            raise

        # PARAMETERS
        N = self._N
        K = self._K
        Cs = self._Cs
        log_sqrt_det_C = np.zeros((K, 1))

        # COMPUTATION
        if self._use_gpu:
            delta = skcuda.misc.zeros((N, K))
            mus = gpuarray.to_gpu(self._mus)
        else:
            delta = np.zeros((N, K))
            mus = self._mus

        # compute for each cluster
        Cs = self._Cs
        for k in range(K):
            mu = mus[:, :, k]
            C = Cs[:, :, k]
            L = np.linalg.cholesky(C)
            log_sqrt_det_C[k] = np.sum(np.log(np.diagonal(L)))
            X = self._spk_x_train.T - mu[:, self._spike_frame_id]
            if self._use_gpu:
                delta[:, k] = skcuda.misc.sum(skcuda.dot(skcuda.linalg.inv(L), X), axis=0).transpose()
            else:
                L_X = np.linalg.lstsq(L, X, rcond=None)[0]
                delta[:, k] = np.sum(L_X ** 2, axis=0)

        return delta

    def _get_subset_model(self,
                          clusterIDs,
                          data_mask=None,
                          data_thresh=0.2):
        """

        This method constructs an MoDT model containing a subset of the clusters present
        in this model. If data is also attached, then the data is subsetted according to
        data_mask, and reweighted according to the posterior likelihood that each spike
        belongs to the selected subset of clusters. If data_mask is not specified by the
        user, it is set as (posterior likelihood > data_thresh)

        :param clusterIDs: List of cluster indices (1..K) or [K, 1] logical vector
            indicating which clusters to include in subset
        :param data_mask: [N, 1] logical specifying data for subset
        :param data_thresh: [1] threshold for inclusion in data subset
        :return: MoDT object with a subset of the clusters/data from self
        """

        # Determine which clusters we're keeping
        cl_subset = np.zeros((self._K, 1)).astype(bool)
        cl_subset[clusterIDs] = True
        assert cl_subset.any(), 'No clusters selected for subsetting.'

        # parse optional parameters
        # TODO: parse optinonal arguments
        # if data_mask is None or (isinstance(x, (bool) and x.size == self._N)):
        #     lambda x: True if (x is None or ) else []
        # ip = inputParser();
        # ip.addParameter('dataMask', [], @ (x)
        # isempty(x) | | ...
        # (islogical(x) & & numel(x) == self.N));
        # ip.addParameter('dataThresh', 0.2, @ isscalar);
        # ip.parse(varargin
        # {:} );
        # prm = ip.Results;

        # Construct the new model
        subModel = self._copy()

        # set the parameters based on the selected subset
        # if this is slow we could bypass setParams() and set the values directly
        subModel.setParams({'alpha': self._alphas / self._alphas(cl_subset).sum(),
                            'mu': self._mus[:, :, cl_subset],
                            'C': self._Cs[:, :, cl_subset]})

        # Attach the subset of data
        if self._spk_x_train is not None:
            # get the posterior likelihood of belonging to the selected subset
            post = self._get_value('posterior')
            post = post[:, cl_subset].sum(axis=1)
            # determine the data to attach
            if data_mask is None:
                data_mask = (post > data_thresh)
            # special case for scalar w
            if self._spk_w_train == 0 and self._N > 1:
                w_masked = self._spk_w_train
            else:
                w_masked = self._spk_w_train[data_mask]
            # attach the selected subset of data
            subModel = subModel.fit(self._spk_x_train[:, data_mask],
                                    self._spk_t_train[data_mask],
                                    w_masked * post[data_mask])

        return subModel

    def _compute_posterior(self, X=None, t=None, clust=None, spike=None):
        """
        computes the posterior likelihood for given data X

        :param X: [N, D] data for which to compute the likelihood
        :param clust: (int, array_like) clusters for which to compute likelihood
        :param spike: (int, array_like) indices of spikes for which to compute likelihood
        :return: posterior data probability
        """
        # PARSE INPUT
        # if X is None:
        #     X = self._spk_x_train

        # if spike is None:
        #     spike = slice(self._N)
        #     N = self._N
        # elif isinstance(spike, (int, float, bool, complex)):
        #     spike = np.array([int(spike)])
        #     N = 1
        # else:
        #     try:
        #         N = len(spike)
        #     except TypeError:
        #         raise TypeError('Not a valid input type for spike.')

        # if clust is None:
        #     clust = np.arange(self._K)
        #     K = len(clust)
        # elif isinstance(clust, (int, float, bool, complex)):
        #     clust = np.array([int(clust)])
        #     K = 1
        # else:
        #     try:
        #         K = len(clust)
        #     except TypeError:
        #         raise TypeError('Not a valid input type for clust.')

        # compute likelihood
        like = self.likelihood(X, t, clust, spike)
        like_scale = np.max(like, axis=1)[:, np.newaxis]
        like /= like_scale
        # Convert like to posterior likelihood like(b,k) = P(num_spikes from cluster k)
        sum_like = np.sum(like, axis=1)
        like /= sum_like[:, np.newaxis]

        posterior = like

        return posterior

    def _compute_data_loglike(self, X=None, t=None):
        """

        returns the log-likelihood of
        :param X: [N, D] data for which to compute the likelihood
        :return:
        """
        # PARSE INPUT
        if (X is None) != (t is None):
            raise ValueError('Provide either both, spike data and spike time data, or none of the two to compute data '
                             'loglikelihood')
        if X is None:
            X = self._spk_x_train
            t = self._spk_t_train

        if X.shape[1] == t.size:
            raise ValueError('The time axis of X and t do not match.')

        # COMPUTATION
        like = self.likelihood(X=X, t=t)
        sum_like = np.sum(like, axis=1)

        # compute iunweighted data log-likelihood
        unweighted_data_loglike = np.log(sum_like)  #
        # unweighted_data_loglike = np.log(np.prod(test_Z*like_scale, axis=1))
        # np.sum(np.log(test_Z), axis=1) + self._N * np.log(like_scale.squeeze())
        # compute weighted data likelihood

        if self._spk_w_train.size == 1:
            weighted_data_loglike = unweighted_data_loglike.sum() * self._spk_w_train
        else:
            weighted_data_loglike = unweighted_data_loglike.T @ self._spk_w_train

        if self._use_gpu:
            weighted_data_loglike = weighted_data_loglike.get()

        return weighted_data_loglike

    def _compute_loglike(self, X=None, t=None):
        """

        :param X: [N, D] data for which to compute the likelihood
        :return:
        """
        if X is None:
            X = self._spk_x_train

        # COMPUTE DATA LOGLIKELIHOOD
        data_loglike = self._data_loglike[-1]
        if not data_loglike:
            data_loglike = self._compute_data_loglike(X=X, t=t)

        # COMPUTE PRIOR LOGLIKELIHOOD
        prior_loglike = self._prior_loglike[-1]
        if not prior_loglike:
            prior_loglike = self._compute_prior_loglike()

        return data_loglike + prior_loglike

    def _compute_penalized_loglike(self, X=None, t=None):
        """

        :param X: [N, D] data for which to compute the likelihood
        :return:
        """
        if X is None:
            X = self._spk_x_train

        # COMPUTE DATA LOGLIKELIHOOD
        data_loglike = self._data_loglike[-1]
        if not data_loglike:
            data_loglike = self._compute_data_loglike(X=X, t=t)

        # COMPUTE PRIOR LOGLIKELIHOOD
        prior_loglike = self._prior_loglike[-1]
        if not prior_loglike:
            prior_loglike = self._compute_prior_loglike()

        # COMPUTE COMPLEXITY PENALTY
        penalty = self._K * self._D * self._cluster_cost

        # return data_loglike + prior_loglike / self._T - penalty
        return data_loglike + prior_loglike - penalty

    def _compute_aic(self, X=None, t=None):
        """
        computes the akaike information criterion for the model
        :return: aic
        """
        # CACHE CLUSTER COST
        clust_cost = self._cluster_cost
        self._cluster_cost = 0

        # PARSE INPUT
        if (X is None) != (t is None):
            raise ValueError('Provide either both, spike data and spike time data, or none of the two to compute '
                             'akaike information criterion')
        if X is None:
            X = self._spk_t_train
            t = self._spk_t_train

        N = X.shape[0]
        aic = -2 * self._compute_data_loglike(X=X, t=t) + self.num_model_params_test
        # aic = -2 * self._compute_penalized_loglike(X=X) + self.num_model_params

        # STORE BACK CLUSTER COST
        self._cluster_cost = clust_cost

        return aic

    def _compute_bic_test(self, X=None, t=None):
        """
        computes the bayesian information criterion for the model
        :return: bic
        """
        # CACHE CLUSTER COST
        clust_cost = self._cluster_cost
        self._cluster_cost = 0

        # PARSE INPUT
        if X is None:
            X = self._spk_t_train

        N = X.shape[0]
        bic = -2 * self._compute_data_loglike(X=X, t=t) + self.num_model_params * np.log(N)

        # STORE BACK CLUSTER COST
        self._cluster_cost = clust_cost

        return bic

    def _compute_bic(self, X=None, t=None):
        """
        computes the bayesian information criterion for the model
        :return: bic
        """
        # CACHE CLUSTER COST
        clust_cost = self._cluster_cost
        self._cluster_cost = 0

        # PARSE INPUT
        if X is None:
            X = self._spk_t_train

        N = X.shape[0]
        bic = -2 * self._compute_penalized_loglike(X=X, t=t) + self.num_model_params * np.log(N)

        # STORE BACK CLUSTER COST
        self._cluster_cost = clust_cost

        return bic

    def _reverse_lookup(self, spk_clustID):
        """

        Given cluster ID of each spike, return spike IDs that belong to each cluster.
        This innfers the number of clusters K from np.max(spk_clustID)

        :param spk_clustID: [N, 1] cluster ID (1..K) for each spike
        :return: []
        """

        # Sort spk_clustID. This groups similar clustIDs together, and the second output
        # argument track the original spike IDs.

        # infer the number of clusters
        K = sorted_clustIDs[-1]

        # count how many spikes are in each cluster.
        N_k = np.bincount()

        # split the sorted spike IDs into a list of arrays

        return clust_

    def _ffill(self, arr):
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:, None], idx]
        return out

    def _bfill(self, arr):
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(mask.shape[1]), mask.shape[1] - 1)
        idx = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
        out = arr[np.arange(idx.shape[0])[:, None], idx]
        return out

    # ----------     CLASS PROPERTIES     ---------- #
    @property
    def mahal_dist(self):
        return self._mahal_dist

    @property
    def posterior(self):
        return self._posterior

    @property
    def assignment(self):
        t = np.concatenate((self._spk_t_train, self._spk_t_test))
        sort_idcs = t.argsort()

        assignments_train = np.argmax(self._compute_posterior(self._spk_x_train, self._spk_t_train), axis=1).squeeze()
        assignments_test = np.argmax(self._compute_posterior(self._spk_x_test, self._spk_t_test), axis=1).squeeze()
        assignments = np.concatenate((assignments_train, assignments_test))[sort_idcs]

        return assignments

    @property
    def clusters(self):
        spk_clust_id = np.argmax(self._posterior, axis=2)
        if self._use_gpu:
            # spk_clust_id = gather (Z)
            pass
        return self._reverse_look_up(spk_clust_id)

    @property
    def confusion_mat(self):
        """

        computes confusion_mat(i,j)

        confusion_mat(i,j) is expected number of spikes assigned to cluster i that were actually generated by cluster j.
        Specifically, this is computed as confusionMat(i,j) = sum(posterior(assignment==i, j)). Also note that
        sum(confusionMat(i,:)) = sum(assignment==i).

        :return: confusion_mat
        """
        Z = self._posterior
        if self._use_gpu:
            # spk_clust_id = gather (Z)
            pass
        spk_clust_id = np.argmax(Z, axis=1)
        confusion_mat = sparse.csr_matrix((np.ones_like(spk_clust_id), (spk_clust_id, np.arange(0, self._N))),
                                          (self._K, self._N)) @ Z
        return confusion_mat

    @property
    def loglike(self):
        return np.array(self._data_loglike) + np.array(self._prior_loglike)

    @property
    def data_loglike(self):
        return np.array(self._data_loglike)

    @property
    def spk_frame(self):
        assert self._spike_frame_id.size is not None, 'spike_frame is not available'
        return self._spike_frame_id

    @property
    def frame_spk_lim(self):
        assert self._frame_spike_lim, 'frame_spike_lim is not available. Please, attach data first.'
        return self._frame_spike_lim

    @property
    def num_model_params(self):
        assert self._D and self._T and self._K, 'model parameters not yet initialized'
        return self._K - 1 + self._D * self._K * self._T + self._K * self._D * (self._D - 1) / 2

    @property
    def num_model_params_test(self):
        assert self._D and self._T and self._K, 'model parameters not yet initialized'
        return self._K - 1 + self._D * self._K + self._K * self._D * (self._D - 1) / 2
