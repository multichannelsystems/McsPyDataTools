"""
.. module:: sorting
        :synopsis: Spike clustering

.. moduleauthor:: Ole Jonas Wenzel <wenzel@multichannelsystems.com>
"""

from .detection import *
from .features import *
from .cluster import *


class Sort(object):
    """

    """

    methods = ['gmeans', 'gmm', 'modt']

    def __init__(self, method: str = 'modt', *args, **kwargs):
        """

        Sorts segment stream of subtype spike.

        :param method: element of
        :return: list of
        """
        # PARSE INPUT
        assert method in Sort.methods, '\'method\' has to be in {}'.format(Sort.methods)

        # init cluster model
        self.cluster_model = None
        if method == 'gmeans':
            print('Not yet implemented.')
            return

        if method == 'gmm':
            self.cluster_model = GMM(*args, **kwargs)

        if method == 'modt':
            self.cluster_model = MoDT(*args, **kwargs)

        if self.cluster_model is None:
            return self.cluster_model

        self._features = None

    def __call__(self, segment_stream, timestamp_stream=None, tetrode: List[int] = [0, 1, 2, 3], *args, **kwargs):
        """
        :param segment_stream: segment stream of subtype spike from MCS file
        :param timestamp_stream: timestamp stream of subtype 'NeuralSpike' from MCS file.
            Note: The function takes the spike train that is associated to the first electrode in the parameter tetrode.
        :param tetrode: list of indices of segment stream entities that make up one tetrode
        :param args: args are passed to the clustering model fit function
        :param kwargs: kwargs are passed to the clustering model fit function
        :return:
        """
        # PARSE INPUT
        assert len(tetrode) == 4, 'Provide a list of exactly four indices that make up your tetrode.'

        # EXTRACT TETRODE DATA
        try:
            err_msg = 'Stream Type, expected: \'Segment\', got \'{}\''.format(segment_stream.stream_type)
            assert segment_stream.stream_type == 'Segment', err_msg
            spikes = np.array([segment_stream.segment_entity[tetrode[0]].data,
                               segment_stream.segment_entity[tetrode[1]].data,
                               segment_stream.segment_entity[tetrode[2]].data,
                               segment_stream.segment_entity[tetrode[3]].data])
            if timestamp_stream is not None:
                err_msg = 'Stream Type, expected: \'TimeStamp\', got \'{}\''.format(timestamp_stream.stream_type)
                assert timestamp_stream.stream_type == 'TimeStamp', err_msg
                ttrain = np.array([timestamp_stream.timestamp_entity[tetrode[0]].data]).astype('float64').squeeze()
                ttrain *= 1e-6  # conversion to seconds
                
        except IndexError:
            print('Provide a  valid list for the parameter tetrode. ' +
                  'got {}, valid: {}'.format(tetrode, list(timestamp_stream.timestamp_entity.keys())))

        w = spikes.transpose((1, 2, 0))  # transpose spikes into the form [#num samples cutout ,#spikes ,#num channels]

        # PCA
        b, _, _ = extract_features(w, method='pca')
        self._features = b

        # CLUSTER THE UNITS
        self.cluster_model.fit(b, ttrain, *args, **kwargs)

        return self.cluster_model.assignment

    @property
    def features(self):
        if self._features is None:
            print('First, sort you spikes')
        return self._features