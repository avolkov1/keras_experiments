'''
TAKEN DIRECTLY FROM TENSORPACK.
'''
from __future__ import print_function

import os
from datetime import datetime

from abc import abstractmethod, ABCMeta
from copy import copy

import six
import numpy as np

import logging as logger

__all__ = ('RepeatedData', 'BatchData',)


# =============================================================================
# UTILS
# =============================================================================

_RNG_SEED = None


def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    if _RNG_SEED is not None:
        seed = _RNG_SEED
    return np.random.RandomState(seed)


# =============================================================================
# DATAFLOWS
# =============================================================================


@six.add_metaclass(ABCMeta)
class DataFlow(object):
    """ Base class for all DataFlow """

    @abstractmethod
    def get_data(self):
        """
        The method to generate datapoints.

        Yields:
            list: The datapoint, i.e. list of components.
        """

    def size(self):
        """
        Returns:
            int: size of this data flow.

        Raises:
            :class:`NotImplementedError` if this DataFlow doesn't have a size.
        """
        raise NotImplementedError()

    def reset_state(self):
        """
        Reset state of the dataflow. It has to be called before producing
        datapoints.

        For example, RNG **has to** be reset if used in the DataFlow,
        otherwise it won't work well with prefetching, because different
        processes will have the same RNG state.
        """
        pass


class RNGDataFlow(DataFlow):
    """ A DataFlow with RNG"""

    def reset_state(self):
        """ Reset the RNG """
        self.rng = get_rng(self)


class ProxyDataFlow(DataFlow):
    """ Base class for DataFlow that proxies another"""

    def __init__(self, ds):
        """
        Args:
            ds (DataFlow): DataFlow to proxy.
        """
        self.ds = ds

    def reset_state(self):
        """
        Reset state of the proxied DataFlow.
        """
        self.ds.reset_state()

    def size(self):
        return self.ds.size()


class RepeatedData(ProxyDataFlow):
    """ Take data points from another DataFlow and produce them until
        it's exhausted for certain amount of times.
    """

    def __init__(self, ds, nr):
        """
        Args:
            ds (DataFlow): input DataFlow
            nr (int): number of times to repeat ds.
                Set to -1 to repeat ``ds`` infinite times.
        """
        self.nr = nr
        super(RepeatedData, self).__init__(ds)

    def size(self):
        """
        Raises:
            :class:`ValueError` when nr == -1.
        """
        if self.nr == -1:
            raise ValueError("size() is unavailable for infinite dataflow")
        return self.ds.size() * self.nr

    def get_data(self):
        if self.nr == -1:
            while True:
                for dp in self.ds.get_data():
                    yield dp
        else:
            for _ in range(self.nr):
                for dp in self.ds.get_data():
                    yield dp


class BatchData(ProxyDataFlow):
    """
    Concat datapoints into batches.
    It produces datapoints of the same number of components as ``ds``, but
    each component has one new extra dimension of size ``batch_size``.
    The new component can be a list of the original datapoints, or an ndarray
    of the original datapoints.
    """

    def __init__(self, ds, batch_size, remainder=False, use_list=False):
        """
        Args:
            ds (DataFlow): Its components must be either scalars or
                :class:`np.ndarray`. Each component has to be of the same shape
                across datapoints.
            batch_size(int): batch size
            remainder (bool): whether to return the remaining data smaller than
                a batch_size. If set True, it will possibly generates a data
                point of a smaller batch size. Otherwise, all generated data
                are guranteed to have the same size.
            use_list (bool): if True, it will run faster by producing a list
                of datapoints instead of an ndarray of datapoints, avoiding an
                extra copy.
        """
        super(BatchData, self).__init__(ds)
        if not remainder:
            try:
                assert batch_size <= ds.size()
            except NotImplementedError:
                pass
        self.batch_size = batch_size
        self.remainder = remainder
        self.use_list = use_list

    def size(self):
        ds_size = self.ds.size()
        div = ds_size // self.batch_size
        rem = ds_size % self.batch_size
        if rem == 0:
            return div
        return div + int(self.remainder)

    def get_data(self):
        """
        Yields:
            Batched data by stacking each component on an extra 0th dimension.
        """
        holder = []
        for data in self.ds.get_data():
            holder.append(data)
            if len(holder) == self.batch_size:
                yield BatchData._aggregate_batch(holder, self.use_list)
                del holder[:]
        if self.remainder and len(holder) > 0:
            yield BatchData._aggregate_batch(holder, self.use_list)

    @staticmethod
    def _aggregate_batch(data_holder, use_list=False):
        size = len(data_holder[0])
        result = []
        for k in range(size):
            if use_list:
                result.append(
                    [x[k] for x in data_holder])
            else:
                dt = data_holder[0][k]
                if type(dt) in [int, bool]:
                    tp = 'int32'
                elif type(dt) == float:
                    tp = 'float32'
                else:
                    try:
                        tp = dt.dtype
                    except Exception:
                        raise TypeError("Unsupported type to batch: {}"
                                        .format(type(dt)))
                try:
                    result.append(
                        np.asarray([x[k] for x in data_holder], dtype=tp))
                except KeyboardInterrupt:
                    raise
                except Exception:
                    logger.exception("Cannot batch data. Perhaps they are of "
                                     "inconsistent shape?")
                    import IPython as IP
                    IP.embed(config=IP
                             .terminal  # @UndefinedVariable
                             .ipapp.load_default_config())
        return result


class MapData(ProxyDataFlow):
    """ Apply a mapper/filter on the DataFlow"""

    def __init__(self, ds, func):
        """
        Args:
            ds (DataFlow): input DataFlow
            func (datapoint -> datapoint | None): takes a datapoint and returns
                a new datapoint. Return None to discard this data point.
                Note that if you use the filter feature, ``ds.size()`` will be
                incorrect.

        Note:
            Please make sure func doesn't modify the components
            unless you're certain it's safe.
        """
        super(MapData, self).__init__(ds)
        self.func = func

    def get_data(self):
        for dp in self.ds.get_data():
            ret = self.func(dp)
            if ret is not None:
                yield ret


class MapDataComponent(MapData):
    """ Apply a mapper/filter on a datapoint component"""
    def __init__(self, ds, func, index=0):
        """
        Args:
            ds (DataFlow): input DataFlow.
            func (TYPE -> TYPE|None): takes ``dp[index]``, returns a new value
                for ``dp[index]``.
                return None to discard this datapoint.
                Note that if you use the filter feature, ``ds.size()`` will be
                incorrect.
            index (int): index of the component.

        Note:
            This proxy itself doesn't modify the datapoints.
            But please make sure func doesn't modify the components
            unless you're certain it's safe.
        """
        def f(dp):
            r = func(dp[index])
            if r is None:
                return None
            dp = copy(dp)   # avoid modifying the list
            dp[index] = r
            return dp
        super(MapDataComponent, self).__init__(ds, f)

