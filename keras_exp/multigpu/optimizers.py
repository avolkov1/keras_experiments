'''
'''

from keras import backend as K
from keras.optimizers import (clip_norm, SGD, RMSprop)

_DEBUG = False
if _DEBUG:
    import traceback

if K.backend() == 'tensorflow':
    import tensorflow as tf  # @UnresolvedImport

    try:
        from tensorflow.contrib import nccl
        have_nccl = True
        print "NCCL support available"
    except ImportError:
        have_nccl = False
        print "WARNING: NCCL support not available"


__all__ = ('OptimizerMultiGPUMixin', 'SGD_MGPU', 'RMSPropMGPU',)


def all_avg_gradients(tower_gradvars, devices, param_server_device='/gpu:0'):
    if len(devices) == 1:
        return tower_gradvars

    num_devices = len(devices)
    avg_gradvars = []
    for layer in zip(*tower_gradvars):
        grads_on_devices, vars_on_devices = zip(*layer)
        if have_nccl:
            # Note: These nccl ops _must_ be run on all devices, else deadlock
            avg_grads_on_devices = nccl.all_sum(grads_on_devices)
            for d, device in enumerate(devices):
                with tf.device(device):
                    avg_grads_on_devices[d] *= 1. / num_devices
        else:
            with tf.device(param_server_device):
                avg_grad = tf.reduce_mean(tf.stack(grads_on_devices), 0)
            avg_grads_on_devices = [avg_grad] * num_devices
        avg_gradvars_on_devices = zip(*(avg_grads_on_devices, vars_on_devices))
        avg_gradvars.append(avg_gradvars_on_devices)

    return list(zip(*avg_gradvars))


class OptimizerMultiGPUMixin(object):
    '''
    Refer to classes below (such a SGD_MGPU) for an example of how to use
    this mixin.
    '''
    def __init__(self, gdev_list=None, baseopt=None):
        '''
        :param list gdev_list: List of gpu devices i.e.
            ['/gpu:0', '/gpu:1', ...]. Use function get_available_gpus to get
            the list of available gpus.
        :param baseopt: A base class keras optimizer such as SGD, RMSprop,...
        '''
        if baseopt is None:
            raise ValueError('Optimizer Base class required for mixin: {}'
                             .format('OptimizerMultiGPUMixin'))
        self._gdev_list = gdev_list

        # This mixin class works fine for 1-gpu case.
        # ngpus = len(gdev_list)
        # if ngpus < 2:
        #     err_msg = 'Multi-GPU requires more than one gpu devices.'
        #     raise RuntimeError(err_msg)

        self._baseopt = baseopt
        self.__idev = 0  # SET STATE
        self._tower_gradvars = None

    @property
    def _device(self):
        '''Device state currently used within get_gradients. This is a
        protected/private property so use it as such i.e. an implementation
        detail not a public property or interface.'''
        return self.__idev

    @_device.setter
    def _device(self, device):
        self.__idev = device

    def _get_tower_gradvars(self, loss, params):
        gdev_list = self._gdev_list
        tower_gradvars = []

        # tower parallelization

        for idev, device in enumerate(gdev_list):
            with tf.device(device), tf.variable_scope('GPU_%i' % idev), \
                    tf.name_scope('tower_%i' % idev):
                # tf.gradients returns list of `sum(dy/dx)`. The gradients
                # are aggregated by all_avg_gradients. Something doesn't seem
                # right though. VERY SLOW.
                # TODO: Need to figure out how to efficiently aggregate.
                grads = tf.gradients(
                    loss, params,
                    colocate_gradients_with_ops=not have_nccl)

                if hasattr(self, 'clipnorm') and self.clipnorm > 0:
                    norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
                    grads = [clip_norm(g, self.clipnorm, norm) for g in grads]

                if hasattr(self, 'clipvalue') and self.clipvalue > 0:
                    grads = [K.clip(g, -self.clipvalue, self.clipvalue)
                             for g in grads]

                gradvars = zip(grads, params)
                tower_gradvars.append(gradvars)

        tower_gradvars = all_avg_gradients(tower_gradvars, gdev_list)

        return tower_gradvars

    def get_gradients(self, loss, params):
        tower_gradvars = self._tower_gradvars \
            if self._tower_gradvars is not None else \
            self._get_tower_gradvars(loss, params)
        idev = self._device
        grads = [tg[0] for tg in tower_gradvars[idev]]

        if _DEBUG:
            # traceback.print_stack()  # DEBUG
            print '\nOptimizerMultiGPUMixin grads:', grads  # DEBUG

        return grads

    def get_updates(self, params, constraints, loss):
        if _DEBUG:
            traceback.print_stack()  # DEBUG
            print 'OptimizerMultiGPUMixin get_updates loss: {}'\
                .format(loss)  # DEBUG
            print 'OptimizerMultiGPUMixin get_updates type(loss): {}'\
                .format(type(loss))  # DEBUG

        tower_gradvars = self._get_tower_gradvars(loss, params)
        self._tower_gradvars = tower_gradvars

        _updates = []

        ngpus = len(self._gdev_list)
        # IMPORTANT when using NCCL to do get updates for all devices otherwise
        # the nccl ops deadlock. Hence the loop below over all gpus.
        for device_num in range(ngpus):
            self._device = device_num  # SET STATE
            # The self._baseopt.get_updates calls get_gradients method.
            # The self._device state is set and the get_gradients uses this
            # state to return the gradients for that device.
            updates_ = self._baseopt.get_updates(self, params, constraints,
                                                 loss)
            _updates += [up for up in updates_ if up not in _updates]

        self._device = 0  # SET STATE
        self.updates = _updates
        # if _DEBUG:
        #     print 'UPDATES:', _updates  # DEBUG

        return self.updates

# Note: Would like to use the mixedomatic decorator for the derived classes
# using OptimizerMultiGPUMixin, but cannot because of kwargs not being allowed
# in the base Optimized class. The code used in keras Optims is in bad style.
#     for k in kwargs:
#         if k not in allowed_kwargs:
#             raise TypeError('Unexpected keyword argument '
#                             'passed to optimizer: ' + str(k))
# List the allowed kwargs in __init__ and don't use **kwargs if the intention
# is not to allow/passthru of uknown kwargs.


class SGD_MGPU(OptimizerMultiGPUMixin, SGD):
    def __init__(self, **kwargs):
        gdev_list = kwargs.pop('gdev_list', [])
        SGD.__init__(self, **kwargs)
        OptimizerMultiGPUMixin.__init__(self, gdev_list=gdev_list,
                                        baseopt=SGD)


class RMSPropMGPU(OptimizerMultiGPUMixin, RMSprop):
    def __init__(self, **kwargs):
        gdev_list = kwargs.pop('gdev_list', [])
        RMSprop.__init__(self, **kwargs)
        OptimizerMultiGPUMixin.__init__(self, gdev_list=gdev_list,
                                        baseopt=RMSprop)

# TODO: Extend as above for any other desired optimizer.

