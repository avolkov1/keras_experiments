'''
'''
from __future__ import print_function
import sys

from keras import backend as K
from keras.optimizers import (
    clip_norm, Optimizer,
    Adagrad, Adadelta, Adam, Adamax, Nadam, RMSprop, SGD)

from keras_exp._mixin_common import mixedomatic

_DEBUG = False
if _DEBUG:
    # import traceback
    pass

if K.backend() == 'tensorflow':
    import tensorflow as tf

    try:
        from tensorflow.contrib import nccl
        have_nccl = True
        print('NCCL support available', file=sys.stderr)
    except ImportError:
        have_nccl = False
        print('WARNING: NCCL support not available', file=sys.stderr)


__all__ = (
    'OptimizerMultiGPUMixin',
    'AdagradMGPU', 'AdadeltaMGPU', 'AdamMGPU', 'AdamaxMGPU', 'NadamMGPU',
    'RMSPropMGPU', 'SGD_MGPU', )


def all_avg_gradients(tower_gradvars, devices, param_server_device='/gpu:0',
                      usenccl=True):
    if len(devices) == 1:
        return tower_gradvars

    num_devices = len(devices)
    avg_gradvars = []
    for layer in zip(*tower_gradvars):
        grads_on_devices, vars_on_devices = zip(*layer)
        if have_nccl and usenccl:
            # Note: These nccl ops _must_ be run on all devices, else deadlock
            # print('ALL_AVG_GRADIENTS GRADS_ON_DEVICES:',
            #       grads_on_devices)  # DEBUG
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
    # :param baseopt: A base class keras optimizer such as SGD, RMSprop,...
    def __init__(self, gdev_list=None, usenccl=True):
        '''
        :param list gdev_list: List of gpu devices i.e.
            ['/gpu:0', '/gpu:1', ...]. Use function get_available_gpus to get
            the list of available gpus.

    :param bool usenccl: Use the contrib.nccl Tensorflow library for gradients
        averaging. Note, the models usenccl option overrides the optimizers
        usenccl option during model compile stage.
        '''
        if len(self.__class__.__bases__) < 2 or \
                not isinstance(self, Optimizer):
            raise RuntimeError(
                'A Keras Optimizer derived class required for mixin: {}.\nUse '
                'multiple inheritance. Ex.:\n{}'.format(
                    'OptimizerMultiGPUMixin',
                    '    @mixedomatic(ignore_kargs_spec=True)\n'
                    '    class RMSPropMGPU(OptimizerMultiGPUMixin, RMSprop):\n'
                    '        pass\n'
                ))

        baseopt = super(OptimizerMultiGPUMixin, self)
        # baseopt = self.__class__.__bases__[-1]
        self._baseopt = baseopt

        self._gdev_list = gdev_list
        # This mixin class works fine for 1-gpu case.
        # ngpus = len(gdev_list)
        # if ngpus < 2:
        #     err_msg = 'Multi-GPU requires more than one gpu devices.'
        #     raise RuntimeError(err_msg)

        self.__idev = 0  # SET STATE: DEVICE
        self._tower_gradvars = None
        self._usenccl = usenccl

    @property
    def ismgpu(self):
        return True

    @property
    def usenccl(self):
        return self._usenccl

    @usenccl.setter
    def usenccl(self, usenccl):
        self._usenccl = usenccl

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

        # tower parallelization
        global_scope = tf.get_variable_scope()
        tower_gradvars = []
        for idev, device in enumerate(gdev_list):
            # tf.variable_scope('GPU_%i' % idev), \
            with tf.device(device), \
                    tf.variable_scope(global_scope, reuse=idev > 0), \
                    tf.name_scope('tower_%i' % idev):
                # tf.gradients returns list of `sum(dy/dx)`. The gradients
                # are aggregated by all_avg_gradients. Something doesn't seem
                # right though. SOMEWHAT SLOW.
                # TODO: Need to figure out how to efficiently aggregate.
                colo = True if not self._usenccl else not have_nccl
                # colo = True
                grads = tf.gradients(
                    loss, params,
                    # # GATE_NONE faster??
                    # gate_gradients=tf.train.Optimizer.GATE_NONE,
                    colocate_gradients_with_ops=colo)  # not have_nccl

                if hasattr(self, 'clipnorm') and self.clipnorm > 0:
                    norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
                    grads = [clip_norm(g, self.clipnorm, norm) for g in grads]

                if hasattr(self, 'clipvalue') and self.clipvalue > 0:
                    grads = [K.clip(g, -self.clipvalue, self.clipvalue)
                             for g in grads]

                gradvars = zip(grads, params)
                tower_gradvars.append(gradvars)

        tower_gradvars = all_avg_gradients(tower_gradvars, gdev_list,
                                           usenccl=self._usenccl)

        return tower_gradvars

    def get_gradients(self, loss, params):
        '''
        :override get_gradients: Overrides the base Optimizer class/sub-class
            get_gradients method to get gradients from tower grads.
        '''
        # READ STATE: TOWER GRADS
        tower_gradvars = self._tower_gradvars \
            if self._tower_gradvars is not None else \
            self._get_tower_gradvars(loss, params)
        idev = self._device  # READ STATE: DEVICE
        grads = [tg[0] for tg in tower_gradvars[idev]]

        if _DEBUG:
            # traceback.print_stack()  # DEBUG
            print('\nOptimizerMultiGPUMixin grads: {}'.format(grads))  # DEBUG

        return grads

    def get_updates(self, params, constraints, loss):
        '''
        :override get_updates: Overrides the base Optimizer class/sub-class
            get_updates method to optionally use nccl for gradient aggregation.
        '''
        tower_gradvars = self._get_tower_gradvars(loss, params)
        self._tower_gradvars = tower_gradvars  # SET STATE: TOWER GRADS

        gdev_list = self._gdev_list
        # ngpus = len(gdev_list)

        global_scope = tf.get_variable_scope()
        updates = []
        # IMPORTANT when using NCCL to get updates for all devices otherwise
        # the nccl ops deadlock. Hence the loop below over all gpus.
        for idev, dev in enumerate(gdev_list):
            # Clear internal updates state. Aggregated and set after for-loop
            self.updates = []
            self._device = idev  # SET STATE: DEVICE
            # The self._baseopt.get_updates calls get_gradients method.
            # The self._device state is set and the get_gradients uses this
            # state to return the gradients for that device.
            with tf.device(dev), \
                    tf.variable_scope(global_scope, reuse=idev > 0), \
                    tf.name_scope('tower_%i' % idev):
                # updates_ = self._baseopt.get_updates(self, params,
                #                                      constraints, loss)
                updates_ = self._baseopt.get_updates(params, constraints,
                                                     loss)
            updates += [up for up in updates_ if up not in updates]

            if (not have_nccl or not self.usenccl) and idev == 0:
                # no need to iterate over all devices
                break

        self._device = 0  # SET STATE: DEVICE
        self.updates = updates
        # if _DEBUG:
        #     print 'UPDATES:', _updates  # DEBUG

        return self.updates

# Note: The code used in keras Optims is in bad style.
#     for k in kwargs:
#         if k not in allowed_kwargs:
#             raise TypeError('Unexpected keyword argument '
#                             'passed to optimizer: ' + str(k))
# List the allowed kwargs in __init__ and don't use **kwargs if the intention
# is not to allow/passthru of uknown kwargs.
# Current workaround is to add ignore_kargs_spec=True to the decorator.

# Implementation without mixedomatic
# class RMSPropMGPU(OptimizerMultiGPUMixin, RMSprop):
#     def __init__(self, **kwargs):
#         gdev_list = kwargs.pop('gdev_list', [])
#         RMSprop.__init__(self, **kwargs)
#         OptimizerMultiGPUMixin.__init__(self, gdev_list=gdev_list)


@mixedomatic(ignore_kargs_spec=True)
class AdagradMGPU(OptimizerMultiGPUMixin, Adagrad):
    pass


@mixedomatic(ignore_kargs_spec=True)
class AdadeltaMGPU(OptimizerMultiGPUMixin, Adadelta):
    pass


@mixedomatic(ignore_kargs_spec=True)
class AdamMGPU(OptimizerMultiGPUMixin, Adam):
    pass


@mixedomatic(ignore_kargs_spec=True)
class AdamaxMGPU(OptimizerMultiGPUMixin, Adamax):
    pass


@mixedomatic(ignore_kargs_spec=True)
class NadamMGPU(OptimizerMultiGPUMixin, Nadam):
    pass


@mixedomatic(ignore_kargs_spec=True)
class RMSPropMGPU(OptimizerMultiGPUMixin, RMSprop):
    pass


@mixedomatic(ignore_kargs_spec=True)
class SGD_MGPU(OptimizerMultiGPUMixin, SGD):
    pass

