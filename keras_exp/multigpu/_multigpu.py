# MODIFIED. Inspiration taken from the ref link below.
# ref: https://raw.githubusercontent.com/kuza55/keras-extras/master/utils/multi_gpu.py @IgnorePep8
# The inspirational one carried license:
#     Apache License
#     Version 2.0, January 2004
# For further info refer to: https://github.com/kuza55/keras-extras
#
# Also used https://github.com/fchollet/keras/issues/2436 which was just
# posted as code snippets in a forum.
from __future__ import print_function

import sys
# import time

try:
    from cStringIO import StringIO
except ImportError:
    # Python 3 compat.
    from io import StringIO

from itertools import chain

from keras import backend as KB
from keras.layers.core import Lambda
from keras.models import Model
from keras.layers.merge import Concatenate  # , Average)
# import keras.layers as KL
import keras.optimizers as KO

if KB.backend() == 'tensorflow':
    # Monkey patch Keras back-end to use Function with enqueue.
    # import keras_exp._patch_tf_backend as tfbpatch
    # tfbpatch.patch()
    from keras_exp._patch_tf_backend import patch as tfbpatch
    tfbpatch()

    import tensorflow as tf
    from tensorflow.python.client import device_lib

    try:
        from tensorflow.contrib import nccl
        have_nccl = True
        print('NCCL support available', file=sys.stderr)
    except ImportError:
        have_nccl = False
        print('WARNING: NCCL support not available', file=sys.stderr)

    from tensorflow.python.ops import data_flow_ops


_DEBUG = False

__all__ = ('get_available_gpus', 'make_parallel', 'print_mgpu_modelsummary',
           'ModelMGPU')


# TODO: Move to some utils module
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout


def get_available_gpus(ngpus=-1):
    '''
    :param int ngpus: GPUs max to use. Default -1 means all gpus.
    :returns: List of gpu devices. Ex.: ['/gpu:0', '/gpu:1', ...]
    '''
    local_device_protos = device_lib.list_local_devices()
    gpus_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return gpus_list[:ngpus] if ngpus > -1 else gpus_list


def print_mgpu_modelsummary(model):
    '''Prints the summary for a multi-GPU keras model.

    :param model: Keras model.
    :type model: Model
    '''
    # print json.dumps(model.get_config(), indent=2)  # DEBUG
    print('\nMULTI-GPU MODEL: {}'.format(model.name))
    print(model.summary())
    for layer in model.layers:
        # print 'layer:', layer, '\ttype:', type(layer)
        if isinstance(layer, Model):
            submodel = layer
            print('\n\tSUBMODEL SUMMARY: {}'.format(layer.name))
            with Capturing() as msum:
                minfo = submodel.summary()
            print('\t{}\n\t{}\n'.format('\n\t'.join(msum), minfo))


def all_sync_params(tower_params, devices, usenccl=True):
    """Assigns the params from the first tower to all others"""
    if len(devices) == 1:
        return tf.no_op()
    sync_ops = []
    if have_nccl and usenccl:
        for param_on_devices in zip(*tower_params):
            # print('PARAM_ON_DEVICES: {}'.format(param_on_devices))  # DEBUG
            # Note: param_on_devices is [paramX_gpu0, paramX_gpu1, ...]
            param0 = param_on_devices[0]
            send_op, received_tensors = nccl.broadcast(param0, devices[1:])
            sync_ops.append(send_op)
            for device, param, received in zip(devices[1:],
                                               param_on_devices[1:],
                                               received_tensors):
                with tf.device(device):
                    sync_op = param.assign(received)
                    sync_ops.append(sync_op)
    else:
        params0 = tower_params[0]
        for device, params in zip(devices, tower_params):
            with tf.device(device):
                for param, param0 in zip(params, params0):
                    sync_op = param.assign(param0.read_value())
                    sync_ops.append(sync_op)

    return tf.group(*sync_ops)


# def stage(tensors):
#     """Stages the given tensors in a StagingArea for asynchronous put/get.
#     """
#     stage_area = data_flow_ops.StagingArea(
#         dtypes=[tensor.dtype for tensor in tensors],
#         shapes=[tensor.get_shape() for tensor in tensors])
#     put_op = stage_area.put(tensors)
#     get_tensors = stage_area.get()
#     if not isinstance(get_tensors, list):
#         get_tensors = [get_tensors]
#     # print('GET_TENSORS: {}'.format(get_tensors))  # DEBUG
#
#     get_tensors = [tf.reshape(gt, t.get_shape())
#                    for (gt, t) in zip(get_tensors, tensors)]
#     return put_op, get_tensors


class ModelMGPU(Model):
    '''Override load and save methods of the multi-gpu model. The load and
    save should correspond to the serial model's load and save.
    If there are other idiosyncracies to handle for multi-gpu model case then
    these can be handled in this subclass. A serial model should always be
    instantiated prior to wrapping it or converting it to a multi-GPU model.
    This multi-gpu implementation uses data-parallelism.

    A copy-constructor is not implemented so optionally pass any additional
    parameters besides inputs/outputs as args/kwargs to initialize the
    multi-gpu model the same way as the serial model. Typically not needed.

    Currently, it seems that using NCCL and synchronizing/averaging gradients
    slows multi-gpu processing down.

    .. seealso::
        Refer to :func:`make_parallel` docstring for scenarios when
        out-of-memory errors might occur and workaround.

    Kwargs:
    :param Model serial_model: Serial i.e. non-multi GPU Keras model. REQUIRED.

    :param list gdev_list: List of gpu devices i.e. ['/gpu:0', '/gpu:1', ...]
        Use function get_available_gpus to get the list of available gpus.
        This can be a list of strings or list of instances of tf.DeviceSpec.
        REQUIRED.

    :param str ps_device: Parameter server device to use.

    :param bool usenccl: Use the contrib.nccl Tensorflow library for initial
        parameter synchronization and gradients averaging. Note, the models
        usenccl option overrides the optimizers usenccl option.
        Default: False
        Raises RuntimeError if specified True and a non-multi-gpu optimizer is
        passed during compile stage.

    :param bool initsync: Synchronize initial Variables i.e. weights,
        biases, etc. Default: True

    :param bool syncopt: Synchronize gradients. Requires a multi-gpu optimizer.
        Default: False

    :param bool enqueue: Use StagingArea in the multi-GPU model. Could
        potentially speed up Host-to-Device transfers.
        Produces a warning that kwargs are ignored for Tensorflow. The
        _patch_tf_backend module mokey patches the Function in
        tensorflow_backend to use the enqueue_ops option.
        Default: False

    '''
    def __init__(self, *args, **kwargs):
        # :param model_creator: Callable that returns a serial i.e. non-multi
        #     GPU Keras model i.e. a keras.models.Model model. REQUIRED.
        #     Suggestion, use partial from functools to setup model_creator.
        # try:
        #     model_creator = kwargs.pop('model_creator')
        # except KeyError:
        #     raise RuntimeError('Keyword argument "model_creator" required '
        #                        'for ModelMGPU.')

        try:
            smodel = kwargs.pop('serial_model')
        except KeyError:
            raise RuntimeError('Keyword argument "serial_model" required '
                               'for ModelMGPU.')

        # SET STATE: Instance of serial model for checkpointing
        self._smodel = smodel  # model_creator()

        try:
            gdev_list = kwargs.pop('gdev_list')
        except KeyError:
            raise RuntimeError('Keyword argument "gdev_list" required '
                               'for ModelMGPU.')
        self._gdev_list = gdev_list

        mname = kwargs.pop('name', self._smodel.name)
        kwargs['name'] = mname

        self._ps_device = kwargs.pop('ps_device', '/cpu:0')
        self._initsync = kwargs.pop('initsync', True)
        self._usenccl = kwargs.pop('usenccl', False)
        self._syncopt = kwargs.pop('syncopt', False)
        self._enqueue = kwargs.pop('enqueue', False)

        # NOTE: To use staging have to patch keras tensorflow_backend.Function.
        #     Function implementation in keras_exp.multigpu._patch_tf_backend
        self._enqueue_ops = []

        self._tower_params = []  # For init/sync'ing of parameters.
        self._init_make_dataparallel(gdev_list, *args,
                                     **kwargs)

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

    # ref: https://github.com/fchollet/keras/issues/2436
    def _init_make_dataparallel(self, gdev_list, *args, **kwargs):
        '''Uses data-parallelism to convert a serial model to multi-gpu. Refer
        to make_parallel doc.
        '''
        gpucopy_ops = []

        def slice_batch(x, ngpus, part, dev):
            '''Divide the input batch into [ngpus] slices, and obtain slice
            no. [part]. i.e. if len(x)=10, then slice_batch(x, 2, 1) will
            return x[5:].
            '''
            sh = KB.shape(x)
            L = sh[0] // ngpus
            if part == ngpus - 1:
                xslice = x[part * L:]
            else:
                xslice = x[part * L:(part + 1) * L]

            # tf.split fails if batch size is not divisible by ngpus. Error:
            #     InvalidArgumentError (see above for traceback): Number of
            #         ways to split should evenly divide the split dimension
            # xslice = tf.split(x, ngpus)[part]

            if not self._enqueue:
                return xslice

            # Did not see any benefit.
            with tf.device(dev):
                # if self._stager is None:
                stager = data_flow_ops.StagingArea(
                    dtypes=[xslice.dtype], shapes=[xslice.shape])
                stage = stager.put([xslice])
                gpucopy_ops.append(stage)
                # xslice_stage = stager.get()
            return stager.get()

        ngpus = len(gdev_list)
        if ngpus < 2:
            raise RuntimeError('Number of gpus < 2. Require two or more GPUs '
                               'for multi-gpu model parallelization.')

        model_ = model = self._smodel
        global_scope = tf.get_variable_scope()
        towers = []
        for idev, dev in enumerate(gdev_list):
            # TODO: The last slice could cause a gradient calculation outlier
            # when averaging gradients. Maybe insure ahead of time that the
            # batch_size is evenly divisible by number of GPUs, or maybe don't
            # use the last slice.
            with tf.device(self._ps_device):
                slices = []  # multi-input case
                for ix, x in enumerate(model.inputs):
                    slice_g = Lambda(
                        slice_batch,  # lambda shape: shape,
                        # lambda shape: x.shape.as_list(),
                        name='stage_cpuSliceIn{}_Dev{}'.format(ix, idev),
                        arguments={'ngpus': ngpus, 'part': idev,
                                   'dev': dev})(x)
                    slices.append(slice_g)
                    # print('SLICE_G: {}'.format(slice_g))  # DEBUG
                # print('SLICES: {}'.format(slices))  # DEBUG

            # with tf.variable_scope('GPU_%i' % idev), \
            # tf.variable_scope(global_scope, reuse=idev > 0), \
            # tf.variable_scope('GPU_{}'.format(idev),
            #                   reuse=idev > 0) as var_scope, \
            with tf.device(dev), \
                    tf.variable_scope(global_scope, reuse=idev > 0), \
                    tf.name_scope('tower_%i' % idev):
                # NOTE: Currently not using model_creator. Did not observe
                #     any benefit in such an implementation.
                # Instantiate model under device context. More complicated.
                # Need to use optimizer synchronization in this scenario.
                # model_ = model_creator()
                # If using NCCL without re-instantiating the model then must
                # set the colocate_gradients_with_ops to False in optimizer.
                # if idev == 0:
                #     # SET STATE: Instance of serial model for checkpointing
                #     self._smodel = model_  # for ability to checkpoint

                modeltower = model_(slices)
                towers.append(modeltower)

                # params = model_.trainable_weights
                # params = tf.get_collection(
                #     tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope.name)
                params = modeltower.graph._collections['trainable_variables']
                # print('PARAMS: {}'.format(params))  # DEBUG

                self._tower_params.append(params)

        with tf.device(self._ps_device):
            merged = Concatenate(axis=0)(towers)
            # print('MERGED: {}'.format(merged))  # DEBUG

        # self._enqueue_ops.append(tf.group(*gpucopy_ops))
        self._enqueue_ops += gpucopy_ops

        kwargs['inputs'] = model.inputs
        kwargs['outputs'] = merged
        super(ModelMGPU, self).__init__(*args, **kwargs)

    def compile(self, *args, **kwargs):
        '''Refer to Model.compile docstring for parameters. Override
        functionality is documented below.

        :override compile: Override Model.compile method to check for options
            that the optimizer is multi-gpu enabled, and synchronize initial
            variables.
        '''
        initsync = self._initsync
        usenccl = self._usenccl

        opt = kwargs['optimizer']
        # if isinstance(opt, str):
        if not isinstance(opt, KO.Optimizer):
            opt = KO.get(opt)
            kwargs['optimizer'] = opt

        if self._syncopt and not getattr(opt, 'ismgpu', False):
            raise RuntimeError(
                'Multi-GPU synchronization model requires a multi-GPU '
                'optimizer. Instead got: {}'.format(opt))

        opt.usenccl = usenccl

        if self._enqueue_ops:
            # Produces a warning that kwargs are ignored for Tensorflow. Patch
            # Function in tensorflow_backend to use the enqueue_ops option.
            kwargs['enqueue_ops'] = self._enqueue_ops

        super(ModelMGPU, self).compile(*args, **kwargs)

        if initsync:
            self._run_initsync()

    def _run_initsync(self):
        # tparams = [list(chain(*tp)) for tp in self._tower_params]
        tparams = self._tower_params

        # Check to prevent from unnecessarily re-initializing and
        # synchronizing, i.e. when the model loads the weights.
        for v in chain.from_iterable(tparams):
            if getattr(v, '_keras_initialized', False):
                return

        KB.manual_variable_initialization(True)
        sess = KB.get_session()
        KB.manual_variable_initialization(False)

        # glob_variables = tf.global_variables()
        # sess.run(tf.variables_initializer(glob_variables))

        # Initialize on GPU0 and sync to other GPUs
        init_op = tf.variables_initializer(tparams[0])
        # init_op = tf.variables_initializer(self._tower_params[0])
        # init_op = tf.variables_initializer(self.trainable_weights)
        sess.run(init_op)

        # Important if using model_creator. Not necessary of model instance is
        # reused in which case the model layers are shared between slices
        # and are automatically sync'd.
        sync_op = all_sync_params(tparams, self._gdev_list,
                                  usenccl=self._usenccl)
        sess.run(sync_op)

        for v in chain.from_iterable(tparams):
            v._keras_initialized = True


# Data-parallel ref: https://github.com/fchollet/keras/issues/2436
# Tower-parallel:
# ref: https://medium.com/autonomous-agents/multi-gpu-training-of-large-sparse-matrix-on-wide-neuralnetwork-cac7afc52ffe @IgnorePep8
# ref: https://gist.github.com/vvpreetham/1379cc4e208ea33ce3e615067e92fc5e
def make_parallel(serial_model, gdev_list, ps_device='/cpu:0', usenccl=False,
                  initsync=True, syncopt=False, enqueue=False,
                  model_class=ModelMGPU):
    '''Given a keras model, return an equivalent model which parallelizes
    the computation over multiple GPUs listed in the gdev_list.

    Data-Parallel:
    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor,
    hence the user sees a model that behaves the same as the original.

    If getting an out-of-memory (OOM) error when scaling the batch size by the
    number of GPUs, there might be input layer(s) in the serial model that runs
    additional special operations (such as tranformation of some sort) on the
    1st GPU as enumerated by Tensorflow. This was an observed behavior for
    Embedding layers. The workaround is to pin such layers to the CPU, or
    simply pin the instantiation of the serial mode to CPU. The parallelization
    will move the operations to GPU.

    :Example:

        if mgpu_flag:
            with tf.device('/cpu:0'):
                # define the serial model.
                model_serial = get_model_serial()

            gdev_list = get_available_gpus()
            model = make_parallel(model_serial, gdev_list)
        else:
            model = def_model_serial()

    :param Model serial_model: Serial i.e. non-multi GPU Keras model.

    :param list gdev_list: List of gpu devices i.e. ['/gpu:0', '/gpu:1', ...]
        Use function get_available_gpus to get the list of available gpus.
        This can be a list of strings or list of instances of tf.DeviceSpec.

    :param str ps_device: Parameter server device to use.

    :param bool usenccl: Use the contrib.nccl Tensorflow library for initial
        parameter synchronization and gradients averaging. Note, the model's
        usenccl option overrides the optimizers usenccl option.
        Default: False

    :param bool initsync: Synchronize initial Variables i.e. weights,
        biases, etc. Default: True

    :param bool syncopt: Synchronize gradients. Requires a multi-gpu optimizer.
        Default: False

    :param bool enqueue: Use StagingArea in the multi-GPU model. Could
        potentially speed up Host-to-Device transfers.
        Produces a warning that kwargs are ignored for Tensorflow. The
        _patch_tf_backend module mokey patches the Function in
        tensorflow_backend to use the enqueue_ops option.
        Default: False

    :param model_class: Class object to instantiate for multi-gpu models. This
        is needed when the ModelMGPU is mixed-in with other classes.
        Default: ModelMGPU

    :returns: Multi-GPU parallelized model. If ngpus < 2 then do nothing and
        return the provided serial_model.
    :rtype: ModelMGPU
    '''
    ngpus = len(gdev_list)
    if ngpus < 2:
        return serial_model  # model_creator()

    return model_class(
        serial_model=serial_model, gdev_list=gdev_list,
        ps_device=ps_device,
        enqueue=enqueue, usenccl=usenccl,
        initsync=initsync, syncopt=syncopt)

