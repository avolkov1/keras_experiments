'''
MODIFIED. Inspiration taken from the ref link below.
ref: https://raw.githubusercontent.com/kuza55/keras-extras/master/utils/multi_gpu.py
The inspirational one carried license:
    Apache License
    Version 2.0, January 2004
For further info refer to: https://github.com/kuza55/keras-extras

Also used https://github.com/fchollet/keras/issues/2436 which was just
posted as code snippets in a forum.
'''  # noqa
from __future__ import print_function

from keras import backend as KB
from keras.layers.core import Lambda
from keras.models import Model
from keras.layers.merge import Concatenate
from keras.utils import multi_gpu_model

from keras_exp._utils import Capturing


if KB.backend() == 'tensorflow':
    # Monkey patch Keras back-end to use Function with enqueue.
    # import keras_exp._patch_tf_backend as tfbpatch
    # tfbpatch.patch()
    # from keras_exp._patch_tf_backend import patch as tfbpatch
    # tfbpatch()

    import tensorflow as tf
    from tensorflow.python.client import device_lib  # @IgnorePep8 pylint: disable=no-name-in-module

_DEBUG = False

__all__ = (
    'GPUListType', 'get_available_gpus', 'make_parallel',
    'print_mgpu_modelsummary', 'ModelKerasMGPU', 'ModelMGPU')


class GPUListType(object):  # pylint: disable=too-few-public-methods
    '''List type for the function get_available_gpus.'''
    name_str = 'name_str'
    int_id = 'int_id'
    dspec = 'dspec'


def get_available_gpus(ngpus=-1, list_type=GPUListType.name_str):
    '''
    :param ngpus: GPUs max to use. Default -1 means all gpus.
    :returns: List of gpu devices. Ex.: ['/gpu:0', '/gpu:1', ...]
    '''
    if ngpus == 0:
        return []

    local_device_protos = device_lib.list_local_devices()
    # gpus_list = [x.name for x in local_device_protos
    #              if x.device_type == 'GPU']
    gpus_list = []
    for dproto in local_device_protos:
        if dproto.device_type != 'GPU':
            continue

        gdev_spec = tf.DeviceSpec.from_string(dproto.name)

        if list_type == GPUListType.name_str:
            gdev = gdev_spec.to_string()

        elif list_type == GPUListType.int_id:
            gdev = gdev_spec.device_index

        elif list_type == GPUListType.dspec:
            gdev = gdev_spec

        else:
            gdev = gdev_spec.to_string()  # default

        gpus_list.append(gdev)

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


class ModelKerasMGPU(Model):
    '''Wrapper class around "keras.utils.multi_gpu_model". This class enables
    loading and saving with multigpu model object.

    Only available with the TensorFlow backend
    for the time being.

    # Arguments
        ser_model: A Keras model instance. To avoid OOM errors,
            this model could have been built on CPU, for instance
            (see usage example below).
        gpus: Integer >= 2 or list of integers, number of GPUs or
            list of GPU IDs on which to create model replicas.
        cpu_merge: A boolean value to identify whether to force
            merging model weights under the scope of the CPU or not.
        cpu_relocation: A boolean value to identify whether to
            create the model's weights under the scope of the CPU.
            If the model is not defined under any preceding device
            scope, you can still rescue it by activating this option.

    # Returns
        A Keras `Model` instance which can be used just like the initial
        `model` argument, but which distributes its workload on multiple GPUs.

    '''
    def __init__(self, ser_model, gpus, *args, **kwargs):  # @IgnorePep8 pylint: disable=super-init-not-called
        pmodel = multi_gpu_model(ser_model, gpus, *args, **kwargs)
        # mimic copy constructor via __dict__ update, hence no super-init
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelKerasMGPU, self).__getattribute__(attrname)


class ModelMGPU(Model):
    '''This implementation is almost identical to keras.utils.multi_gpu_model

    Override load and save methods of the multi-gpu model. The load and
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
        super(ModelMGPU, self).__init__()

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

        kwargs_ = self._init_make_dataparallel(gdev_list, **kwargs)
        super(ModelMGPU, self).__init__(*args, **kwargs_)

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

    # ref: https://github.com/fchollet/keras/issues/2436
    def _init_make_dataparallel(self, gdev_list, **kwargs):
        '''Uses data-parallelism to convert a serial model to multi-gpu. Refer
        to make_parallel doc.
        '''
        def slice_batch(xin, ngpus, part):
            '''Divide the input batch into [ngpus] slices, and obtain slice
            no. [part]. i.e. if len(x)=10, then slice_batch(x, 2, 1) will
            return x[5:].
            '''
            sh = KB.shape(xin)
            sub_batch_size = sh[0] // ngpus
            if part == ngpus - 1:
                xslice = xin[part * sub_batch_size:]
            else:
                xslice = xin[part * sub_batch_size:(part + 1) * sub_batch_size]

            return xslice

        ngpus = len(gdev_list)
        if ngpus < 2:
            raise RuntimeError('Number of gpus < 2. Require two or more GPUs '
                               'for multi-gpu model parallelization.')

        model = self._smodel
        noutputs = len(self._smodel.outputs)
        global_scope = tf.get_variable_scope()
        towers = [[] for _ in range(noutputs)]
        for idev, dev in enumerate(gdev_list):
            # TODO: The last slice could cause a gradient calculation outlier
            # when averaging gradients. Maybe insure ahead of time that the
            # batch_size is evenly divisible by number of GPUs, or maybe don't
            # use the last slice.
            with tf.device(self._ps_device):
                slices = []  # multi-input case
                for ix, xin in enumerate(model.inputs):
                    slice_g = Lambda(
                        slice_batch,  # lambda shape: shape,
                        # lambda shape: x.shape.as_list(),
                        name='stage_cpuSliceIn{}_Dev{}'.format(ix, idev),
                        arguments={'ngpus': ngpus, 'part': idev}
                    )(xin)
                    slices.append(slice_g)
                    # print('SLICE_G: {}'.format(slice_g))  # DEBUG
                # print('SLICES: {}'.format(slices))  # DEBUG

            with tf.device(dev), \
                    tf.variable_scope(global_scope, reuse=idev > 0), \
                    tf.name_scope('tower_%i' % idev):
                # Handle multi-output case
                modeltower = model(slices)
                if not isinstance(modeltower, list):
                    modeltower = [modeltower]

                for imt, mt in enumerate(modeltower):
                    towers[imt].append(mt)

        with tf.device(self._ps_device):
            # Tower list for each output.
            merged = [Concatenate(axis=0)(tw) for tw in towers]

        kwargs['inputs'] = model.inputs
        kwargs['outputs'] = merged

        return kwargs


# Data-parallel ref: https://github.com/fchollet/keras/issues/2436
# Tower-parallel:
# ref: https://medium.com/autonomous-agents/multi-gpu-training-of-large-sparse-matrix-on-wide-neuralnetwork-cac7afc52ffe @IgnorePep8
# ref: https://gist.github.com/vvpreetham/1379cc4e208ea33ce3e615067e92fc5e
def make_parallel(serial_model, gdev_list, ps_device='/cpu:0',
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
        serial_model=serial_model, gdev_list=gdev_list, ps_device=ps_device)
