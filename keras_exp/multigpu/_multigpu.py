# MODIFIED. Inspiration taken from the ref link below.
# ref: https://raw.githubusercontent.com/kuza55/keras-extras/master/utils/multi_gpu.py @IgnorePep8
# The inspirational onw carried license:
#     Apache License
#     Version 2.0, January 2004
# For further info refer to: https://github.com/kuza55/keras-extras

# Also used https://github.com/fchollet/keras/issues/2436 which was just
# posted as a code snippet in a forum.

from cStringIO import StringIO
import sys

from keras import backend as K
from keras.layers.core import Lambda
from keras.models import Model
from keras.layers.merge import (Concatenate, Average)

if K.backend() == 'tensorflow':
    import tensorflow as tf  # @UnresolvedImport
    from tensorflow.python.client import device_lib  # @UnresolvedImport


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
    print '\nMULTI-GPU MODEL: {}'.format(model.name)
    print(model.summary())
    for layer in model.layers:
        # print 'layer:', layer, '\ttype:', type(layer)
        if isinstance(layer, Model):
            submodel = layer
            print '\n\tSUBMODEL SUMMARY: {}'.format(layer.name)
            with Capturing() as msum:
                minfo = submodel.summary()
            print('\t{}\n\t{}\n'.format('\n\t'.join(msum), minfo))


# Data-parallel ref: https://github.com/fchollet/keras/issues/2436
# Tower-parallel:
# ref: https://medium.com/autonomous-agents/multi-gpu-training-of-large-sparse-matrix-on-wide-neuralnetwork-cac7afc52ffe @IgnorePep8
# ref: https://gist.github.com/vvpreetham/1379cc4e208ea33ce3e615067e92fc5e
def make_parallel(model, gdev_list, partype='dp'):
    '''Given a keras [model], return an equivalent model which parallelizes
    the computation over [n_gpus] GPUs.

    Data-Parallel: partype == "dp"
    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor,
    hence the user sees a model that behaves the same as the original.

    Tower-Parallel: partype == "tp"
    Load the data, cost functions and gradients and regularizers all on the
    same GPU. This allows one to run an exact replica of the entire model on
    multiple GPUs and then aggregate the results using a mixing routine or a
    simple mean.

    :param Model model: Serial i.e. non-multi GPU Keras model.

    :param list gdev_list: List of gpu devices i.e. ['/gpu:0', '/gpu:1', ...]
        Use function get_available_gpus to get the list of available gpus.

    :param str partype: Parallelism type. OPTIONAL. Types supported:
        "dp" - data-parallel (default)
        "tp" - tower-parallel

    :returns: Multi-GPU parallelized model. If ngpus < 2 then return in model.
    :rtype: ModelMGPU
    '''
    n_gpus = len(gdev_list)
    if n_gpus < 2:
        return model

    return ModelMGPU(serial_model=model, gdev_list=gdev_list, partype=partype)


class ModelMGPU(Model):
    '''Override load and save methods of the multi-gpu model. The load and
    save should correspond to the serial model's load and save.
    If there are other idiosyncracies to handle for multi-gpu model case then
    these can be handled in this subclass. A serial model should always be
    instantiated prior to wrapping it or converting it to a multi-GPU model.
    This multi-gpu implementation uses data-parallelism.

    A copy-constructor is not implemented so optionally pass any additional
    parameters besides inputes/outputs as args/kwargs to initialize the
    multi-gpu model the same way as the serial model. Typically not needed.

    Kwargs:
    :param Model serial_model: Serial i.e. non-multi GPU Keras model. REQUIRED.

    :param list gdev_list: List of gpu devices i.e. ['/gpu:0', '/gpu:1', ...]
        Use function get_available_gpus to get the list of available gpus.
        REQUIRED.

    :param str partype: Parallelism type. OPTIONAL. Types supported:
        "dp" - data-parallel (default)
        "tp" - tower-parallel

    '''
    def __init__(self, *args, **kwargs):
        try:
            smodel = kwargs.pop('serial_model')
        except KeyError:
            raise RuntimeError('Keyword argument "serial_model" required '
                               'for ModelMGPU.')

        self._smodel = smodel

        try:
            gdev_list = kwargs.pop('gdev_list')
        except KeyError:
            raise RuntimeError('Keyword argument "gdev_list" required '
                               'for ModelMGPU.')

        partype = kwargs.pop('partype', 'dp')

        mname = kwargs.pop('name', smodel.name)
        if partype == 'tp':
            mname = '{}-TowerParallel'.format(mname)
        else:
            mname = '{}-DataParallel'.format(mname)
        kwargs['name'] = mname

        if partype == 'tp':
            self._init_make_towerparallel(smodel, gdev_list, *args, **kwargs)
        else:
            self._init_make_dataparallel(smodel, gdev_list, *args, **kwargs)

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

    # ref: https://github.com/fchollet/keras/issues/2436
    def _init_make_dataparallel(self, model, gdev_list, *args, **kwargs):
        '''Uses data-parallelism to convert a serial model to multi-gpu. Refer
        to make_parallel doc.
        '''
        def slice_batch(x, n_gpus, part):
            '''Divide the input batch into [n_gpus] slices, and obtain slice
            no. [part]. i.e. if len(x)=10, then slice_batch(x, 2, 1) will
            return x[5:].
            '''
            sh = K.shape(x)
            L = sh[0] / n_gpus
            if part == n_gpus - 1:
                return x[part * L:]
            return x[part * L:(part + 1) * L]

        n_gpus = len(gdev_list)
        if n_gpus < 2:
            raise RuntimeError('Number of gpus < 2. Require two or more GPUs '
                               'for multi-gpu model parallelization.')

        towers = []
        for idev in range(n_gpus):
            with tf.device(gdev_list[idev]):
                for x in model.inputs:
                    slice_g = Lambda(
                        slice_batch, lambda shape: shape,
                        arguments={'n_gpus': n_gpus, 'part': idev})(x)
                    towers.append(model(slice_g))

        with tf.device('/cpu:0'):
            merged = Concatenate(axis=0)(towers)

        kwargs['inputs'] = model.inputs
        kwargs['outputs'] = merged
        # Model.__init__(self, inputs=model.inputs, outputs=merged)
        super(ModelMGPU, self).__init__(*args, **kwargs)

    # ref: https://medium.com/autonomous-agents/multi-gpu-training-of-large-sparse-matrix-on-wide-neuralnetwork-cac7afc52ffe @IgnorePep8
    def _init_make_towerparallel(self, model, gdev_list, *args, **kwargs):
        '''Uses tower-parallelism to convert a serial model to multi-gpu. Refer
        to make_parallel doc.
        '''
        def slice_batch(x):
            return x[:]

        n_gpus = len(gdev_list)
        towers = []
        for idev in range(n_gpus):
            with tf.device(gdev_list[idev]):
                for x in model.inputs:
                    slice_g = Lambda(slice_batch, lambda shape: shape)(x)
                    towers.append(model(slice_g))

        merged = []
        with tf.device('/cpu:0'):
            merged.append(Average()(towers))

        kwargs['inputs'] = model.inputs
        kwargs['outputs'] = merged
        # Model.__init__(self, inputs=model.inputs, outputs=merged)
        super(ModelMGPU, self).__init__(*args, **kwargs)

