'''
Run VGG19 on Imagenet data.

Generate TFrecords for Imagenet data by following instructions in:
    examples/build_imagenet_data/README.md

Contrived example to demonstrate model-parallelism.

Example run 8 GPU machine with 4 GPUs per socket:

# 1 GPU per rank per model using 8 ranks.  8 x 1 = 8 GPUs
TMPDIR=/tmp mpirun --report-bindings --map-by ppr:4:socket -oversubscribe -np 8 \
    python ./examples/resnet/vgg_tfrecord_horovod.py  --imgs_per_epoch=6400 \
    --batch_size=128 --ngpus_per_model=1  # OOM on Tesla P100-SXM2-16GB

# 2 GPUs per rank per model using 4 ranks. 4 x 2 = 8 GPUs
TMPDIR=/tmp mpirun --report-bindings --map-by ppr:2:socket -oversubscribe -np 4 \
    python ./examples/resnet/vgg_tfrecord_horovod.py  --imgs_per_epoch=6400 \
    --batch_size=128 --ngpus_per_model=2  # runs on Tesla P100-SXM2-16GB

Use option --datadir to specify path to directory. Default:
    --datadir=/datasets/imagenet/train-val-tfrecord-480-subset

'''  # noqa
from __future__ import print_function
import sys

import argparse as ap
from textwrap import dedent

import time
import warnings

import tensorflow as tf
import horovod.tensorflow as hvd
import horovod.keras as hvd_keras

import keras.backend as KB
import keras.optimizers as KO
import keras.layers as KL
# from keras.models import Model
# from keras import backend as KB
# from keras.layers import Input
# from keras.applications.vgg19 import VGG19
from keras.models import Model

from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.layers import (Flatten, Dense, Input, Conv2D, MaxPooling2D,
                          GlobalAveragePooling2D, GlobalMaxPooling2D)

from keras_exp.callbacks.timing import SamplesPerSec, BatchTiming
# from keras_tqdm import TQDMCallback

from resnet_common import RecordInputImagenetPreprocessor

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'  # @IgnorePep8
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'  # @IgnorePep8


# noqa ref: https://stackoverflow.com/questions/27803059/conditional-with-statement-in-python
class dummy_context_mgr(object):  # pylint: disable=invalid-name
    '''
    Use contexts/with conditionally. Ex:
      with tf.device(gpu0) if ngpus > 1 else dummy_context_mgr():  # @NoEffect
    '''
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):  # @UnusedVariable
        return False


def VGG19(include_top=True, weights='imagenet',
          input_tensor=None, input_shape=None,
          pooling=None,
          classes=1000,
          mpar_gpus=None):
    """Instantiates the VGG19 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization)
            or 'imagenet' (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    # Determine proper input shape
    input_shape = _obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=48,
        data_format=KB.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not KB.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    ngpus = 1 if mpar_gpus is None else len(mpar_gpus)
    mpar_gpus = [0, 0] if mpar_gpus is None else mpar_gpus
    gpu0 = '/gpu:{}'.format(mpar_gpus[0])
    # il - intermediate layer input/output
    with tf.device(gpu0) if ngpus > 1 else dummy_context_mgr():  # @NoEffect
        # Block 1
        il = Conv2D(64, (3, 3), activation='relu', padding='same',
                    name='block1_conv1')(img_input)
        il = Conv2D(64, (3, 3), activation='relu',
                    padding='same', name='block1_conv2')(il)
        il = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(il)

    gpu1 = '/gpu:{}'.format(mpar_gpus[1])
    with tf.device(gpu1) if ngpus > 1 else dummy_context_mgr():  # @NoEffect
        # Block 2
        il = Conv2D(128, (3, 3), activation='relu',
                    padding='same', name='block2_conv1')(il)
        il = Conv2D(128, (3, 3), activation='relu',
                    padding='same', name='block2_conv2')(il)
        il = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(il)

        # Block 3
        il = Conv2D(256, (3, 3), activation='relu',
                    padding='same', name='block3_conv1')(il)
        il = Conv2D(256, (3, 3), activation='relu',
                    padding='same', name='block3_conv2')(il)
        il = Conv2D(256, (3, 3), activation='relu',
                    padding='same', name='block3_conv3')(il)
        il = Conv2D(256, (3, 3), activation='relu',
                    padding='same', name='block3_conv4')(il)
        il = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(il)

        # Block 4
        il = Conv2D(512, (3, 3), activation='relu',
                    padding='same', name='block4_conv1')(il)
        il = Conv2D(512, (3, 3), activation='relu',
                    padding='same', name='block4_conv2')(il)
        il = Conv2D(512, (3, 3), activation='relu',
                    padding='same', name='block4_conv3')(il)
        il = Conv2D(512, (3, 3), activation='relu',
                    padding='same', name='block4_conv4')(il)
        il = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(il)

        # Block 5
        il = Conv2D(512, (3, 3), activation='relu',
                    padding='same', name='block5_conv1')(il)
        il = Conv2D(512, (3, 3), activation='relu',
                    padding='same', name='block5_conv2')(il)
        il = Conv2D(512, (3, 3), activation='relu',
                    padding='same', name='block5_conv3')(il)
        il = Conv2D(512, (3, 3), activation='relu',
                    padding='same', name='block5_conv4')(il)
        il = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(il)

    if include_top:
        # Classification block
        il = Flatten(name='flatten')(il)
        il = Dense(4096, activation='relu', name='fc1')(il)
        il = Dense(4096, activation='relu', name='fc2')(il)
        il = Dense(classes, activation='softmax', name='predictions')(il)
    else:
        if pooling == 'avg':
            il = GlobalAveragePooling2D()(il)
        elif pooling == 'max':
            il = GlobalMaxPooling2D()(il)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, il, name='vgg19')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
        if KB.backend() == 'theano':
            layer_utils.convert_all_kernels_in_model(model)

        if KB.image_data_format() == 'channels_first':
            if include_top:
                maxpool = model.get_layer(name='block5_pool')
                shape = maxpool.output_shape[1:]
                dense = model.get_layer(name='fc1')
                layer_utils.convert_dense_weights_data_format(dense, shape,
                                                              'channels_first')

            if KB.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image data format convention '
                              '(`image_data_format="channels_first"`). '
                              'For best performance, set '
                              '`image_data_format="channels_last"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')
    return model


class SmartFormatterMixin(ap.HelpFormatter):
    '''Arguments parser formatter that splits on \n. Start help with "S|".'''
    # ref:
    # http://stackoverflow.com/questions/3853722/python-argparse-how-to-insert-newline-in-the-help-text
    # @IgnorePep8

    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('S|'):
            return text[2:].splitlines()
        return ap.HelpFormatter._split_lines(self, text, width)


class CustomFormatter(ap.RawDescriptionHelpFormatter, SmartFormatterMixin):
    '''Convenience formatter_class for argparse help print out.'''


def _parser(desc):
    parser = ap.ArgumentParser(description=dedent(desc),
                               formatter_class=CustomFormatter)

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to run training for.\n'
                        '(Default: %(default)s)\n')

    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='S|Batch size. Default: %(default)s')

    parser.add_argument(
        '--ngpus_per_model', type=int, default=1, choices=(1, 2),
        help='S|GPUs for Model parallelism. Max set to 2 for now. '
        'Default: %(default)s')

    parser.add_argument(
        '--imgs_per_epoch', type=int, default=0,
        help='S|Number of images to run during epoch. Use for timing.\n'
        'Default uses all the images for an epoch.')

    imagenet_datadir = '/datasets/imagenet/train-val-tfrecord-480-subset'
    parser.add_argument(
        '--datadir', default=imagenet_datadir,
        help='S|Data directory with Imagenet TFrecord dataset. Assumes\n'
        'TFrecord subsets prefixed with train-* and validation-* are in the\n'
        'directory. Default: %(default)s')

    parser.add_argument(
        '--distort_color', action='store_true', default=False,
        help='S|Distort color during training on imagenet to "enrich" the\n'
        'dataset. Default no distortion. Set this flag to enable distortion.')

    args = parser.parse_args()

    return args


def main(argv=None):
    main.__doc__ = __doc__
    argv = sys.argv if argv is None else sys.argv.extend(argv)
    desc = main.__doc__  # .format(os.path.basename(__file__))
    # CLI parser
    args = _parser(desc)

    # Initialize Horovod.
    hvd.init()

    local_rank = hvd.local_rank()
    ngpus_per_model = args.ngpus_per_model
    local_gpus = range(local_rank * ngpus_per_model,
                       local_rank * ngpus_per_model + ngpus_per_model)
    print('RANK: {}; LOCAL RANK: {}; LOCAL_GPUS: {}'.format(
        hvd.rank(), local_rank, local_gpus))
    lgpus_str = ','.join(str(ig) for ig in local_gpus)

    # Pin GPU(s) to be used.
    gpu_options = tf.GPUOptions(
        allow_growth=True,
        visible_device_list=lgpus_str)
    config = tf.ConfigProto(gpu_options=gpu_options)
    KB.set_session(tf.Session(config=config))

    # mpar_gpus - Normalized ID list of GPUs to use for model parallelism
    mpar_gpus = range(ngpus_per_model) if ngpus_per_model > 1 else None

    ngpus = hvd.size()

    num_devices_tfrecord = 1
    height, width = 224, 224  # Image dimensions. Gets resized if not match.
    distort_color = args.distort_color
    data_dir = args.datadir
    batch_size = args.batch_size  # * ngpus
    epochs = args.epochs
    imgs_per_epoch = args.imgs_per_epoch

    # Fit the model using data from the TFRecord data tensors.
    device_minibatches = RecordInputImagenetPreprocessor.device_minibatches
    images_tfrecord, labels_tfrecord, nrecords = device_minibatches(
        num_devices_tfrecord, data_dir, batch_size,
        height, width, distort_color, val=False)
    images_tfrecord = images_tfrecord[0]
    labels_tfrecord = labels_tfrecord[0]

    # CASTING FOR KERAS
    # labels[device_num] = tf.cast(labels_tfrecord, dtype)
    nclasses = 1000
    labels_tfrecord = tf.one_hot(labels_tfrecord, nclasses)

    nimgs_to_use = imgs_per_epoch if imgs_per_epoch > 0 else nrecords
    steps_per_epoch = nimgs_to_use // batch_size // hvd.size()
    # steps_per_epoch = 100

    # batch_shape = images_tfrecord.get_shape().as_list()
    # images = Input(tensor=images_tfrecord, batch_shape=x_batch_shape)
    images = Input(tensor=images_tfrecord)
    model = VGG19(input_tensor=images, weights=None, mpar_gpus=mpar_gpus)

    # print('NLAYERS: {}'.format(len(model.layers)))

    if hvd.rank() == 0:
        model.summary()

        print('Num images: {}'.format(nrecords))

        if nimgs_to_use < nrecords:
            print('Using {} images per epoch'.format(nimgs_to_use))

        # print('IMAGES_TFRECORD: {}'.format(images_tfrecord))
        # print('LABELS_TFRECORD: {}'.format(labels_tfrecord))

    # lr = 0.001 * ngpus
    # opt = tf.train.AdamOptimizer()
    # opt = hvd.DistributedOptimizer(opt)  # , use_locking=True)
    # opt = KO.TFOptimizer(opt)  # Required for tf.train based optimizers

    opt = KO.Adam()
    opt = hvd_keras.DistributedOptimizer(opt)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  # metrics=['accuracy'],
                  target_tensors=[labels_tfrecord])

    # tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    KB.get_session().run(init)
    # Broadcast variables from rank 0 to all other processes.
    KB.get_session().run(hvd.broadcast_global_variables(0))

    callbacks = [hvd_keras.callbacks.BroadcastGlobalVariablesCallback(0)]
    if hvd.rank() == 0:
        callbacks += [BatchTiming(),
                      SamplesPerSec(ngpus * batch_size)]

    # RecordInput is a yield op which doesn't use queue runners or queues.
    # Start the queue runners.
    # sess = KB.get_session()

    # sess.run([tf.local_variables_initializer(),
    #           tf.global_variables_initializer()])

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess, coord)

    start_time = time.time()
    model.fit(
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1)
    # verbose=hvd.rank() == 0)
    elapsed_time = time.time() - start_time

    if hvd.rank() == 0:
        print('[{}] finished in {} s'
              .format('TRAINING', round(elapsed_time, 3)))
        # loss = model.evaluate(None, None, steps=steps_per_epoch_val)

        images_tfrecord_val, labels_tfrecord_val, nrecords_val = \
            device_minibatches(num_devices_tfrecord, data_dir, batch_size,
                               height, width, distort_color, val=True)
        images_tfrecord_val = images_tfrecord_val[0]
        labels_tfrecord_val = labels_tfrecord_val[0]
        labels_tfrecord_val = tf.one_hot(labels_tfrecord_val, nclasses)

        # print('IMAGES_TFRECORD_VAL: {}'.format(images_tfrecord_val))
        # print('labels_tfrecord_val: {}'.format(labels_tfrecord_val))

        steps_per_epoch_val = nrecords_val // batch_size

        images_val = Input(tensor=images_tfrecord_val)
        model_val = model
        model_val.layers[0] = KL.InputLayer(input_tensor=images_val)
        model_val.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'],
            target_tensors=[labels_tfrecord_val])
        # model.summary()
        loss = model_val.evaluate(x=None, y=None, steps=steps_per_epoch_val)

        print('\nNum images evaluated, steps: {}, {}'.
              format(nrecords_val, steps_per_epoch_val))
        print('\nTest loss, acc: {}'.format(loss))
        # print('\nTest accuracy: {0}'.format(acc))

    # Clean up the TF session.
    # coord.request_stop()
    # coord.join(threads)

    KB.clear_session()  # do this for Horovod


if __name__ == '__main__':
    main()
