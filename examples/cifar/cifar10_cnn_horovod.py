#!/usr/bin/env python
'''Train a simple deep CNN on the CIFAR10 small images dataset.

MultiGPU Horovod implementation.

Some run command examples (download data ahead of time):
TMPDIR=/tmp mpirun --report-bindings --map-by ppr:4:socket:pe=20 \
  -oversubscribe -np 4 python \
  ./examples/cifar/cifar10_cnn_horovod.py --epochs=4 --aug

TMPDIR=/tmp mpirun --report-bindings --bind-to none --map-by slot -np 4 \
  python ./examples/cifar/cifar10_cnn_horovod.py --epochs=4 --aug

With overubscribing if MPS is enabled (highly experimental and unstable):
# Ex.: Assume --ntasks-per-node=8 with 4 GPUs per node using 2 nodes.
NGPUS=4 NNODES=2 RANKS_PER_GPU=2 && \
  TMPDIR=/tmp mpirun --report-bindings --bind-to none --map-by slot \
  -np $(($RANKS_PER_GPU * $NNODES * $NGPUS)) \
  python ./examples/cifar/cifar10_cnn_horovod.py --epochs=50 \
  --nranks_per_gpu=$RANKS_PER_GPU

TMPDIR=/tmp mpirun --report-bindings --bind-to none --map-by slot -np 4 \
  python ./examples/cifar/cifar10_cnn_horovod.py --epochs=4 --aug

With singularity containers:

TMPDIR=/tmp mpirun --report-bindings -mca btl_tcp_if_exclude docker0,lo \
  --bind-to none --map-by slot -np 8 singularity exec --nv \
  /cm/shared/singularity/tf1.4.0_hvd_ompi3.0.0-2017-11-23-154091b4d08c.img \
  bash -c 'LD_LIBRARY_PATH=/.singularity.d/libs:$LD_LIBRARY_PATH; \
  source ~/.virtualenvs/py-keras_theano/bin/activate && \
  python ./examples/cifar/cifar10_cnn_horovod.py --epochs=3'


TMPDIR=/tmp mpirun --report-bindings -mca btl_tcp_if_exclude docker0,lo \
  --bind-to none --map-by slot -np 8 \
  run_psgcluster_singularity.sh \
    --container=/cm/shared/singularity/tf1.4.0_hvd_ompi3.0.0-2017-11-23-154091b4d08c.img \
    --venvpy=~/.virtualenvs/py-keras_theano \
    --scripts=./examples/cifar/cifar10_cnn_horovod.py \
    --epochs=20

'''

from __future__ import print_function
import sys
import os
import time

from argparse import SUPPRESS

import numpy as np

from parser_common import parser_def_mgpu, remove_options

import tensorflow as tf
import horovod.tensorflow as hvd

# from keras.utils.data_utils import get_file
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import keras.layers as KL
from keras import backend as KB

# from keras.callbacks import ModelCheckpoint

from keras.optimizers import TFOptimizer

from keras_exp.callbacks.timing import SamplesPerSec, BatchTiming


_DEVPROF = False


def parser_(desc):
    parser = parser_def_mgpu(desc)

    remove_options(parser, ['--nccl', '--enqueue', '--syncopt', '--rdma',
                            '--mgpu', '--network'])

    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='S|Batch size. Default: %(default)s')

    # parser.add_argument(
    #     '--ngpus_per_node', type=int, default=-1,
    #     help='S|Number of GPUs per node. Default: Horovod local_size()')

    parser.add_argument(
        '--nranks_per_gpu', type=int, default=1,
        help='S|Number of ranks to run on each GPUs. Use this parameter to\n'
        'oversubscribe a GPU. When oversubscribing a GPU use in combination\n'
        'with MPS (multi-process service). Default: %(default)s')

    checkptfile = 'cifar10_cnn_hvd.weights.best.hdf5'
    parser.add_argument(
        '--checkpt', action='store', nargs='?',
        const=checkptfile, default=SUPPRESS,
        help='S|Save (overwrites) and load the model weights if available.'
        '\nOptionally specify a file/filepath if the default name is '
        'undesired.\n(default: {})'.format(checkptfile))

    parser.add_argument('--aug', action='store_true', default=False,
                        help='S|Perform data augmentation on cifar10 set.\n')

    parser.add_argument('--logdevp', action='store_true', default=False,
                        help='S|Log device placement in Tensorflow.\n')

    parser.add_argument('--datadir', default=SUPPRESS,
                        help='Data directory with Cifar10 dataset.')

    args = parser.parse_args()

    return args


def make_model(inshape, num_classes, weights_file=None):
    return make_model_full(inshape, num_classes, weights_file)
    # return make_model_small(inshape, num_classes, weights_file)


def make_model_full(inshape, num_classes, weights_file=None):
    model = Sequential()
    model.add(KL.InputLayer(input_shape=inshape[1:]))
    # model.add(KL.Conv2D(32, (3, 3), padding='same', input_shape=inshape[1:]))
    model.add(KL.Conv2D(32, (3, 3), padding='same'))
    model.add(KL.Activation('relu'))
    model.add(KL.Conv2D(32, (3, 3)))
    model.add(KL.Activation('relu'))
    model.add(KL.MaxPooling2D(pool_size=(2, 2)))
    model.add(KL.Dropout(0.25))

    model.add(KL.Conv2D(64, (3, 3), padding='same'))
    model.add(KL.Activation('relu'))
    model.add(KL.Conv2D(64, (3, 3)))
    model.add(KL.Activation('relu'))
    model.add(KL.MaxPooling2D(pool_size=(2, 2)))
    model.add(KL.Dropout(0.25))

    model.add(KL.Flatten())
    model.add(KL.Dense(512))
    model.add(KL.Activation('relu'))
    model.add(KL.Dropout(0.5))
    model.add(KL.Dense(num_classes))
    model.add(KL.Activation('softmax'))

    if weights_file is not None and os.path.exists(weights_file):
        model.load_weights(weights_file)

    return model


def make_model_small(inshape, num_classes, weights_file=None):
    model = Sequential()
    model.add(KL.InputLayer(input_shape=inshape[1:]))
    model.add(KL.Conv2D(32, (3, 3), padding='same'))
    model.add(KL.Activation('relu'))
    model.add(KL.Flatten())
    # model.add(Dropout(0.5))
    model.add(KL.Dense(num_classes))
    model.add(KL.Activation('softmax'))

    if weights_file is not None and os.path.exists(weights_file):
        model.load_weights(weights_file)

    return model


def cifar10_load_data(path):
    """Loads CIFAR10 dataset.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    dirname = 'cifar-10-batches-py'
    # origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    # path = get_file(dirname, origin=origin, untar=True)
    path_ = os.path.join(path, dirname)

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path_, 'data_batch_' + str(i))
        data, labels = cifar10.load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path_, 'test_batch')
    x_test, y_test = cifar10.load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if KB.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def main(argv=None):
    '''
    '''
    main.__doc__ = __doc__
    argv = sys.argv if argv is None else sys.argv.extend(argv)
    desc = main.__doc__  # .format(os.path.basename(__file__))
    # CLI parser
    args = parser_(desc)

    # Initialize Horovod.
    hvd.init()

    logdevp = args.logdevp  # For debugging
    log_device_placement, allow_soft_placement = (True, True) \
        if _DEVPROF or logdevp else (False, False)

    nranks_per_gpu = args.nranks_per_gpu
    local_rank = hvd.local_rank()
    gpu_local_rank = local_rank // nranks_per_gpu
    print('local_rank, GPU_LOCAL_RANK: {}, {}'.format(
        local_rank, gpu_local_rank))

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto(log_device_placement=log_device_placement,
                            allow_soft_placement=allow_soft_placement)
    config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.visible_device_list = str(gpu_local_rank)
    KB.set_session(tf.Session(config=config))

    hvdsize = hvd.size()

    checkpt = getattr(args, 'checkpt', None)
    checkpt_flag = False if checkpt is None else True
    filepath = checkpt
    # print('CHECKPT:', checkpt)

    batch_size = args.batch_size
    num_classes = 10
    epochs = args.epochs
    data_augmentation = args.aug

    datadir = getattr(args, 'datadir', None)

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10_load_data(datadir) \
        if datadir is not None else cifar10.load_data()
    train_samples = x_train.shape[0]
    test_samples = x_test.shape[0]
    steps_per_epoch = train_samples // batch_size // hvdsize
    test_batches = test_samples // batch_size
    print(train_samples, 'train samples')
    print(test_samples, 'test samples')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes).squeeze()
    y_test = to_categorical(y_test, num_classes).squeeze()

    callbacks = []
    if hvd.rank() == 0:
        callbacks += [BatchTiming(), SamplesPerSec(batch_size * hvdsize)]

    print(x_train.shape, 'train shape')
    # with tf.device('/cpu:0'):
    model = make_model(x_train.shape, num_classes,
                       filepath if checkpt_flag else None)

    lr = 0.0001 * hvdsize
    opt = tf.train.RMSPropOptimizer(lr)
    # Add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)  # , use_locking=True)
    opt = TFOptimizer(opt)  # Required for tf.train based optimizers

    # ------------------------------------- HAVE TO GET SESSION AFTER OPTIMIZER
    # sess = KB.get_session()
    # -------------------------------------------------------------------------

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    if hvd.rank() == 0:
        model.summary()

    KB.get_session().run(hvd.broadcast_global_variables(0))
    if not data_augmentation:
        print('Not using data augmentation.')
        # model.fit(x_train, y_train,
        #           batch_size=batch_size,
        #           epochs=epochs,
        #           validation_data=(x_test, y_test),
        #           shuffle=True,
        #           callbacks=callbacks)

        train_gen = ImageDataGenerator()
        test_gen = ImageDataGenerator()
        # Train the model. The training will randomly sample 1 / N batches of
        # training data and 3 / N batches of validation data on every worker,
        # where N is the number of workers. Over-sampling of validation data
        # helps to increase probability that every validation example will be
        # evaluated.
        start_time = time.time()
        model.fit_generator(
            train_gen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks,
            epochs=epochs,
            verbose=hvd.rank() == 0,
            validation_data=test_gen.flow(x_test, y_test,
                                          batch_size=batch_size),
            validation_steps=3 * test_batches // hvdsize)

    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            # divide inputs by std of the dataset
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            # randomly rotate images in the range (degrees, 0 to 180)
            rotation_range=0,
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.1,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.1,
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        start_time = time.time()
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(x_test, y_test),
            verbose=hvd.rank() == 0,
            callbacks=callbacks)

    if hvd.rank() == 0:
        elapsed_time = time.time() - start_time
        print('[{}] finished in {} s'
              .format('TRAINING', round(elapsed_time, 3)))

        metrics = model.evaluate(x=x_test, y=y_test, batch_size=batch_size)
        print('\nCIFAR VALIDATION LOSS, ACC: {}, {}'.format(*metrics))

    KB.clear_session()


if __name__ == '__main__':
    main()

