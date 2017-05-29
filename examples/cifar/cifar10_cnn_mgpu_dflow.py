#!/usr/bin/env python
'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically
used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatx=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50
epochs. (it's still underfitting at that point, though).
'''
from __future__ import print_function
import sys
import os

from argparse import SUPPRESS

import numpy as np
from datetime import datetime
import threading

# -------------------------------------------------------- Helpers for main app
from parser_common import parser_def_mgpu

# --------------------------------------------------------------- Keras package
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.models import Sequential
import keras.layers as KL
from keras import backend as KB

from keras.callbacks import ModelCheckpoint

from keras.optimizers import RMSprop

# -------------------------------------------------- Keras experimental package
from keras_exp.multigpu import (get_available_gpus, print_mgpu_modelsummary)
# from keras_exp.multigpu import ModelMGPU
from keras_exp.multigpu import make_parallel
from keras_exp.multigpu.optimizers import RMSPropMGPU

from keras_exp.dataflow import (
    BatchData, RepeatedData, PrefetchDataZMQ)

from keras_exp.dataflow.imgaug import AugmentImageComponent


class dataset(object):
    from keras_exp.dataflow.dataset import (Cifar10, Cifar100)  # @UnusedImport

# ---------------------------------------------------------- Tensorpack package


# from tensorpack import imgaug
class imgaug(object):
    from keras_exp.dataflow.imgaug import ImgScale  # @UnusedImport
    from tensorpack.dataflow.imgaug.imgproc import (
        Brightness, Contrast)  # @UnusedImport
    from tensorpack.dataflow.imgaug.noname import Flip  # @UnusedImport
    from tensorpack.dataflow.imgaug.deform import (
        GaussianDeform)  # @UnusedImport


# -------------------------------------------------- Main functions and routine

_DEVPROF = False


def parser_(desc):
    parser = parser_def_mgpu(desc)

    checkptfile = 'cifar10_cnn_mgpu.weights.best.hdf5'
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

    args = parser.parse_args()

    return args


class threadsafe_iter(object):
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


@threadsafe_generator
def mygenerator(nsamples, batch_size, x_train, y_train):
    steps_per_epoch = nsamples // batch_size
    seed = (id(None) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    np.random.RandomState(seed)
    while 1:
        # np.random.shuffle()
        idx_shuffle = np.random.permutation(nsamples)
        for i in range(steps_per_epoch):
            start_ = i * batch_size
            end_ = min((i + 1) * batch_size, nsamples)
            slice_shuffle = idx_shuffle[start_:end_]
            yield x_train[slice_shuffle], y_train[slice_shuffle]


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


def get_data(train_or_test, cifar_classnum, batch_size=128, aug=False,
             epochs=100):
    '''Return a dataset wrapped in a dataflow object.'''
    isTrain = train_or_test == 'train'
    if cifar_classnum == 10:
        ds = dataset.Cifar10(train_or_test)
    else:
        ds = dataset.Cifar100(train_or_test)

    if isTrain:
        ds = RepeatedData(ds, epochs + 1)

    if isTrain and aug:
        # Not sure if augmentation is working.
        augmentors = [
            imgaug.Flip(horiz=True),
            imgaug.Brightness(63),
            imgaug.Contrast((0.2, 1.8)),
            imgaug.GaussianDeform(
                [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
                (32, 32), 0.2, 3),
            imgaug.ImgScale(255)
        ]
    else:
        augmentors = [
            imgaug.ImgScale()
        ]
    ds = AugmentImageComponent(ds, augmentors)
    # Tradeoffs between batch_size, convergence rate, and number of GPUs
    # batch_size = 128  # * number of GPUs sometimes helps
    ds = BatchData(ds, batch_size, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, 16)  # at 10 or 16 sometimes better
    return ds


def main(argv=None):
    '''
    '''
    main.__doc__ = __doc__
    argv = sys.argv if argv is None else sys.argv.extend(argv)
    desc = main.__doc__  # .format(os.path.basename(__file__))
    # CLI parser
    args = parser_(desc)
    mgpu = 0 if getattr(args, 'mgpu', None) is None else args.mgpu
    enqueue = args.enqueue
    usenccl = args.nccl
    syncopt = args.syncopt

    checkpt = getattr(args, 'checkpt', None)
    checkpt_flag = False if checkpt is None else True
    filepath = checkpt
    # print('CHECKPT:', checkpt)

    batch_size = 32  # 64
    num_classes = 10
    epochs = args.epochs
    data_augmentation = args.aug

    logdevp = args.logdevp

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    callbacks = None

    if _DEVPROF or logdevp:
        import tensorflow as tf

        # Setup Keras session using Tensorflow
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True)
        # config.gpu_options.allow_growth = True
        tfsess = tf.Session(config=config)
        KB.set_session(tfsess)

    print(x_train.shape, 'train shape')
    print(y_train.shape, 'label shape')
    model_init = make_model(x_train.shape, num_classes,
                            filepath if checkpt_flag else None)

    # model_init = partial(make_model, x_train.shape, num_classes,
    #                      filepath if checkpt_flag else None)

    if checkpt_flag:
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='max')
        callbacks = [checkpoint]

    if mgpu > 1 or mgpu == -1:
        gpus_list = get_available_gpus(mgpu)
        ngpus = len(gpus_list)
        print('Using GPUs: {}'.format(', '.join(gpus_list)))
        batch_size = batch_size * ngpus  #
        # batch_size = 40000  # split over four devices works fine no grad avg
        # batch_size = 25000  # split over four devices works fine w/ grad avg

        # Data-Parallelize the model via function or class.
        model = make_parallel(model_init, gpus_list, usenccl=usenccl,
                              syncopt=syncopt, enqueue=enqueue)
        # model = ModelMGPU(serial_model=model_init, gdev_list=gpus_list,
        #                   syncopt=syncopt, usenccl=usenccl, enqueue=enqueue)
        print_mgpu_modelsummary(model)
        if not syncopt:
            opt = RMSprop(lr=0.0001, decay=1e-6)
        else:
            opt = RMSPropMGPU(lr=0.0001, decay=1e-6, gdev_list=gpus_list)

    else:
        model = model_init
        # batch_size = batch_size * 3
        # batch_size = 25000  # exhaust GPU memory. Crashes.
        print(model.summary())

        # initiate RMSprop optimizer
        opt = RMSprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    nsamples = x_train.shape[0]
    steps_per_epoch = nsamples // batch_size

    # prepare dataset
    dataset_train = get_data('train', num_classes, batch_size=batch_size,
                             aug=data_augmentation, epochs=epochs)
    # dataset_test = get_data('test', cifar_classnum)

    if not data_augmentation:
        print('Not using data augmentation.')

    # Plain ol'd fit is faster than dataflow generator below.
    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_data=(x_test, y_test),
    #           shuffle=True,
    #           callbacks=callbacks)

    # Fit the model on the batches generated by datagen.flow().
    # mygen = mygenerator(nsamples, batch_size, x_train, y_train)
    dataset_train.reset_state()
    mygen = dataset_train.get_data()
    # for dp in mygen:
    #     print('DP SHAPE: {}'.format(dp[0].shape))
    model.fit_generator(mygen,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)

    # Using fit_dataflow method that's mixed into ModelMGPU class.
    # model.fit_dataflow(dataset_train,
    #                    steps_per_epoch=steps_per_epoch,
    #                    epochs=epochs,
    #                    validation_data=(x_test, y_test),
    #                    callbacks=callbacks)


if __name__ == '__main__':
    main()

