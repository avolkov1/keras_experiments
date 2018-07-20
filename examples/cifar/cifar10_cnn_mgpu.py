#!/usr/bin/env python
'''Train a simple deep CNN on the CIFAR10 small images dataset.

MultiGPU implementation.
'''

from __future__ import print_function
import sys
import time

import tensorflow as tf

from keras import backend as KB
from keras.models import Model
import keras.layers as KL
from keras.utils import to_categorical
import keras.losses as keras_losses
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from keras.optimizers import RMSprop

from keras_exp.multigpu import (
    GPUListType, get_available_gpus, print_mgpu_modelsummary)
from keras_exp.multigpu import ModelMGPU
# from keras_exp.multigpu import make_parallel
from keras_exp.multigpu import ModelKerasMGPU
from keras_exp.multigpu.optimizers import RMSPropMGPU

from keras_exp.callbacks.timing import BatchTiming, SamplesPerSec

from cifar_common import (
    CifarTrainDefaults, cifar10_load_data, make_model, wrap_as_tfdataset)

from parser_common import parser_def_mgpu, remove_options


_DEVPROF = False


def parser_(desc):
    '''CLI parser for Cifar10 multigpu example.'''
    parser = parser_def_mgpu(desc)

    remove_options(parser, ['--rdma', '--network'])

    parser.add_argument(
        '--batch_size', type=int, default=CifarTrainDefaults.batch_size,
        help='S|Batch size. Default: %(default)s')

    checkptfile = 'cifar10_cnn_mgpu.weights.best.hdf5'
    parser.add_argument(
        '--checkpt', action='store', nargs='?',
        const=checkptfile,
        help='S|Save (overwrites) and load the model weights if available.'
        '\nOptionally specify a file/filepath if the default name is '
        'undesired.\n(default: {})'.format(checkptfile))

    parser.add_argument(
        '--mgpu-type', action='store', nargs='?', type=str.lower,
        const='expmgpu', default='expmgpu',
        choices=['expmgpu', 'kerasmgpu'],
        help='S|Use experimental or Keras multigpu conversion. For\n'
        'experimental uses ModelMGPU and for Keras uses ModelKerasMGPU\n'
        'which is a wrapper around multi_gpu_model function.\n'
        'Default: expmgpu')

    parser.add_argument(
        '--syncopt', action='store_true', default=False,
        help='S|Use gradient synchronization in Optimizer. Not sure if this\n'
        'feature is working correctly. Default: %(default)s')

    parser.add_argument(
        '--aug', action='store_true', default=False,
        help='S|Perform data augmentation on cifar10 set.\n')

    parser.add_argument(
        '--logdevp', action='store_true', default=False,
        help='S|Log device placement in Tensorflow.\n')

    parser.add_argument(
        '--datadir',
        help='Data directory with Cifar10 dataset.')

    parser.add_argument(
        '--use-dataset-api', action='store_true', default=False,
        help='S|Use Tensorflow Dataset API for Keras model training.')

    args = parser.parse_args()

    return args


def main(argv=None):
    '''Multigpu example using Keras for Cifar10 training.'''
    argv = sys.argv if argv is None else sys.argv.extend(argv)
    # CLI parser
    args = parser_(main.__doc__)

    logdevp = args.logdevp

    gpu_options = tf.GPUOptions(allow_growth=True)
    if _DEVPROF or logdevp:  # or True:
        # Setup Keras session using Tensorflow
        config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True,
            gpu_options=gpu_options)
        # config.gpu_options.allow_growth = True
        KB.set_session(tf.Session(config=config))
    else:
        config = tf.ConfigProto(gpu_options=gpu_options)
        KB.set_session(tf.Session(config=config))

    mgpu = 0 if args.mgpu is None else args.mgpu
    gpus_list = get_available_gpus(mgpu)
    ngpus = len(gpus_list)

    syncopt = args.syncopt

    checkpt = args.checkpt
    filepath = checkpt
    # print('CHECKPT:', checkpt)

    batch_size = args.batch_size * ngpus if ngpus > 1 else args.batch_size
    num_classes = 10
    epochs = args.epochs

    datadir = args.datadir

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10_load_data(datadir)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    if not args.use_dataset_api:
        traingen = ImageDataGenerator()
        if args.aug:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            traingen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of the dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # randomly rotate images in the range (degrees, 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally (fraction of total width)
                width_shift_range=0.1,
                # randomly shift images vertically (fraction of total height)
                height_shift_range=0.1,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False)

            # Compute quantities required for feature-wise normalization
            # (std, mean, and principal components if ZCA whitening is applied)
            traingen.fit(x_train)

        # x_train_input = KL.Input(shape=x_train.shape[1:])
        model_init = make_model(
            x_train.shape[1:], num_classes, filepath)
    else:
        print('USING TF DATASET API.')
        dataset = wrap_as_tfdataset(
            x_train, y_train, args.aug, batch_size)
        iterator = dataset.make_one_shot_iterator()

        # Model creation using tensors from the get_next() graph node.
        inputs, targets = iterator.get_next()
        x_train_input = KL.Input(tensor=inputs)

        model_init_ = make_model(x_train_input, num_classes, filepath)
        x_train_out = model_init_.output

        model_init = Model(inputs=[x_train_input], outputs=[x_train_out])

    lr = 0.0001
    if ngpus > 1:
        print('Using GPUs: {}'.format(', '.join(gpus_list)))
        lr = lr * ngpus

        # Data-Parallelize the model via function or class.
        if args.mgpu_type == 'kerasmgpu':
            gpus_list_int = get_available_gpus(
                ngpus, list_type=GPUListType.int_id)
            model = ModelKerasMGPU(model_init, gpus_list_int)
        else:
            model = ModelMGPU(
                serial_model=model_init, gdev_list=gpus_list)

        print_mgpu_modelsummary(model)
        if not syncopt:
            opt = RMSprop(lr=lr, decay=1e-6)
        else:
            opt = RMSPropMGPU(lr=lr, decay=1e-6, gdev_list=gpus_list)  # @IgnorePep8 pylint: disable=unexpected-keyword-arg

    else:
        model = model_init
        # batch_size = batch_size * 3
        # batch_size = 25000  # exhaust GPU memory. Crashes.
        print(model.summary())

        # initiate RMSprop optimizer
        opt = RMSprop(lr=lr, decay=1e-6)

    model.compile(
        loss=keras_losses.categorical_crossentropy,
        optimizer=opt,
        metrics=['accuracy'],
        target_tensors=None if not args.use_dataset_api else [targets])

    callbacks = []
    if checkpt:
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='max')
        callbacks = [checkpoint]

    callbacks += [BatchTiming(), SamplesPerSec(batch_size)]

    nsamples = x_train.shape[0]
    steps_per_epoch = nsamples // batch_size

    if not args.use_dataset_api:
        start_time = time.time()
        # Fit the model on the batches generated by traingen.flow().
        model.fit_generator(
            traingen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks)

    else:
        # augmentation incorporated in the Dataset pipeline
        start_time = time.time()
        # Validation during training can be incorporated via callback:
        # noqa ref: https://github.com/keras-team/keras/blob/c8bef99ec7a2032b9bea6e9a1260d05a2b6a80f1/examples/mnist_tfrecord.py#L56
        model.fit(
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks)

    elapsed_time = time.time() - start_time
    print('[{}] finished in {} s'
          .format('TRAINING', round(elapsed_time, 3)))

    test_model = model_init
    if args.use_dataset_api:
        # Create a test-model without Dataset pipeline in the model graph.
        test_model = make_model(x_test.shape[1:], num_classes)
        print('SETTING WEIGHTS FOR EVAL WITH DATASET API...')
        test_model.set_weights(model.get_weights())
        print('WEIGHTS SET!!!')

    test_model.compile(
        loss=keras_losses.categorical_crossentropy,
        optimizer=opt,
        metrics=['accuracy'])

    metrics = test_model.evaluate(x_test, y_test)
    print('\nCIFAR VALIDATION LOSS, ACC: {}, {}'.format(*metrics))

    KB.clear_session()


if __name__ == '__main__':
    main()
