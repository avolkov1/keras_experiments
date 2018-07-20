#!/usr/bin/env python
'''Train a simple deep CNN on the CIFAR10 small images dataset.

MultiGPU Horovod implementation.

For help run:
    python ./examples/cifar/cifar10_cnn_horovod.py --help

General example of run command:
NGPUS=4 NNODES=1 RANKS_PER_GPU=1 np=$(($RANKS_PER_GPU * $NNODES * $NGPUS)) && \
    TMPDIR=/tmp mpirun -x NCCL_RINGS="$( seq -s' ' 0 $((np - 1)) )"  \
    --report-bindings --bind-to none --map-by slot -np ${np} \
    python ./examples/cifar/cifar10_cnn_horovod.py --epochs=5 \
    --nranks_per_gpu=$RANKS_PER_GPU

'''
from __future__ import print_function
import sys
import time

import tensorflow as tf
import horovod.tensorflow as hvd
import horovod.keras as hvd_keras

from keras import backend as KB
from keras.models import Model
import keras.layers as KL
from keras.utils import to_categorical
# import keras.optimizers as KO
import keras.losses as keras_losses
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint

from keras_exp.callbacks.timing import SamplesPerSec, BatchTiming

from cifar_common import (
    CifarTrainDefaults, make_model, cifar10_load_data, wrap_as_tfdataset,
    print_rank0)

from parser_common import parser_def_mgpu, remove_options


_DEVPROF = False


def parser_(desc):
    '''Parser for Cifar10 CNN Horovod training script.'''
    parser = parser_def_mgpu(desc)

    remove_options(parser, ['--nccl', '--enqueue', '--syncopt', '--rdma',
                            '--mgpu', '--network'])

    parser.add_argument(
        '--batch_size', type=int, default=CifarTrainDefaults.batch_size,
        help='S|Batch size. Default: %(default)s')

    parser.add_argument(
        '--nranks_per_gpu', type=int, default=1,
        help='S|Number of ranks to run on each GPUs. Use this parameter to\n'
        'oversubscribe a GPU. When oversubscribing a GPU use in combination\n'
        'with MPS (multi-process service). Default: %(default)s')

    checkptfile = 'cifar10_cnn_hvd.weights.best.hdf5'
    parser.add_argument(
        '--checkpt', action='store', nargs='?',
        const=checkptfile,
        help='S|Save (overwrites) and load the model weights if available.'
        '\nOptionally specify a file/filepath if the default name is '
        'undesired.\n(default: {})'.format(checkptfile))

    parser.add_argument('--aug', action='store_true', default=False,
                        help='S|Perform data augmentation on cifar10 set.\n')

    parser.add_argument('--logdevp', action='store_true', default=False,
                        help='S|Log device placement in Tensorflow.\n')

    parser.add_argument(
        '--datadir',
        help='S|Data directory with Cifar10 dataset. Otherwise Keras\n'
        'downloads to "<HOME>/.keras/datasets" directory by default.')

    parser.add_argument(
        '--use-dataset-api', action='store_true', default=False,
        help='S|Use Tensorflow Dataset API for Keras model training.')

    args = parser.parse_args()

    return args


def main(argv=None):
    '''Train a simple deep CNN on the CIFAR10 small images dataset on multigpu
    (and optionally multinode+multigpu) systems via Horovod implementation.
    '''
    argv = sys.argv if argv is None else sys.argv.extend(argv)
    desc = main.__doc__
    # CLI parser
    # args = parser_(argv[1:], desc)
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

    # Pin GPU to local rank. Typically one GPU per process unless
    # oversubscribing GPUs (experimental MPS). In model parallelism it's
    # possible to have multiple GPUs per process.
    # visible_device_list = str(hvd.local_rank()
    gpu_options = tf.GPUOptions(
        allow_growth=True,
        visible_device_list=str(gpu_local_rank))
    config = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement,
        gpu_options=gpu_options)
    KB.set_session(tf.Session(config=config))

    hvdsize = hvd.size()

    checkpt = args.checkpt
    filepath = checkpt

    batch_size = args.batch_size
    num_classes = 10
    epochs = args.epochs

    datadir = args.datadir

    # The data, shuffled and split between train and test sets:
    if hvd.rank() == 0:
        # download only in rank0 i.e. single process
        (x_train, y_train), (x_test, y_test) = cifar10_load_data(datadir)

    hvd_keras.allreduce([0], name="Barrier")
    if hvd.rank() != 0:
        # Data should be downloaded already so load in the other ranks.
        (x_train, y_train), (x_test, y_test) = cifar10_load_data(datadir)

    train_samples = x_train.shape[0]
    test_samples = x_test.shape[0]
    steps_per_epoch = train_samples // batch_size // hvdsize

    print_rank0('{} train samples'.format(train_samples), hvd)
    print_rank0('{} test samples'.format(test_samples), hvd)

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
            print_rank0('Using real-time data augmentation.', hvd)
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

        model = make_model(
            x_train.shape[1:], num_classes, filepath)
    else:
        print_rank0('USING TF DATASET API.', hvd)
        dataset = wrap_as_tfdataset(
            x_train, y_train, args.aug, batch_size, gpu_local_rank,
            prefetch_to_device=True, comm=hvd_keras)
        iterator = dataset.make_one_shot_iterator()

        # Model creation using tensors from the get_next() graph node.
        inputs, targets = iterator.get_next()
        x_train_input = KL.Input(tensor=inputs)

        model_init = make_model(x_train_input, num_classes, filepath)
        x_train_out = model_init.output

        model = Model(inputs=[x_train_input], outputs=[x_train_out])

    # Let's train the model using RMSprop
    lr = 0.0001 * hvdsize

    # opt = KO.RMSprop(lr=lr, decay=1e-6)
    # opt = hvd_keras.DistributedOptimizer(opt)

    opt = tf.train.RMSPropOptimizer(lr)
    # Add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)

    model.compile(
        loss=keras_losses.categorical_crossentropy,
        optimizer=opt,
        metrics=['accuracy'],
        target_tensors=None if not args.use_dataset_api else [targets])

    if hvd.rank() == 0:
        model.summary()

    callbacks = []
    if checkpt and hvd.rank() == 0:
        checkpoint = ModelCheckpoint(
            filepath, monitor='loss', mode='min', verbose=1,
            save_best_only=True)
        callbacks.append(checkpoint)

    if hvd.rank() == 0:
        callbacks += [BatchTiming(), SamplesPerSec(batch_size * hvdsize)]

    # Broadcast initial variable states from rank 0 to all other procs.
    # This is necessary to ensure consistent initialization of all
    # workers when training is started with random weights or restored
    # from a checkpoint.
    # Callback when using horovod.keras as hvd
    # callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    KB.get_session().run(hvd.broadcast_global_variables(0))

    if not args.use_dataset_api:
        start_time = time.time()
        # Fit the model on the batches generated by traingen.flow().
        model.fit_generator(
            traingen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(x_test, y_test) if hvd.rank() == 0 else None,
            verbose=hvd.rank() == 0,
            callbacks=callbacks)
    else:
        # augmentation incorporated in the Dataset pipeline
        start_time = time.time()
        # Validation during training can be incorporated via callback:
        # noqa ref: https://github.com/keras-team/keras/blob/c8bef99ec7a2032b9bea6e9a1260d05a2b6a80f1/examples/mnist_tfrecord.py#L56
        model.fit(
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=hvd.rank() == 0,
            callbacks=callbacks)

    if hvd.rank() != 0:
        return

    elapsed_time = time.time() - start_time
    print('[{}] finished in {} s'
          .format('TRAINING', round(elapsed_time, 3)))

    test_model = model
    if args.use_dataset_api:
        # Create a test-model without Dataset pipeline in the model graph.
        test_model = make_model(x_test.shape[1:], num_classes)
        test_model.compile(
            loss=keras_losses.categorical_crossentropy,
            optimizer=opt,
            metrics=['accuracy'])
        print('SETTING WEIGHTS FOR EVAL WITH DATASET API...')
        test_model.set_weights(model.get_weights())
        print('WEIGHTS SET!!!')

    metrics = test_model.evaluate(x_test, y_test)
    print('\nCIFAR VALIDATION LOSS, ACC: {}, {}'.format(*metrics))


if __name__ == '__main__':
    main()
    # join all ranks and cleanup Keras/Tensorflow session.
    hvd_keras.allreduce([0], name="Barrier")
    KB.clear_session()
