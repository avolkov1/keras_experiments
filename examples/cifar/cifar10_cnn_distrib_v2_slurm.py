#!/usr/bin/env python
'''Train a simple deep CNN on the CIFAR10 small images dataset.

Distributed training.
'''
from __future__ import print_function

import sys
import os

from argparse import SUPPRESS

# from time import sleep

from parser_common import (parser_def_mgpu, remove_options)

from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import keras.layers as KL
from keras import backend as KB

from keras.callbacks import ModelCheckpoint

from keras.optimizers import RMSprop

from keras_exp.multigpu import (get_available_gpus, print_mgpu_modelsummary)
from keras_exp.multigpu import make_parallel

import tensorflow as tf

from keras_exp.distrib.slurm import SlurmClusterParser
from keras_exp.distrib import TFClusterManagerFacade, JobType  # , DevType

# from functools import partial

_DEVPROF = False


def parser_(desc):
    parser = parser_def_mgpu(desc)
    remove_options(parser, ['--mgpu', '--nccl'])

    checkptfile = 'cifar10_cnn_distrib_v2.weights.best.hdf5'
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


def make_model(inshape, num_classes, weights_file=None):
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


def main(argv=None):
    '''
    '''
    main.__doc__ = __doc__

    argv = sys.argv if argv is None else sys.argv.extend(argv)
    desc = main.__doc__  # .format(os.path.basename(__file__))
    # CLI parser
    args = parser_(desc)
    # mgpu = 0 if getattr(args, 'mgpu', None) is None else args.mgpu
    # enqueue = args.enqueue
    # usenccl = args.nccl
    # syncopt = args.syncopt
    rdma = args.rdma

    checkpt = getattr(args, 'checkpt', None)
    checkpt_flag = False if checkpt is None else True
    filepath = checkpt
    # print('CHECKPT:', checkpt)

    batch_size = 32
    num_classes = 10
    epochs = args.epochs
    data_augmentation = args.aug

    logdevp = args.logdevp

    # ---------------------------------------------- Distributed setup on SLURM
    scpar = SlurmClusterParser()
    cmgr_facade = TFClusterManagerFacade(scpar)

    logdevp_flag = True if _DEVPROF or logdevp else False
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=logdevp_flag,  # True,
                            allow_soft_placement=True,
                            gpu_options=gpu_options)

    # TF 1.2.x RDMA: specify protocol='grpc+verbs' in server below.
    server = cmgr_facade.get_server(
        config,
        protocol='grpc+verbs' if rdma else None)
    tfsess = cmgr_facade.get_session(server)
    KB.set_session(tfsess)

    #: :type cluster_spec: tf.train.ClusterSpec
    # cluster_spec = cmgr_facade.get_cluster_spec()
    job_type = cmgr_facade.myjobtype
    # task_id = cmgr_facade.mytask_id

    is_chief = cmgr_facade.is_chief

    if job_type == JobType.ps:
        # JOIN PARAMETER SERVERS
        # server.join()
        cmgr_facade.join(server)

    ps_device = cmgr_facade.get_mypsdevice()
    print('MYPS_DEVICE: {}'.format(ps_device))  # DEBUG

    # sleep(2)  # Have the chief wait just in case. Occasionally get errors.

    # The ngpus per host needs to be done with MPI or somehow sync'd. Currently
    # assuming all hosts have the same number of GPUs.
    gdev_list = get_available_gpus()
    ngpus = len(gdev_list)

    # List of all devices. The devices might be associated to the same worker.
    wgdev_list = cmgr_facade.get_allworkers_devlist(ngpus)
    # If 2 workers ea. w/ 4 devices then nworker_devices_total == 2 * 4 = 8
    # If 4 workers ea. w/ 1 devices then nworker_devices_total == 4 * 1 = 4
    # nworker_devices_total = len(wgdev_list)

    # Number of workers, not devices. Each worker can have multiple devices.
    num_workers = cmgr_facade.num_workers

    # List of devices associated with current worker/task.
    mydevlist = cmgr_facade.get_mydevlist(ngpus)
    nmydevs = len(mydevlist)
    batch_size = batch_size * nmydevs

    # ------------------------------------ Data loading and basic preprocessing
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    nsamples = x_train.shape[0]
    steps_per_epoch = (nsamples // num_workers) // batch_size

    # NOTE: Naive dataset below split. With such a naive approach the random
    #     sampling gets screwed up. The convergence rate is slower as a
    #     result (hence defeats the purpose of scaling since more iterations
    #     are required when using more nodes), and if scaling to very many
    #     nodes might not converge. Instead using a generator that
    #     randomly chooses the samples for "mypart". Maybe implement a
    #     custom ImageDataGenerator for distributed case.
    # split train dataset for myrank
    # mytaskid = mypart = cmgr_facade.mytask_id
    # nn = x_train.shape[0] // num_workers
    # i1 = mypart * nn
    # if mypart == num_workers - 1:
    #     x_train = x_train[i1:, ...]
    #     y_train = y_train[i1:, ...]
    # else:
    #     i2 = (mypart + 1) * nn
    #     x_train = x_train_[i1:i2, ...]
    #     y_train = y_train[i1:i2, ...]
    # print('TASK {}: train samples {}'.format(mytaskid, x_train.shape[0]))
    # print('TASK {}: test samples {}'.format(mytaskid, x_test.shape[0]))
    # nsamples = x_train.shape[0]
    # steps_per_epoch = nsamples // batch_size

    # --------------------------------------------- Setup model and parallelize
    def _load_fn(unused_op):
        return 1

    cspec = cmgr_facade.get_cluster_spec()
    num_ps = cmgr_facade.num_ps
    ps_strategy = \
        tf.contrib.training.GreedyLoadBalancingStrategy(num_ps, _load_fn)

    rdsetter = tf.train.replica_device_setter(
        cluster=cspec,
        ps_strategy=ps_strategy,
    )
    with tf.device(rdsetter):
        model_init = make_model(
            x_train.shape, num_classes,
            filepath if checkpt_flag else None
        )

    # if using checkpointing callback enable it on chief or use unique
    # filepath for each worker task.
    callbacks = None
    if checkpt_flag and is_chief:
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='max')
        callbacks = [checkpoint]

    if is_chief:
        print('\n\tCLUSTER_SPEC_DICT: {}\n\tWGDEV_LIST: {}\n'
              .format(cmgr_facade.clusterspec_dict,
                      [dev.to_string() for dev in wgdev_list]))  # DEBUG

    print('\n\tMYWGDEV_LIST: {}\n'
          .format([dev.to_string() for dev in mydevlist]))  # DEBUG

    # Data-Parallelize the model via function or class.
    model = make_parallel(model_init, mydevlist, ps_device=ps_device)
    print_mgpu_modelsummary(model)

    # ------------------------------------------------------------ Run training
    lr = 0.0001 * nmydevs
    # lr = 0.0001 * nworker_devices_total
    opt = RMSprop(lr=lr, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    if not data_augmentation:
        print('Not using data augmentation.')
        # model.fit(x_train, y_train,
        #           batch_size=batch_size,
        #           epochs=epochs,
        #           validation_data=(x_test, y_test),
        #           shuffle=True,
        #           callbacks=callbacks)  # verbose=is_chief)

        datagen = ImageDataGenerator()
        datagen.fit(x_train)
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            callbacks=callbacks)

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

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            validation_data=(x_test, y_test),
                            callbacks=callbacks)

    # ------------------------------------------------------------- STOP SERVER
    if not is_chief:
        # JOIN WORKERS EXCEPT FOR CHIEF
        cmgr_facade.join(server)

    cmgr_facade.stop_chief(server)


if __name__ == '__main__':
    main()

