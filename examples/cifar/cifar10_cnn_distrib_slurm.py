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

# from time import sleep

import numpy as np
from datetime import datetime
import threading

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
# from keras_exp.multigpu import ModelMGPU
from keras_exp.multigpu import make_parallel
# from keras_exp.multigpu.optimizers import RMSPropMGPU

import tensorflow as tf

from keras_exp.distrib.slurm import SlurmClusterParser
from keras_exp.distrib import (TFClusterManagerFacade, JobType, DevType)

# from functools import partial

_DEVPROF = False


def parser_(desc):
    parser = parser_def_mgpu(desc)
    remove_options(parser, ['--mgpu', '--nccl'])

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

    logdevp_flag = True if _DEVPROF or logdevp else False
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=logdevp_flag,  # True,
                            allow_soft_placement=True,
                            gpu_options=gpu_options)

    cmgr_facade = TFClusterManagerFacade(
        scpar.num_tasks_per_host, scpar.hostnames,
        scpar.num_parameter_servers, scpar.my_proc_id)
    # TF 1.2.x RDMA: specify protocol='grpc+verbs' in server below.
    server = cmgr_facade.get_server(config)
    tfsess = cmgr_facade.get_session(server)
    KB.set_session(tfsess)

    # TODO: Try
    #     sv = tf.train.Supervisor(...)
    #     with sv.managed_session(server.target, config=config) ...
    #     sess = sv.prepare_or_wait_for_session(server.target,
    #                                           config=sess_config)
    #     KB.set_session(tfsess)  # based on this managed session.

    #: :type cluster_spec: tf.train.ClusterSpec
    # cluster_spec = cmgr_facade.get_cluster_spec()
    job_type = cmgr_facade.myjobtype
    # task_id = cmgr_facade.mytask_id

    is_chief = cmgr_facade.is_chief

    if job_type == JobType.ps:
        # JOIN PARAMETER SERVERS
        # server.join()
        cmgr_facade.join(server)

    # Once the server is started everything but the chief worker can join
    # the server and wait to process/service graph computations. Chief pushes
    # the compute graph.
    if not is_chief:
        # JOIN WORKERS (PS also) EXCEPT FOR CHIEF
        cmgr_facade.join(server)

    # sleep(2)  # Have the chief wait just in case. Occasionally get errors.

    # The ngpus per host needs to be done with MPI or somehow sync'd. Currently
    # assuming all hosts have the same number of GPUs.
    gdev_list = get_available_gpus()
    ngpus = len(gdev_list)

    #: :type mywgdev: tf.DeviceSpec
    # mywgdev, wgdev_list = cmgr_facade.get_workers_dev_list(ngpus)
    _, wgdev_list = cmgr_facade.get_workers_dev_list(ngpus)
    nworker_devices_total = len(wgdev_list)
    # print('\n\tCLUSTER_SPEC_DICT: {}\n\tWGDEV_LIST: {}\n'
    #       .format(cmgr_facade.clusterspec_dict,
    #               [dev.to_string() for dev in wgdev_list]))  # DEBUG

    # ------------------------------------ Data loading and basic preprocessing
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

    print(x_train.shape, 'train shape')

    # --------------------------------------------- Setup model and parallelize
    def _load_fn(unused_op):
        return 1

    cspec = cmgr_facade.get_cluster_spec()
    num_ps = cspec.num_tasks(JobType.ps)
    ps_strategy = \
        tf.contrib.training.GreedyLoadBalancingStrategy(num_ps, _load_fn)

    ps_device = tf.DeviceSpec(job=JobType.ps, device_type=DevType.cpu,
                              device_index=0).to_string()

    rdsetter = tf.train.replica_device_setter(
        cluster=cspec,
        ps_strategy=ps_strategy,
        ps_device=ps_device,  # '/job:ps/cpu:0'  # seems to work
    )
    with tf.device(rdsetter):
        model_init = make_model(x_train.shape, num_classes,
                                filepath if checkpt_flag else None)

    # model_init = partial(make_model, x_train.shape, num_classes,
    #                      filepath if checkpt_flag else None)

    if checkpt_flag:
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1,
                                     save_best_only=True, mode='max')
        callbacks = [checkpoint]

    print('\n\tCLUSTER_SPEC_DICT: {}\n\tWGDEV_LIST: {}\n'
          .format(cmgr_facade.clusterspec_dict,
                  [dev.to_string() for dev in wgdev_list]))  # DEBUG

    batch_size = batch_size * nworker_devices_total
    # batch_size = 40000  # split over four devices works fine no grad avg
    # batch_size = 25000  # split over four devices works fine w/ grad avg

    # ps_device = rdsetter
    # ps_device = '/job:ps/cpu:0'
    # ps_device = '/cpu:0'
    # ps_device = tf.train.replica_device_setter(
    #     ps_device="/job:ps/cpu:0",
    #     worker_device=mywgdev.to_string(),
    #     cluster=cmgr_facade.get_cluster_spec())

    # TODO: Use replica_device_setter and do not join workers above. Try using
    # managed session.
    # Need to think about this because what I want is to have the workers
    # load the relevant data on the node they are running from instead of
    # loading on chief rank's node and transferring slices over network.
    # Maybe parameter servers can do this via ZeroMQ.
    # Ref: https://gist.github.com/fchollet/2c9b029f505d94e6b8cd7f8a5e244a4e

    # Data-Parallelize the model via function or class.
    model = make_parallel(model_init, wgdev_list, ps_device=ps_device)
    # model = ModelMGPU(serial_model=model_init, gdev_list=gpus_list,
    #                   syncopt=syncopt, usenccl=usenccl, enqueue=enqueue)
    print_mgpu_modelsummary(model)

    # ------------------------------------------------------------ Run training
    opt = RMSprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    nsamples = x_train.shape[0]
    steps_per_epoch = nsamples // batch_size

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=callbacks)

        # Fit the model on the batches generated by datagen.flow().
        # mygen = mygenerator(nsamples, batch_size, x_train, y_train)
        # model.fit_generator(mygen,
        #                     steps_per_epoch=steps_per_epoch,
        #                     epochs=epochs,
        #                     validation_data=(x_test, y_test),
        #                     callbacks=callbacks)

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
    # if not is_chief:
    #     # JOIN WORKERS (PS also) EXCEPT FOR CHIEF
    #     cmgr_facade.join(server)
    cmgr_facade.stop_chief(server)


if __name__ == '__main__':
    main()

