'''
Common functions for Cifar10 examples.
'''
from __future__ import print_function
import os
import numpy as np
import tensorflow as tf
from keras import backend as KB
from keras.models import Sequential
import keras.layers as KL

from keras.datasets import cifar10

TFVER = tf.__version__  # @UndefinedVariable pylint: disable=no-member

if TFVER >= '1.8.0':
    from tensorflow.contrib.data.python.ops import prefetching_ops  # @IgnorePep8 pylint: disable=no-name-in-module,ungrouped-imports


__all__ = (
    'CifarTrainDefaults', 'cifar10_load_data', 'make_model',
    'wrap_as_tfdataset', 'print_rank0',)


class CifarTrainDefaults(object):  # pylint: disable=too-few-public-methods
    '''Cifar common training defaults.'''
    batch_size = 128
    epochs = 200


def cifar10_load_data(datadir=None):
    '''Loads CIFAR10 dataset.

    Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    '''

    if datadir is None:
        return cifar10.load_data()

    dirname = 'cifar-10-batches-py'
    # origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    # path = get_file(dirname, origin=origin, untar=True)
    path_ = os.path.join(datadir, dirname)

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for ii in range(1, 6):
        fpath = os.path.join(path_, 'data_batch_' + str(ii))
        data, labels = cifar10.load_batch(fpath)
        x_train[(ii - 1) * 10000: ii * 10000, :, :, :] = data
        y_train[(ii - 1) * 10000: ii * 10000] = labels

    fpath = os.path.join(path_, 'test_batch')
    x_test, y_test = cifar10.load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if KB.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


# def stand_img(xin):
#     '''Use as: model.add(KL.Lambda(stand_img))
#     Seems to make the code run very slow. Pre-processing the data is faster.
#     '''
#     # KB.map_fn(fn, elems, name, dtype)  # maybe KB.map_fn also works.
#     with tf.device(xin.device):
#         img_std = tf.map_fn(tf.image.per_image_standardization, xin)
#
#     return img_std


def make_model(train_input, num_classes, weights_file=None, small=False):
    '''Return Cifar10 DL model.'''

    if small:
        return make_model_small(train_input, num_classes, weights_file)

    return make_model_full(train_input, num_classes, weights_file)


def make_model_full(train_input, num_classes, weights_file=None):
    '''Return Cifar10 DL model with many layers.

    :param train_input: Either a tf.Tensor input placeholder/pipeline, or a
        tuple input shape.
    '''
    model = Sequential()

    # model.add(KL.InputLayer(input_shape=inshape[1:]))
    if isinstance(train_input, tf.Tensor):
        model.add(KL.InputLayer(input_tensor=train_input))
    else:
        model.add(KL.InputLayer(input_shape=train_input))

    # if standardize:
    #     model.add(KL.Lambda(stand_img))

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


def make_model_small(train_input, num_classes, weights_file=None):
    '''Return Cifar10 DL model with small number layers.'''
    model = Sequential()

    # model.add(KL.InputLayer(input_shape=inshape[1:]))
    if isinstance(train_input, tf.Tensor):
        model.add(KL.InputLayer(input_tensor=train_input))
    else:
        model.add(KL.InputLayer(input_shape=train_input))

    # if standardize:
    #     model.add(KL.Lambda(stand_img))

    model.add(KL.Conv2D(32, (3, 3), padding='same'))
    model.add(KL.Activation('relu'))
    model.add(KL.Flatten())
    # model.add(Dropout(0.5))
    model.add(KL.Dense(num_classes))
    model.add(KL.Activation('softmax'))

    if weights_file is not None and os.path.exists(weights_file):
        model.load_weights(weights_file)

    return model


class DummyComm(object):
    '''Look like a comm object.'''

    @staticmethod
    def size():
        '''Total number of processes running.'''
        return 1

    @staticmethod
    def rank():
        '''Current process enumeration or rank.'''
        return 0

    @staticmethod
    def allreduce(*args, **kwargs):
        '''An allreduce operation.'''
        return


def print_rank0(msg, comm):
    '''Print a message in rank0 process.

    :param comm: An mpi like communicator. Requires rank() interface.
        If using Horovod pass the hvd or hvd_keras comm.
            import horovod.tensorflow as hvd
            import horovod.keras as hvd_keras
    '''
    if comm.rank() == 0:
        print(msg)


def print_in_order(msg, comm):
    '''Horovod/MPI print in order.

    :param comm: An mpi like communicator. Requires size(), rank(), and
        allreduce interfaces. If using Horovod pass the hvd_keras comm.
            import horovod.keras as hvd_keras
    '''
    # ref: https://stackoverflow.com/questions/5305061/ordering-output-in-mpi

    for irank in range(comm.size()):
        # https://github.com/uber/horovod/issues/159
        comm.allreduce([0], name="Barrier")
        if comm.rank() == irank:
            print(msg)


def aug_fn(image):
    '''Cifar10 images distortion for data augmentation.'''
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(image)

    # Because these operations are not commutative, consider
    # randomizing the order their operation.
    # NOTE: since per_image_standardization zeros the mean and
    # makes the stddev unit, this likely has no effect see
    # tensorflow#1458.
    distorted_image = tf.image.random_brightness(
        distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(
        distorted_image, lower=0.2, upper=1.8)

    return image


def wrap_as_tfdataset(
        x_train, y_train, data_augmentation, batch_size,
        gpu_local_rank=None, prefetch_to_device=False,
        comm=DummyComm()):
    '''Wrap numpy data in TF Datasets API.'''
    # ref: https://www.tensorflow.org/versions/master/performance/datasets_performance @IgnorePep8
    buffer_size = tf.contrib.data.AUTOTUNE if TFVER >= '1.8.0' \
        else 1000
    shuffle_buffer = 1000
    # Create the dataset and its associated one-shot iterator.
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    if TFVER >= '1.5.0':
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
            shuffle_buffer))  # , seed=1234 + hvdrank))
    else:
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.repeat()

    if data_augmentation:
        print_rank0('USING IMAGE AUGMENTATION IN DATASET PIPELINE.', comm)

        def proc_dataset(images, labels):
            '''Aug/proc function for map_and_batch.'''
            images = aug_fn(images)
            # The per_image_standardization could be part of the model
            # layers (incorporate via Lambda layer), but preprocessing
            # via dataset pipeline seems to be faster.
            # images = tf.image.per_image_standardization(images)

            # NOTE: If using per_image_standardization then during
            # eval/inference the images have to be standardized as well.
            # Code for that would be:
            #   xtest_dset = tf.data.Dataset.from_tensor_slices(x_test)
            #   xtest_dset = xtest_dset.map(tf.image.per_image_standardization)
            #   test_samples = x_test.shape[0]
            #   xtest_dset = xtest_dset.batch(test_samples)
            #   xtest_gen = xtest_dset.make_one_shot_iterator().get_next()
            #   x_test = KB.get_session().run([xtest_gen])

            return images, labels

        dataset = dataset.apply(tf.contrib.data.map_and_batch(
            map_func=proc_dataset,
            batch_size=batch_size))  # ,num_parallel_batches=4))
    else:
        dataset = dataset.batch(batch_size)

    if TFVER >= '1.8.0' and prefetch_to_device:
        print_in_order(
            'RANK {} PREFETCHING TO GPU: {}'.format(
                comm.rank(), gpu_local_rank),
            comm)
        # Note: In horovod once the visible device list is set you prefetch
        #     to device starting from 0 even when that device is not
        #     physically 0 device.
        # gdev = '/gpu:{}'.format(gpu_local_rank)  # incorrect per Note ^.
        gdev = '/gpu:0'
        # Prefetch to GPU doesn't seem to help much
        dataset = dataset.apply(prefetching_ops.prefetch_to_device(gdev))
        # Don't know what buffer_size to use. Some value is automatically set.
        # , buffer_size=10000
        # Hangs if using AUTOTUNE???
        # , buffer_size=tf.contrib.data.AUTOTUNE
        # failed to query event: CUDA_ERROR_DEINITIALIZED
    else:
        dataset.prefetch(buffer_size=buffer_size)

    # dataset.prefetch(buffer_size=buffer_size)

    return dataset
