#!/usr/bin/env python
'''Train a simple deep CNN on the CIFAR10 small images dataset.

Using TFRecord queues with Keras and multi-GPU enabled. Compare with:
https://github.com/avolkov1/keras_experiments/blob/master/examples/cifar/cifar10_cnn_mgpu.py
https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py

Use Tensorflow version 1.2.x and above for good performance.
'''

from __future__ import print_function
import sys
import os
import time

from argparse import SUPPRESS

import numpy as np

from parser_common import parser_def_mgpu, remove_options

import tensorflow as tf

# from keras.utils.data_utils import get_file
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.models import Sequential, Model
import keras.layers as KL
from keras import backend as KB

from keras.callbacks import ModelCheckpoint

from keras.optimizers import RMSprop

from keras_exp.multigpu import (get_available_gpus, print_mgpu_modelsummary)
# from keras_exp.multigpu import ModelMGPU
from keras_exp.multigpu import make_parallel


_DEVPROF = False

checkptfile = 'cifar10_cnn_tfqueue_var.weights.hdf5'


def parser_(desc):
    parser = parser_def_mgpu(desc)

    remove_options(parser, ['--nccl', '--enqueue', '--syncopt', '--rdma'])

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


def make_model(train_input, num_classes, weights_file=None):
    '''
    :param train_input: Either tensorflow Tensor or tuple/list shape. Bad style
        since the parameter can be of different types, but seems Ok here.
    :type train_input: tf.Tensor or tuple/list
    '''
    model = Sequential()
    # model.add(KL.InputLayer(input_shape=inshape[1:]))
    if isinstance(train_input, tf.Tensor):
        model.add(KL.InputLayer(input_tensor=train_input))
    else:
        model.add(KL.InputLayer(input_shape=train_input))
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
    mgpu = 0 if getattr(args, 'mgpu', None) is None else args.mgpu

    checkpt = getattr(args, 'checkpt', None)
    checkpt_flag = False if checkpt is None else True
    filepath = checkpt
    # print('CHECKPT:', checkpt)

    gdev_list = get_available_gpus(mgpu or 1)
    ngpus = len(gdev_list)

    batch_size_1gpu = 32
    batch_size = batch_size_1gpu * ngpus
    num_classes = 10
    epochs = args.epochs
    data_augmentation = args.aug

    logdevp = args.logdevp

    datadir = getattr(args, 'datadir', None)

    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10_load_data(datadir) \
        if datadir is not None else cifar10.load_data()
    train_samples = x_train.shape[0]
    test_samples = y_test.shape[0]
    steps_per_epoch = train_samples // batch_size
    # validations_steps = test_samples // batch_size
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Squeeze is to deal with to_categorical bug in Keras 2.1.0 which
    # was fixed in Keras 2.1.1
    y_train = to_categorical(y_train, num_classes).astype(np.float32).squeeze()
    y_test = to_categorical(y_test, num_classes).astype(np.float32).squeeze()

    x_train_feed = x_train.reshape(train_samples, -1)
    y_train_feed = y_train.reshape(train_samples, -1)

    # The capacity variable controls the maximum queue size
    # allowed when prefetching data for training.
    capacity = 10000

    # min_after_dequeue is the minimum number elements in the queue
    # after a dequeue, which ensures sufficient mixing of elements.
    # min_after_dequeue = 3000

    # Force input pipeline to CPU:0 to avoid data operations ending up on GPU
    # and resulting in a slow down for multigpu case due to comm overhead.
    with tf.device('/cpu:0'):
        # ref: https://www.tensorflow.org/api_guides/python/reading_data#Preloaded_data @IgnorePep8
        # Using tf.Variable instead of tf.constant uses less memory, because
        # the constant is stored inline in the graph data structure which may
        # be duplicated a few times. The placeholder/variable either is not
        # duplicated or the duplication will not consume memory since it's a
        # placeholder.
        with tf.name_scope('input'):
            # Input data
            images_initializer = tf.placeholder(
                dtype=x_train.dtype,
                shape=x_train_feed.shape)
            labels_initializer = tf.placeholder(
                dtype=y_train.dtype,
                shape=y_train_feed.shape)
            # Setting trainable=False keeps the variable out of the
            # GraphKeys.TRAINABLE_VARIABLES collection in the graph, so we
            # won't try and update it when training. Setting collections=[]
            # keeps the variable out of the GraphKeys.GLOBAL_VARIABLES
            # collection used for saving and restoring checkpoints
            input_images = tf.Variable(
                images_initializer, trainable=False, collections=[])
            input_labels = tf.Variable(
                labels_initializer, trainable=False, collections=[])

        image, label = tf.train.slice_input_producer(
            [input_images, input_labels], shuffle=True)
        # If using num_epochs=epochs have to:
        #     sess.run(tf.local_variables_initializer())
        #     and maybe also: sess.run(tf.global_variables_initializer())
        image = tf.reshape(image, x_train.shape[1:])

        test_images = tf.constant(x_test.reshape(test_samples, -1))
        test_image, test_label = tf.train.slice_input_producer(
            [test_images, y_test], shuffle=False)
        test_image = tf.reshape(test_image, x_train.shape[1:])

        if data_augmentation:
            print('Using real-time data augmentation.')
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

            # Subtract off the mean and divide by the variance of the
            # pixels.
            image = tf.image.per_image_standardization(distorted_image)

            # Do this for testing as well if standardizing
            test_image = tf.image.per_image_standardization(test_image)

        # Use tf.train.batch if slice_input_producer shuffle=True,
        # otherwise use tf.train.shuffle_batch. Not sure which way is faster.
        x_train_batch, y_train_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            capacity=capacity,
            num_threads=8)

        x_test_batch, y_test_batch = tf.train.batch(
            [test_image, test_label],
            batch_size=test_samples,
            capacity=capacity,
            num_threads=1,
            name='test_batch',
            shared_name='test_batch')

    x_train_input = KL.Input(tensor=x_train_batch)

    callbacks = None

    if _DEVPROF or logdevp:  # or True:
        # Setup Keras session using Tensorflow
        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=True)
        # config.gpu_options.allow_growth = True
        tfsess = tf.Session(config=config)
        KB.set_session(tfsess)

    model_init = make_model(x_train_input, num_classes,
                            filepath if checkpt_flag else None)
    x_train_out = model_init.output
    # model_init.summary()

    lr = 0.0001 * ngpus
    if ngpus > 1:
        model = make_parallel(model_init, gdev_list)
    else:
        # Must re-instantiate model per API below otherwise doesn't work.
        model_init = Model(inputs=[x_train_input], outputs=[x_train_out])
        model = model_init

    opt = RMSprop(lr=lr, decay=1e-6)
    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'],
                  target_tensors=[y_train_batch])

    print_mgpu_modelsummary(model)  # will print non-mgpu model as well

    if checkpt_flag:
        checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1,
                                     save_best_only=True)
        callbacks = [checkpoint]

    # Start the queue runners.
    sess = KB.get_session()

    # Create the op for initializing variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Run the Op to initialize the variables.
    sess.run(init_op)
    sess.run(input_images.initializer,
             feed_dict={images_initializer: x_train_feed})
    sess.run(input_labels.initializer,
             feed_dict={labels_initializer: y_train_feed})

    # Fit the model using data from the TFRecord data tensors.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    val_in_train = False  # not sure how the validation part works during fit.
    start_time = time.time()
    model.fit(
        # validation_data=(x_test_batch, y_test_batch)
        # if val_in_train else None,  # validation data is not used???
        # validation_steps=validations_steps if val_in_train else None,
        validation_steps=val_in_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks)
    elapsed_time = time.time() - start_time
    print('[{}] finished in {} s'.format('TRAINING', round(elapsed_time, 3)))

    weights_file = checkptfile  # './saved_cifar10_wt.h5'
    if not checkpt_flag:  # empty list
        model.save_weights(checkptfile)

    KB.clear_session()

    # Second Session. Demonstrate that the model works
    # test_model = make_model(x_test.shape[1:], num_classes,
    #                         weights_file=weights_file)
    test_model = make_model(x_test.shape[1:], num_classes)
    test_model.load_weights(weights_file)
    test_model.compile(loss='categorical_crossentropy',
                       optimizer=opt,
                       metrics=['accuracy'])

    if data_augmentation:
        # Need to run x_test through per_image_standardization otherwise
        # results get messed up.
        x_processed, y_processed = sess.run([x_test_batch, y_test_batch])
        # DEBUGGING
        # xdiff = np.abs(x_test - x_processed)
        # print('MAX XDIFF: {}'.format(np.max(xdiff)))
        # ydiff = np.abs(y_test - y_processed)
        # print('y_test: {}'.format(y_test[0:5, :]))
        # print('y_processed: {}'.format(y_processed[0:5, :]))
        # print('ydiff: {}'.format(ydiff[-10:, :]))
        # print('SUM YDIFF: {}'.format(np.sum(ydiff)))

        loss, acc = test_model.evaluate(x_processed, y_processed)
    else:
        loss, acc = test_model.evaluate(x_test, y_test)

    print('\nTest loss: {0}'.format(loss))
    print('\nTest accuracy: {0}'.format(acc))

    # Clean up the TF session.
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()

