# https://stackoverflow.com/questions/42184863/how-do-you-make-tensorflow-keras-fast-with-a-tfrecord-dataset
'''
MNIST dataset with TensorFlow TFRecords.

Call this example:
    # best achieved performance with 2 GPUs
    time CUDA_VISIBLE_DEVICES=0,1 python mnist_tfrecord_mgpu.py
    # or no timeing info:
    CUDA_VISIBLE_DEVICES=0,1 python mnist_tfrecord_mgpu.py

    # vary CUDA_VISIBLE_DEVICES for more than 2 GPUs. Performance starts to
    # degrade.
    time CUDA_VISIBLE_DEVICES=0,1,3,4 python mnist_tfrecord_mgpu.py

    # compare 2 GPUs to one GPU
    time CUDA_VISIBLE_DEVICES=0 python mnist_tfrecord_mgpu.py

    time CUDA_VISIBLE_DEVICES=0,1 python mnist_tfrecord_mgpu.py
    # The overall walltime for this whole example might be faster with 1 GPU
    # due to startup time being longer for multigpu case, but the training
    # portion is faster with 2 GPUs. Running on P100s 50 epochs:
    #     nGPUs | training (sec). | walltime (sec.)
    #         1 | ~ 4.3           | ~ 11
    #         2 | ~ 3.4           | ~ 14
    #
    # Degrades with > 2 GPUs because the mnist model is not significant enough
    # to stress the computing of the GPUs so the startup/comm. overhead is
    # greater than speedup achieved due to data-parallelism.

Using TFRecord queues with Keras can be a significant performance booster. When
the model and batch sizes are signifcantly large, using multigpu with TFRecord
queue can give additional performance boost to Keras.

'''
import os
import time

import numpy as np

import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from keras import backend as KB
from keras.models import Model
from keras.layers import (
    Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D)
from keras.optimizers import RMSprop

from keras.models import Sequential
import keras.layers as KL

from keras.objectives import categorical_crossentropy
from keras.utils import np_utils
from keras.utils.generic_utils import Progbar

from keras.datasets import mnist

from keras_exp.multigpu import (
    get_available_gpus, make_parallel, print_mgpu_modelsummary, ModelMGPU)

from keras_exp.mixin_models.tfrecord import (
    ModelTFRecordMixin, ModelCheckpointTFRecord)


class Model_TFrecord(ModelTFRecordMixin, Model):
    pass


class ModelMGPU_TFrecord(ModelTFRecordMixin, ModelMGPU):
    pass


if KB.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend for the time being, '
                       'because it requires TFRecords, which '
                       'are not supported on other platforms.')


def images_to_tfrecord(images, labels, filename):
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    """ Save data into TFRecord """
    if not os.path.isfile(filename):
        num_examples = images.shape[0]

        rows = images.shape[1]
        cols = images.shape[2]
        depth = images.shape[3]

        print('Writing', filename)
        writer = tf.python_io.TFRecordWriter(filename)
        for index in range(num_examples):
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(rows),
                'width': _int64_feature(cols),
                'depth': _int64_feature(depth),
                'label': _int64_feature(int(labels[index])),
                'image_raw': _bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        writer.close()
    else:
        print('tfrecord %s already exists' % filename)


def read_and_decode_recordinput(
        tf_glob, one_hot=True, classes=None, is_train=None,
        batch_shape=[1000, 28, 28, 1], parallelism=1):
    """ Return tensor to read from TFRecord """
    print('\nCreating graph for loading {} TFRecords...'.format(tf_glob))
    with tf.variable_scope("TFRecords"):
        record_input = data_flow_ops.RecordInput(
            tf_glob, batch_size=batch_shape[0], parallelism=parallelism)
        records_op = record_input.get_yield_op()
        records_op = tf.split(records_op, batch_shape[0], 0)
        records_op = [tf.reshape(record, []) for record in records_op]
        progbar = Progbar(len(records_op))

        images = []
        labels = []
        for i, serialized_example in enumerate(records_op):
            progbar.update(i)
            with tf.variable_scope("parse_images", reuse=True):
                features = tf.parse_single_example(
                    serialized_example,
                    features={
                        'label': tf.FixedLenFeature([], tf.int64),
                        'image_raw': tf.FixedLenFeature([], tf.string),
                    })
                img = tf.decode_raw(features['image_raw'], tf.uint8)
                img.set_shape(batch_shape[1] * batch_shape[2])
                img = tf.reshape(img, [1] + batch_shape[1:])

                img = tf.cast(img, tf.float32) * (1. / 255) - 0.5

                label = tf.cast(features['label'], tf.int32)
                if one_hot and classes:
                    label = tf.one_hot(label, classes)

                images.append(img)
                labels.append(label)

        progbar.update(i + 1)

        images = tf.parallel_stack(images, 0)
        labels = tf.parallel_stack(labels, 0)
        images = tf.cast(images, tf.float32)

        images = tf.reshape(images, shape=batch_shape)

        return images, labels


def read_and_decode_recordinput2(
        tf_glob, one_hot=True, classes=None, is_train=None,
        batch_shape=[1000, 28, 28, 1], parallelism=1):
    '''Return tensor to read from TFRecord'''
    filename_queue = tf.train.string_input_producer([tf_glob])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['image_raw'], tf.uint8)
    img.set_shape([batch_shape[1] * batch_shape[2]])
    img = tf.reshape(img, batch_shape[1:])

    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    # img = tf.cast(img, tf.float32) * (1. / 255)

    label = tf.cast(features['label'], tf.int32)
    if one_hot and classes:
        label = tf.one_hot(label, classes)

    batch_size = batch_shape[0]
    if is_train:
        x_train_batch, y_train_batch = tf.train.shuffle_batch(
            [img, label],
            batch_size=batch_size,
            capacity=2000,
            min_after_dequeue=1000,
            num_threads=parallelism)  # set the number of threads here
    else:
        x_train_batch, y_train_batch = tf.train.batch(
            [img, label],
            batch_size=batch_size,
            capacity=2000,
            num_threads=parallelism)  # set the number of threads here

    return x_train_batch, y_train_batch

    # num_gpus = 1
    # batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
    #     [x_train_batch, y_train_batch], capacity=2 * num_gpus)
    #
    # image_batch, label_batch = batch_queue.dequeue()
    # return image_batch, label_batch


def save_mnist_as_tfrecord(X_train, y_train, X_test, y_test):
    images_to_tfrecord(images=X_train, labels=y_train,
                       filename='train.mnist.tfrecord')
    images_to_tfrecord(images=X_test, labels=y_test,
                       filename='test.mnist.tfrecord')


def cnn_layers_list(nclasses):
    ll = []
    ll.append(Conv2D(32, (3, 3), activation='relu', padding='valid'))
    ll.append(Conv2D(64, (3, 3), activation='relu'))
    ll.append(MaxPooling2D(pool_size=(2, 2)))
    ll.append(Dropout(0.25))
    ll.append(Flatten())
    ll.append(Dense(128, activation='relu'))
    ll.append(Dropout(0.5))
    ll.append(Dense(nclasses, activation='softmax', name='x_train_out'))

    return ll


def cnn_layers(x_train_input, nclasses):
    ll = cnn_layers_list(nclasses)
    x = x_train_input
    for il in ll:
        x = il(x)

    return x


def make_model(x_train_input, nclasses):
    model = Sequential()
    model.add(KL.InputLayer(input_tensor=x_train_input))
    ll = cnn_layers_list(nclasses)
    for il in ll:
        model.add(il)

    return model


def main():
    sess = tf.Session()
    KB.set_session(sess)

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    save_mnist_as_tfrecord(X_train, y_train, X_test, y_test)
    print('X_train shape: {}'.format(X_train.shape))

    gdev_list = get_available_gpus()
    ngpus = len(gdev_list)

    batch_size = 100 * ngpus
    train_samples = X_train.shape[0]  # 60000
    height_nrows = X_train.shape[1]  # 28
    width_ncols = X_train.shape[2]  # 28
    batch_shape = [batch_size, height_nrows, width_ncols, 1]
    epochs = 50
    steps_per_epoch = train_samples / batch_size
    nclasses = 10
    parallelism = 10  # threads for tf queue readers

    def rdi():
        # x_train_batch, y_train_batch = read_and_decode_recordinput2(
        x_train_batch, y_train_batch = read_and_decode_recordinput(
            './train.mnist.tfrecord',
            one_hot=True,
            classes=nclasses,
            is_train=True,
            batch_shape=batch_shape,
            parallelism=parallelism)

        return x_train_batch, y_train_batch

    with tf.device('/cpu:0'):
        x_train_batch, y_train_batch = rdi()

    # x_test_batch, y_test_batch = read_and_decode_recordinput2(
    x_test_batch, y_test_batch = read_and_decode_recordinput(
        './test.mnist.tfrecord',
        one_hot=True,
        classes=nclasses,
        is_train=False,
        batch_shape=batch_shape,
        parallelism=parallelism)

    x_batch_shape = x_train_batch.get_shape().as_list()
    y_batch_shape = y_train_batch.get_shape().as_list()

    x_train_input = Input(tensor=x_train_batch, batch_shape=x_batch_shape)
    y_train_in_out = Input(tensor=y_train_batch, batch_shape=y_batch_shape,
                           name='y_labels')

    if ngpus < 2:
        # x_train_out = cnn_layers(x_train_input, nclasses)
        # train_model = Model_TFrecord(inputs=[x_train_input],
        #                              outputs=[x_train_out])
        model_init = make_model(x_train_input, nclasses)
        x_train_out = model_init.output
        train_model = Model_TFrecord(inputs=[x_train_input],
                                     outputs=[x_train_out])
    else:
        model_init = make_model(x_train_input, nclasses)
        train_model = make_parallel(model_init, gdev_list,
                                    model_class=ModelMGPU_TFrecord)
        x_train_out = train_model.output

    # cce = categorical_crossentropy(y_train_batch, x_train_out)  # works too
    cce = categorical_crossentropy(y_train_in_out, x_train_out)
    train_model.add_loss(cce)

    lr = 0.001 * ngpus
    opt = RMSprop(lr=lr)
    train_model.compile(optimizer=opt,  # 'rmsprop',
                        loss=None,
                        metrics=['accuracy'])
    if ngpus > 1:
        print_mgpu_modelsummary(train_model)
    else:
        train_model.summary()

    # Callbacks
    checkpoint = ModelCheckpointTFRecord(
        'saved_wt.h5', monitor='val_loss', verbose=0,
        save_best_only=True,
        save_weights_only=True)
    callbacks = []  # [checkpoint]  # []
    # Training slower with callback. Multigpu slower with callback during
    # training than 1 GPU. Again, mnist is too trivial of a model and dataset
    # to benchmark or stress GPU compute capabilities. I set up this example
    # to illustrate potential for speedup of multigpu case trying to use mnist
    # as a stressor.
    # It's like comparing a 5 ft race between a person and a truck. A truck is
    # obviously faster than a person but in a 5 ft race the person will likely
    # win due to slower startup for the truck.
    # I will re-implement this with Cifar that should be a better benchmark.

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    start_time = time.time()
    train_model.fit_tfrecord(
        batch_size=batch_size,
        validation_data=(x_test_batch, y_test_batch),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks)
    elapsed_time = time.time() - start_time
    print('[{}] finished in {} ms'.format('TRAINING',
                                          int(elapsed_time * 1000)))

    if not callbacks:  # empty list
        train_model.save_weights('saved_wt.h5')

    KB.clear_session()

    # Second Session, pure Keras. Demonstrate that the model works and is
    # independent of the TFRecord pipeline.
    x_test_inp = Input(batch_shape=(None,) + (X_test.shape[1:]))
    test_out = cnn_layers(x_test_inp, nclasses)
    test_model = Model(inputs=x_test_inp, outputs=test_out)

    test_model.load_weights('saved_wt.h5')

    test_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                       metrics=['accuracy'])
    test_model.summary()

    loss, acc = test_model.evaluate(
        X_test, np_utils.to_categorical(y_test), nclasses)
    print('\nTest loss: {0}'.format(loss))
    print('\nTest accuracy: {0}'.format(acc))


if __name__ == '__main__':
    main()

