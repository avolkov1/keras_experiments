'''
MNIST dataset with TensorFlow TFRecords. Refer to:
    https://github.com/fchollet/keras/blob/master/examples/mnist_tfrecord.py

This is an implementation for multi-GPU systems.

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
    #         1 | ~ 2             | ~ 16
    #         2 | ~ 1             | ~ 12
    #
    # Degrades with > 2 GPUs because the mnist model is not significant enough
    # to stress the computing of the GPUs so the startup/comm. Overhead is
    # greater than speedup achieved due to data-parallelism.

Using TFRecord queues with Keras can be a significant performance booster. When
the model and batch sizes are signifcantly large, using multigpu with TFRecord
queue can give additional performance boost to Keras.

'''
import time

import numpy as np

import tensorflow as tf
from keras import backend as KB
from keras.models import Model
from keras.layers import (
    Dense, Dropout, Flatten, Input, Conv2D, MaxPooling2D)
from keras.optimizers import RMSprop

from keras.models import Sequential
import keras.layers as KL

from keras.callbacks import ModelCheckpoint

from keras.utils import to_categorical

from tensorflow.contrib.learn.python.learn.datasets import mnist

from keras_exp.multigpu import (
    get_available_gpus, make_parallel, print_mgpu_modelsummary)


if KB.backend() != 'tensorflow':
    raise RuntimeError('This example can only run with the '
                       'TensorFlow backend, '
                       'because it requires TFRecords, which '
                       'are not supported on other platforms.')


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
    '''Non-functional model definition.'''
    model = Sequential()
    model.add(KL.InputLayer(input_tensor=x_train_input))
    ll = cnn_layers_list(nclasses)
    for il in ll:
        model.add(il)

    return model


def main():
    # user options
    batch_size = 128
    val_in_train = False  # not sure how the validation part works during fit.
    use_model_checkpt = False

    # demo processing
    sess = tf.Session()
    KB.set_session(sess)

    gdev_list = get_available_gpus()
    ngpus = len(gdev_list)
    batch_size = batch_size * ngpus

    data = mnist.load_mnist()
    X_train = data.train.images
    # X_test = data.test.images
    train_samples = X_train.shape[0]  # 60000
    # test_samples = X_test.shape[0]  # 10000
    height_nrows = 28
    width_ncols = 28
    batch_shape = [batch_size, height_nrows, width_ncols, 1]
    epochs = 5
    steps_per_epoch = train_samples / batch_size
    # validations_steps = test_samples / batch_size
    nclasses = 10

    # The capacity variable controls the maximum queue size
    # allowed when prefetching data for training.
    capacity = 10000

    # min_after_dequeue is the minimum number elements in the queue
    # after a dequeue, which ensures sufficient mixing of elements.
    min_after_dequeue = 3000

    # If `enqueue_many` is `False`, `tensors` is assumed to represent a
    # single example.  An input tensor with shape `[x, y, z]` will be output
    # as a tensor with shape `[batch_size, x, y, z]`.
    #
    # If `enqueue_many` is `True`, `tensors` is assumed to represent a
    # batch of examples, where the first dimension is indexed by example,
    # and all members of `tensors` should have the same size in the
    # first dimension.  If an input tensor has shape `[*, x, y, z]`, the
    # output will have shape `[batch_size, x, y, z]`.
    enqueue_many = True

    x_train_batch, y_train_batch = tf.train.shuffle_batch(
        tensors=[data.train.images, data.train.labels.astype(np.int32)],
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        enqueue_many=enqueue_many,
        num_threads=8)

    x_train_batch = tf.cast(x_train_batch, tf.float32)
    x_train_batch = tf.reshape(x_train_batch, shape=batch_shape)

    y_train_batch = tf.cast(y_train_batch, tf.int32)
    y_train_batch = tf.one_hot(y_train_batch, nclasses)

    x_train_input = Input(tensor=x_train_batch)

    # x_test_batch, y_test_batch = tf.train.batch(
    #     tensors=[data.test.images, data.test.labels.astype(np.int32)],
    #     batch_size=batch_size,
    #     capacity=capacity,
    #     enqueue_many=enqueue_many,
    #     num_threads=8)

    # I like the non-functional definition of model more.
    # model_init = make_model(x_train_input, nclasses)
    # x_train_out = model_init.output
    # train_model = Model(inputs=[x_train_input], outputs=[x_train_out])

    x_train_out = cnn_layers(x_train_input, nclasses)
    train_model = Model(inputs=[x_train_input], outputs=[x_train_out])
    if ngpus > 1:
        train_model = make_parallel(train_model, gdev_list)

    lr = 2e-3 * ngpus
    train_model.compile(optimizer=RMSprop(lr=lr, decay=1e-5),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'],
                        target_tensors=[y_train_batch])

    if ngpus > 1:
        print_mgpu_modelsummary(train_model)
    else:
        train_model.summary()

    # Callbacks
    if use_model_checkpt:
        mon = 'val_acc' if val_in_train else 'acc'
        checkpoint = ModelCheckpoint(
            'saved_wt.h5', monitor=mon, verbose=0,
            save_best_only=True,
            save_weights_only=True)
        checkpoint = [checkpoint]
    else:
        checkpoint = []

    callbacks = checkpoint
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

    # Fit the model using data from the TFRecord data tensors.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    start_time = time.time()
    train_model.fit(
        # validation_data=(x_test_batch, y_test_batch)
        # if val_in_train else None, # validation data is not used???
        # validations_steps if val_in_train else None,
        # validation_steps=val_in_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks)
    elapsed_time = time.time() - start_time
    print('[{}] finished in {} ms'.format('TRAINING',
                                          int(elapsed_time * 1000)))

    if not checkpoint:  # empty list
        train_model.save_weights('./saved_wt.h5')

    # Clean up the TF session.
    coord.request_stop()
    coord.join(threads)

    KB.clear_session()

    # Second Session. Demonstrate that the model works and is independent of
    # the TFRecord pipeline, and to test loading trained model without tensors.
    x_test = np.reshape(data.validation.images,
                        (data.validation.images.shape[0], 28, 28, 1))
    y_test = data.validation.labels
    x_test_inp = KL.Input(shape=(x_test.shape[1:]))
    test_out = cnn_layers(x_test_inp, nclasses)
    test_model = Model(inputs=x_test_inp, outputs=test_out)

    test_model.load_weights('saved_wt.h5')
    test_model.compile(optimizer='rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])
    test_model.summary()

    loss, acc = test_model.evaluate(x_test, to_categorical(y_test))
    print('\nTest loss: {0}'.format(loss))
    print('\nTest accuracy: {0}'.format(acc))


if __name__ == '__main__':
    main()

