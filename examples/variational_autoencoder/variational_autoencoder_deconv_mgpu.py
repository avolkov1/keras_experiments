'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114


Multigpu modifications. Not entirely sure if this is working correctly. The
loss during training does not decrease/converge the way it does in single gpu
case. Refer to original at:
https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder_deconv.py

'''

import sys
import argparse as ap

import numpy as np

# try:
#     import Tkinter  # @UnusedImport
#     import matplotlib.pyplot as plt
# except ImportError:
#     import matplotlib
#     matplotlib.use('Agg')
#     import matplotlib.pyplot as plt

from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.optimizers import RMSprop  # , TFOptimizer

from keras_exp.multigpu import (get_available_gpus, print_mgpu_modelsummary)
from keras_exp.multigpu import make_parallel

# import tensorflow as tf


def make_vae_and_codec(
        original_img_size, img_chns, img_rows, img_cols, batch_size, filters,
        num_conv, intermediate_dim, latent_dim, epsilon_std):

    x = Input(shape=original_img_size)
    conv_1 = Conv2D(img_chns,
                    kernel_size=(2, 2),
                    padding='same', activation='relu')(x)
    conv_2 = Conv2D(filters,
                    kernel_size=(2, 2),
                    padding='same', activation='relu',
                    strides=(2, 2))(conv_1)
    conv_3 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)(conv_2)
    conv_4 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)(conv_3)
    flat = Flatten()(conv_4)
    hidden = Dense(intermediate_dim, activation='relu')(flat)

    z_mean = Dense(latent_dim)(hidden)
    z_log_var = Dense(latent_dim)(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_hid = Dense(intermediate_dim, activation='relu')
    decoder_upsample = Dense(filters * 14 * 14, activation='relu')

    if K.image_data_format() == 'channels_first':
        output_shape = (batch_size, filters, 14, 14)
    else:
        output_shape = (batch_size, 14, 14, filters)

    decoder_reshape = Reshape(output_shape[1:])
    decoder_deconv_1 = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
                                       activation='relu')
    decoder_deconv_2 = Conv2DTranspose(filters,
                                       kernel_size=num_conv,
                                       padding='same',
                                       strides=1,
                                       activation='relu')
    if K.image_data_format() == 'channels_first':
        output_shape = (batch_size, filters, 29, 29)
    else:
        output_shape = (batch_size, 29, 29, filters)
    decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                              kernel_size=(3, 3),
                                              strides=(2, 2),
                                              padding='valid',
                                              activation='relu')
    decoder_mean_squash = Conv2D(img_chns,
                                 kernel_size=2,
                                 padding='valid',
                                 activation='sigmoid')

    hid_decoded = decoder_hid(z)
    up_decoded = decoder_upsample(hid_decoded)
    reshape_decoded = decoder_reshape(up_decoded)
    deconv_1_decoded = decoder_deconv_1(reshape_decoded)
    deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
    x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
    x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

    # Custom loss layer
    class CustomVariationalLayer(Layer):
        def __init__(self, img_rows, img_cols, **kwargs):
            self._img_rows, self._img_cols = img_rows, img_cols
            self.is_placeholder = True
            super(CustomVariationalLayer, self).__init__(**kwargs)

        def vae_loss(self, x, x_decoded_mean_squash):
            x = K.flatten(x)
            x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
            img_rows, img_cols = self._img_rows, self._img_cols
            xent_loss = img_rows * img_cols * \
                metrics.binary_crossentropy(x, x_decoded_mean_squash)
            kl_loss = - 0.5 * K.mean(
                1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return K.mean(xent_loss + kl_loss)

        def call(self, inputs):
            x = inputs[0]
            x_decoded_mean_squash = inputs[1]
            loss = self.vae_loss(x, x_decoded_mean_squash)
            self.add_loss(loss, inputs=inputs)
            # We don't use this output.
            return x

    y = CustomVariationalLayer(
        img_rows, img_cols)([x, x_decoded_mean_squash])

    vae = Model(x, y)

    # build a model to project inputs on the latent space
    encoder = Model(x, z_mean)

    # build a digit generator that can sample from the learned distribution
    decoder_input = Input(shape=(latent_dim,))
    _hid_decoded = decoder_hid(decoder_input)
    _up_decoded = decoder_upsample(_hid_decoded)
    _reshape_decoded = decoder_reshape(_up_decoded)
    _deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
    _deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
    _x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
    _x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
    generator = Model(decoder_input, _x_decoded_mean_squash)

    return vae, encoder, generator


def parser_(desc):
    parser = ap.ArgumentParser(description=desc)

    parser.add_argument(
        '--mgpu', action='store', nargs='?', type=int,
        const=-1,  # if mgpu is specified but value not provided then -1
        # if mgpu is not specified then defaults to 0 - single gpu
        # mgpu = 0 if getattr(args, 'mgpu', None) is None else args.mgpu
        default=ap.SUPPRESS,
        help='Run on multiple-GPUs using all available GPUs on a system. If'
        '\nnot passed does not use multiple GPU. If passed uses all GPUs.'
        '\nOptionally specify a number to use that many GPUs. Another approach'
        '\nis to specify CUDA_VISIBLE_DEVICES=0,1,... when calling script and'
        '\nspecify --mgpu to use this specified device list.'
        '\nThis option is only supported with TensorFlow backend.')

    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to run training for.')

    args = parser.parse_args()

    return args


def main(argv=None):
    '''
    '''
    main.__doc__ = __doc__
    argv = sys.argv if argv is None else sys.argv.extend(argv)
    desc = main.__doc__  # .format(os.path.basename(__file__))
    # CLI parser
    args = parser_(desc)
    mgpu = 1 if getattr(args, 'mgpu', None) is None else args.mgpu

    # input image dimensions
    img_rows, img_cols, img_chns = 28, 28, 1
    # number of convolutional filters to use
    filters = 64
    # convolution kernel size
    num_conv = 3

    gpus_list = get_available_gpus(mgpu)
    ngpus = len(gpus_list)

    batch_size = 100 * ngpus
    if K.image_data_format() == 'channels_first':
        original_img_size = (img_chns, img_rows, img_cols)
    else:
        original_img_size = (img_rows, img_cols, img_chns)
    latent_dim = 2
    intermediate_dim = 128
    epsilon_std = 1.0
    epochs = args.epochs  # 5

    vae, encoder, generator = make_vae_and_codec(
        original_img_size, img_chns, img_rows, img_cols, batch_size,
        filters, num_conv, intermediate_dim, latent_dim, epsilon_std)
    # :  :type vae: Model

    vae = make_parallel(vae, gpus_list)
    lr = 0.001 * ngpus  # / (10 ** (ngpus - 1))
    opt = RMSprop(lr)  # 'rmsprop'
    # tf_opt = tf.train.RMSPropOptimizer(lr)
    # opt = TFOptimizer(tf_opt)
    vae.compile(optimizer=opt, loss=None)
    # vae.summary()
    print_mgpu_modelsummary(vae)

    # train the VAE on MNIST digits
    (x_train, _), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

    print('x_train.shape:', x_train.shape)

    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))

    # display a 2D plot of the digit classes in the latent space
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    # # plt.figure(figsize=(6, 6))
    # # plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
    # # plt.colorbar()
    # # plt.show()

    # display a 2D manifold of the digits
    n = 15  # figure with 15x15 digits
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # Linearly spaced coordinates on the unit square were transformed through
    # the inverse CDF (ppf) of the Gaussian
    # To produce values of the latent variables z, since the prior of the
    # latent space is Gaussian
    grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
            x_decoded = generator.predict(z_sample, batch_size=batch_size)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    # # plt.figure(figsize=(10, 10))
    # # plt.imshow(figure, cmap='Greys_r')
    # # plt.show()


if __name__ == '__main__':
    main()
