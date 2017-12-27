'''
This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114

Original implementation:
https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder_deconv.py  # @IgnorePep8

'''

from argparse import HelpFormatter, RawDescriptionHelpFormatter

from keras.layers import (
    Input, Layer, Dense, Lambda, Flatten, Reshape, Conv2D, Conv2DTranspose)
from keras.models import Model
from keras import backend as K
from keras import metrics


__all__ = ('CustomFormatter', 'make_vae_and_codec', 'make_shared_layers_dict',
           'make_vae', 'get_encoded', 'get_decoded',)


class SmartFormatterMixin(HelpFormatter):
    # ref:
    # http://stackoverflow.com/questions/3853722/python-argparse-how-to-insert-newline-in-the-help-text
    # @IgnorePep8

    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('S|'):
            return text[2:].splitlines()
        return HelpFormatter._split_lines(self, text, width)


# class CustomFormatter(ArgumentDefaultsHelpFormatter,
#                       RawDescriptionHelpFormatter, SmartFormatterMixin):
class CustomFormatter(RawDescriptionHelpFormatter, SmartFormatterMixin):
    '''Convenience formatter_class for argparse help print out.'''


class vae_lnames(object):
    conv_1 = 'conv_1'
    conv_2 = 'conv_2'
    conv_3 = 'conv_3'
    conv_4 = 'conv_4'
    flat = 'flat'
    hidden = 'hidden'

    z_mean = 'z_mean'
    z_log_var = 'z_log_var'
    z_sampling = 'z_sampling'

    decoder_hid = 'decoder_hid'
    decoder_upsample = 'decoder_upsample'

    decoder_reshape = 'decoder_reshape'
    decoder_deconv_1 = 'decoder_deconv_1'
    decoder_deconv_2 = 'decoder_deconv_2'

    decoder_deconv_3_upsamp = 'decoder_deconv_3_upsamp'
    decoder_mean_squash = 'decoder_mean_squash'

    custom_var_layer = 'custom_var_layer'


# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, img_rows, img_cols, **kwargs):
        self._img_rows, self._img_cols = img_rows, img_cols
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean_squash, z_mean, z_log_var):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        img_rows, img_cols = self._img_rows, self._img_cols
        # generative or reconstruction loss
        xent_loss = img_rows * img_cols * \
            metrics.binary_crossentropy(x, x_decoded_mean_squash)
        # Kullback-Leibler divergence loss
        kl_loss = - 0.5 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean_squash = inputs[1]
        z_mean = inputs[2]
        z_log_var = inputs[3]
        loss = self.vae_loss(x, x_decoded_mean_squash, z_mean, z_log_var)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x


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

    # Not USED
    # if K.image_data_format() == 'channels_first':
    #     output_shape = (batch_size, filters, 29, 29)
    # else:
    #     output_shape = (batch_size, 29, 29, filters)

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

    y = CustomVariationalLayer(img_rows, img_cols)(
        [x, x_decoded_mean_squash, z_mean, z_log_var])

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


# TODO: This should be memoized or implemented as a singleton class.
def make_shared_layers_dict(
        img_chns, img_rows, img_cols, batch_size, filters,
        num_conv, intermediate_dim, latent_dim, epsilon_std):
    '''Returns layers to be shared by the variational autoencoder. This
    function should be memoized or implemented as a singleton. Until then
    don't call this function more then once. Re-use the returned dictionary
    with the instantiated layers.
    Keras shared layers are explained here:
        https://keras.io/getting-started/functional-api-guide/#shared-layers

    :returns: Shared layers for vae in a dictionary structure keyed by
        :class:`vae_lnames`.
    :rtype: dict

    '''
    ldict = dict()

    conv_1 = Conv2D(img_chns,
                    kernel_size=(2, 2),
                    padding='same', activation='relu')
    conv_2 = Conv2D(filters,
                    kernel_size=(2, 2),
                    padding='same', activation='relu',
                    strides=(2, 2))
    conv_3 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)
    conv_4 = Conv2D(filters,
                    kernel_size=num_conv,
                    padding='same', activation='relu',
                    strides=1)
    flat = Flatten()
    hidden = Dense(intermediate_dim, activation='relu')

    z_mean = Dense(latent_dim)
    z_log_var = Dense(latent_dim)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z_sampling = Lambda(sampling, output_shape=(latent_dim,))

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

    # Not USED
    # if K.image_data_format() == 'channels_first':
    #     output_shape = (batch_size, filters, 29, 29)
    # else:
    #     output_shape = (batch_size, 29, 29, filters)

    decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                              kernel_size=(3, 3),
                                              strides=(2, 2),
                                              padding='valid',
                                              activation='relu')
    decoder_mean_squash = Conv2D(img_chns,
                                 kernel_size=2,
                                 padding='valid',
                                 activation='sigmoid')

    custom_var_layer = CustomVariationalLayer(img_rows, img_cols)

    ldict[vae_lnames.conv_1] = conv_1
    ldict[vae_lnames.conv_2] = conv_2
    ldict[vae_lnames.conv_3] = conv_3
    ldict[vae_lnames.conv_4] = conv_4
    ldict[vae_lnames.flat] = flat
    ldict[vae_lnames.hidden] = hidden

    ldict[vae_lnames.z_mean] = z_mean
    ldict[vae_lnames.z_log_var] = z_log_var
    ldict[vae_lnames.z_sampling] = z_sampling

    ldict[vae_lnames.decoder_hid] = decoder_hid
    ldict[vae_lnames.decoder_upsample] = decoder_upsample

    ldict[vae_lnames.decoder_reshape] = decoder_reshape
    ldict[vae_lnames.decoder_deconv_1] = decoder_deconv_1
    ldict[vae_lnames.decoder_deconv_2] = decoder_deconv_2

    ldict[vae_lnames.decoder_deconv_3_upsamp] = decoder_deconv_3_upsamp
    ldict[vae_lnames.decoder_mean_squash] = decoder_mean_squash

    ldict[vae_lnames.custom_var_layer] = custom_var_layer

    return ldict


def get_encoded(ldict, x):
    '''Variational Autoencoder Keras (vae) encoder output Tensor.

    :param ldict: Dictionary with instantiated layers keyed by
        :class:`vae_lnames`. Use the :func:`make_shared_layers_dict` function
        to obtain the dict.

    :param x: Keras input for the vae encoder.
    :type x: keras.layers.Input

    :returns: The vae encoder output Tensor. If backend is TF then rtype is
        tensorflow.python.framework.ops.Tensor
    :rtype: keras.backend.T
    '''
    # x = Input(shape=original_img_size) or Input(tensor=tensor_in)
    conv_1 = ldict[vae_lnames.conv_1](x)
    conv_2 = ldict[vae_lnames.conv_2](conv_1)
    conv_3 = ldict[vae_lnames.conv_3](conv_2)
    conv_4 = ldict[vae_lnames.conv_4](conv_3)
    flat = ldict[vae_lnames.flat](conv_4)
    hidden = ldict[vae_lnames.hidden](flat)

    z_mean = ldict[vae_lnames.z_mean](hidden)
    z_log_var = ldict[vae_lnames.z_log_var](hidden)

    # I don't know why the code below doesn't work. Instantiating the Model
    # outside of this function works fine.
    # build a model to project inputs on the latent space
    # encoder = Model(x, z_mean)

    return z_mean, z_log_var


def get_decoded(ldict, decoder_input):
    '''Variational Autoencoder Keras (vae) decoder output Tensor

    :param ldict: Dictionary with instantiated layers keyed by
        :class:`vae_lnames`. Use the :func:`make_shared_layers_dict` function
        to obtain the dict.

    :param decoder_input: Keras input for the vae decoder.
    :type decoder_input: keras.layers.Input

    :returns: The vae decoder output Tensor. If backend is TF then rtype is
        tensorflow.python.framework.ops.Tensor
    :rtype: keras.backend.T

    '''
    # build a digit generator that can sample from the learned distribution
    # decoder_input = Input(shape=(latent_dim,)) or z from make_vae
    hid_decoded = ldict[vae_lnames.decoder_hid](decoder_input)
    up_decoded = ldict[vae_lnames.decoder_upsample](hid_decoded)
    reshape_decoded = ldict[vae_lnames.decoder_reshape](up_decoded)
    deconv_1_decoded = ldict[vae_lnames.decoder_deconv_1](reshape_decoded)
    deconv_2_decoded = ldict[vae_lnames.decoder_deconv_2](deconv_1_decoded)
    x_decoded_relu = \
        ldict[vae_lnames.decoder_deconv_3_upsamp](deconv_2_decoded)
    x_decoded_mean_squash = \
        ldict[vae_lnames.decoder_mean_squash](x_decoded_relu)
    # I don't know why the code below doesn't work. Instantiating the Model
    # outside of this function works fine.
    # generator = Model(decoder_input, x_decoded_mean_squash)

    return x_decoded_mean_squash


def make_vae(ldict, x):
    '''Instantiates the Variational Autoencoder (vae) Keras Model.

    :param ldict: Dictionary with instantiated layers keyed by
        :class:`vae_lnames`. Use the :func:`make_shared_layers_dict` function
        to obtain the dict.

    :param x: Keras input for the vae.
    :type x: keras.layers.Input

    :returns: The vae Keras Model.
    :rtype: keras.models.Model

    '''
    # x could be Input(tensor=tensor_in) or Input(shape=original_img_size)
    z_mean, z_log_var = get_encoded(ldict, x)

    z = ldict[vae_lnames.z_sampling]([z_mean, z_log_var])

    x_decoded_mean_squash = get_decoded(ldict, z)

    y = ldict[vae_lnames.custom_var_layer](
        [x, x_decoded_mean_squash, z_mean, z_log_var])

    vae = Model(x, y)

    return vae
