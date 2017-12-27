'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114

Using Horovod.

original implementation:
https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder_deconv.py


run:

    mpirun -np 2 --map-by ppr:4:socket python \
      variational_autoencoder_deconv_horovod.py --epochs=30

    TMPDIR=/tmp mpirun --report-bindings --bind-to none --map-by slot -np 4 \
      python ./examples/variational_autoencoder/variational_autoencoder_deconv_horovod.py \
      --epochs=30  --nranks_per_gpu=1

# Below via singularity. NGPUS is GPUs per node
NNODES=1 NGPUS=1 RANKS_PER_GPU=2 && \
time TMPDIR=/tmp mpirun --report-bindings -mca btl_tcp_if_exclude docker0,lo \
  --bind-to none --map-by slot -np $(($NNODES * $RANKS_PER_GPU * $NGPUS)) \
  run_psgcluster_singularity.sh \
    --container=/cm/shared/singularity/tf1.4.0_hvd_ompi3.0.0-2017-11-23-154091b4d08c.img \
    --venvpy=~/.virtualenvs/py-keras-gen \
    --scripts=./examples/variational_autoencoder/variational_autoencoder_deconv_horovod.py \
    --epochs=4 \
    --nranks_per_gpu=$RANKS_PER_GPU

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
import matplotlib
try:
    matplotlib.use('Agg')
except Exception:
    raise
import matplotlib.pyplot as plt

from scipy.stats import norm

import tensorflow as tf
import horovod.tensorflow as hvd

try:
    # Initialize Horovod.
    hvd.init()
except Exception:
    raise

from keras import backend as K
from keras.datasets import mnist
from keras.optimizers import TFOptimizer

from keras_exp.callbacks.timing import BatchTiming, SamplesPerSec

from vae_common import CustomFormatter, make_vae_and_codec


def parser_(desc):
    parser = ap.ArgumentParser(description=desc,
                               formatter_class=CustomFormatter)

    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to run training for.')

    parser.add_argument(
        '--nranks_per_gpu', type=int, default=1,
        help='S|Number of ranks to run on each GPUs. Use this parameter to\n'
        'oversubscribe a GPU. When oversubscribing a GPU use in combination\n'
        'with MPS (multi-process service). Default: %(default)s')

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

    nranks_per_gpu = args.nranks_per_gpu
    local_rank = hvd.local_rank()
    gpu_local_rank = local_rank // nranks_per_gpu
    print('local_rank, GPU_LOCAL_RANK: {}, {}'.format(
        local_rank, gpu_local_rank))

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = str(hvd.local_rank())
    config.gpu_options.visible_device_list = str(gpu_local_rank)
    K.set_session(tf.Session(config=config))

    # input image dimensions
    img_rows, img_cols, img_chns = 28, 28, 1
    # number of convolutional filters to use
    filters = 64
    # convolution kernel size
    num_conv = 3

    hvdsize = hvd.size()

    batch_size = 128  # 100
    if K.image_data_format() == 'channels_first':
        original_img_size = (img_chns, img_rows, img_cols)
    else:
        original_img_size = (img_rows, img_cols, img_chns)
    latent_dim = 2
    intermediate_dim = 128
    epsilon_std = 1.0
    epochs = args.epochs  # 5

    # train the VAE on MNIST digits
    (x_train, _), (x_test, y_test) = mnist.load_data()

    # Data split if going for reduction in each iteration step. Using
    # tf-queue or dataset is better to preserve uniform random sampling.
    # nsamples = x_train.shape[0]
    # mysamples = nsamples // hvdsize
    # start_sam = hvd.local_rank() * mysamples
    # stop_sam = min((hvd.local_rank() + 1) * mysamples, nsamples)
    # x_train = x_train[start_sam:stop_sam, ...]

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

    if hvd.rank() == 0:
        print('x_train.shape:', x_train.shape)

    vae, encoder, generator = make_vae_and_codec(
        original_img_size, img_chns, img_rows, img_cols, batch_size,
        filters, num_conv, intermediate_dim, latent_dim, epsilon_std)
    # :  :type vae: Model

    lr = 0.001  # * hvdsize
    opt = tf.train.RMSPropOptimizer(lr)
    # Add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)  # , use_locking=True)
    opt = TFOptimizer(opt)

    vae.compile(optimizer=opt, loss=None)
    if hvd.rank() == 0:
        vae.summary()

    callbacks = []
    if hvd.rank() == 0:
        callbacks += [BatchTiming(), SamplesPerSec(batch_size * hvdsize)]

    sess = K.get_session()
    sess.run(hvd.broadcast_global_variables(0))

    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None),
            callbacks=callbacks)

    if hvd.rank() == 0:
        vae_val = vae
        loss = vae_val.evaluate(x=x_test, y=None, batch_size=batch_size)
        print('\n\nVAE VALIDATION LOSS: {}'.format(loss))

        # display a 2D plot of the digit classes in the latent space
        x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
        plt.figure(figsize=(6, 6))
        plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
        plt.colorbar()
        # plt.show()
        plt.savefig('vae_scatter.ps')
        plt.close()

        # display a 2D manifold of the digits
        n = 15  # figure with 15x15 digits
        digit_size = 28
        figure = np.zeros((digit_size * n, digit_size * n))
        # Linearly spaced coordinates on the unit square were transformed
        # through the inverse CDF (ppf) of the Gaussian
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

        plt.figure(figsize=(10, 10))
        plt.imshow(figure, cmap='Greys_r')
        # plt.show()
        plt.savefig('vae_digit.ps')
        plt.close()

    K.clear_session()


if __name__ == '__main__':
    main()
