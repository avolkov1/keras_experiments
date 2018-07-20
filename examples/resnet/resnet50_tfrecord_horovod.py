'''
Run Resnet50 on Imagenet data.

Generate TFrecords for Imagenet data by following instructions in:
    examples/build_imagenet_data/README.md


run:
    # 4 GPU machine 2 sockets typically --map-by ppr:2:socket works well.
    TMPDIR=/tmp mpirun --report-bindings \
      --map-by ppr:2:socket -oversubscribe -np 4 python \
      ./examples/resnet/resnet50_tfrecord_horovod.py \
        --imgs_per_epoch=6400 # to speed up epoch

TMPDIR=/tmp mpirun --report-bindings -mca btl_tcp_if_exclude docker0,lo \
  --bind-to none --map-by slot -np 8 \
run_psgcluster_singularity.sh --datamnt=/datasets \
  --container=/cm/shared/singularity/tf1.4.0_hvd_ompi3.0.0-2017-11-23-154091b4d08c.img \
  --venvpy=~/.virtualenvs/py-keras-gen \
  --scripts=./examples/resnet/resnet50_tfrecord_horovod.py \
    --datadir=/datasets/imagenet/train-val-tfrecord-480-subset \
    --batch_size=64 --epochs=2 --imgs_per_epoch=6400 # to speed up epoch

TMPDIR=/tmp mpirun --report-bindings -mca btl_tcp_if_exclude docker0,lo \
  --bind-to none --map-by slot -np 8 \
run_psgcluster_singularity.sh --datamnts=/datasets \
  --container=/cm/shared/singularity/tf17.12_tf1.4.0_hvd_ompi3.0.0_ibverbs-2018-02-01-5540d30e4dc5.img \
  --venvpy=~/.virtualenvs/py-keras-gen \
  --scripts=./examples/resnet/resnet50_tfrecord_horovod.py \
    --datadir=/datasets/imagenet/train-val-tfrecord-480-subset \
    --batch_size=64 --epochs=2 --imgs_per_epoch=6400 # to speed up epoch

'''
from __future__ import print_function
import sys

import argparse as ap
from textwrap import dedent

import time

import tensorflow as tf
import horovod.tensorflow as hvd
import horovod.keras as hvd_keras

import keras.backend as KB
import keras.optimizers as KO
import keras.layers as KL
# from keras.models import Model
# from keras import backend as KB
from keras.layers import Input
from keras.applications.resnet50 import ResNet50

from keras_exp.callbacks.timing import SamplesPerSec, BatchTiming
# from keras_tqdm import TQDMCallback

from resnet_common import RecordInputImagenetPreprocessor


class SmartFormatterMixin(ap.HelpFormatter):
    # ref:
    # http://stackoverflow.com/questions/3853722/python-argparse-how-to-insert-newline-in-the-help-text
    # @IgnorePep8

    def _split_lines(self, text, width):
        # this is the RawTextHelpFormatter._split_lines
        if text.startswith('S|'):
            return text[2:].splitlines()
        return ap.HelpFormatter._split_lines(self, text, width)


class CustomFormatter(ap.RawDescriptionHelpFormatter, SmartFormatterMixin):
    '''Convenience formatter_class for argparse help print out.'''


def _parser(desc):
    parser = ap.ArgumentParser(description=dedent(desc),
                               formatter_class=CustomFormatter)

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to run training for.\n'
                        '(Default: %(default)s)\n')

    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='S|Batch size. Default: %(default)s')

    parser.add_argument(
        '--imgs_per_epoch', type=int, default=0,
        help='S|Number of images to run during epoch. Use for timing.\n'
        'Default uses all the images for an epoch.')

    imagenet_datadir = '/datasets/imagenet/train-val-tfrecord-480-subset'
    parser.add_argument(
        '--datadir', default=imagenet_datadir,
        help='S|Data directory with Imagenet TFrecord dataset. Assumes\n'
        'TFrecord subsets prefixed with train-* and validation-* are in the\n'
        'directory. Default: %(default)s')

    parser.add_argument(
        '--distort_color', action='store_true', default=False,
        help='S|Distort color during training on imagenet to "enrich" the\n'
        'dataset. Default no distortion. Set this flag to enable distortion.')

    args = parser.parse_args()

    return args


def main(argv=None):
    # Initialize Horovod.
    hvd.init()

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    KB.set_session(tf.Session(config=config))

    # print('LOCAL RANK, OVERAL RANK: {}, {}'.format(hvd.local_rank(),
    #                                                hvd.rank()))

    ngpus = hvd.size()

    main.__doc__ = __doc__
    argv = sys.argv if argv is None else sys.argv.extend(argv)
    desc = main.__doc__  # .format(os.path.basename(__file__))
    # CLI parser
    args = _parser(desc)

    num_devices_tfrecord = 1
    height, width = 224, 224  # Image dimensions. Gets resized if not match.
    distort_color = args.distort_color
    data_dir = args.datadir
    batch_size = args.batch_size  # * ngpus
    epochs = args.epochs
    imgs_per_epoch = args.imgs_per_epoch

    # Fit the model using data from the TFRecord data tensors.
    device_minibatches = RecordInputImagenetPreprocessor.device_minibatches
    images_tfrecord, labels_tfrecord, nrecords = device_minibatches(
        num_devices_tfrecord, data_dir, batch_size,
        height, width, distort_color, val=False)
    images_tfrecord = images_tfrecord[0]
    labels_tfrecord = labels_tfrecord[0]

    # CASTING FOR KERAS
    # labels[device_num] = tf.cast(labels_tfrecord, dtype)
    nclasses = 1000
    labels_tfrecord = tf.one_hot(labels_tfrecord, nclasses)

    nimgs_to_use = imgs_per_epoch if imgs_per_epoch > 0 else nrecords
    steps_per_epoch = nimgs_to_use // batch_size // hvd.size()
    # steps_per_epoch = 100

    # batch_shape = images_tfrecord.get_shape().as_list()
    # images = Input(tensor=images_tfrecord, batch_shape=x_batch_shape)
    images = Input(tensor=images_tfrecord)
    model = ResNet50(input_tensor=images, weights=None)
    if hvd.rank() == 0:
        model.summary()

        print('Num images: {}'.format(nrecords))

        if nimgs_to_use < nrecords:
            print('Using {} images per epoch'.format(nimgs_to_use))

        # print('IMAGES_TFRECORD: {}'.format(images_tfrecord))
        # print('LABELS_TFRECORD: {}'.format(labels_tfrecord))

    # Add Horovod Distributed Optimizer from nvcnn.py
    # momentum = 0.9
    # lr = 0.1
    # learning_rate = tf.train.exponential_decay(
    #             lr,
    #             self.global_step,
    #             decay_steps=FLAGS.lr_decay_epochs * nstep_per_epoch,
    #             decay_rate=FLAGS.lr_decay_rate,
    #             staircase=True)
    # opt = tf.train.MomentumOptimizer(self.learning_rate, momentum,
    #                                  use_nesterov=True)

    # lr = 0.001 * ngpus
    # opt = tf.train.AdamOptimizer()
    # opt = hvd.DistributedOptimizer(opt)  # , use_locking=True)
    # opt = KO.TFOptimizer(opt)  # Required for tf.train based optimizers

    opt = KO.Adam()
    opt = hvd_keras.DistributedOptimizer(opt)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  # metrics=['accuracy'],
                  target_tensors=[labels_tfrecord])

    # Broadcast variables from rank 0 to all other processes.
    KB.get_session().run(hvd.broadcast_global_variables(0))

    callbacks = []
    if hvd.rank() == 0:
        callbacks += [BatchTiming(),
                      SamplesPerSec(ngpus * batch_size)]

    # RecordInput is a yield op which doesn't use queue runners or queues.
    # Start the queue runners.
    # sess = KB.get_session()

    # sess.run([tf.local_variables_initializer(),
    #           tf.global_variables_initializer()])

    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess, coord)

    start_time = time.time()
    model.fit(
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1)
    # verbose=hvd.rank() == 0)
    elapsed_time = time.time() - start_time

    if hvd.rank() == 0:
        print('[{}] finished in {} s'
              .format('TRAINING', round(elapsed_time, 3)))
        # loss = model.evaluate(None, None, steps=steps_per_epoch_val)

        images_tfrecord_val, labels_tfrecord_val, nrecords_val = \
            device_minibatches(num_devices_tfrecord, data_dir, batch_size,
                               height, width, distort_color, val=True)
        images_tfrecord_val = images_tfrecord_val[0]
        labels_tfrecord_val = labels_tfrecord_val[0]
        labels_tfrecord_val = tf.one_hot(labels_tfrecord_val, nclasses)

        # print('IMAGES_TFRECORD_VAL: {}'.format(images_tfrecord_val))
        # print('labels_tfrecord_val: {}'.format(labels_tfrecord_val))

        steps_per_epoch_val = nrecords_val // batch_size

        images_val = Input(tensor=images_tfrecord_val)
        model_val = model
        model_val.layers[0] = KL.InputLayer(input_tensor=images_val)
        model_val.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'],
            target_tensors=[labels_tfrecord_val])
        # model.summary()
        loss = model_val.evaluate(x=None, y=None, steps=steps_per_epoch_val)

        print('\nNum images evaluated, steps: {}, {}'.
              format(nrecords_val, steps_per_epoch_val))
        print('\nTest loss, acc: {}'.format(loss))
        # print('\nTest accuracy: {0}'.format(acc))

    # Clean up the TF session.
    # coord.request_stop()
    # coord.join(threads)

    KB.clear_session()  # do this for Horovod


if __name__ == '__main__':
    main()
