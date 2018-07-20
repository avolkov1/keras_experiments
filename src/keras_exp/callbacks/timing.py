'''
Callbacks for basic benchmarking and timing of Keras model training.

MIT Licensed portion of the code. Credit to Bohumir Zamecnik
Taken from: https://github.com/rossumai/keras-multi-gpu
'''  # noqa
from __future__ import print_function
import time
import numpy as np
from keras.callbacks import Callback


__all__ = ('BatchTiming', 'SamplesPerSec',)


# Taken from: https://github.com/rossumai/keras-multi-gpu/blob/8cc6c5328af3dd643c5772c4f9fe1d275f33c926/keras_tf_multigpu/callbacks.py#L262 @IgnorePep8
# MIT Licensed portion of the code. Credit to Bohumir Zamecnik
# https://github.com/rossumai/keras-multi-gpu
class BatchTiming(Callback):
    """
    It measure robust stats for timing of batches and epochs.
    Useful for measuring the training process.
    For each epoch it prints median batch time and total epoch time.
    After training it prints overall median batch time and median epoch time.
    Usage: model.fit(X_train, Y_train, callbacks=[BatchTiming()])
    All times are in seconds.
    More info: https://keras.io/callbacks/
    """
    def __init__(self):
        Callback.__init__(self)
        self.train_beg_time = None
        self.all_batch_times = None
        self.all_epoch_times = None

        self.epoch_batch_times = None
        self._epoch_start_time = None

        self.start_time = None

    def on_train_begin(self, logs=None):
        self.train_beg_time = time.time()
        self.all_batch_times = []
        self.all_epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_batch_times = []
        self._epoch_start_time = time.time()

    def on_batch_begin(self, batch, logs=None):
        self.start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self.epoch_batch_times.append(elapsed_time)
        self.all_batch_times.append(elapsed_time)

    def on_epoch_end(self, epoch, logs=None):
        # epoch_batch_time = np.sum(self.epoch_batch_times)
        epoch_time = time.time() - self._epoch_start_time
        self.all_epoch_times.append(epoch_time)
        median_batch_time = np.median(self.epoch_batch_times)
        epoch_batch_time = median_batch_time * len(self.epoch_batch_times)
        overhead_time = epoch_time - epoch_batch_time
        print('\nEpoch timing - batch (median): {:0.5f}, epoch: {:0.5f} (sec),'
              ' overhead (epoch - steps*batch): {:0.5f} (sec)'
              .format(median_batch_time, epoch_time, overhead_time))

    def on_train_end(self, logs=None):
        train_time = time.time() - self.train_beg_time
        train_time_m1st_epoch = train_time - self.all_epoch_times[0]
        median_batch_time = np.median(self.all_batch_times)
        median_epoch_time = np.median(self.all_epoch_times)
        print(
            '\nOverall - batch (median): {:0.5f}, '
            'epoch (median): {:0.5f} (sec), '
            'Total Train: {:0.5f} (sec), '
            'Total Train - 1st epoch: {:0.5f} (sec)\n'
            .format(median_batch_time, median_epoch_time, train_time,
                    train_time_m1st_epoch))


# Taken from: https://github.com/rossumai/keras-multi-gpu/blob/8cc6c5328af3dd643c5772c4f9fe1d275f33c926/keras_tf_multigpu/callbacks.py#L305 @IgnorePep8
# MIT Licensed portion of the code. Credit to Bohumir Zamecnik
# https://github.com/rossumai/keras-multi-gpu
class SamplesPerSec(Callback):
    '''Measure generic samples per sec processing during each batch step.'''
    def __init__(self, batch_size, **kwargs):
        Callback.__init__(self, **kwargs)
        self.batch_size = batch_size

        self.all_samples_per_sec = None
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.all_samples_per_sec = []

    def on_batch_begin(self, batch, logs=None):
        self.start_time = time.time()
        # self.batch_size = logs['size']

    def on_batch_end(self, batch, logs=None):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        samples_per_sec = self.batch_size / elapsed_time
        self.all_samples_per_sec.append(samples_per_sec)
        # self.progbar.update(self.seen, [('samples/sec', samples_per_sec)])

    def on_epoch_end(self, epoch, logs=None):
        print('\nSamples/sec: {:0.2f}'
              .format(np.median(self.all_samples_per_sec)))
