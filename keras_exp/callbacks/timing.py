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

    def on_train_begin(self, logs={}):
        self.all_batch_times = []
        self.all_epoch_times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_batch_times = []
        self._epoch_start_time = time.time()

    def on_batch_begin(self, batch, logs={}):
        self.start_time = time.time()

    def on_batch_end(self, batch, logs={}):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        self.epoch_batch_times.append(elapsed_time)
        self.all_batch_times.append(elapsed_time)

    def on_epoch_end(self, epoch, logs={}):
        # epoch_batch_time = np.sum(self.epoch_batch_times)
        epoch_time = time.time() - self._epoch_start_time
        self.all_epoch_times.append(epoch_time)
        median_batch_time = np.median(self.epoch_batch_times)
        epoch_batch_time = median_batch_time * len(self.epoch_batch_times)
        overhead_time = epoch_time - epoch_batch_time
        print('\nEpoch timing - batch (median): {:0.5f}, epoch: {:0.5f} (sec),'
              ' overhead (epoch - steps*batch): {:0.5f} (sec)'
              .format(median_batch_time, epoch_time, overhead_time))

    def on_train_end(self, logs={}):
        median_batch_time = np.median(self.all_batch_times)
        median_epoch_time = np.median(self.all_epoch_times)
        print('\nOverall - batch (median): {:0.5f}, '
              'epoch (median): {:0.5f} (sec)'
              .format(median_batch_time, median_epoch_time))


# Taken from: https://github.com/rossumai/keras-multi-gpu/blob/8cc6c5328af3dd643c5772c4f9fe1d275f33c926/keras_tf_multigpu/callbacks.py#L305 @IgnorePep8
# MIT Licensed portion of the code. Credit to Bohumir Zamecnik
# https://github.com/rossumai/keras-multi-gpu
class SamplesPerSec(Callback):
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.all_samples_per_sec = []

    def on_batch_begin(self, batch, logs={}):
        self.start_time = time.time()
        # self.batch_size = logs['size']

    def on_batch_end(self, batch, logs={}):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        samples_per_sec = self.batch_size / elapsed_time
        self.all_samples_per_sec.append(samples_per_sec)
        # self.progbar.update(self.seen, [('samples/sec', samples_per_sec)])

    def on_epoch_end(self, epoch, logs={}):
        print('\nSamples/sec: {:0.2f}'
              .format(np.median(self.all_samples_per_sec)))
