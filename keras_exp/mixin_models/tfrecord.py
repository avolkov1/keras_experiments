'''
'''
import sys
import warnings
import copy

import time

import numpy as np

from keras import callbacks as cbks
import keras.backend as K

# unstable APIs
from keras.engine.training import (
    _batch_shuffle, _make_batches, _slice_arrays)


__all__ = ('ModelTFRecordMixin', 'ModelCheckpointTFRecord',)


class ModelCheckpointTFRecord(cbks.Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpointTFRecord, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                current = np.mean(current)  # PATCH of ModelCheckpoint
                if current is None:
                    warnings.warn(
                        'Can save best model only with %s available, '
                        'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(
                                'Epoch %05d: %s improved from %0.5f to %0.5f,'
                                ' saving model to %s'
                                % (epoch, self.monitor, self.best,
                                   current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


# From generic_utils.py
# ref: https://github.com/fchollet/keras/pull/7060
class Progbar(object):
    """Displays a progress bar.

    # Arguments
        target: Total number of steps expected, None if unknown.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05):
        self.width = width
        if target is None:
            target = -1
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, force=False):
        """Updates the progress bar.

        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            force: Whether to force visual progress update.
        """
        values = values or []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            if not force and (now - self.last_update) < self.interval:
                return

            prev_total_width = self.total_width
            sys.stdout.write('\b' * prev_total_width)
            sys.stdout.write('\r')

            if self.target is not -1:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
                bar = barstr % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
                sys.stdout.write(bar)
                self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target and self.target is not -1:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                info += ' - %s:' % k
                if isinstance(self.sum_values[k], list):
                    # avg = \
                    #     self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    avg = np.mean(self.sum_values[k][0] /
                                  max(1, self.sum_values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * ' ')

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write('\n')

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s:' % k
                    # avg = \
                    #     self.sum_values[k][0] / max(1, self.sum_values[k][1])
                    avg = np.mean(self.sum_values[k][0] /
                                  max(1, self.sum_values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                sys.stdout.write(info + "\n")

        self.last_update = now

    def add(self, n, values=None):
        self.update(self.seen_so_far + n, values)


class ProgbarLogger_TFRecord(cbks.Callback):
    """Callback that prints metrics to stdout.

    # Arguments
        count_mode: One of "steps" or "samples".
            Whether the progress bar should
            count samples seens or steps (batches) seen.

    # Raises
        ValueError: In case of invalid `count_mode`.
    """

    def __init__(self, count_mode='samples'):
        super(ProgbarLogger_TFRecord, self).__init__()
        if count_mode == 'samples':
            self.use_steps = False
        elif count_mode == 'steps':
            self.use_steps = True
        else:
            raise ValueError('Unknown `count_mode`: ' + str(count_mode))

    def on_train_begin(self, logs=None):  # @UnusedVariable
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):  # @UnusedVariable
        if self.verbose:
            print('Epoch %d/%d' % (epoch + 1, self.epochs))
            if self.use_steps:
                target = self.params['steps']
            else:
                target = self.params['samples']
            self.target = target
            self.progbar = Progbar(target=self.target,
                                   verbose=self.verbose)
        self.seen = 0

    def on_batch_begin(self, batch, logs=None):  # @UnusedVariable
        if self.seen < self.target:
            self.log_values = []

    def on_batch_end(self, batch, logs=None):  # @UnusedVariable
        logs = logs or {}
        batch_size = logs.get('size', 0)
        if self.use_steps:
            self.seen += 1
        else:
            self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and self.seen < self.target:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs=None):  # @UnusedVariable
        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            self.progbar.update(self.seen, self.log_values, force=True)


# TODO: Figure out how to get accuracy to print out in front of the
#    progress bar.
class ModelTFRecordMixin(object):
    def _fit_loop(self, f, ins, out_labels=None, batch_size=32,
                  epochs=100, verbose=1, callbacks=None,
                  val_f=None, val_ins=None, shuffle=True,
                  callback_metrics=None, initial_epoch=0,
                  steps_per_epoch=None):
        """Abstract fit function for `f(ins)`.

        Assume that f returns a list, labeled by out_labels.

        # Arguments
            f: Keras function returning a list of tensors
            ins: list of tensors to be fed to `f`
            out_labels: list of strings, display names of
                the outputs of `f`
            batch_size: integer batch size
            epochs: number of times to iterate over the data
            verbose: verbosity mode, 0, 1 or 2
            callbacks: list of callbacks to be called during training
            val_f: Keras function to call for validation
            val_ins: list of tensors to be fed to `val_f`
            shuffle: whether to shuffle the data at the beginning of each epoch
            callback_metrics: list of strings, the display names of the metrics
                passed to the callbacks. They should be the
                concatenation of list the display names of the outputs of
                 `f` and the list of display names of the outputs of `f_val`.
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)
            steps_per_epoch: Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. The default `None` is equal to the number
                of unique samples in your dataset divided by the batch
                size, or 1 if that cannot be determined.

        # Returns
            `History` object.
        """
        do_validation = False
        if val_f and val_ins:
            do_validation = True
            if verbose and ins and hasattr(ins[0], 'shape'):
                print('Train on %d samples, validate on %d samples' %
                      (ins[0].shape[0], val_ins[0].shape[0]))

        if steps_per_epoch is not None:
            num_train_samples = steps_per_epoch
        else:
            if ins and hasattr(ins[0], 'shape'):
                num_train_samples = ins[0].shape[0]
            else:
                # May happen if we are running `fit` without Numpy input data,
                # i.e. if all inputs to the models are data tensors
                # instead of placeholders.
                # In that case we will run `fit` over a single batch.
                num_train_samples = batch_size
                verbose = 2
        index_array = np.arange(num_train_samples)

        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
        if verbose:
            # callbacks += [cbks.ProgbarLogger()]
            callbacks += [ProgbarLogger_TFRecord()]
        callbacks = cbks.CallbackList(callbacks)
        out_labels = out_labels or []

        # it's possible to callback a different model than self
        # (used by Sequential models)
        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self

        callbacks.set_model(callback_model)
        callbacks.set_params({
            'batch_size': batch_size,
            'epochs': epochs,
            'samples': num_train_samples,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics or [],
        })
        callbacks.on_train_begin()
        callback_model.stop_training = False
        for cbk in callbacks:
            cbk.validation_data = val_ins

        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            if shuffle == 'batch':
                index_array = _batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = _make_batches(num_train_samples, batch_size)
            epoch_logs = {}
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    if isinstance(ins[-1], float):
                        # Do not slice the training phase flag.
                        ins_batch = \
                            _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
                    else:
                        ins_batch = _slice_arrays(ins, batch_ids)
                except TypeError:
                    raise TypeError('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = f(ins_batch)
                if not isinstance(outs, list):
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)
                if callback_model.stop_training:
                    break

                if batch_index == len(batches) - 1:  # Last batch.
                    if do_validation:
                        val_outs = self._test_loop(val_f, val_ins,
                                                   batch_size=batch_size,
                                                   verbose=0)
                        if not isinstance(val_outs, list):
                            val_outs = [val_outs]
                        # Same labels assumed.
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o
            callbacks.on_epoch_end(epoch, epoch_logs)
            if callback_model.stop_training:
                break
        callbacks.on_train_end()
        return self.history

    def fit_tfrecord(
            self,
            x=None,
            y=None,
            batch_size=32,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            **kwargs):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        # Arguments
            x: Numpy array of training data,
                or list of Numpy arrays if the model has multiple inputs.
                If all inputs in the model are named,
                you can also pass a dictionary
                mapping input names to Numpy arrays.
            y: Numpy array of target data,
                or list of Numpy arrays if the model has multiple outputs.
                If all outputs in the model are named,
                you can also pass a dictionary
                mapping output names to Numpy arrays.
            batch_size: integer. Number of samples per gradient update.
            epochs: integer, the number of times to iterate
                over the training data arrays.
            verbose: 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = verbose, 2 = one log line per epoch.
            callbacks: list of callbacks to be called during training.
                See [callbacks](/callbacks).
            validation_split: float between 0 and 1:
                fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
            validation_data: data on which to evaluate
                the loss and any model metrics
                at the end of each epoch. The model will not
                be trained on this data.
                This could be a tuple (x_val, y_val)
                or a tuple (x_val, y_val, val_sample_weights).
            shuffle: boolean, whether to shuffle the training data
                before each epoch.
            class_weight: optional dictionary mapping
                class indices (integers) to
                a weight (float) to apply to the model's loss for the samples
                from this class during training.
                This can be useful to tell the model to "pay more attention" to
                samples from an under-represented class.
            sample_weight: optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape (samples, sequence_length),
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                sample_weight_mode="temporal" in compile().
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)
            steps_per_epoch: Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. The default `None` is equal to the number
                of unique samples in your dataset divided by the batch
                size, or 1 if that cannot be determined.

        # Returns
            A `History` instance. Its `history` attribute contains
            all information collected during training.

        # Raises
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        # Legacy support
        if 'nb_epoch' in kwargs:
            warnings.warn('The `nb_epoch` argument in `fit` '
                          'has been renamed `epochs`.', stacklevel=2)
            epochs = kwargs.pop('nb_epoch')
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        # Validate user data.
        x, y, sample_weights = self._standardize_user_data(
            x, y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            check_batch_axis=False,
            batch_size=batch_size)
        # Prepare validation data.
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError('When passing validation_data, '
                                 'it must contain 2 (x_val, y_val) '
                                 'or 3 (x_val, y_val, val_sample_weights) '
                                 'items, however it contains %d items' %
                                 len(validation_data))

            val_x, val_y, val_sample_weights = self._standardize_user_data(
                val_x, val_y,
                sample_weight=val_sample_weight,
                check_batch_axis=False,
                batch_size=batch_size)
            self._make_test_function()
            val_f = self.test_function
            if self.uses_learning_phase and \
                    not isinstance(K.learning_phase(), int):
                val_ins = val_x + val_y + val_sample_weights + [0.]
            else:
                val_ins = val_x + val_y + val_sample_weights

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (_slice_arrays(x, 0, split_at),
                        _slice_arrays(x, split_at))
            y, val_y = (_slice_arrays(y, 0, split_at),
                        _slice_arrays(y, split_at))
            sample_weights, val_sample_weights = (
                _slice_arrays(sample_weights, 0, split_at),
                _slice_arrays(sample_weights, split_at))
            self._make_test_function()
            val_f = self.test_function
            if self.uses_learning_phase and \
                    not isinstance(K.learning_phase(), int):
                val_ins = val_x + val_y + val_sample_weights + [0.]
            else:
                val_ins = val_x + val_y + val_sample_weights
        else:
            do_validation = False
            val_f = None
            val_ins = None

        # Prepare input arrays and training function.
        if self.uses_learning_phase and \
                not isinstance(K.learning_phase(), int):
            ins = x + y + sample_weights + [1.]
        else:
            ins = x + y + sample_weights
        self._make_train_function()
        f = self.train_function

        # Prepare display labels.
        out_labels = self._get_deduped_metrics_names()

        if do_validation:
            callback_metrics = \
                copy.copy(out_labels) + ['val_' + n for n in out_labels]
        else:
            callback_metrics = copy.copy(out_labels)

        # Delegate logic to `_fit_loop`.
        return self._fit_loop(f, ins, out_labels=out_labels,
                              batch_size=batch_size, epochs=epochs,
                              verbose=verbose, callbacks=callbacks,
                              val_f=val_f, val_ins=val_ins, shuffle=shuffle,
                              callback_metrics=callback_metrics,
                              initial_epoch=initial_epoch,
                              steps_per_epoch=steps_per_epoch)

