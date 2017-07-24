'''
When mixing in these classes requires tensorpack (or tensorpack's dependencies
such as ZeroMQ) to be installed.

I did not observe any speedup from tensorpack's dataflow. Perhaps someone can
fix my implementation. In the meantime it's there for referece. Corresponding
example: cifar10_cnn_mgpu_dflow.py

@author: avolkov
'''
from keras.legacy import interfaces
from keras import callbacks as cbks


__all__ = ('ModelDataflowMixin',)


class ModelDataflowMixin(object):
    @interfaces.legacy_generator_methods_support
    def fit_dataflow(self, dflow,
                     steps_per_epoch,
                     epochs=1,
                     verbose=1,
                     callbacks=None,
                     validation_data=None,
                     validation_steps=None,
                     class_weight=None,
                     max_q_size=10,
                     workers=1,
                     pickle_safe=False,
                     initial_epoch=0):
        """Fits the model on data yielded batch-by-batch by a Python generator.

        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.

        # Arguments
            dflow: a dataflow object a-la-carte Tensorpack.
                The output of the generator must be either
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
                All arrays should contain the same number of samples.
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `steps_per_epoch`
                samples have been seen by the model.
            steps_per_epoch: Total number of steps (batches of samples)
                to yield from `generator` before declaring one epoch
                finished and starting the next epoch. It should typically
                be equal to the number of unique samples if your dataset
                divided by the batch size.
            epochs: integer, total number of iterations on the data.
            verbose: verbosity mode, 0, 1, or 2.
            callbacks: list of callbacks to be called during training.
            validation_data: this can be either
                - a generator for the validation data
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
            validation_steps: Only relevant if `validation_data`
                is a generator. Total number of steps (batches of samples)
                to yield from `generator` before stopping.
            class_weight: dictionary mapping class indices to a weight
                for the class.
            max_q_size: maximum size for the generator queue
            workers: maximum number of processes to spin up
                when using process based threading
            pickle_safe: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
            initial_epoch: epoch at which to start training
                (useful for resuming a previous training run)

        # Returns
            A `History` object.

        # Example

        ```python
            def generate_arrays_from_file(path):
                while 1:
                    f = open(path)
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x1, x2, y = process_line(line)
                        yield ({'input_1': x1, 'input_2': x2}, {'output': y})
                    f.close()

            model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                                steps_per_epoch=10000, epochs=10)
        ```

        # Raises
            ValueError: In case the generator yields
                data in an invalid format.
        """
        # wait_time = 0.01  # in seconds
        epoch = initial_epoch

        do_validation = bool(validation_data)
        self._make_train_function()
        if do_validation:
            self._make_test_function()

        # python 2 has 'next', 3 has '__next__'
        # avoid any explicit version checks
        val_gen = (hasattr(validation_data, 'next') or
                   hasattr(validation_data, '__next__'))
        if val_gen and not validation_steps:
            raise ValueError('When using a generator for validation data, '
                             'you must specify a value for '
                             '`validation_steps`.')

        out_labels = self.metrics_names
        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        # prepare callbacks
        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger(count_mode='steps')]
        callbacks = cbks.CallbackList(callbacks)

        # it's possible to callback a different model than self:
        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self
        callbacks.set_model(callback_model)
        callbacks.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })
        callbacks.on_train_begin()

        if do_validation and not val_gen:
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError('validation_data should be a tuple '
                                 '`(val_x, val_y, val_sample_weight)` '
                                 'or `(val_x, val_y)`. Found: ' +
                                 str(validation_data))
            val_x, val_y, val_sample_weights = self._standardize_user_data(
                val_x, val_y, val_sample_weight)
            for cbk in callbacks:
                cbk.validation_data = val_x + [val_y, val_sample_weights]
        # enqueuer = None

        # TODO: Tensorpack does some kind of acceleratn using
        #     QueueInputTrainer, QueueInput, and EnqueueThread. The
        #     implementation below corresponds to SimpleTrainer which
        #     Tensorpack notes as being slow. I still cannot decipher what
        #     exactly is going on in Tensorpack. For the same per-GPU batchsize
        #     the runtime per epoch seems on par. Perhaps with Tensorpack
        #     implementation using Queue+Thread for datafalow the feed_dict
        #     would be faster. The keras fit_generator does use an enqueuer,
        #     but I did not notice performance difference between using
        #     fit_generator or this mixed-in fit_dataflow method.

        try:
            # enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
            # enqueuer.start(max_q_size=max_q_size, workers=workers)

            dflow.reset_state()
            _generator = dflow.get_data()

            callback_model.stop_training = False
            while epoch < epochs:
                callbacks.on_epoch_begin(epoch)
                steps_done = 0
                batch_index = 0
                while steps_done < steps_per_epoch:
                    # generator_output = None
                    generator_output = next(_generator)
                    # while enqueuer.is_running():
                    #     if not enqueuer.queue.empty():
                    #         generator_output = enqueuer.queue.get()
                    #         break
                    #     else:
                    #         time.sleep(wait_time)

                    if not hasattr(generator_output, '__len__'):
                        raise ValueError('output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
                    if len(generator_output) == 2:
                        x, y = generator_output
                        sample_weight = None
                    elif len(generator_output) == 3:
                        x, y, sample_weight = generator_output
                    else:
                        raise ValueError('output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
                    # build batch logs
                    batch_logs = {}
                    if isinstance(x, list):
                        batch_size = x[0].shape[0]
                    elif isinstance(x, dict):
                        batch_size = list(x.values())[0].shape[0]
                    else:
                        batch_size = x.shape[0]
                    batch_logs['batch'] = batch_index
                    batch_logs['size'] = batch_size
                    callbacks.on_batch_begin(batch_index, batch_logs)

                    outs = self.train_on_batch(x, y,
                                               sample_weight=sample_weight,
                                               class_weight=class_weight)

                    if not isinstance(outs, list):
                        outs = [outs]
                    for l, o in zip(out_labels, outs):
                        batch_logs[l] = o

                    callbacks.on_batch_end(batch_index, batch_logs)

                    # Construct epoch logs.
                    epoch_logs = {}
                    batch_index += 1
                    steps_done += 1

                    # Epoch finished.
                    if steps_done >= steps_per_epoch and do_validation:
                        if val_gen:
                            val_outs = self.evaluate_generator(
                                validation_data,
                                validation_steps,
                                max_q_size=max_q_size,
                                workers=workers,
                                pickle_safe=pickle_safe)
                        else:
                            # No need for try/except because
                            # data has already been validated.
                            val_outs = self.evaluate(
                                val_x, val_y,
                                batch_size=batch_size,
                                sample_weight=val_sample_weights,
                                verbose=0)
                        if not isinstance(val_outs, list):
                            val_outs = [val_outs]
                        # Same labels assumed.
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o

                callbacks.on_epoch_end(epoch, epoch_logs)
                epoch += 1
                if callback_model.stop_training:
                    break

        finally:
            # if enqueuer is not None:
            #     enqueuer.stop()
            pass

        callbacks.on_train_end()
        return self.history

