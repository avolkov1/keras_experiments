# Keras Experiments
Experimental Keras libraries and examples.


#### Examples
* [cifar10\_cnn\_mgpu.py](examples/cifar/cifar10_cnn_mgpu.py)

    Cifar10 example with multi-GPU options.


#### Usage
Refer to the example above for detailed usage. Typical usage is to define
a Keras model and then call the model conversion function or class to make it
run on multiple GPUs.

##### Function: make\_parallel
```python
def make_parallel(serial_model, gdev_list, usenccl=False, syncopt=False,
                  enqueue=False):
    '''Given a keras [model], return an equivalent model which parallelizes
    the computation over [ngpus] GPUs.

    Data-Parallel:
    Each GPU gets a slice of the input batch, applies the model on that slice
    and later the outputs of the models are concatenated to a single tensor,
    hence the user sees a model that behaves the same as the original.

    :param Model serial_model: Serial i.e. non-multi GPU Keras model.

    :param list gdev_list: List of gpu devices i.e. ['/gpu:0', '/gpu:1', ...]
        Use function get_available_gpus to get the list of available gpus.

    :param bool usenccl: Use the contrib.nccl Tensorflow library for initial
        parameter synchronization and gradients averaging. Note, the model's
        usenccl option overrides the optimizers usenccl option.
        Default: False

    :param bool syncopt: Synchronize gradients. Requires a multi-gpu optimizer.
        Default: False

    :param bool enqueue: Use StagingArea in the multi-GPU model. Could
        potentially speed up Host-to-Device transfers.
        Produces a warning that kwargs are ignored for Tensorflow. The
        _patch_tf_backend module mokey patches the Function in
        tensorflow_backend to use the enqueue_ops option.
        Default: False

    :returns: Multi-GPU parallelized model. If ngpus < 2 then do nothing and
        return the provided serial_model.
    :rtype: ModelMGPU
    '''
```

Usage summary:

```python
# EXAMPLE
from keras_exp.multigpu import get_available_gpus
from keras_exp.multigpu import make_parallel
# or using class # from keras_exp.multigpu import ModelMGPU

# serial_model = ... # some Keras model
gdev_list = get_available_gpus()
mgpu_model = make_parallel(serial_model, gdev_list)
# or using class
# mgpu_model = ModelMGPU(serial_model=serial_model, gdev_list=gdev_list)
# ... Setup optimizer, compile, etc. Run training.
# mgpu_model.compile(...)
# mgpu_model.fit(...)
```

